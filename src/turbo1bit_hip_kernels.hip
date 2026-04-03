/**
 * turbo1bit_hip_kernels.hip — HIP compute kernels for Turbo1Bit compressed KV cache.
 *
 * Ported from turbo1bit_metal.metal for AMD ROCm/HIP support.
 *
 * Six kernels for compressed KV cache operations:
 *   1. t1b_hip_mse_score:     Compute attention scores against MSE-quantized keys
 *   2. t1b_hip_qjl_score:     Add QJL residual contribution to scores
 *   3. t1b_hip_fused_attn:    Fused decode attention with online softmax
 *   4. t1b_hip_dequant_values: Unpack and dequantize group-quantized values
 *   5. t1b_hip_value_qd_inplace: In-place value quantize-dequantize (group quantization roundtrip)
 *   6. t1b_hip_matvec:        Matrix-vector multiply for rotation/projection
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cmath>

extern "C" {

// ── Constants ───────────────────────────────────────────────────────

// Lloyd-Max codebook centroids for d=128
// 2-bit (4 centroids) — used for MSE stage of 3-bit Prod quantizer
__constant__ float centroids_d128_b2[4] = {
    -0.1330401982533685f, -0.039990945215356365f,
     0.039990945215356365f,  0.1330401982533685f
};

// QJL scale constant: sqrt(pi/2) / d
__constant__ float QJL_SCALE_128 = 0.009801714f;  // sqrt(pi/2) / 128

// ── Kernel 1: MSE Score ─────────────────────────────────────────────
//
// Computes: score[t] = sum_j query_rot[j] * centroids[indices[t][j]] * norms[t]
// where query_rot = query @ Pi^T (pre-rotated by host)
//
// Each thread handles one KV token.

__global__ void t1b_hip_mse_score(
    const float    *query_rot,     // [head_dim]
    const uint8_t  *mse_packed,    // [n_tokens, packed_dim]
    const float    *norms,         // [n_tokens]
    float          *scores,        // [n_tokens]
    const unsigned int n_tokens,
    const unsigned int head_dim,
    const unsigned int packed_dim,
    const unsigned int bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tokens) return;

    unsigned int vals_per_byte = 8 / bits;
    unsigned int mask = (1u << bits) - 1u;

    float dot = 0.0f;
    const uint8_t *packed = mse_packed + tid * packed_dim;

    for (unsigned int byte_idx = 0; byte_idx < packed_dim; byte_idx++) {
        uint8_t byte_val = packed[byte_idx];
        for (unsigned int v = 0; v < vals_per_byte; v++) {
            unsigned int coord = byte_idx * vals_per_byte + v;
            if (coord >= head_dim) break;

            unsigned int idx = (byte_val >> (v * bits)) & mask;
            // Use the 2-bit codebook (for 3-bit Prod = 2-bit MSE + 1-bit QJL)
            float centroid = centroids_d128_b2[idx];
            dot += query_rot[coord] * centroid;
        }
    }

    scores[tid] = dot * norms[tid];
}

// ── Kernel 2: QJL Score ─────────────────────────────────────────────
//
// Adds QJL residual contribution to existing MSE scores:
//   scores[t] += (sqrt(pi/2)/d) * residual_norms[t] * sum_j q_sketch[j] * sign[t][j]
//
// q_sketch = query @ S^T (pre-computed by host)

__global__ void t1b_hip_qjl_score(
    const float    *q_sketch,       // [head_dim]
    const uint8_t  *qjl_packed,     // [n_tokens, sign_packed_dim]
    const float    *residual_norms, // [n_tokens]
    float          *scores,         // [n_tokens] (accumulated)
    const unsigned int n_tokens,
    const unsigned int head_dim,
    const unsigned int sign_packed_dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tokens) return;

    float dot = 0.0f;
    const uint8_t *packed = qjl_packed + tid * sign_packed_dim;

    for (unsigned int byte_idx = 0; byte_idx < sign_packed_dim; byte_idx++) {
        uint8_t byte_val = packed[byte_idx];
        for (unsigned int b = 0; b < 8; b++) {
            unsigned int coord = byte_idx * 8 + b;
            if (coord >= head_dim) break;

            float sign = ((byte_val >> b) & 1) ? 1.0f : -1.0f;
            dot += q_sketch[coord] * sign;
        }
    }

    scores[tid] += QJL_SCALE_128 * residual_norms[tid] * dot;
}

// ── Kernel 3: Fused Decode Attention ────────────────────────────────
//
// Full fused attention kernel with online softmax:
//   1. Compute MSE + QJL scores per KV token
//   2. Online softmax
//   3. Dequantize values and accumulate weighted output
//
// This kernel processes one head. Currently simplified to single-thread
// per head for correctness matching Metal implementation.

__global__ void t1b_hip_fused_attn(
    // Query (pre-transformed)
    const float    *query_rot,      // [head_dim]
    const float    *q_sketch,       // [head_dim]
    // Compressed keys
    const uint8_t  *mse_packed,     // [n_tokens, mse_packed_dim]
    const uint8_t  *qjl_packed,     // [n_tokens, sign_packed_dim]
    const float    *key_norms,      // [n_tokens]
    const float    *residual_norms, // [n_tokens]
    // Compressed values
    const uint8_t  *val_packed,     // [n_tokens, val_packed_dim]
    const float    *val_scales,     // [n_tokens, n_groups]
    const float    *val_zeros,      // [n_tokens, n_groups]
    // Output
    float          *output,         // [head_dim]
    // Params
    const unsigned int n_compressed,
    const unsigned int head_dim,
    const unsigned int mse_packed_dim,
    const unsigned int sign_packed_dim,
    const unsigned int val_packed_dim,
    const unsigned int n_groups,
    const unsigned int group_size,
    const float     attn_scale,
    // Buffer KV (full precision, appended after compressed)
    const float    *buf_keys,       // [n_buffered, head_dim]
    const float    *buf_values,     // [n_buffered, head_dim]
    const unsigned int n_buffered)
{
    // Simplified single-thread implementation matching Metal
    unsigned int head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (head_idx != 0) return; // Only first thread does work (simplified)

    unsigned int total_tokens = n_compressed + n_buffered;
    if (total_tokens == 0) {
        for (unsigned int d = 0; d < head_dim; d++) output[d] = 0.0f;
        return;
    }

    // Phase 1: Compute all scores and accumulate output
    float m_i = -INFINITY;
    float l_i = 0.0f;

    // Initialize accumulator
    float acc[128]; // max head_dim
    for (unsigned int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    // Process compressed tokens
    for (unsigned int t = 0; t < n_compressed; t++) {
        // MSE score
        float score = 0.0f;
        const uint8_t *mp = mse_packed + t * mse_packed_dim;
        for (unsigned int byte_idx = 0; byte_idx < mse_packed_dim; byte_idx++) {
            uint8_t bv = mp[byte_idx];
            for (unsigned int v = 0; v < 4; v++) { // 2-bit = 4 per byte
                unsigned int coord = byte_idx * 4 + v;
                if (coord >= head_dim) break;
                unsigned int idx = (bv >> (v * 2)) & 0x3;
                score += query_rot[coord] * centroids_d128_b2[idx];
            }
        }
        score *= key_norms[t];

        // QJL score
        float qjl_dot = 0.0f;
        const uint8_t *qp = qjl_packed + t * sign_packed_dim;
        for (unsigned int byte_idx = 0; byte_idx < sign_packed_dim; byte_idx++) {
            uint8_t bv = qp[byte_idx];
            for (unsigned int b = 0; b < 8; b++) {
                unsigned int coord = byte_idx * 8 + b;
                if (coord >= head_dim) break;
                float sign = ((bv >> b) & 1) ? 1.0f : -1.0f;
                qjl_dot += q_sketch[coord] * sign;
            }
        }
        score += QJL_SCALE_128 * residual_norms[t] * qjl_dot;
        score *= attn_scale;

        // Online softmax update
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        for (unsigned int d = 0; d < head_dim; d++) acc[d] *= alpha;

        // Dequantize value and accumulate
        const uint8_t *vp = val_packed + t * val_packed_dim;
        const float *vs = val_scales + t * n_groups;
        const float *vz = val_zeros + t * n_groups;
        for (unsigned int g = 0; g < n_groups; g++) {
            float sc = vs[g];
            float zr = vz[g];
            unsigned int base = g * group_size;
            for (unsigned int i = 0; i < group_size; i += 4) {
                unsigned int byte_off = (base + i) / 4;
                uint8_t bv = vp[byte_off];
                acc[base + i + 0] += p * (float((bv >> 0) & 0x3) * sc + zr);
                acc[base + i + 1] += p * (float((bv >> 2) & 0x3) * sc + zr);
                acc[base + i + 2] += p * (float((bv >> 4) & 0x3) * sc + zr);
                acc[base + i + 3] += p * (float((bv >> 6) & 0x3) * sc + zr);
            }
        }

        m_i = m_new;
    }

    // Process buffered tokens (full precision)
    for (unsigned int t = 0; t < n_buffered; t++) {
        float score = 0.0f;
        const float *k = buf_keys + t * head_dim;
        for (unsigned int d = 0; d < head_dim; d++) {
            score += query_rot[d] * k[d]; // Note: buf keys are NOT rotated
        }
        score *= attn_scale;

        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float p = expf(score - m_new);
        l_i = l_i * alpha + p;
        for (unsigned int d = 0; d < head_dim; d++) acc[d] *= alpha;

        const float *v = buf_values + t * head_dim;
        for (unsigned int d = 0; d < head_dim; d++) {
            acc[d] += p * v[d];
        }

        m_i = m_new;
    }

    // Normalize
    float inv_l = 1.0f / (l_i + 1e-10f);
    for (unsigned int d = 0; d < head_dim; d++) {
        output[d] = acc[d] * inv_l;
    }
}

// ── Kernel 4: Value Dequantization ──────────────────────────────────
//
// Unpacks and dequantizes group-quantized values.
// Each thread handles one token's value vector.

__global__ void t1b_hip_dequant_values(
    const uint8_t  *packed,         // [n_tokens, packed_dim]
    const float    *scales,         // [n_tokens, n_groups]
    const float    *zeros,          // [n_tokens, n_groups]
    float          *output,         // [n_tokens, head_dim]
    const unsigned int n_tokens,
    const unsigned int head_dim,
    const unsigned int packed_dim,
    const unsigned int n_groups,
    const unsigned int group_size,
    const unsigned int bits)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tokens) return;

    const uint8_t *p = packed + tid * packed_dim;
    const float *s = scales + tid * n_groups;
    const float *z = zeros + tid * n_groups;
    float *out = output + tid * head_dim;

    if (bits == 2) {
        for (unsigned int g = 0; g < n_groups; g++) {
            float scale = s[g];
            float zero = z[g];
            unsigned int base = g * group_size;

            for (unsigned int i = 0; i < group_size; i += 4) {
                unsigned int byte_idx = (base + i) / 4;
                uint8_t byte_val = p[byte_idx];

                out[base + i + 0] = float((byte_val >> 0) & 0x3) * scale + zero;
                out[base + i + 1] = float((byte_val >> 2) & 0x3) * scale + zero;
                out[base + i + 2] = float((byte_val >> 4) & 0x3) * scale + zero;
                out[base + i + 3] = float((byte_val >> 6) & 0x3) * scale + zero;
            }
        }
    } else if (bits == 4) {
        for (unsigned int g = 0; g < n_groups; g++) {
            float scale = s[g];
            float zero = z[g];
            unsigned int base = g * group_size;

            for (unsigned int i = 0; i < group_size; i += 2) {
                unsigned int byte_idx = (base + i) / 2;
                uint8_t byte_val = p[byte_idx];

                out[base + i + 0] = float(byte_val & 0xF) * scale + zero;
                out[base + i + 1] = float((byte_val >> 4) & 0xF) * scale + zero;
            }
        }
    }
}

// ── Kernel 5: In-place Value Quantize-Dequantize ────────────────────
//
// Performs group quantization then dequantization in-place on FP16 data.
// This simulates compression: values are replaced with their lossy reconstructed versions.
// Each thread handles one group of `group_size` elements.

__global__ void t1b_hip_value_qd_inplace(
    __half          *data,           // [n_elements] FP16 in/out
    const unsigned int n_elements,
    const unsigned int group_size,
    const unsigned int bits)
{
    unsigned int n_groups = n_elements / group_size;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) return;

    unsigned int base = gid * group_size;
    unsigned int n_levels = (1u << bits) - 1u;

    // Find min/max of group
    __half h_min = data[base];
    __half h_max = h_min;
    float vmin = __half2float(h_min);
    float vmax = vmin;

    for (unsigned int i = 1; i < group_size; i++) {
        float v = __half2float(data[base + i]);
        vmin = fminf(vmin, v);
        vmax = fmaxf(vmax, v);
    }

    float scale = (vmax - vmin) / float(n_levels);
    if (scale < 1e-10f) scale = 1e-10f;

    // Quantize then immediately dequantize
    for (unsigned int i = 0; i < group_size; i++) {
        float v = __half2float(data[base + i]);
        float normalized = (v - vmin) / scale;
        unsigned int q = (unsigned int)(fminf(normalized + 0.5f, float(n_levels)));
        data[base + i] = __float2half(float(q) * scale + vmin);
    }
}

// ── Kernel 6: Matrix-Vector Multiply (for rotation) ─────────────────
//
// Computes y = x @ M^T where M is a d x d matrix (rotation or QJL).
// Used for: query rotation (x @ Pi^T) and query sketching (x @ S^T).

__global__ void t1b_hip_matvec(
    const float *x,                 // [dim]
    const float *M,                 // [dim, dim] row-major
    float       *y,                 // [dim]
    const unsigned int   dim)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim) return;

    // y[tid] = sum_j x[j] * M[tid][j]  (= x @ M^T row tid)
    float sum = 0.0f;
    const float *row = M + tid * dim;
    for (unsigned int j = 0; j < dim; j++) {
        sum += x[j] * row[j];
    }
    y[tid] = sum;
}

} // extern "C"
