cmake_minimum_required(VERSION 3.16)
project(Turbo1Bit C)

set(CMAKE_C_STANDARD 11)

# ── Turbo1Bit core library ──────────────────────────────────────────

add_library(turbo1bit STATIC
    src/turbo1bit_codebook.c
    src/turbo1bit_rotation.c
    src/turbo1bit_quantizer.c
    src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit PUBLIC src)
target_link_libraries(turbo1bit PRIVATE m)

if(APPLE)
    target_link_libraries(turbo1bit PRIVATE "-framework Accelerate")
endif()

# ── GPU acceleration ────────────────────────────────────────────────

if(APPLE)
    # Metal acceleration (macOS)
    enable_language(OBJC)
    add_library(turbo1bit-metal STATIC src/turbo1bit_metal.m)
    target_include_directories(turbo1bit-metal PUBLIC src)
    target_link_libraries(turbo1bit-metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    set_source_files_properties(src/turbo1bit_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
elseif(UNIX AND NOT APPLE)
    # ROCm/HIP acceleration (Linux)
    # Find HIP compiler directly
    find_program(HIPCC hipcc HINTS "/opt/rocm/bin" "/opt/rocm-7.0.0/bin" "$ENV{ROCM_PATH}/bin")

    if(HIPCC)
        # Find HIP libraries
        find_library(AMDHIP64_LIB amdhip64 HINTS "/opt/rocm/lib" "/opt/rocm-7.0.0/lib")

        if(AMDHIP64_LIB)
            # Enable CXX language for HIP
            enable_language(CXX)

            # Find hiprtc library
            find_library(HIPRTC_LIB hiprtc HINTS "/opt/rocm/lib" "/opt/rocm-7.0.0/lib")

            # Create HIP library
            add_library(turbo1bit-hip STATIC
                src/turbo1bit_hip.cpp
            )
            target_include_directories(turbo1bit-hip PUBLIC
                src
                /opt/rocm/include
                /opt/rocm-7.0.0/include
            )
            target_link_libraries(turbo1bit-hip PRIVATE
                ${AMDHIP64_LIB}
                ${HIPRTC_LIB}
            )

            # Set HIP architecture for MI50 (gfx906)
            target_compile_definitions(turbo1bit-hip PRIVATE
                __HIP_ARCH_GFX906__=1
                __HIP_PLATFORM_AMD__=1
            )

            message(STATUS "ROCm/HIP support enabled for AMD GPU acceleration")
            message(STATUS "  HIP compiler: ${HIPCC}")
            message(STATUS "  HIP library: ${AMDHIP64_LIB}")
            if(HIPRTC_LIB)
                message(STATUS "  HIPRTC library: ${HIPRTC_LIB}")
            endif()
        else()
            message(WARNING "AMD HIP64 library not found. GPU acceleration disabled.")
        endif()
    else()
        message(WARNING "HIP compiler not found. GPU acceleration disabled.")
    endif()
endif()

# ── Benchmark executables ───────────────────────────────────────────

add_executable(turbo1bit-bench tools/turbo1bit/turbo1bit_bench.c)
target_link_libraries(turbo1bit-bench PRIVATE turbo1bit m)
target_include_directories(turbo1bit-bench PRIVATE src)

add_executable(turbo1bit-stress tools/turbo1bit/turbo1bit_stress.c)
target_link_libraries(turbo1bit-stress PRIVATE turbo1bit m)
target_include_directories(turbo1bit-stress PRIVATE src)

# Link GPU acceleration library based on platform
if(APPLE)
    target_link_libraries(turbo1bit-bench PRIVATE turbo1bit-metal)
    target_link_libraries(turbo1bit-stress PRIVATE turbo1bit-metal)
    target_link_libraries(turbo1bit-bench PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-stress PRIVATE "-framework Accelerate")
elseif(UNIX AND NOT APPLE AND TARGET turbo1bit-hip)
    target_link_libraries(turbo1bit-bench PRIVATE turbo1bit-hip)
    target_link_libraries(turbo1bit-stress PRIVATE turbo1bit-hip)
endif()
