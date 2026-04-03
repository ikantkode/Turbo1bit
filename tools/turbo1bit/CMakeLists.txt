add_executable(turbo1bit-stress
    turbo1bit_stress.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit-stress PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(turbo1bit-stress PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-stress PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-stress PRIVATE turbo1bit-metal)
elseif(UNIX AND NOT APPLE AND TARGET turbo1bit-hip)
    target_link_libraries(turbo1bit-stress PRIVATE turbo1bit-hip)
endif()
set_target_properties(turbo1bit-stress PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── GPU acceleration libraries ───────────────────────────────────────

if(APPLE)
    # Metal acceleration (macOS)
    add_library(turbo1bit-metal STATIC
        ${CMAKE_SOURCE_DIR}/src/turbo1bit_metal.m
    )
    target_include_directories(turbo1bit-metal PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(turbo1bit-metal PRIVATE
        "-framework Metal"
        "-framework Foundation"
    )
    target_compile_options(turbo1bit-metal PRIVATE -fobjc-arc)
    set_source_files_properties(
        ${CMAKE_SOURCE_DIR}/src/turbo1bit_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Metal shaders compiled at runtime from source (no Metal toolchain needed)
elseif(UNIX AND NOT APPLE)
    # ROCm/HIP acceleration (Linux)
    find_program(HIPCC hipcc HINTS "/opt/rocm/bin" "/opt/rocm-7.0.0/bin" "$ENV{ROCM_PATH}/bin")

    if(HIPCC)
        find_library(AMDHIP64_LIB amdhip64 HINTS "/opt/rocm/lib" "/opt/rocm-7.0.0/lib")

        if(AMDHIP64_LIB)
            add_library(turbo1bit-hip STATIC
                ${CMAKE_SOURCE_DIR}/src/turbo1bit_hip.cpp
            )
            target_include_directories(turbo1bit-hip PRIVATE
                ${CMAKE_SOURCE_DIR}/src
                /opt/rocm/include
                /opt/rocm-7.0.0/include
            )
            target_link_libraries(turbo1bit-hip PRIVATE
                ${AMDHIP64_LIB}
            )

            # Set HIP architecture for MI50 (gfx906)
            target_compile_definitions(turbo1bit-hip PRIVATE
                __HIP_ARCH_GFX906__=1
            )
        endif()
    endif()
endif()

add_executable(turbo1bit-debug
    turbo1bit_debug.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)
target_include_directories(turbo1bit-debug PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(turbo1bit-debug PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-debug PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-debug PRIVATE turbo1bit-metal)
elseif(UNIX AND NOT APPLE AND TARGET turbo1bit-hip)
    target_link_libraries(turbo1bit-debug PRIVATE turbo1bit-hip)
endif()
set_target_properties(turbo1bit-debug PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(turbo1bit-bench
    turbo1bit_bench.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_codebook.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_rotation.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_quantizer.c
    ${CMAKE_SOURCE_DIR}/src/turbo1bit_kv_cache.c
)

target_include_directories(turbo1bit-bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Link math library and Accelerate (macOS) or ROCm (Linux) for LAPACK
target_link_libraries(turbo1bit-bench PRIVATE m)
if(APPLE)
    target_link_libraries(turbo1bit-bench PRIVATE "-framework Accelerate")
    target_link_libraries(turbo1bit-bench PRIVATE turbo1bit-metal)
elseif(UNIX AND NOT APPLE AND TARGET turbo1bit-hip)
    target_link_libraries(turbo1bit-bench PRIVATE turbo1bit-hip)
endif()

set_target_properties(turbo1bit-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
