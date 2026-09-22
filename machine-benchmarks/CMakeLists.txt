if(NOT PFNG_TARGET_PLATFORM STREQUAL "spacemit-k1-a60")
    message(STATUS
        "Machine benchmarks are unavailable for ${PFNG_TARGET_PLATFORM}")
    return()
endif()

# All targets in this directory characterize the SpacemiT K1/A60 machine and
# are intentionally excluded from every other platform build.
add_executable(k1-l2-bandwidth k1_l2_bandwidth.cpp)
target_compile_features(k1-l2-bandwidth PRIVATE cxx_std_20)
target_compile_options(k1-l2-bandwidth PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
)
target_link_libraries(k1-l2-bandwidth PRIVATE pthread)

add_executable(dynpreunpack-inner-loop
    dynpreunpack_inner_loop.cpp
    dynpreunpack_inner_loop.S
)
target_compile_features(dynpreunpack-inner-loop PRIVATE cxx_std_20)
target_compile_options(dynpreunpack-inner-loop PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dynpreunpack-inner-loop PRIVATE pthread)

add_executable(m8b-inner-issue
    m8b_inner_issue.cpp
    m8b_inner_issue.S
)
target_compile_features(m8b-inner-issue PRIVATE cxx_std_20)
target_compile_options(m8b-inner-issue PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(m8b-inner-issue PRIVATE pthread)

add_executable(dynpreunpack-full-realistic
    dynpreunpack_full_realistic.cpp
    dynpreunpack_full_realistic.S
)
target_compile_features(dynpreunpack-full-realistic PRIVATE cxx_std_20)
target_compile_options(dynpreunpack-full-realistic PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dynpreunpack-full-realistic PRIVATE pthread)

add_executable(unpack-address-conflict
    unpack_address_conflict.cpp
    unpack_address_conflict.S
)
target_compile_features(unpack-address-conflict PRIVATE cxx_std_20)
target_compile_options(unpack-address-conflict PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(unpack-address-conflict PRIVATE pthread)

add_executable(perf-pc-semantics
    perf_pc_semantics.cpp
    perf_pc_semantics.S
)
target_compile_features(perf-pc-semantics PRIVATE cxx_std_20)
target_compile_options(perf-pc-semantics PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_options(perf-pc-semantics PRIVATE -no-pie)
target_link_libraries(perf-pc-semantics PRIVATE pthread)

add_executable(prefetch-threshold
    prefetch_threshold.cpp
    prefetch_threshold.S
)
target_compile_features(prefetch-threshold PRIVATE cxx_std_20)
target_compile_options(prefetch-threshold PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
)
target_link_libraries(prefetch-threshold PRIVATE pthread)

add_executable(prefetch-multistream
    prefetch_multistream.cpp
    prefetch_multistream.S
)
target_compile_features(prefetch-multistream PRIVATE cxx_std_20)
target_compile_options(prefetch-multistream PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
)
target_link_libraries(prefetch-multistream PRIVATE pthread)

add_executable(dpu-cb-prefetch
    dpu_cb_prefetch.cpp
    dpu_cb_prefetch.S
)
target_compile_features(dpu-cb-prefetch PRIVATE cxx_std_20)
target_compile_options(dpu-cb-prefetch PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dpu-cb-prefetch PRIVATE pthread)

add_executable(dpu-cb-inner-issue
    dpu_cb_inner_issue.cpp
    dpu_cb_inner_issue.S
)
target_compile_features(dpu-cb-inner-issue PRIVATE cxx_std_20)
target_compile_options(dpu-cb-inner-issue PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dpu-cb-inner-issue PRIVATE pthread)

add_executable(m8b-aw-miss
    m8b_aw_miss.cpp
    m8b_aw_miss.S
)
target_compile_features(m8b-aw-miss PRIVATE cxx_std_20)
target_compile_options(m8b-aw-miss PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_options(m8b-aw-miss PRIVATE -no-pie)
target_link_libraries(m8b-aw-miss PRIVATE pthread)

add_executable(k1-memory-latency
    k1_memory_latency.cpp
    k1_memory_latency.S
)
target_compile_features(k1-memory-latency PRIVATE cxx_std_20)
target_compile_options(k1-memory-latency PRIVATE
    -O3 -Wall -Wextra
    -march=${PFNG_PLATFORM_ARCH}
    -mtune=${PFNG_PLATFORM_TUNE}
)
target_link_libraries(k1-memory-latency PRIVATE pthread)
