add_executable(k1-l2-bandwidth k1_l2_bandwidth.cpp)
target_compile_features(k1-l2-bandwidth PRIVATE cxx_std_20)
target_compile_options(k1-l2-bandwidth PRIVATE
    -O3 -Wall -Wextra
    -march=rv64gcv_zba_zbb
    -mtune=spacemit-x60
)
target_link_libraries(k1-l2-bandwidth PRIVATE pthread)

add_executable(dynpreunpack-inner-loop
    dynpreunpack_inner_loop.cpp
    dynpreunpack_inner_loop.S
)
target_compile_features(dynpreunpack-inner-loop PRIVATE cxx_std_20)
target_compile_options(dynpreunpack-inner-loop PRIVATE
    -O3 -Wall -Wextra
    -march=${IME_ARCH}
    -mtune=${IME_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dynpreunpack-inner-loop PRIVATE pthread)

add_executable(dynpreunpack-full-realistic
    dynpreunpack_full_realistic.cpp
    dynpreunpack_full_realistic.S
)
target_compile_features(dynpreunpack-full-realistic PRIVATE cxx_std_20)
target_compile_options(dynpreunpack-full-realistic PRIVATE
    -O3 -Wall -Wextra
    -march=${IME_ARCH}
    -mtune=${IME_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_libraries(dynpreunpack-full-realistic PRIVATE pthread)

add_executable(perf-pc-semantics
    perf_pc_semantics.cpp
    perf_pc_semantics.S
)
target_compile_features(perf-pc-semantics PRIVATE cxx_std_20)
target_compile_options(perf-pc-semantics PRIVATE
    -O3 -Wall -Wextra
    -march=${IME_ARCH}
    -mtune=${IME_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_link_options(perf-pc-semantics PRIVATE -no-pie)
target_link_libraries(perf-pc-semantics PRIVATE pthread)

add_executable(k1-memory-path-bandwidth
    k1_memory_path_bandwidth.cpp
    k1_memory_path_load.S
)
target_compile_features(k1-memory-path-bandwidth PRIVATE cxx_std_20)
target_compile_options(k1-memory-path-bandwidth PRIVATE
    -O3 -Wall -Wextra
    -march=rv64gcv_zba_zbb
    -mtune=spacemit-x60
)
target_include_directories(k1-memory-path-bandwidth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tcm
)
target_link_libraries(k1-memory-path-bandwidth PRIVATE
    pthread
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tcm/libfine_tcm_mm.a
)