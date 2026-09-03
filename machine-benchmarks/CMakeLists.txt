add_executable(k1-l2-bandwidth k1_l2_bandwidth.cpp)
target_compile_features(k1-l2-bandwidth PRIVATE cxx_std_20)
target_compile_options(k1-l2-bandwidth PRIVATE
    -O3 -Wall -Wextra
    -march=rv64gcv_zba_zbb
    -mtune=spacemit-x60
)
target_link_libraries(k1-l2-bandwidth PRIVATE pthread)

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