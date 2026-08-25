cmake_minimum_required(VERSION 3.20)

project(ime_llama LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(IME_ARCH "rv64gcv_zvfh_zicbop_zba_xsmtvdot" CACHE STRING
    "RISC-V architecture used by the SpacemiT IME kernels")
set(IME_TUNE "spacemit-x60" CACHE STRING
    "RISC-V CPU tuning target")

add_library(ime_compile_options INTERFACE)
target_compile_features(ime_compile_options INTERFACE cxx_std_20)
target_compile_definitions(ime_compile_options INTERFACE SPACEMIT_X60)
target_compile_options(ime_compile_options INTERFACE
    -march=${IME_ARCH}
    -mtune=${IME_TUNE}
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)
target_include_directories(ime_compile_options INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(ime-llama-bench
    src/ime.cpp
    src/ime1_kernels.cpp
    src/selected_cycles.cpp
    bench/main.cpp
)
target_include_directories(ime-llama-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
)
target_link_libraries(ime-llama-bench PRIVATE ime_compile_options)
target_compile_options(ime-llama-bench PRIVATE -Wall -Wextra)

# Compile-check experimental kernels without linking them into the benchmark.
add_library(ime-experiments OBJECT
    experiments/m4_batch_reduction.cpp
    experiments/m8_batch_reduction.cpp
)
target_link_libraries(ime-experiments PRIVATE ime_compile_options)
target_compile_options(ime-experiments PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-unused-but-set-variable
)

set(IME_ASM_DIR ${CMAKE_CURRENT_BINARY_DIR}/asm)
set(IME_ASM_OUTPUTS)
foreach(source IN ITEMS
    src/ime.cpp
    src/ime1_kernels.cpp
    src/selected_cycles.cpp
    bench/main.cpp
)
    get_filename_component(stem ${source} NAME_WE)
    set(output ${IME_ASM_DIR}/${stem}.s)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${IME_ASM_DIR}
        COMMAND ${CMAKE_CXX_COMPILER}
            -std=c++20
            -DSPACEMIT_X60
            -march=${IME_ARCH}
            -mtune=${IME_TUNE}
            -mllvm -riscv-v-vector-bits-min=256
            -mllvm -riscv-v-vector-bits-max=256
            -O3
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench
            -S ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -o ${output}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ggml.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ime.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ime1.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/selected_cycles.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/perf.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/timer.hpp
        VERBATIM
        COMMENT "Generating ${output}"
    )
    list(APPEND IME_ASM_OUTPUTS ${output})
endforeach()

add_custom_target(assembly DEPENDS ${IME_ASM_OUTPUTS})