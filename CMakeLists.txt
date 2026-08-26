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

add_library(ime-kernels STATIC
    kernels/common/adapter_common.cpp
    kernels/llama_dispatch/wrapper.cpp
    kernels/llama_dispatch/ime1_kernels.cpp
    kernels/llama_dispatch/adapter.cpp
    kernels/m4_batch_reduction/kernel.cpp
    kernels/m4_batch_reduction/adapter.cpp
    kernels/m4_immediate_reduction/kernel.cpp
    kernels/m4_immediate_reduction/adapter.cpp
    kernels/m8_batch_reduction/kernel.cpp
    kernels/m8_batch_reduction/adapter.cpp
)
target_include_directories(ime-kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
)
target_link_libraries(ime-kernels PRIVATE ime_compile_options)
target_compile_options(ime-kernels PRIVATE -Wall -Wextra)

set_source_files_properties(
    kernels/m4_batch_reduction/kernel.cpp
    kernels/m4_immediate_reduction/kernel.cpp
    kernels/m8_batch_reduction/kernel.cpp
    PROPERTIES COMPILE_OPTIONS
    "-Wno-unused-parameter;-Wno-unused-variable;-Wno-unused-but-set-variable"
)

add_executable(ime-llama-bench
    bench/main.cpp
    bench/framework/kernel_registry.cpp
    bench/framework/benchmark.cpp
    bench/llama_reference/llama_reference.cpp
)
target_include_directories(ime-llama-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
)
target_link_libraries(ime-llama-bench PRIVATE ime_compile_options ime-kernels)
target_compile_options(ime-llama-bench PRIVATE -Wall -Wextra)

set(IME_ASM_DIR ${CMAKE_CURRENT_BINARY_DIR}/asm)
set(IME_ASM_OUTPUTS)
function(add_ime_assembly source stem)
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
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
            -I${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
            -S ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -o ${output}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ggml.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ime.h
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ime1.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/perf.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/timer.hpp
        VERBATIM
        COMMENT "Generating ${output}"
    )
    set(IME_ASM_OUTPUTS ${IME_ASM_OUTPUTS} ${output} PARENT_SCOPE)
endfunction()

add_ime_assembly(kernels/llama_dispatch/wrapper.cpp llama_dispatch_wrapper)
add_ime_assembly(kernels/llama_dispatch/ime1_kernels.cpp llama_dispatch_kernels)
add_ime_assembly(kernels/m4_batch_reduction/kernel.cpp m4_batch_reduction)
add_ime_assembly(kernels/m4_immediate_reduction/kernel.cpp m4_immediate_reduction)
add_ime_assembly(kernels/m8_batch_reduction/kernel.cpp m8_batch_reduction)
add_ime_assembly(bench/main.cpp main)

add_custom_target(assembly DEPENDS ${IME_ASM_OUTPUTS})