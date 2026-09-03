cmake_minimum_required(VERSION 3.20)

project(ime_llama LANGUAGES CXX ASM)

add_subdirectory(machine-benchmarks)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(IME_ARCH "rv64gcv_zvfh_zicbop_zba_xsmtvdot" CACHE STRING
    "RISC-V architecture used by the SpacemiT IME kernels")
set(IME_TUNE "spacemit-x60" CACHE STRING
    "RISC-V CPU tuning target")

add_library(ime_base_compile_options INTERFACE)
target_compile_features(ime_base_compile_options INTERFACE cxx_std_20)
target_compile_definitions(ime_base_compile_options INTERFACE SPACEMIT_X60)
target_compile_options(ime_base_compile_options INTERFACE
    -march=${IME_ARCH}
    -mtune=${IME_TUNE}
    "SHELL:-mllvm --misched-cluster=false"
)
target_include_directories(ime_base_compile_options INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(ime_compile_options INTERFACE)
target_link_libraries(ime_compile_options INTERFACE ime_base_compile_options)
target_compile_options(ime_compile_options INTERFACE
    "SHELL:-mllvm -riscv-v-vector-bits-min=256"
    "SHELL:-mllvm -riscv-v-vector-bits-max=256"
)

add_library(ime-kernels STATIC
    kernels/common/adapter_common.cpp
    kernels/common/iq2_xxs_adapter.cpp
    kernels/q4_0-q8_0-IME-upstream/wrapper.cpp
    kernels/q4_0-q8_0-IME-upstream/ime1_kernels.cpp
    kernels/q4_0-q8_0-IME-upstream/adapter.cpp
    kernels/q4_0-q8_0-IME-m4b/kernel.cpp
    kernels/q4_0-q8_0-IME-m4b/adapter.cpp
    kernels/q4_0-q8_0-IME-m4i/kernel.cpp
    kernels/q4_0-q8_0-IME-m4i/adapter.cpp
    kernels/q4_0-q8_0-IME-m8b/kernel.cpp
    kernels/q4_0-q8_0-IME-m8b/adapter.cpp
    kernels/q4_0-q8_0-RVV-my/kernel.cpp
    kernels/q4_0-q8_0-RVV-my/adapter.cpp
    kernels/q4_0-q8_0-RVV-upstream/kernel.cpp
    kernels/q4_0-q8_0-RVV-upstream/adapter.cpp
    kernels/q4_K-q8_K-RVV-group/adapter.cpp
    kernels/q4_K-q8_K-RVV-my/kernel.cpp
    kernels/q4_K-q8_K-RVV-my/adapter.cpp
    kernels/q4_K-q8_K-RVV-upstream/kernel.cpp
    kernels/q4_K-q8_K-RVV-upstream/adapter.cpp
    kernels/iq2_xxs-q8_K-RVV-my-br/kernel.cpp
    kernels/iq2_xxs-q8_K-RVV-my-ir/kernel.cpp
    kernels/iq2_xxs-q8_K-RVV-upstream/kernel.cpp
    $<TARGET_OBJECTS:q4_K_rvv_group_objects>
    $<TARGET_OBJECTS:q4_0_rvv_group_objects>
)

add_library(q4_0_rvv_group_objects OBJECT
    kernels/q4_0-q8_0-RVV-group/kernel.cpp
    kernels/q4_0-q8_0-RVV-group/adapter.cpp
)
target_include_directories(q4_0_rvv_group_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
)
target_link_libraries(q4_0_rvv_group_objects PRIVATE ime_base_compile_options)
target_compile_options(q4_0_rvv_group_objects PRIVATE -Wall -Wextra)

add_library(q4_K_rvv_group_objects OBJECT kernels/q4_K-q8_K-RVV-group/kernel.cpp)
target_include_directories(q4_K_rvv_group_objects PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_K-q8_K-RVV-group
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
)
target_link_libraries(q4_K_rvv_group_objects PRIVATE ime_base_compile_options)
target_compile_options(q4_K_rvv_group_objects PRIVATE -Wall -Wextra)

function(add_iq2_xxs_tcm_object target source)
    add_library(${target} OBJECT ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
    )
    target_link_libraries(${target} PRIVATE ime_compile_options)
    target_compile_definitions(${target} PRIVATE USE_TCM)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
endfunction()

option(IME_ENABLE_TCM "Build IQ2_XXS kernels that use the fine_mm TCM allocator" ON)
set(IME_TCM_RUNTIME "" CACHE FILEPATH
    "Library providing fine_mm_*; required to enable IQ2_XXS TCM kernels")
set(IME_TCM_RUNTIME_CANDIDATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../tcm/libfine_tcm_mm.a")
if(IME_ENABLE_TCM AND NOT IME_TCM_RUNTIME AND EXISTS "${IME_TCM_RUNTIME_CANDIDATE}")
    set(IME_TCM_RUNTIME "${IME_TCM_RUNTIME_CANDIDATE}")
endif()
if(IME_ENABLE_TCM AND IME_TCM_RUNTIME)
    if(NOT EXISTS "${IME_TCM_RUNTIME}")
        message(FATAL_ERROR "IME_TCM_RUNTIME does not exist: ${IME_TCM_RUNTIME}")
    endif()
    add_iq2_xxs_tcm_object(iq2_xxs_rvv_br_tcm_objects
                           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-my-br/kernel.cpp)
    add_iq2_xxs_tcm_object(iq2_xxs_rvv_ir_tcm_objects
                           ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-my-ir/kernel.cpp)
    target_compile_definitions(iq2_xxs_rvv_br_tcm_objects PRIVATE
        ggml_gemm_iq2_xxs_q8_K_br=ggml_gemm_iq2_xxs_q8_K_br_tcm
        pack_a_iq2_xxs_br=pack_a_iq2_xxs_br_tcm
    )
    target_compile_definitions(iq2_xxs_rvv_ir_tcm_objects PRIVATE
        ggml_gemm_iq2_xxs_q8_K_ir=ggml_gemm_iq2_xxs_q8_K_ir_tcm
        pack_a_iq2_xxs_ir=pack_a_iq2_xxs_ir_tcm
    )
    target_sources(ime-kernels PRIVATE
        $<TARGET_OBJECTS:iq2_xxs_rvv_br_tcm_objects>
        $<TARGET_OBJECTS:iq2_xxs_rvv_ir_tcm_objects>
    )
    target_compile_definitions(ime-kernels PRIVATE IME_HAVE_TCM)
    target_link_libraries(ime-kernels PUBLIC "${IME_TCM_RUNTIME}")
    message(STATUS "IQ2_XXS TCM kernels enabled with ${IME_TCM_RUNTIME}")
elseif(IME_ENABLE_TCM)
    message(STATUS "IQ2_XXS TCM kernels disabled: set IME_TCM_RUNTIME to the fine_mm runtime library")
else()
    message(STATUS "IQ2_XXS TCM kernels disabled by IME_ENABLE_TCM=OFF")
endif()

target_include_directories(ime-kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
)
target_link_libraries(ime-kernels PRIVATE ime_compile_options)
target_compile_options(ime-kernels PRIVATE -Wall -Wextra)

set_source_files_properties(
    kernels/q4_0-q8_0-IME-m4b/kernel.cpp
    kernels/q4_0-q8_0-IME-m4i/kernel.cpp
    kernels/q4_0-q8_0-IME-m8b/kernel.cpp
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
    set(IME_ASM_VLEN_OPTIONS
        -mllvm -riscv-v-vector-bits-min=256
        -mllvm -riscv-v-vector-bits-max=256
    )
    if(ARGC GREATER 2 AND ARGV2 STREQUAL "NO_VLEN")
        set(IME_ASM_VLEN_OPTIONS)
    endif()
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${IME_ASM_DIR}
        COMMAND ${CMAKE_CXX_COMPILER}
            -std=c++20
            -DSPACEMIT_X60
            -march=${IME_ARCH}
            -mtune=${IME_TUNE}
            ${IME_ASM_VLEN_OPTIONS}
            -mllvm --misched-cluster=false
            -O3
            -I${CMAKE_CURRENT_SOURCE_DIR}/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
            -I${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
            -I${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
            -I${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
            -S ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            -o ${output}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${source}
            ${CMAKE_CURRENT_SOURCE_DIR}/include/ggml.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/types.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/ime1_dispatch.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/rvv_vl.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_0-q8_0-IME-upstream/wrapper.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_0-q8_0-IME-m4b/kernel.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_0-q8_0-IME-m4i/kernel.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/q4_0-q8_0-IME-m8b/kernel.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/perf.h
            ${CMAKE_CURRENT_SOURCE_DIR}/bench/timer.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common/gemm.h
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common/ggml_min.h
        VERBATIM
        COMMENT "Generating ${output}"
    )
    set(IME_ASM_OUTPUTS ${IME_ASM_OUTPUTS} ${output} PARENT_SCOPE)
endfunction()

add_ime_assembly(kernels/q4_0-q8_0-IME-upstream/wrapper.cpp q4_0_q8_0_ime_upstream_wrapper)
add_ime_assembly(kernels/q4_0-q8_0-IME-upstream/ime1_kernels.cpp q4_0_q8_0_ime_upstream_kernels)
add_ime_assembly(kernels/q4_0-q8_0-IME-m4b/kernel.cpp q4_0_q8_0_ime_m4b)
add_ime_assembly(kernels/q4_0-q8_0-IME-m4i/kernel.cpp q4_0_q8_0_ime_m4i)
add_ime_assembly(kernels/q4_0-q8_0-IME-m8b/kernel.cpp q4_0_q8_0_ime_m8b)
add_ime_assembly(kernels/q4_0-q8_0-RVV-group/kernel.cpp q4_0_q8_0_rvv_group NO_VLEN)
add_ime_assembly(kernels/q4_0-q8_0-RVV-my/kernel.cpp q4_0_q8_0_rvv_my)
add_ime_assembly(kernels/q4_0-q8_0-RVV-upstream/kernel.cpp q4_0_q8_0_rvv_upstream)
add_ime_assembly(kernels/q4_K-q8_K-RVV-group/kernel.cpp q4_K_q8_K_rvv_group NO_VLEN)
add_ime_assembly(kernels/q4_K-q8_K-RVV-my/kernel.cpp q4_K_q8_K_rvv_my)
add_ime_assembly(kernels/q4_K-q8_K-RVV-upstream/kernel.cpp q4_K_q8_K_rvv_upstream)
add_ime_assembly(kernels/iq2_xxs-q8_K-RVV-my-br/kernel.cpp iq2_xxs_q8_K_rvv_my_br)
add_ime_assembly(kernels/iq2_xxs-q8_K-RVV-my-ir/kernel.cpp iq2_xxs_q8_K_rvv_my_ir)
add_ime_assembly(kernels/iq2_xxs-q8_K-RVV-upstream/kernel.cpp iq2_xxs_q8_K_rvv_upstream)
add_ime_assembly(bench/main.cpp main)

add_custom_target(assembly DEPENDS ${IME_ASM_OUTPUTS})