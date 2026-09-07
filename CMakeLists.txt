cmake_minimum_required(VERSION 3.20)

project(ime_llama LANGUAGES CXX ASM)

set(IME_ARCH "rv64gcv_zvfh_zicbop_zba_xsmtvdot" CACHE STRING
    "RISC-V architecture used by the SpacemiT IME kernels")
set(IME_TUNE "spacemit-x60" CACHE STRING
    "RISC-V CPU tuning target")

add_subdirectory(machine-benchmarks)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

set(IME_KERNEL_DEFAULT_COMPILE_OPTIONS
    -Wall
    -Wextra
    -mllvm
    -riscv-v-vector-bits-min=256
    -mllvm
    -riscv-v-vector-bits-max=256
)
set(IME_ASSEMBLY_DEFAULT_COMPILE_OPTIONS
    -O3
    -mllvm
    -riscv-v-vector-bits-min=256
    -mllvm
    -riscv-v-vector-bits-max=256
)

add_library(ime-kernels STATIC)
target_include_directories(ime-kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
)
target_link_libraries(ime-kernels PRIVATE ime_base_compile_options)

function(ime_resolve_sources output)
    set(resolved)
    foreach(source IN LISTS ARGN)
        if(IS_ABSOLUTE "${source}")
            list(APPEND resolved "${source}")
        else()
            list(APPEND resolved "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        endif()
    endforeach()
    set(${output} "${resolved}" PARENT_SCOPE)
endfunction()

function(ime_apply_default_options output defaults removals additions)
    set(options ${defaults})
    foreach(option IN LISTS removals)
        list(REMOVE_ITEM options "${option}")
    endforeach()
    list(APPEND options ${additions})
    set(${output} "${options}" PARENT_SCOPE)
endfunction()

function(ime_add_kernel_sources)
    cmake_parse_arguments(ARG "" "" "SOURCES;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "ime_add_kernel_sources requires SOURCES")
    endif()
    ime_resolve_sources(sources ${ARG_SOURCES})
    ime_apply_default_options(options
        "${IME_KERNEL_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    target_sources(ime-kernels PRIVATE ${sources})
    set_property(SOURCE ${sources} TARGET_DIRECTORY ime-kernels
                 PROPERTY COMPILE_OPTIONS ${options})
    if(ARG_COMPILE_DEFINITIONS)
        set_property(SOURCE ${sources} TARGET_DIRECTORY ime-kernels
                     PROPERTY COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS})
    endif()
endfunction()

function(ime_add_kernel_object target)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDE_DIRECTORIES;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "ime_add_kernel_object requires SOURCES")
    endif()
    ime_resolve_sources(sources ${ARG_SOURCES})
    ime_apply_default_options(options
        "${IME_KERNEL_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    add_library(${target} OBJECT ${sources})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/bench/framework
        ${CMAKE_SOURCE_DIR}/bench/llama_reference
        ${CMAKE_SOURCE_DIR}/kernels/common
        ${ARG_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${target} PRIVATE ime_base_compile_options)
    set_property(SOURCE ${sources} TARGET_DIRECTORY ${target}
                 PROPERTY COMPILE_OPTIONS ${options})
    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${target} PRIVATE ${ARG_COMPILE_DEFINITIONS})
    endif()
    target_sources(ime-kernels PRIVATE $<TARGET_OBJECTS:${target}>)
endfunction()

set(IME_ASM_DIR ${CMAKE_CURRENT_BINARY_DIR}/asm)
set_property(GLOBAL PROPERTY IME_ASM_TARGETS "")
function(add_ime_assembly source stem)
    cmake_parse_arguments(ARG "" "" "DEPENDS;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS;INCLUDE_DIRECTORIES" ${ARGN})
    ime_resolve_sources(sources "${source}")
    list(GET sources 0 source_path)
    ime_resolve_sources(dependencies ${ARG_DEPENDS})
    ime_apply_default_options(options
        "${IME_ASSEMBLY_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    set(definitions)
    foreach(definition IN LISTS ARG_COMPILE_DEFINITIONS)
        list(APPEND definitions "-D${definition}")
    endforeach()
    set(include_directories)
    foreach(directory IN LISTS ARG_INCLUDE_DIRECTORIES)
        list(APPEND include_directories "-I${directory}")
    endforeach()
    set(output ${IME_ASM_DIR}/${stem}.s)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${IME_ASM_DIR}
        COMMAND ${CMAKE_CXX_COMPILER}
            -std=c++20
            -DSPACEMIT_X60
            ${definitions}
            -march=${IME_ARCH}
            -mtune=${IME_TUNE}
            -mllvm --misched-cluster=false
            ${options}
            -I${CMAKE_SOURCE_DIR}/include
            -I${CMAKE_SOURCE_DIR}/bench
            -I${CMAKE_SOURCE_DIR}/bench/framework
            -I${CMAKE_SOURCE_DIR}/bench/llama_reference
            -I${CMAKE_SOURCE_DIR}/kernels/common
            -I${CMAKE_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
            ${include_directories}
            -S ${source_path}
            -o ${output}
        DEPENDS
            ${source_path}
            ${dependencies}
            ${CMAKE_SOURCE_DIR}/include/ggml.h
            ${CMAKE_SOURCE_DIR}/kernels/common/types.h
            ${CMAKE_SOURCE_DIR}/kernels/common/ime1_dispatch.h
            ${CMAKE_SOURCE_DIR}/kernels/common/rvv_vl.h
            ${CMAKE_SOURCE_DIR}/bench/perf.h
            ${CMAKE_SOURCE_DIR}/bench/timer.hpp
        VERBATIM
        COMMENT "Generating ${output}"
    )
    set(assembly_target ime_asm_${stem})
    add_custom_target(${assembly_target} DEPENDS ${output})
    set_property(GLOBAL APPEND PROPERTY IME_ASM_TARGETS ${assembly_target})
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
    set(IME_HAVE_TCM ON)
else()
    set(IME_HAVE_TCM OFF)
endif()

add_subdirectory(kernels)

if(IME_HAVE_TCM)
    target_compile_definitions(ime-kernels PRIVATE IME_HAVE_TCM)
    target_link_libraries(ime-kernels PUBLIC "${IME_TCM_RUNTIME}")
    message(STATUS "IQ2_XXS TCM kernels enabled with ${IME_TCM_RUNTIME}")
elseif(IME_ENABLE_TCM)
    message(STATUS "IQ2_XXS TCM kernels disabled: set IME_TCM_RUNTIME to the fine_mm runtime library")
else()
    message(STATUS "IQ2_XXS TCM kernels disabled by IME_ENABLE_TCM=OFF")
endif()

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

add_ime_assembly(bench/main.cpp main)
get_property(IME_ASM_TARGETS GLOBAL PROPERTY IME_ASM_TARGETS)
add_custom_target(assembly)
add_dependencies(assembly ${IME_ASM_TARGETS})