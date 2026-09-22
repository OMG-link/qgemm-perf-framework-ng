cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Local cross-compilation setup.
#
# CMake needs the target system before project(), so these values cannot be
# normal project options. They are stable project facts and contain no
# machine-local paths.
#
# The compiler is intentionally NOT pinned here: it is machine-local, so it must
# come from -DCMAKE_CXX_COMPILER, from the CXX environment variable, or from
# -DCMAKE_TOOLCHAIN_FILE.
# ---------------------------------------------------------------------------
if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
endif()

if(NOT CMAKE_TOOLCHAIN_FILE
   AND NOT DEFINED CMAKE_CXX_COMPILER
   AND NOT DEFINED ENV{CXX})
    message(FATAL_ERROR
        "CMAKE_CXX_COMPILER is not configured. Pass "
        "-DCMAKE_CXX_COMPILER=<RISC-V clang++>, run the wrapper with a "
        ".pfng-workspace.json entry, or pass -DCMAKE_TOOLCHAIN_FILE=<file>.")
endif()

project(pfng LANGUAGES CXX ASM)

find_package(OpenMP REQUIRED COMPONENTS CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PlatformConfig)
include(Kernel)
include(BuildManifest)
pfng_generate_build_manifest()

add_subdirectory(machine-benchmarks)

set(PFNG_ALIGNED_NEW_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework/aligned_new.cpp)
set(PFNG_MACHINE_BENCHMARK_TARGETS
    k1-l2-bandwidth
    dynpreunpack-inner-loop
    m8b-inner-issue
    dynpreunpack-full-realistic
    unpack-address-conflict
    perf-pc-semantics
    prefetch-threshold
    prefetch-multistream
    dpu-cb-prefetch
    dpu-cb-inner-issue
    m8b-aw-miss
    k1-memory-latency)
foreach(target IN LISTS PFNG_MACHINE_BENCHMARK_TARGETS)
    if(TARGET ${target})
        target_sources(${target} PRIVATE ${PFNG_ALIGNED_NEW_SOURCE})
    endif()
endforeach()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(pfng_base_compile_options INTERFACE)
target_compile_features(pfng_base_compile_options INTERFACE cxx_std_20)
target_compile_definitions(pfng_base_compile_options INTERFACE
    ${PFNG_PLATFORM_COMPILE_DEFINITIONS})
target_compile_options(pfng_base_compile_options INTERFACE
    ${PFNG_PLATFORM_KERNEL_COMPILE_OPTIONS}
)
target_include_directories(pfng_base_compile_options INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(PFNG_KERNEL_DEFAULT_COMPILE_OPTIONS
    -Wall
    -Wextra
    ${PFNG_PLATFORM_FIXED_VLEN_COMPILE_OPTIONS}
)
set(PFNG_ASSEMBLY_DEFAULT_COMPILE_OPTIONS
    -O3
    ${PFNG_PLATFORM_FIXED_VLEN_COMPILE_OPTIONS}
)

add_library(pfng-kernels STATIC)
target_include_directories(pfng-kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
)
target_link_libraries(pfng-kernels PRIVATE pfng_base_compile_options)

function(pfng_resolve_sources output)
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

function(pfng_apply_default_options output defaults removals additions)
    set(options ${defaults})
    foreach(option IN LISTS removals)
        list(REMOVE_ITEM options "${option}")
    endforeach()
    list(APPEND options ${additions})
    set(${output} "${options}" PARENT_SCOPE)
endfunction()

function(pfng_add_kernel_sources)
    cmake_parse_arguments(ARG "" "" "SOURCES;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "pfng_add_kernel_sources requires SOURCES")
    endif()
    pfng_resolve_sources(sources ${ARG_SOURCES})
    pfng_apply_default_options(options
        "${PFNG_KERNEL_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    target_sources(pfng-kernels PRIVATE ${sources})
    set_property(SOURCE ${sources} TARGET_DIRECTORY pfng-kernels
                 PROPERTY COMPILE_OPTIONS ${options})
    if(ARG_COMPILE_DEFINITIONS)
        set_property(SOURCE ${sources} TARGET_DIRECTORY pfng-kernels
                     PROPERTY COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS})
    endif()
endfunction()

function(pfng_add_kernel_object target)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDE_DIRECTORIES;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS" ${ARGN})
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "pfng_add_kernel_object requires SOURCES")
    endif()
    pfng_resolve_sources(sources ${ARG_SOURCES})
    pfng_apply_default_options(options
        "${PFNG_KERNEL_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    add_library(${target} OBJECT ${sources})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/bench/framework
        ${CMAKE_SOURCE_DIR}/bench/llama_reference
        ${CMAKE_SOURCE_DIR}/kernels/common
        ${ARG_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${target} PRIVATE pfng_base_compile_options)
    set_property(SOURCE ${sources} TARGET_DIRECTORY ${target}
                 PROPERTY COMPILE_OPTIONS ${options})
    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${target} PRIVATE ${ARG_COMPILE_DEFINITIONS})
    endif()
    target_sources(pfng-kernels PRIVATE $<TARGET_OBJECTS:${target}>)
endfunction()

set(PFNG_ASM_DIR ${CMAKE_CURRENT_BINARY_DIR}/asm)
set_property(GLOBAL PROPERTY PFNG_ASM_TARGETS "")
function(add_pfng_assembly source stem)
    cmake_parse_arguments(ARG "" "" "DEPENDS;COMPILE_OPTIONS;REMOVE_COMPILE_OPTIONS;COMPILE_DEFINITIONS;INCLUDE_DIRECTORIES" ${ARGN})
    pfng_resolve_sources(sources "${source}")
    list(GET sources 0 source_path)
    pfng_resolve_sources(dependencies ${ARG_DEPENDS})
    pfng_apply_default_options(options
        "${PFNG_ASSEMBLY_DEFAULT_COMPILE_OPTIONS}"
        "${ARG_REMOVE_COMPILE_OPTIONS}"
        "${ARG_COMPILE_OPTIONS}")
    set(definitions)
    foreach(definition IN LISTS PFNG_PLATFORM_COMPILE_DEFINITIONS)
        list(APPEND definitions "-D${definition}")
    endforeach()
    foreach(definition IN LISTS ARG_COMPILE_DEFINITIONS)
        list(APPEND definitions "-D${definition}")
    endforeach()
    set(include_directories)
    foreach(directory IN LISTS ARG_INCLUDE_DIRECTORIES)
        list(APPEND include_directories "-I${directory}")
    endforeach()
    set(output ${PFNG_ASM_DIR}/${stem}.s)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PFNG_ASM_DIR}
        COMMAND ${CMAKE_CXX_COMPILER}
            -std=c++20
            ${definitions}
            -march=${PFNG_PLATFORM_ARCH}
            -mtune=${PFNG_PLATFORM_TUNE}
            -mllvm --misched-cluster=false
            ${options}
            -I${CMAKE_SOURCE_DIR}/include
            -I${CMAKE_SOURCE_DIR}/bench
            -I${CMAKE_SOURCE_DIR}/bench/framework
            -I${CMAKE_SOURCE_DIR}/bench/llama_reference
            -I${CMAKE_SOURCE_DIR}/kernels/common
            -I${CMAKE_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
            -I${CMAKE_BINARY_DIR}/generated
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
        VERBATIM
        COMMENT "Generating ${output}"
    )
    set(assembly_target pfng_asm_${stem})
    add_custom_target(${assembly_target} DEPENDS ${output})
    set_property(GLOBAL APPEND PROPERTY PFNG_ASM_TARGETS ${assembly_target})
endfunction()

add_subdirectory(kernels)

set(PFNG_BUILTIN_KERNELS_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/builtin_kernels.cpp")
pfng_generate_builtin_kernels("${PFNG_BUILTIN_KERNELS_SOURCE}")

add_executable(pfng-bench
    bench/main.cpp
    bench/framework/aligned_new.cpp
    bench/framework/kernel_registry.cpp
    bench/framework/benchmark.cpp
    bench/framework/platform.cpp
    bench/llama_reference/llama_reference.cpp
    ${PFNG_BUILTIN_KERNELS_SOURCE}
)
target_include_directories(pfng-bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bench
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/llama_reference
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_features(pfng-bench PRIVATE cxx_std_20)
target_link_libraries(pfng-bench PRIVATE pfng-kernels OpenMP::OpenMP_CXX)
target_compile_options(pfng-bench PRIVATE -Wall -Wextra)
target_link_options(pfng-bench PRIVATE -fopenmp=libomp)

add_pfng_assembly(bench/main.cpp main)
get_property(PFNG_ASM_TARGETS GLOBAL PROPERTY PFNG_ASM_TARGETS)
add_custom_target(assembly)
add_dependencies(assembly ${PFNG_ASM_TARGETS})

add_executable(pfng-platform-test
    tests/platform_test.cpp
    bench/framework/platform.cpp)
target_compile_features(pfng-platform-test PRIVATE cxx_std_20)
target_include_directories(pfng-platform-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/framework
    ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_options(pfng-platform-test PRIVATE -Wall -Wextra)