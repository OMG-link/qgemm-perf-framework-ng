if(NOT DEFINED PFNG_TARGET_PLATFORM OR PFNG_TARGET_PLATFORM STREQUAL "")
    message(FATAL_ERROR
        "PFNG_TARGET_PLATFORM is required (supported: spacemit-k1-a60, spacemit-k1-x60)")
endif()

set(_pfng_platform_file
    "${CMAKE_SOURCE_DIR}/platforms/${PFNG_TARGET_PLATFORM}.cmake")
if(NOT EXISTS "${_pfng_platform_file}")
    message(FATAL_ERROR "Unknown PFNG_TARGET_PLATFORM: ${PFNG_TARGET_PLATFORM}")
endif()
include("${_pfng_platform_file}")

foreach(required
        PFNG_PLATFORM_ID
        PFNG_PLATFORM_DISPLAY_NAME
        PFNG_PLATFORM_ARCH
        PFNG_PLATFORM_TUNE
        PFNG_PLATFORM_COMPILE_DEFINITIONS
        PFNG_PLATFORM_DEVICE_TREE_COMPATIBLES
        PFNG_PLATFORM_FIRST_CPU
        PFNG_PLATFORM_LAST_CPU
        PFNG_PLATFORM_MAX_THREADS
        PFNG_PLATFORM_KERNEL_COMPILE_OPTIONS
        PFNG_PLATFORM_FIXED_VLEN_COMPILE_OPTIONS)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR
            "Platform profile '${_pfng_platform_file}' must define ${required}")
    endif()
endforeach()

if(NOT PFNG_PLATFORM_ID STREQUAL PFNG_TARGET_PLATFORM)
    message(FATAL_ERROR
        "Platform profile '${_pfng_platform_file}' declares '${PFNG_PLATFORM_ID}', "
        "expected '${PFNG_TARGET_PLATFORM}'")
endif()

if(DEFINED PFNG_CONFIGURED_PLATFORM AND
   NOT PFNG_CONFIGURED_PLATFORM STREQUAL PFNG_TARGET_PLATFORM)
    message(FATAL_ERROR
        "This build tree was configured for '${PFNG_CONFIGURED_PLATFORM}'. "
        "Use a separate build directory for '${PFNG_TARGET_PLATFORM}'.")
endif()
set(PFNG_CONFIGURED_PLATFORM "${PFNG_TARGET_PLATFORM}" CACHE INTERNAL
    "Target platform used to create this build tree")

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "[Rr][Ii][Ss][Cc][Vv]")
    message(FATAL_ERROR
        "Only RISC-V targets are supported; CMAKE_SYSTEM_PROCESSOR is "
        "'${CMAKE_SYSTEM_PROCESSOR}'")
endif()

execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -dumpmachine
    RESULT_VARIABLE _pfng_dumpmachine_result
    OUTPUT_VARIABLE PFNG_COMPILER_TARGET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT _pfng_dumpmachine_result EQUAL 0 OR
   NOT PFNG_COMPILER_TARGET MATCHES "riscv")
    message(FATAL_ERROR
        "CMAKE_CXX_COMPILER must target RISC-V; -dumpmachine returned "
        "'${PFNG_COMPILER_TARGET}'")
endif()

message(STATUS "PFNG target platform: ${PFNG_PLATFORM_ID}")
message(STATUS "PFNG platform display name: ${PFNG_PLATFORM_DISPLAY_NAME}")
message(STATUS "PFNG compiler target: ${PFNG_COMPILER_TARGET}")
message(STATUS "PFNG target architecture: ${PFNG_PLATFORM_ARCH}")
message(STATUS "PFNG compiler tune: ${PFNG_PLATFORM_TUNE}")
message(STATUS "PFNG allowed CPUs: ${PFNG_PLATFORM_FIRST_CPU}-${PFNG_PLATFORM_LAST_CPU}")