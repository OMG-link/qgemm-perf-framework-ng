function(pfng_generate_build_manifest)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
    set(PFNG_PLATFORM_COMPATIBLE_INITIALIZERS "")
    foreach(compatible IN LISTS PFNG_PLATFORM_DEVICE_TREE_COMPATIBLES)
        if(NOT PFNG_PLATFORM_COMPATIBLE_INITIALIZERS STREQUAL "")
            string(APPEND PFNG_PLATFORM_COMPATIBLE_INITIALIZERS ", ")
        endif()
        string(APPEND PFNG_PLATFORM_COMPATIBLE_INITIALIZERS "\"${compatible}\"")
    endforeach()
    configure_file(
        "${CMAKE_SOURCE_DIR}/bench/framework/build_manifest.h.in"
        "${CMAKE_BINARY_DIR}/generated/build_manifest.h" @ONLY)
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/build-manifest.json" CONTENT
"{
  \"platform\": \"${PFNG_PLATFORM_ID}\",
  \"platform_display_name\": \"${PFNG_PLATFORM_DISPLAY_NAME}\",
  \"target_arch\": \"${PFNG_PLATFORM_ARCH}\",
  \"target_tune\": \"${PFNG_PLATFORM_TUNE}\",
  \"first_cpu\": ${PFNG_PLATFORM_FIRST_CPU},
  \"last_cpu\": ${PFNG_PLATFORM_LAST_CPU},
  \"max_threads\": ${PFNG_PLATFORM_MAX_THREADS},
  \"compiler\": \"${CMAKE_CXX_COMPILER}\",
  \"compiler_id\": \"${CMAKE_CXX_COMPILER_ID}\",
  \"compiler_version\": \"${CMAKE_CXX_COMPILER_VERSION}\",
  \"remote_prologue\": \"${PFNG_PLATFORM_REMOTE_PROLOGUE}\",
  \"tcm_enabled\": false
}
")
endfunction()