pfng_declare_kernel(
    ID q4_0-q8_0-IME-upstream
    NAME "Q4_0 x Q8_0 IME upstream"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_upstream
    PLATFORMS spacemit-k1-a60 spacemit-k3-a100)

if(PFNG_CURRENT_KERNEL_ENABLED)
    # smt-ime1 targets the K1 / A60 (XSMTVDot, VLEN 256, four-by-eight times
    # eight-by-four i4 vmadot), smt-ime2 the K3 / A100 (XSMTVDotII, VLEN 1024,
    # eight-by-many i8 and i4 vmadot). The two differ in packed layout, ABI and
    # ISA, so they are separate implementations behind one kernel ID.
    if(PFNG_TARGET_PLATFORM STREQUAL "spacemit-k3-a100")
        set(PFNG_UPSTREAM_IMPLEMENTATION implementations/smt-ime2)
        set(PFNG_UPSTREAM_ASSEMBLY_STEM q4_0_q8_0_ime_upstream_ime2)
        pfng_add_kernel_sources(SOURCES
            ${PFNG_UPSTREAM_IMPLEMENTATION}/wrapper.cpp
            ${PFNG_UPSTREAM_IMPLEMENTATION}/ime2_kernels.cpp
            COMPILE_OPTIONS
                -Wno-unused-parameter
                -Wno-unused-variable
                -Wno-unused-but-set-variable)
        pfng_add_kernel_sources(SOURCES ${PFNG_UPSTREAM_IMPLEMENTATION}/adapter.cpp
            COMPILE_OPTIONS -fopenmp)
    else()
        set(PFNG_UPSTREAM_IMPLEMENTATION implementations/smt-ime1)
        set(PFNG_UPSTREAM_ASSEMBLY_STEM q4_0_q8_0_ime_upstream_wrapper)
        pfng_add_kernel_sources(SOURCES
            ${PFNG_UPSTREAM_IMPLEMENTATION}/wrapper.cpp
            ${PFNG_UPSTREAM_IMPLEMENTATION}/ime1_kernels.cpp
            ${PFNG_UPSTREAM_IMPLEMENTATION}/adapter.cpp)
    endif()

    add_pfng_assembly(${PFNG_UPSTREAM_IMPLEMENTATION}/wrapper.cpp ${PFNG_UPSTREAM_ASSEMBLY_STEM}
        DEPENDS ${PFNG_UPSTREAM_IMPLEMENTATION}/wrapper.h)
    if(PFNG_TARGET_PLATFORM STREQUAL "spacemit-k3-a100")
        add_pfng_assembly(${PFNG_UPSTREAM_IMPLEMENTATION}/ime2_kernels.cpp
            q4_0_q8_0_ime_upstream_ime2_kernels
            DEPENDS ${PFNG_UPSTREAM_IMPLEMENTATION}/ime2_dispatch.h)
    else()
        add_pfng_assembly(${PFNG_UPSTREAM_IMPLEMENTATION}/ime1_kernels.cpp
            q4_0_q8_0_ime_upstream_kernels)
    endif()
endif()
