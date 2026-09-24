pfng_declare_kernel(
    ID q4_0-q8_0-IME-m8b
    NAME "Q4_0 x Q8_0 IME M8 batch reduction"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_m8b
    PLATFORMS spacemit-k1-a60 spacemit-k3-a100)
if(PFNG_CURRENT_KERNEL_ENABLED)
    # smt-ime1 targets the K1 / A60 (XSMTVDot, VLEN 256, A(4x8)xB(8x4)),
    # smt-ime2 the K3 / A100 (XSMTVDotII, VLEN 1024, A(8x16)xB(16x8)). The two
    # differ in tile shape, packed layout and reduction, so they are separate
    # implementations behind one kernel ID.
    if(PFNG_TARGET_PLATFORM STREQUAL "spacemit-k3-a100")
        set(PFNG_M8B_IMPLEMENTATION implementations/smt-ime2)
        set(PFNG_M8B_ASSEMBLY_STEM q4_0_q8_0_ime_m8b_ime2)
    else()
        set(PFNG_M8B_IMPLEMENTATION implementations/smt-ime1)
        set(PFNG_M8B_ASSEMBLY_STEM q4_0_q8_0_ime_m8b)
    endif()
    pfng_add_kernel_sources(SOURCES ${PFNG_M8B_IMPLEMENTATION}/kernel.cpp
        COMPILE_OPTIONS
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-but-set-variable
    )
    pfng_add_kernel_sources(SOURCES ${PFNG_M8B_IMPLEMENTATION}/adapter.cpp
        COMPILE_OPTIONS -fopenmp)
    if(PFNG_TARGET_PLATFORM STREQUAL "spacemit-k3-a100")
        pfng_add_kernel_sources(SOURCES ${PFNG_M8B_IMPLEMENTATION}/tcm_arena.cpp)
    endif()
    add_pfng_assembly(${PFNG_M8B_IMPLEMENTATION}/kernel.cpp ${PFNG_M8B_ASSEMBLY_STEM}
        DEPENDS ${PFNG_M8B_IMPLEMENTATION}/kernel.h)
endif()
