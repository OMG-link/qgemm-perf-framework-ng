pfng_declare_kernel(
    ID q4_0-q8_0-IME-m8b-DPU-CB
    NAME "Q4_0 x Q8_0 IME M8 dynamic pre-unpack cache blocking"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_m8b_dpu_cb
    PLATFORMS spacemit-k1-a60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/spacemit-k1-a60/kernel.cpp
        COMPILE_OPTIONS
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-but-set-variable
    )
    pfng_add_kernel_sources(SOURCES implementations/spacemit-k1-a60/adapter.cpp
        COMPILE_OPTIONS -fopenmp)

    add_pfng_assembly(implementations/spacemit-k1-a60/kernel.cpp q4_0_q8_0_ime_m8b_DPU_CB
        DEPENDS implementations/spacemit-k1-a60/kernel.h)
endif()
