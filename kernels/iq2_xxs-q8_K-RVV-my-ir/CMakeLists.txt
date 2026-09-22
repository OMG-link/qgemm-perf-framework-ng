pfng_declare_kernel(
    ID iq2_xxs-q8_K-RVV-my-ir
    NAME "IQ2_XXS/Q8_K RVV custom 12x32 immediate-reduce"
    QUANTIZATION WeightIQ2_XXSActivationQ8_K
    REGISTER_FUNCTION register_iq2_xxs_rvv_my_ir
    PLATFORMS spacemit-k1-a60 spacemit-k1-x60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/rvv-vlen256/kernel.cpp)
    add_pfng_assembly(implementations/rvv-vlen256/kernel.cpp iq2_xxs_q8_K_rvv_my_ir
        DEPENDS
            ../iq2_xxs-q8_K-RVV-common/gemm.h
            ../iq2_xxs-q8_K-RVV-common/ggml_min.h
    )
endif()
