pfng_declare_kernel(
    ID q4_K-q8_K-RVV-upstream
    NAME "Q4_K/Q8_K RVV upstream dot"
    QUANTIZATION WeightQ4_KActivationQ8_K
    REGISTER_FUNCTION register_q4_K_rvv_upstream
    PLATFORMS spacemit-k1-a60 spacemit-k1-x60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/rvv/kernel.cpp implementations/rvv/adapter.cpp)
    add_pfng_assembly(implementations/rvv/kernel.cpp q4_K_q8_K_rvv_upstream)
endif()
