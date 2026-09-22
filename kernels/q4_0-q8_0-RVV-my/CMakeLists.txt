pfng_declare_kernel(
    ID q4_0-q8_0-RVV-my
    NAME "Q4_0/Q8_0 RVV custom 8x32"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_rvv_my
    PLATFORMS spacemit-k1-a60 spacemit-k1-x60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/rvv-vlen256/kernel.cpp implementations/rvv-vlen256/adapter.cpp)
    add_pfng_assembly(implementations/rvv-vlen256/kernel.cpp q4_0_q8_0_rvv_my)
endif()
