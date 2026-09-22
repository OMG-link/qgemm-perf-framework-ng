pfng_declare_kernel(
    ID q4_K-q8_K-RVV-group
    NAME "Q4_K/Q8_K RVV grouped 4x32"
    QUANTIZATION WeightQ4_KActivationQ8_K
    REGISTER_FUNCTION register_q4_K_rvv_group
    PLATFORMS spacemit-k1-a60 spacemit-k1-x60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    set(vlen_options
        -mllvm
        -riscv-v-vector-bits-min=256
        -riscv-v-vector-bits-max=256
    )

    pfng_add_kernel_sources(SOURCES implementations/rvv-vlen256/adapter.cpp)
    pfng_add_kernel_object(q4_K_rvv_group_objects
        SOURCES implementations/rvv-vlen256/kernel.cpp
        REMOVE_COMPILE_OPTIONS ${vlen_options}
    )

    add_pfng_assembly(implementations/rvv-vlen256/kernel.cpp q4_K_q8_K_rvv_group
        REMOVE_COMPILE_OPTIONS ${vlen_options})
endif()
