pfng_declare_kernel(
    ID q4_0-q8_0-RVV-group
    NAME "Q4_0/Q8_0 RVV grouped 12x32"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_rvv_group
    PLATFORMS spacemit-k1-a60 spacemit-k1-x60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    set(vlen_options
        -mllvm
        -riscv-v-vector-bits-min=256
        -riscv-v-vector-bits-max=256
    )

    pfng_add_kernel_object(q4_0_rvv_group_objects
        SOURCES implementations/rvv-vlen256/kernel.cpp implementations/rvv-vlen256/adapter.cpp
        REMOVE_COMPILE_OPTIONS ${vlen_options}
    )

    add_pfng_assembly(implementations/rvv-vlen256/kernel.cpp q4_0_q8_0_rvv_group
        REMOVE_COMPILE_OPTIONS ${vlen_options})
endif()
