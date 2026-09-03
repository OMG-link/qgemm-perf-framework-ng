set(vlen_options
    -mllvm
    -riscv-v-vector-bits-min=256
    -riscv-v-vector-bits-max=256
)

ime_add_kernel_object(q4_0_rvv_group_objects
    SOURCES kernel.cpp adapter.cpp
    REMOVE_COMPILE_OPTIONS ${vlen_options}
)

add_ime_assembly(kernel.cpp q4_0_q8_0_rvv_group
    REMOVE_COMPILE_OPTIONS ${vlen_options})