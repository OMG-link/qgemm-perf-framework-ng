set(vlen_options
    -mllvm
    -riscv-v-vector-bits-min=256
    -riscv-v-vector-bits-max=256
)

ime_add_kernel_sources(SOURCES adapter.cpp)
ime_add_kernel_object(q4_K_rvv_group_objects
    SOURCES kernel.cpp
    REMOVE_COMPILE_OPTIONS ${vlen_options}
)

add_ime_assembly(kernel.cpp q4_K_q8_K_rvv_group
    REMOVE_COMPILE_OPTIONS ${vlen_options})