ime_add_kernel_sources(SOURCES
    wrapper.cpp
    ime1_kernels.cpp
    adapter.cpp
)

add_ime_assembly(wrapper.cpp q4_0_q8_0_ime_upstream_wrapper
    DEPENDS wrapper.h)
add_ime_assembly(ime1_kernels.cpp q4_0_q8_0_ime_upstream_kernels)