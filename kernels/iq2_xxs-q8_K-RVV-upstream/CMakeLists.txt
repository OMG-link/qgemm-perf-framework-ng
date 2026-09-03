ime_add_kernel_sources(SOURCES kernel.cpp)
add_ime_assembly(kernel.cpp iq2_xxs_q8_K_rvv_upstream
    DEPENDS tables.h)