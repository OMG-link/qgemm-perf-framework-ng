ime_add_kernel_sources(SOURCES kernel.cpp
    COMPILE_OPTIONS
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-but-set-variable
)
ime_add_kernel_sources(SOURCES adapter.cpp)

add_ime_assembly(kernel.cpp q4_0_q8_0_ime_m4b
    DEPENDS kernel.h)