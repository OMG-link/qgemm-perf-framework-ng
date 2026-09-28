pfng_declare_kernel(
    ID q4_0-q8_0-IME-RR
    NAME "Q4_0 x Q8_0 IME scale outer-product reduction"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_rr
    PLATFORMS spacemit-k3-a100)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/smt-ime2/kernel.cpp
        COMPILE_OPTIONS
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-but-set-variable)
    pfng_add_kernel_sources(SOURCES implementations/smt-ime2/adapter.cpp
        COMPILE_OPTIONS -fopenmp)

    add_pfng_assembly(implementations/smt-ime2/kernel.cpp q4_0_q8_0_ime_rr
        DEPENDS implementations/smt-ime2/kernel.h)
endif()