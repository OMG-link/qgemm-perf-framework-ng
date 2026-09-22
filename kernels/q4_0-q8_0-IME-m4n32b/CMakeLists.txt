pfng_declare_kernel(
    ID q4_0-q8_0-IME-m4n32b
    NAME "Q4_0 x Q8_0 IME M4xN32 batch reduction"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_m4n32b
    PLATFORMS spacemit-k1-a60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES implementations/smt-ime1/kernel.cpp
        COMPILE_OPTIONS
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-unused-but-set-variable
    )
    pfng_add_kernel_sources(SOURCES implementations/smt-ime1/adapter.cpp)

    add_pfng_assembly(implementations/smt-ime1/kernel.cpp q4_0_q8_0_ime_m4n32b
        DEPENDS implementations/smt-ime1/kernel.h implementations/smt-ime1/types.h)
endif()
