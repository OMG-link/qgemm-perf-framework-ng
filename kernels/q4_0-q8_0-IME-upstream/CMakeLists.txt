pfng_declare_kernel(
    ID q4_0-q8_0-IME-upstream
    NAME "Q4_0 x Q8_0 IME upstream"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_upstream
    PLATFORMS spacemit-k1-a60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    pfng_add_kernel_sources(SOURCES
        implementations/smt-ime1/wrapper.cpp
        implementations/smt-ime1/ime1_kernels.cpp
        implementations/smt-ime1/adapter.cpp
    )

    add_pfng_assembly(implementations/smt-ime1/wrapper.cpp q4_0_q8_0_ime_upstream_wrapper
        DEPENDS implementations/smt-ime1/wrapper.h)
    add_pfng_assembly(implementations/smt-ime1/ime1_kernels.cpp q4_0_q8_0_ime_upstream_kernels)
endif()
