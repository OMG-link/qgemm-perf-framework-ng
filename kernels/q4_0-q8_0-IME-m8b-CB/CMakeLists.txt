pfng_declare_kernel(
    ID q4_0-q8_0-IME-m8b-CB
    NAME "Q4_0 x Q8_0 IME M8 batch reduction cache blocking"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_m8b_cb
    PLATFORMS spacemit-k1-a60)

if(PFNG_CURRENT_KERNEL_ENABLED)
    # The cache-blocked variant reuses the M4 immediate-reduction microkernel that
    # `q4_0-q8_0-IME-m4i` owns, so this directory only provides the blocked adapter
    # and needs no extra microkernel source or assembly target.
    pfng_add_kernel_sources(SOURCES implementations/spacemit-k1-a60/adapter.cpp
        COMPILE_OPTIONS -fopenmp)
endif()
