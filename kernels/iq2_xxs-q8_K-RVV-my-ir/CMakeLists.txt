ime_add_kernel_sources(SOURCES kernel.cpp)
add_ime_assembly(kernel.cpp iq2_xxs_q8_K_rvv_my_ir
    DEPENDS
        ../iq2_xxs-q8_K-RVV-common/gemm.h
        ../iq2_xxs-q8_K-RVV-common/ggml_min.h
)

if(IME_HAVE_TCM)
    ime_add_kernel_object(iq2_xxs_rvv_ir_tcm_objects
        SOURCES kernel.cpp
        INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/kernels/iq2_xxs-q8_K-RVV-common
        COMPILE_DEFINITIONS
            USE_TCM
            ggml_gemm_iq2_xxs_q8_K_ir=ggml_gemm_iq2_xxs_q8_K_ir_tcm
            pack_a_iq2_xxs_ir=pack_a_iq2_xxs_ir_tcm
    )
endif()