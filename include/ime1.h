#ifndef IME_LLAMA_IME1_H
#define IME_LLAMA_IME1_H

#include "ggml.h"
#include "types.h"

namespace sqnbitgemm_spacemit_ime::ime1 {

void quantize_a_row_i8(size_t block_length, const float *input, size_t count_k,
                       std::byte *output);
void quantize_a_4row_i8(size_t block_length, const float *input, size_t count_k,
                        std::byte *output);
size_t gemm_kernel_i8i4(size_t block_length, const std::byte *quant_a,
                        const std::byte *quant_b, const float *quant_b_scale,
                        const std::byte *quant_b_zero_point, float *output,
                        size_t count_m, size_t count_n, size_t count_k,
                        size_t block_count_k, size_t ldc, const float *bias,
                        size_t scale_stride);

} // namespace sqnbitgemm_spacemit_ime::ime1

void quant_a(const float *data, block_q8_0_ime_m4 *buffer, size_t count_m, size_t count_k);
void quant_a_m8(const float *data, block_q8_0_ime_m8 *buffer, size_t count_m, size_t count_k);

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(const uint8_t * GGML_RESTRICT QuantA,
    const uint8_t * GGML_RESTRICT QuantBData, float * GGML_RESTRICT C, size_t CountN, size_t BlockCountK, const size_t ldc);

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed(const uint8_t * GGML_RESTRICT QuantA,
    const uint8_t * GGML_RESTRICT QuantBData, float * GGML_RESTRICT C, size_t CountN, size_t BlockCountK, const size_t ldc);

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin(size_t BlkLen, const uint8_t * QuantA, 
    const uint8_t * QuantBData, float * C, size_t CountN, size_t BlockCountK, const size_t ldc);

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(const uint8_t * QuantA,
    const uint8_t * QuantBQs, const uint16_t * QuantBScales, float * C, size_t CountN, size_t BlockCountK, const size_t ldc);
    
void forward_mul_mat(void *w_data, const float *feature, float *output, const int64_t gemm_m, 
    const int64_t gemm_n, const int64_t gemm_k);

int repack_q4_0_to_q4_0_16_bl(void *t_dst, int ncol, int nrow, int interleave_block, 
    const void * GGML_RESTRICT data, size_t data_size);

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl(size_t BlkLen, const uint8_t * QuantA, 
    const uint8_t * QuantBData, float *C, size_t CountN, size_t BlockCountK, const size_t ldc);

inline constexpr size_t vl(size_t e, size_t m) {
    return m * 256 / e;
}
inline constexpr size_t vl_mf(size_t e, size_t mf) {
    return 256 / e / mf;
}

#endif
