#ifndef IME_KERNEL_COMMON_IME1_DISPATCH_H
#define IME_KERNEL_COMMON_IME1_DISPATCH_H

#include <cstddef>

namespace sqnbitgemm_spacemit_ime::ime1 {

void quantize_a_row_i8(
    size_t block_length,
    const float *input,
    size_t count_k,
    std::byte *output);

void quantize_a_4row_i8(
    size_t block_length,
    const float *input,
    size_t count_k,
    std::byte *output);

size_t gemm_kernel_i8i4(
    size_t block_length,
    const std::byte *quant_a,
    const std::byte *quant_b,
    const float *quant_b_scale,
    const std::byte *quant_b_zero_point,
    float *output,
    size_t count_m,
    size_t count_n,
    size_t count_k,
    size_t block_count_k,
    size_t ldc,
    const float *bias,
    size_t scale_stride);

} // namespace sqnbitgemm_spacemit_ime::ime1

#endif