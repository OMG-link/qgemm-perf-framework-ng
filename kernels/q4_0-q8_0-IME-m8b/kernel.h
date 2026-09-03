#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M8B_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M8B_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(
    const uint8_t *GGML_RESTRICT quant_a,
    const uint8_t *GGML_RESTRICT quant_b_qs,
    const uint16_t *GGML_RESTRICT quant_b_scales,
    float *GGML_RESTRICT output,
    size_t count_n,
    size_t block_count_k,
    size_t ldc);

#endif