#ifndef IME_KERNEL_M4_BATCH_REDUCTION_KERNEL_H
#define IME_KERNEL_M4_BATCH_REDUCTION_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed(
    const uint8_t *GGML_RESTRICT quant_a,
    const uint8_t *GGML_RESTRICT quant_b_data,
    float *GGML_RESTRICT output,
    size_t count_n,
    size_t block_count_k,
    size_t ldc);

#endif