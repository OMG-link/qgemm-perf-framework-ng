#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M4N32B_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M4N32B_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

void q4_0_q8_0_ime_m4n32b(const uint8_t *GGML_RESTRICT quant_a,
                           const uint8_t *GGML_RESTRICT quant_b,
                           float *GGML_RESTRICT output, size_t block_count_k,
                           size_t ldc, bool accumulate);

#endif
