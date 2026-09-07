#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M8B_DYN_PRE_UNPACK_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M8B_DYN_PRE_UNPACK_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

[[gnu::noinline]] void unpack_q4_0_ime_m8b_rvv(const uint8_t *GGML_RESTRICT packed_b_qs, int8_t *GGML_RESTRICT unpacked_b_qs, size_t block_count);

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack(
    const uint8_t *GGML_RESTRICT quant_a,
    const int8_t *GGML_RESTRICT unpacked_b_qs,
    const uint16_t *GGML_RESTRICT quant_b_scales,
    float *GGML_RESTRICT output,
    size_t count_n,
    size_t block_count_k,
    size_t ldc);

#endif
