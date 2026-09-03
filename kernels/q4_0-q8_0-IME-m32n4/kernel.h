#ifndef Q4_0_Q8_0_IME_M32N4_KERNEL_H
#define Q4_0_Q8_0_IME_M32N4_KERNEL_H

#include <cstddef>
#include <cstdint>

#include "ggml.h"

struct block_q8_0_ime_m32 {
    _Float16 d[32];
    int8_t qs[32 * QK8_0];
};

struct block_q4_0_ime_n4 {
    _Float16 d[4];
    uint8_t qs[4 * QK4_0 / 2];
};

static_assert(sizeof(block_q8_0_ime_m32) == 32 * sizeof(_Float16) + 32 * QK8_0);
static_assert(sizeof(block_q4_0_ime_n4) == 4 * sizeof(_Float16) + 4 * QK4_0 / 2);

void q4_0_q8_0_ime_m32n4(const block_q8_0_ime_m32 *a, const block_q4_0_ime_n4 *b, float *c, size_t block_count_k, size_t ldc);

#endif
