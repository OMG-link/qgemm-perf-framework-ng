#ifndef IME_KERNEL_TYPES_H
#define IME_KERNEL_TYPES_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

template <size_t Columns> struct block_q4_0_rvv {
    _Float16 d[Columns];
    alignas(8) int8_t qs[Columns * QK4_0 / 2];
};

template <size_t Rows> struct block_q8_0_rvv {
    _Float16 d[Rows];
    alignas(8) int8_t qs[Rows * QK8_0];
};

using block_q4_0_rvv_n32 = block_q4_0_rvv<32>;
using block_q8_0_rvv_m8 = block_q8_0_rvv<8>;
using block_q8_0_rvv_m12 = block_q8_0_rvv<12>;

template <size_t Columns> struct block_q4_0_ime {
    _Float16 d[Columns];
    alignas(8) int8_t qs[Columns * QK4_0 / 2];
};

using block_q4_0_ime_n16 = block_q4_0_ime<16>;

template <size_t Rows> struct block_q8_0_ime {
    float d[Rows];
    int8_t qs[Rows * QK8_0];
};

using block_q8_0_ime_m4 = block_q8_0_ime<4>;
using block_q8_0_ime_m8 = block_q8_0_ime<8>;

inline constexpr int kQKSubBlock = 32;

template <size_t Columns> struct block_q4_K_rvv {
    _Float16 d[Columns];
    _Float16 dmin[Columns];
    int8_t scales[Columns * (QK_K / kQKSubBlock)];
    int8_t mins[Columns * (QK_K / kQKSubBlock)];
    uint8_t qs[Columns * QK_K / 2];
};

template <size_t Rows> struct block_q8_K_rvv {
    float d[Rows];
    int8_t qs[Rows * QK_K];
    uint16_t bsums[Rows * QK_K / kQKSubBlock];
};

using block_q4_K_rvv_n32 = block_q4_K_rvv<32>;
using block_q8_K_rvv_m4 = block_q8_K_rvv<4>;
using block_q8_K_rvv_m12 = block_q8_K_rvv<12>;

static_assert(sizeof(_Float16) == sizeof(ggml_half));

static_assert(sizeof(block_q4_0_rvv_n32) ==
              32 * sizeof(_Float16) + 32 * QK4_0 / 2);
static_assert(sizeof(block_q8_0_rvv_m8) ==
              8 * sizeof(_Float16) + 8 * QK8_0);
static_assert(sizeof(block_q8_0_rvv_m12) ==
              12 * sizeof(_Float16) + 12 * QK8_0);
static_assert(sizeof(block_q4_0_ime_n16) ==
              16 * sizeof(_Float16) + 16 * QK4_0 / 2);
static_assert(alignof(block_q4_0_rvv_n32) >= 8);
static_assert(alignof(block_q8_0_rvv_m8) >= 8);
static_assert(alignof(block_q8_0_rvv_m12) >= 8);
static_assert(alignof(block_q4_0_ime_n16) >= 8);
static_assert(sizeof(block_q8_0_ime_m4) ==
              4 * sizeof(float) + 4 * QK8_0);
static_assert(sizeof(block_q8_0_ime_m8) ==
              8 * sizeof(float) + 8 * QK8_0);

static_assert(sizeof(block_q4_K_rvv_n32) ==
              32 * sizeof(_Float16) * 2 +
              32 * (QK_K / 32) * 2 +
              32 * QK_K / 2);
static_assert(offsetof(block_q4_K_rvv_n32, d) == 0);
static_assert(offsetof(block_q4_K_rvv_n32, dmin) == 32 * sizeof(_Float16));
static_assert(offsetof(block_q4_K_rvv_n32, scales) ==
              64 * sizeof(_Float16));
static_assert(offsetof(block_q4_K_rvv_n32, mins) ==
              64 * sizeof(_Float16) + 32 * (QK_K / 32));
static_assert(offsetof(block_q4_K_rvv_n32, qs) ==
              64 * sizeof(_Float16) + 32 * (QK_K / 32) * 2);

static_assert(sizeof(block_q8_K_rvv_m4) ==
              4 * sizeof(float) + 4 * QK_K +
              4 * (QK_K / 32) * sizeof(uint16_t));
static_assert(sizeof(block_q8_K_rvv_m12) ==
              12 * sizeof(float) + 12 * QK_K +
              12 * (QK_K / 32) * sizeof(uint16_t));
static_assert(offsetof(block_q8_K_rvv_m4, d) == 0);
static_assert(offsetof(block_q8_K_rvv_m4, qs) == 4 * sizeof(float));
static_assert(offsetof(block_q8_K_rvv_m4, bsums) ==
              4 * sizeof(float) + 4 * QK_K);
static_assert(offsetof(block_q8_K_rvv_m12, d) == 0);
static_assert(offsetof(block_q8_K_rvv_m12, qs) == 12 * sizeof(float));
static_assert(offsetof(block_q8_K_rvv_m12, bsums) ==
              12 * sizeof(float) + 12 * QK_K);

#endif