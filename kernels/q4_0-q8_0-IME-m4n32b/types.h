#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M4N32B_TYPES_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M4N32B_TYPES_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ime::m4n32b {

inline constexpr size_t MR = 4;
inline constexpr size_t NR = 32;
inline constexpr size_t MC = 128;
inline constexpr size_t KC = 512;
inline constexpr size_t ReductionBatch = 8;

struct block_q8_0_m4 {
    _Float16 d[MR];
    int8_t qs[MR * QK8_0];
};

// Scales are contiguous. qs is [K16 plane][four-column group][32 bytes].
struct block_q4_0_n32 {
    _Float16 d[NR];
    alignas(8) uint8_t qs[2][8][32];
};

static_assert(sizeof(block_q8_0_m4) == MR * sizeof(_Float16) + MR * QK8_0);
static_assert(offsetof(block_q4_0_n32, qs) == NR * sizeof(_Float16));
static_assert(sizeof(block_q4_0_n32) == NR * sizeof(_Float16) + NR * QK4_0 / 2);

} // namespace ime::m4n32b

#endif
