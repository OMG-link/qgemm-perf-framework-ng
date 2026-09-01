#ifndef IME_LLAMA_GGML_H
#define IME_LLAMA_GGML_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif

#ifdef __cplusplus
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__) || defined(_MSC_VER)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT
#endif
#else
#if defined(_MSC_VER) && (__STDC_VERSION__ < 201112L)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif
#endif

#define GGML_UNUSED(x) (void)(x)
#define UNUSED(x) GGML_UNUSED(x)

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define QK_K 256
#define QK_SB_K 32
#define QK4_0 32
#define QK8_0 32
#define K_SCALE_SIZE 12

#ifndef NDEBUG
#define ASSERT_MSG(cond, fmt, ...)                                                                                                                                                                     \
    do {                                                                                                                                                                                               \
        if (!(cond)) {                                                                                                                                                                                 \
            std::fprintf(stderr, "[ASSERT] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);                                                                                                      \
            std::fprintf(stderr, "  Failed: %s\n", #cond);                                                                                                                                             \
            std::abort();                                                                                                                                                                              \
        }                                                                                                                                                                                              \
    } while (0)
#else
#define ASSERT_MSG(cond, fmt, ...) ((void)0)
#endif

// Compatibility helper for older RVV intrinsic naming.
#define __riscv_vzext_vf2_u16m2(x, vl) __riscv_vwaddu_vx_u16m2((x), 0, (vl))

struct block_q4_0 {
    _Float16 d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q8_0 {
    _Float16 d;
    int8_t qs[QK8_0];
};

template <int Bits> constexpr int qk_0() {
    if constexpr (Bits == 4) {
        return QK4_0;
    } else if constexpr (Bits == 8) {
        return QK8_0;
    } else {
        return -1;
    }
}

template <int Bits, int Rows> struct block {
    _Float16 d[Rows];
    alignas(8) int8_t qs[(qk_0<Bits>() * Rows * Bits) / 8];
};

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q4_0x32 = block<4, 32>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x12 = block<8, 12>;

template <size_t Rows> struct block_q8_0x_scale32 {
    float d[Rows];
    int8_t qs[Rows * QK8_0];
};

using block_q8_0x4_scale32 = block_q8_0x_scale32<4>;
using block_q8_0x8_scale32 = block_q8_0x_scale32<8>;

template <int VL> struct block_q4_Kx {
    _Float16 d[VL];
    _Float16 dmin[VL];
    uint8_t scales[VL * (QK_K / QK_SB_K) * 2];
    uint8_t qs[VL * QK_K / 2];
};

template <int VL> struct block_q8_Kx {
    float d[VL];
    uint8_t qs[VL * QK_K];
    uint16_t bsums[VL * QK_K / QK_SB_K];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S

struct block_q4_K {
    GGML_EXTENSION union {
        struct {
            _Float16 d;
            _Float16 dmin;
        } GGML_COMMON_AGGR_S;
        uint32_t dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

static_assert(sizeof(block_q4_0) == sizeof(_Float16) + QK4_0 / 2);
static_assert(sizeof(block_q8_0) == sizeof(_Float16) + QK8_0);
static_assert(sizeof(block_q4_0x16) == 16 * sizeof(_Float16) + 16 * QK4_0 / 2);
static_assert(sizeof(block_q4_0x32) == 32 * sizeof(_Float16) + 32 * QK4_0 / 2);
static_assert(sizeof(block_q8_0x12) == 12 * sizeof(_Float16) + 12 * QK8_0);
static_assert(sizeof(block_q8_0x4_scale32) == 4 * sizeof(float) + 4 * QK8_0);
static_assert(sizeof(block_q8_0x8_scale32) == 8 * sizeof(float) + 8 * QK8_0);

#endif