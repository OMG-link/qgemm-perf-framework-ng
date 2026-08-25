#ifndef IME_LLAMA_GGML_H
#define IME_LLAMA_GGML_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
#define ASSERT_MSG(cond, fmt, ...)                                                \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "[ASSERT] %s:%d: " fmt "\n", __FILE__, __LINE__, \
                         ##__VA_ARGS__);                                           \
            std::fprintf(stderr, "  Failed: %s\n", #cond);                       \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#else
#define ASSERT_MSG(cond, fmt, ...) ((void)0)
#endif

// Compatibility helper for older RVV intrinsic naming.
#define __riscv_vzext_vf2_u16m2(x, vl) __riscv_vwaddu_vx_u16m2((x), 0, (vl))

using ggml_half = uint16_t;
using ggml_half2 = uint32_t;
using ggml_fp16_t = uint16_t;

inline float fp32_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t fp32_to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
    constexpr float exp_scale = 0x1.0p-112f;
    const float normalized = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    constexpr uint32_t magic_mask = UINT32_C(126) << 23;
    constexpr float magic_bias = 0.5f;
    const float denormalized = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    return fp32_from_bits(sign | (two_w < denormalized_cutoff
                                      ? fp32_to_bits(denormalized)
                                      : fp32_to_bits(normalized)));
}

inline ggml_fp16_t ggml_compute_fp32_to_fp16(float value) {
    constexpr float scale_to_inf = 0x1.0p+112f;
    constexpr float scale_to_zero = 0x1.0p-110f;
    float base = (std::fabs(value) * scale_to_inf) * scale_to_zero;
    const uint32_t w = fp32_to_bits(value);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }
    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return static_cast<ggml_fp16_t>(
        (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)
#define GGML_CPU_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)

struct block_q4_0 {
    ggml_half d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q8_0 {
    ggml_half d;
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
    ggml_half d[Rows];
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
    ggml_half d[VL];
    ggml_half dmin[VL];
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
            ggml_half d;
            ggml_half dmin;
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
};

static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2);
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0);
static_assert(sizeof(block_q4_0x16) == 16 * sizeof(ggml_half) + 16 * QK4_0 / 2);
static_assert(sizeof(block_q4_0x32) == 32 * sizeof(ggml_half) + 32 * QK4_0 / 2);
static_assert(sizeof(block_q8_0x12) == 12 * sizeof(ggml_half) + 12 * QK8_0);
static_assert(sizeof(block_q8_0x4_scale32) == 4 * sizeof(float) + 4 * QK8_0);
static_assert(sizeof(block_q8_0x8_scale32) == 8 * sizeof(float) + 8 * QK8_0);

#endif