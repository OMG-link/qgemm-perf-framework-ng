#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

#define QK_K 256
#define QK_SK 32

#ifdef __cplusplus
// restrict not standard in C++
#if defined(__GNUC__)
#define GGML_RESTRICT __restrict__
#elif defined(__clang__)
#define GGML_RESTRICT __restrict
#elif defined(_MSC_VER)
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

#define UNUSED(x) (void)(x)

typedef _Float16 float16_t;
typedef float float32_t;

template <int NR> struct block_q4_Kx {
    float16_t d[NR];                    // super-block scale for quantized scales
    float16_t dmin[NR];                 // super-block scale for quantized mins
    int8_t scales[NR * (QK_K / QK_SK)]; // sub-block scales, quantized with 6 bits, but stored with 8 bits
    int8_t mins[NR * (QK_K / QK_SK)];   // sub-block mins, quantized with 6 bits, but stored with 8 bits
    uint8_t qs[NR * QK_K / 2];          // 4--bit quants
};

template <int MR> struct block_q8_Kx {
    float32_t d[MR];                   // delta
    int8_t qs[MR * QK_K];              // quants
    uint16_t bsums[MR * QK_K / QK_SK]; // sum of quants in groups of 32
};

void ggml_gemm_q4_K_8x32_q8_K(int n, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int nr, int nc);
