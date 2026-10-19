#ifndef PFNG_KERNEL_Q4_0_Q8_0_IME_UPSTREAM_IME2_DISPATCH_H
#define PFNG_KERNEL_Q4_0_Q8_0_IME_UPSTREAM_IME2_DISPATCH_H

// Local ABI header for the vendored SpacemiT IME2 q4_0 x q8_0 sources.
//
// The upstream extracts include ggml-impl.h (for GGML_ASSERT) and
// rvv_kernels.h (for q8_blk_size). This header supplies both from this
// project's dependency-free ggml compatibility header plus one local
// definition, so the extracts build without the external checkout. See
// ../../provenance.md.

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define GGML_ASSERT(condition)                                                           \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            std::fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__,         \
                         #condition);                                                    \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

namespace spacemit_kernels {

// rvv_kernels.h: Q8 block layout [f32 scale][s16 block sum][int8 quants].
constexpr size_t q8_blk_size(size_t blk_len, bool with_blk_sum = false) {
    return sizeof(float) + blk_len * sizeof(int8_t) + (with_blk_sum ? sizeof(int16_t) : 0);
}

namespace ime2 {

// ime2_kernels.cpp. quant_a points at a row group of quantized activations,
// quant_b at one 32-column block of repacked q4_0 weights, quant_b_zero_point
// is null for q4_0 (the kernel then uses the format's implicit 8), output is
// row-major with ldc leading elements. Returns the number of activation rows
// consumed: 4 for the M4 microkernel, 1 for M1.
size_t gemm_kernel_i8i4(size_t          block_length,
                        const uint8_t * quant_a,
                        const uint8_t * quant_b,
                        const uint8_t * quant_b_zero_point,
                        float *         output,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          block_count_k,
                        size_t          ldc);

} // namespace ime2

// Upstream's activation quantizer (rvv_kernels.cpp quantize_a_row_i8 /
// quantize_a_4row_i8) is not imported: it stores an fp32 amax/127 scale and
// rounds to nearest even, so its blocks are not the q8_0 the reference defines.
// The adapter writes the same block layout from the reference quantization.
// See ../../provenance.md.

} // namespace spacemit_kernels

#endif
