#ifndef PFNG_KERNEL_Q4_0_Q8_0_IME_UPSTREAM_IME2_WRAPPER_H
#define PFNG_KERNEL_Q4_0_Q8_0_IME_UPSTREAM_IME2_WRAPPER_H

#include <cstddef>
#include <cstdint>

// Local reproduction of the llama.cpp IME2 call site for the i8-i4 GEMM
// (`ggml/src/ggml-cpu/spacemit/ime.cpp`, `forward_mul_mat`): one B block covers
// 32 columns and is consumed as a whole, while the A operand is consumed in
// whatever row count `gemm_kernel_i8i4` reports, so a call with row_count > 4
// walks the activation row groups and the output rows in steps of the returned
// value.
//
// `row_stride_a` is the distance between two consecutive activation rows and
// `col_stride_b` the distance between two consecutive 32-column weight blocks,
// both measured over every K block of the gemm.
void sqnbitgemm_spacemit_ime_ime2_i8i4(size_t          block_length,
                                       size_t          block_count_k,
                                       const uint8_t * quant_a,
                                       size_t          row_stride_a,
                                       const uint8_t * quant_b,
                                       size_t          col_stride_b,
                                       float *         output,
                                       size_t          ldc,
                                       size_t          row_count,
                                       size_t          column_count);

#endif
