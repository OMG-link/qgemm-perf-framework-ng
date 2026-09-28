#ifndef PFNG_KERNEL_Q4_0_Q8_0_IME_RR_KERNEL_H
#define PFNG_KERNEL_Q4_0_Q8_0_IME_RR_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

inline constexpr size_t kImeRrMr = 8;
inline constexpr size_t kImeRrColumnGroups = 4;
inline constexpr size_t kImeRrNr = kImeRrColumnGroups * 8;
inline constexpr size_t kImeRrScaleMatrixElements = 8 * 8;
inline constexpr size_t kImeRrActivationTilesPerScaleLoad = 8;
inline constexpr size_t kImeRrPackedCElements = kImeRrMr * kImeRrNr;

// Holds one N32 x K32 weight tile in registers and applies it to `m_tile_count`
// consecutive M8 activation tiles. This is the kernel's reuse boundary: all
// eight unpacked int8 weight matrices and all four fp16 weight-scale matrices
// are loaded before the M loop and remain live while activation tiles stream
// through them.
//
// Each activation tile contains two A(8x16) int8 matrices. Activation scales
// are packed in groups of eight tiles as one row-major fp16 A(8x8) matrix: row
// m contains that row's scale for tile 0..7. The kernel loads it once and slides
// it down by one lane after each tile, bringing the next scale to column zero.
//
// The weight tile contains four packed Q4_0 matrices; each produces low/high
// W(16x8) int8 matrices, plus eight fp16 scales stored at lanes 0,8,...,56 of
// an 8x8 operand footprint. A masked unit-stride load reads only those lanes and
// supplies zero for every other lane. Consequently smt.vfwmadot computes the
// required 8x8 scale outer product without relying on the unused memory lanes.
//
// packed_c contains one native-order M8 x N32 tile per activation tile. Each
// such tile is four consecutive row-major 8x8 matrix-unit output blocks. The
// kernel reads, accumulates and writes these tiles for the current K block.
void q4_0_q8_0_ime_rr_w_stationary(const int8_t *GGML_RESTRICT a_qs,
                                   const uint16_t *GGML_RESTRICT a_scales,
                                   const uint8_t *GGML_RESTRICT w_qs,
                                   const uint16_t *GGML_RESTRICT w_scales,
                                   float *GGML_RESTRICT packed_c,
                                   size_t m_tile_count);

#endif