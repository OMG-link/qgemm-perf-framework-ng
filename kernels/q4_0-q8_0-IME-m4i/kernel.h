#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M4I_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M4I_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Immediate-reduction microkernel for a single kMr x kNr output tile.
//
// The kernel owns the tile shape and the K loop only: it consumes
// `block_count_k` K blocks of one activation tile (`quant_a`, kMr rows) plus the
// matching packed weight tile (`quant_b_data`, kNr columns, already offset to
// the first K block by the caller) and accumulates the fp32 result into `acc`.
//
// `acc` holds the tile in MIV (matrix-in-vector) order: one 16-lane accumulator
// per row group, i.e. eight 8-float chunks that each carry the 4-float group
// `c / 2` of row `2 * (c % 2)` followed by the same group of the next row. The
// kernel loads that tile once before the K loop and stores it once afterwards,
// so no unpack work is repeated per K panel.
//
// The caller owns the N tiling, the packed B tile offsets, the acc -> C(output)
// unpack and the acc initialization: `acc` is deliberately not initialized by
// the kernel, which only adds to what is already stored there.
void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(
    const uint8_t *GGML_RESTRICT quant_a,
    const uint8_t *GGML_RESTRICT quant_b_data,
    float *GGML_RESTRICT acc,
    size_t block_count_k);

// Write one MIV tile (`kMr * kNr` floats) to the four rows of C at `ldc` stride.
void c_unpack_ime_m4_n16(const float *GGML_RESTRICT miv, float *GGML_RESTRICT c, size_t ldc);

#endif