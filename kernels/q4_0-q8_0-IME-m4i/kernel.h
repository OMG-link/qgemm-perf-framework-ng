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
// the first K block by the caller) and accumulates the fp32 result into `acc`,
// which is a contiguous row-major kMr x kNr tile of kMr * kNr floats.
//
// The caller owns the N tiling, the packed B tile offsets and the acc ->
// C(output) copy. `acc` is deliberately not initialized by the kernel: the
// unpacked reduction adds to the values already stored in `acc`, so a caller
// that accumulates over K panels can hand in the (zeroed) C tile itself.
void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(
    const uint8_t *GGML_RESTRICT quant_a,
    const uint8_t *GGML_RESTRICT quant_b_data,
    float *GGML_RESTRICT acc,
    size_t block_count_k);

#endif