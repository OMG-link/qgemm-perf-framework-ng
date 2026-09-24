#ifndef PFNG_KERNEL_Q4_0_Q8_0_M8B_IME2_KERNEL_H
#define PFNG_KERNEL_Q4_0_Q8_0_M8B_IME2_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Batch-reduction microkernel for a single kMr x kNr output tile on the
// SpacemiT K3 / A100 matrix unit (XSMTVDotII, VLEN = 1024 bit).
//
// This is a 1:1 port of the A60 implementation in ../smt-ime1: same batched
// reduction, same row-major int32 component buffer, same compact scale layout,
// same "scatter the accumulator into row-major order on writeback" strategy.
// Only the sizes move, because one dot instruction grew:
//
//   A60, XSMTVDot    one smt.vmadot is A(4x8)  x B(8x4)  = C(4x4)
//   A100, XSMTVDotII one smt.vmadot is A(8x16) x B(16x8) = C(8x8)
//
// The register budget is held constant: 8 int32m2 accumulators arranged as
// 2 row groups x 4 column groups, 4 A registers, 8 unpacked B registers. With
// the larger dot that covers kMr = 16 rows and kNr = 32 columns instead of
// 8 x 16, and one K block (QK8_0 = 32) is 16 dots instead of 32. The kernel ID
// still says m8b; m16b would be the accurate name.
//
// The kernel owns the K loop only: it consumes `block_count_k` K blocks of one
// activation tile (kMr rows) plus the matching packed weight tile (kNr columns,
// already offset to the first K block by the caller) and accumulates the fp32
// result into `acc`, which is a contiguous row-major kMr x kNr tile of
// kMr * kNr floats.
//
// Operand layout, confirmed on hardware by
// machine-benchmarks/spacemit-k3-a100/k3_madot_layout.cpp. Row group r covers
// rows 8r..8r+7, column group g covers columns 8g..8g+7, K window w covers
// K 16w..16w+15, and m / n / k below are local to a group or window:
//
//   quant_a_qs     per K block 512 bytes, four 128-byte A registers at
//                  (2 * r + w) * 128, byte index m * 16 + k.
//   quant_a_scales per K block kMr floats, one per row.
//   quant_b_qs     per K block 512 bytes, four 128-byte B registers at
//                  g * 128, byte index n * 16 + k, where the low nibble is
//                  K window 0 and the high nibble is K window 1. That is the
//                  native Q4_0 nibble order, so packing copies bytes through.
//   quant_b_scales per K block kNr fp16 values, one per column.
//
// Unlike the A60 kernel there is no inner K-window loop: on the A60 one B
// register holds 8 K values, so a nibble pair spans 16 K and a QK8_0 block
// needs two iterations. On the A100 one B register holds 16 K, so a nibble pair
// already spans the whole 32-element block.
//
// `reduction_components` is the batched int32 component buffer, which the A60
// kernel keeps on its own stack. It is a caller parameter here so the adapter
// can place it in the A100's tightly coupled memory, where whole-register stores
// run about 2.9x faster than to L1D-resident ordinary memory. It must hold
// kIme2ReductionComponentCount int32 and start on a 256-byte boundary.
//
// The caller owns the N tiling, the packed B tile offsets and the acc ->
// C(output) copy. `acc` is deliberately not initialized by the kernel: every
// batch flush adds to the values already stored in `acc`, so a caller that
// accumulates over K panels can hand in the (zeroed) C tile itself.
inline constexpr size_t kIme2ReductionComponentCount = 16 * 32 * 8;  // kOutputM * kOutputN * kRedBatchSize

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_IME2_BatchRed(const int8_t *GGML_RESTRICT quant_a_qs, const float *GGML_RESTRICT quant_a_scales, const uint8_t *GGML_RESTRICT quant_b_qs,
                                                         const uint16_t *GGML_RESTRICT quant_b_scales, float *GGML_RESTRICT acc, size_t block_count_k,
                                                         int32_t *GGML_RESTRICT reduction_components);

#endif
