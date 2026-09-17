#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M4I_CB_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M4I_CB_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// The cache-blocked variant reuses the M4 immediate-reduction microkernel owned
// by `q4_0-q8_0-IME-m4i`; no second implementation is needed because the blocking
// only changes how the microkernel is called. The microkernel covers one
// kMr x kNr output tile for a contiguous run of K blocks and accumulates into a
// caller-owned acc tile, so the blocked loop passes its packed C tile directly:
//
//   * `quant_b_data` points at the first K block of a K panel for that tile and
//     `block_count_k` is the panel block count, so the packed B panel
//     (KC * 288 B) stays L1D-resident while the activation chunk streams past
//     it;
//   * `acc` is the packed C tile itself (`kMr * kNr` floats, row-major). The
//     caller zeroes it once before the first K panel and the kernel adds every
//     panel's K contributions to it.
void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(const uint8_t *GGML_RESTRICT quant_a, const uint8_t *GGML_RESTRICT quant_b_data, float *GGML_RESTRICT acc, size_t block_count_k);

#endif
