#ifndef IME_KERNEL_Q4_0_Q8_0_IME_M8B_CB_KERNEL_H
#define IME_KERNEL_Q4_0_Q8_0_IME_M8B_CB_KERNEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// The cache-blocked variant reuses the M8 batch-reduction microkernel owned by
// `q4_0-q8_0-IME-m8b`; no second implementation is needed because the blocking
// only changes how the microkernel is called. The cache-blocked loop passes
// `count_n = 16` and `ldc = 16`, which selects the single N16 tile form:
//
//   * `quant_b_qs`/`quant_b_scales` point at the first K block of a K panel for
//     that tile, and `block_count_k` is the panel block count, so the packed B
//     panel (KC * (kNr * QK4_0/2) bytes of qs plus KC * kNr scales) stays
//     L1D-resident while the activation chunk streams past it;
//   * `output` is a contiguous kMr * kNr C tile that the caller accumulates
//     across K panels.
void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(const int8_t *GGML_RESTRICT quant_a_qs, const float *GGML_RESTRICT quant_a_scales, const uint8_t *GGML_RESTRICT quant_b_qs,
                                                                const uint16_t *GGML_RESTRICT quant_b_scales, float *GGML_RESTRICT output, size_t count_n, size_t block_count_k, size_t ldc);

#endif
