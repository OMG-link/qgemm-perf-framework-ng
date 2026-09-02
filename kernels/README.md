# Imported kernels

This directory contains source-level imports of the requested RVV and SpacemiT
kernels. Each implementation is kept in its own directory using the requested
name. The imported source is intentionally preserved rather than rewritten:
its original quantized layout, ABI, and RVV intrinsic assumptions remain
reviewable and can be adapted to a benchmark adapter independently.

| Directory | Source / entry point | Notes |
|---|---|---|
| `q4_K-q8_K-RVV-group` | `duyl-gemm18.cpp`, `ggml_gemm_q4_K_8x32_q8_K` | 4x32 grouped kernel |
| `q4_K-q8_K-RVV-my` | `kernel.cpp`, `ggml_gemm_q4_K_8x32_q8_K` | 12x32 kernel |
| `q4_0-q8_0-RVV-group` | triton-cpu commit `45018883019239550daf638222fb23588d203d47`, `ggml_gemm_q4_0_12x32_q8_0` | 12x32 kernel |
| `q4_0-q8_0-RVV-my` | triton-cpu commit `94fd845b386a0be8bd67baa76d06c47d81eaf6fb`, `ggml_gemm_q4_0_8x32_q8_0` | 8x32 kernel |
| `iq2_xxs-q8_K-RVV-my-br` | `gemm_br.cpp`, `ggml_gemm_iq2_xxs_q8_K` | on-the-fly decompression |
| `iq2_xxs-q8_K-RVV-my-ir` | `gemm_ir.cpp`, `ggml_gemm_iq2_xxs_q8_K` | pre-decompressed weights |
| `*-RVV-upstream` | llama.cpp `ggml/src/ggml-cpu/arch/riscv/quants.c` | contains the upstream q4_0, q4_K, and iq2_xxs RVV dot kernels |
| `q4_0-q8_0-IME-upstream` | llama.cpp `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp` | IME i8 x i4 path; supporting headers included |

The `iq2_xxs-q8_K-RVV-my-{br,ir}-tcm` directories are the TCM-enabled
variants. They preserve the original `#ifdef USE_TCM` branches and include
`fine_tcm_mm.h`; enable `USE_TCM` and link the corresponding TCM runtime when
building them.

The upstream imports are source archives, not yet registered benchmark kernels:
they depend on llama.cpp's internal `ggml-common.h`, quantization headers, and
build macros. Likewise, the standalone imported GEMMs use their original
`ggml_def.h`/`gemm.h` layouts. This avoids silently changing correctness while
bringing the kernels into the framework for subsequent ABI-specific adapters.
