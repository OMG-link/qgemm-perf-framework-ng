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
| `iq2_xxs-q8_K-RVV-my-br` | `kernel.cpp`, `ggml_gemm_iq2_xxs_q8_K_br` | on-the-fly decompression |
| `iq2_xxs-q8_K-RVV-my-ir` | `kernel.cpp`, `ggml_gemm_iq2_xxs_q8_K_ir` | pre-decompressed weights |
| `*-RVV-upstream` | llama.cpp `ggml/src/ggml-cpu/arch/riscv/quants.c` | dependency-free extracts of the upstream q4_0, q4_K, and iq2_xxs RVV dot kernels; see each directory's provenance |
| `q4_0-q8_0-IME-upstream` | llama.cpp `ggml/src/ggml-cpu/spacemit/ime1_kernels.cpp` | IME i8 x i4 path; supporting headers included |
| `q4_0-q8_0-IME-m4i` | `kernel.cpp`, `SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin` | 4x16 immediate reduction kernel |
| `q4_0-q8_0-IME-m4b` | `kernel.cpp`, `SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed` | 4x16 batch reduction kernel |
| `q4_0-q8_0-IME-m8b` | `kernel.cpp`, `SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed` | 8x16 batch reduction kernel |

The regular and TCM IQ2_XXS variants share the same `kernel.cpp` in each of
the `iq2_xxs-q8_K-RVV-my-br` and `iq2_xxs-q8_K-RVV-my-ir` directories. CMake
defines `USE_TCM` for the TCM object targets and supplies the shared
`iq2_xxs-q8_K-RVV-common` include directory, which contains `ggml_min.h`,
`gemm.h`, and `fine_tcm_mm.h`. TCM variants no longer have separate source
directories; CMake still gives them their distinct exported symbols and
benchmark IDs.

Each kernel directory owns its source and assembly registration in a local
`CMakeLists.txt`. `ime_add_kernel_sources()` and `ime_add_kernel_object()` apply
the root project's default kernel options. A directory can extend them with
`COMPILE_OPTIONS` or remove exact default option tokens with
`REMOVE_COMPILE_OPTIONS`. `add_ime_assembly()` provides the same controls for
assembly generation. The grouped q4_0 and q4_K kernels use this mechanism to
remove the default fixed 256-bit vector-length options, while the IQ2_XXS
directories use local object targets for their additional `USE_TCM` builds.

The registered upstream RVV kernels are local, dependency-free extracts rather
than complete copies of llama.cpp's `quants.c`. Standard GGML block definitions
live in `include/ggml.h`; custom packed kernel layouts live in
`kernels/common/types.h` so adapters and kernels share the same ABI definitions.
