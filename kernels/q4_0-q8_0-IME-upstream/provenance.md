# q4_0/q8_0 IME upstream provenance

Two implementations of this kernel exist, one per IME generation:

- `implementations/smt-ime1` vendors the IME1 (K1 / A60) sources;
- `implementations/smt-ime2` is a local extract of the IME2 (K3 / A100) q4_0
  path described in this document.

## smt-ime2 sources

Repository: `/home/linkai/projects/llama.cpp` (worktree clean, `v0.4.1-54`).

| Local file | Upstream file | Revision | Lines |
|---|---|---|---|
| `ime2_kernels.cpp` | `ggml/src/ggml-cpu/spacemit/ime2_kernels.cpp` | `81b0d882ae7fcd3118920ff26c5213abecbc8c08` | 217-322, 2430-2881, 3007-3358, 5583-5610 |

The extract carries `gemm_kernel_i8i4_mrow_ref` (the layout reference),
`gemm_kernel_i8i4_m1`, `gemm_kernel_i8i4_m4` and `gemm_kernel_i8i4` (the
dispatcher) verbatim. Everything else in `ime2_kernels.cpp` (i8-i2k, i8-i3k,
i8-i8, i8-i5, mxfp4, MoE) is not imported, because this kernel is the q4_0 x
q8_0 one.

Four changes are made to the extracted text; nothing else differs from
upstream.

1. **Includes.** `ggml-impl.h`, `ggml.h`, `ime_kernels.h` and `rvv_kernels.h`
   are replaced by one local header, `ime2_dispatch.h`, which supplies
   `GGML_ASSERT` and `q8_blk_size` on top of this project's dependency-free
   `include/ggml.h`. The `RISCV64_SPACEMIT_IME2` gate and the
   `#pragma GCC diagnostic` preamble of the upstream translation unit are
   dropped; the platform gate is now the kernel's declared platform list.
2. **Instruction prefixes.** This toolchain assembles the XSMTVDotII dot,
   pack and unpack mnemonics only with their `smt.` prefix. Upstream writes them
   bare (`vmadotsu`, `smt`-less `vpack.vv`, ...). The prefix is added to every
   occurrence, which is the same adaptation the smt-ime1 implementation already
   carries.
3. **The int32 dot variant is enabled in both microkernels.** Upstream ships two
   inner loops per microkernel and disables the first one: the `#else` branch
   stays active and uses the `.hp` dot forms, whose accumulator is fp16, while
   the `#if 0` branch is an exact int32 accumulation. This is not cosmetic, see
   "Dot precision" below. The dispatcher's own `#if 0` (which selects the
   portable `_mrow_ref` over the assembly microkernel) is left as upstream has
   it.
4. **`wrapper.cpp`** reproduces the upstream call site (`ime.cpp`,
   `forward_mul_mat`): B is consumed in 32-column blocks and A in whatever row
   count the dispatcher reports. `adapter.cpp` is this project's glue and does
   not exist upstream.

Upstream's activation quantizer (`rvv_kernels.cpp`, `quantize_a_row_i8` and
`quantize_a_4row_i8`) is deliberately **not** imported. See
"Activation quantization" below.

## ABI the adapter depends on

Activation, per K32 block and row group:

```text
|| f32 scale x rows | s16 block sum x rows | int8 quants x rows * 32 ||
```

with `rows = 4` for the M4 microkernel (152 B per K32 per four rows, i.e. 38 B
per row) and `rows = 1` for M1. The block sum is the negated sum of the stored
quants; the kernel adds `sum * 8`, which is what turns the raw 0..15 q4_0
nibbles into the -8..7 centered weights of the format.

Weight, per K32 block and 32-column block (576 B, scales first):

```text
|| fp16 scale x 32 | int4 pairs x 32 columns x 16 bytes ||
```

Byte `b` of a column carries K = 2b in its low nibble and K = 2b + 1 in its high
nibble, which is what upstream's `repack_q4_0_to_q4_0_32_bl` produces. The
adapter's repack moves the q4_0 quants and scales without re-deriving them, so it
is lossless.

The per-row and per-column strides are 38 and 18 bytes; a four-row group is four
row strides long and a 32-column group 32 column strides long, which is how the
adapter sizes its buffers.

## Activation quantization

**The upstream activation quantizer is not the q8_0 of the reference.** It
computes the block scale as a fp32 `amax / 127` and keeps it in fp32 instead of
rounding it to fp16, and it rounds the quants with the vector unit's default
round-to-nearest-even instead of `roundf`. The stored int8 values agree with the
reference except at rounding ties, but the fp32 scale makes a dequantized block
`fp32_d * q` instead of `fp16_d * q`, a systematic relative difference of about
2^-11.

`prepare()` therefore writes the upstream block layout from
`llama_reference::quantize_row_q8_0`: the reference fp16 scale widened to fp32,
the reference int8 quants, and their negated sum. Feeding upstream's own quants
instead leaves one +-1 LSB difference per rounding tie, and a single tie is
enough to push an output element past the harness tolerance: on
480 x 1536 x 1536 that costs 337 of 737280 elements (0.046%) with a worst
absolute error of 0.0077.

## Dot precision

Upstream's enabled `.hp` dot path rounds the integer K32 dot of every block to
fp16 before multiplying by the block scales, so a block contributes
`fp16(dot) * a_scale * b_scale` instead of `dot * a_scale * b_scale`. The error
is per block and does not cancel. Measured against the q4_0 x q8_0 reference at
80 x 160 with the reference quantization:

| K | blocks | failing elements | max abs | max rel |
|---|---:|---:|---:|---:|
| 32 | 1 | 0 / 12800 | 0.00197 | 0.00049 |
| 64 | 2 | 105 / 12800 | 0.00290 | 1.95 |
| 128 | 4 | 280 / 12800 | 0.00475 | 1.28 |
| 320 | 10 | 815 / 12800 | 0.00564 | 24.9 |

The enabled path is 11% faster than the int32 one on 480 x 1536 x 1536,
67,904,944 against 76,474,880 cycles, but it cannot reproduce the format within
the harness tolerance. With the int32 variant enabled both shapes verify bit
exact (`max_abs = 0`, `max_rel = 0`).

## Adapter differences

The adapter is written for this benchmark harness, not for ggml:

- the activation is quantized and packed once in `prepare`, because the harness
  reuses one input for every timed iteration, where ggml re-quantizes per call;
- the 32-column weight blocks are distributed over OpenMP threads;
- `ldc = n`, so the kernel writes the output row-major;
- every call passes four activation rows, which is what the M4 microkernel
  takes, so the M1 microkernel is imported but not exercised by any benchmark
  shape (`validate_shape<4, 32, 32>` keeps `m` a multiple of four).

Measured on `spacemit-k3`, cpu8, 480 x 1536 x 1536, one thread, int32 variant:
76,474,880 cycles, 14.81 i8-i32 MAC/cycle, 1.45% of one matrix unit.
