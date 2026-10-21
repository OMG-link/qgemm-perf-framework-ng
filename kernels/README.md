# Kernel organization

Each kernel directory describes one logical benchmark kernel. Its local
`CMakeLists.txt` must declare the platforms for which a reviewed implementation
exists:

```cmake
pfng_declare_kernel(
    ID q4_0-q8_0-IME-m8b
    NAME "Q4_0 x Q8_0 IME M8 batch reduction"
    QUANTIZATION WeightQ4_0ActivationQ8_0
    REGISTER_FUNCTION register_q4_0_ime_m8b
    PLATFORMS spacemit-k1-a60 spacemit-k3-a100)
```

The platform list is fail-closed: do not add a platform until its implementation
has been built and correctness-tested on that board.

## Platform implementations

Adapters may differ between implementations and live beside the kernel source.
Implementation directories are named for their actual dependency rather than
the target platform whenever possible:

```text
kernels/<kernel-id>/
  CMakeLists.txt
  provenance.md                    # optional source-level provenance
  implementations/
    rvv-vlen256/
      adapter.cpp
      kernel.cpp
      kernel.h
    rvv/
      adapter.cpp
      kernel.cpp
      kernel.h
    smt-ime1/
      adapter.cpp
      kernel.cpp
      kernel.h
    smt-ime2/                       # IME2 (K3 / A100)
      adapter.cpp
      kernel.cpp
      kernel.h
    spacemit-k1-a60/                # board/cache-specific implementation
      adapter.cpp
      performance.md
```

Only genuinely shared ABI and data-layout definitions belong in
`kernels/common/` or another explicitly shared directory. CMake includes only
the implementation selected by `PFNG_TARGET_PLATFORM`, and generates the list of
built-in registration calls from the enabled declarations.

`rvv-vlen256` identifies implementations that require at least 256-bit VLEN;
some have an exact 256-bit layout and enforce `vlenb == 32`. The plain `rvv`
directory is reserved for implementations without that minimum requirement.

RVV kernels declare both `spacemit-k1-a60` and `spacemit-k1-x60`. IME1 kernels
declare `spacemit-k1-a60`, because the X60 target profile does not include
`xsmtvdot`. IME2 kernels declare `spacemit-k3-a100`, whose A100 cores provide
`xsmtvdotii` and a fixed 1024-bit vector length. Kernels with both IME
generations (`q4_0-q8_0-IME-m8b` and `q4_0-q8_0-IME-upstream`) declare A60 and
A100 and pick `smt-ime1` or `smt-ime2` in their local `CMakeLists.txt`.
Cache-blocked implementations remain A60-specific because they embed assumptions
about that target's cache size.

The target profiles are independent and each declares its complete compiler,
ISA, Device Tree, performance, CPU-range, and optional deployment-prologue
metadata. Equal values between profiles are intentional duplication, not shared
defaults.

## Imported implementations

The source-level imports retain their original quantized layout, ABI, and RVV
intrinsic assumptions. Upstream provenance documents stay at the logical kernel
root, while platform performance measurements stay beside the implementation
that produced them.

Cache-blocked M4 and M8 variants reuse their corresponding base microkernels and
provide platform-specific blocked adapters. IQ2 custom kernels share common
layout headers and A60 adapter support through `iq2_xxs-q8_K-RVV-common`.

## TCM

`kernels/common/tcm_arena.{h,cpp}` provides the shared process-wide TCM
allocator. It maps the complete platform TCM once before benchmark input
creation, then performs aligned secondary allocations. A60 CPUs 0-3 share one
512 KiB arena; A100 CPUs 8-15 each use their corresponding 384 KiB block within
the 3 MiB mapping; X60 uses a no-op backend. TCM-backed IQ2 variants are still
not part of the active registry.