#ifndef PFNG_KERNEL_Q4_0_Q8_0_M8B_IME2_TCM_ARENA_H
#define PFNG_KERNEL_Q4_0_Q8_0_M8B_IME2_TCM_ARENA_H

#include <cstddef>
#include <cstdint>

namespace pfng::bench::tcm {

// Per-thread scratch buffer in the A100's tightly coupled memory.
//
// The /dev/tcm driver is fragile: a process must perform exactly one mmap, and
// the length must be a multiple of 384 KiB. Violating either leaves the driver
// in a broken state that hangs the *next* TCM call in an unkillable D state.
// This arena therefore performs a single 384 KiB mmap, once per process, on
// first use, and never unmaps it (process exit releases it). 384 KiB is the
// smallest legal request; asking for the whole 3 MiB would only widen the
// window for colliding with another holder.
//
// TCM blocks are private to a hart pair. The device tree (soc/tcm@0) pairs
// blocks 0/1 with cpus 8 and 9, blocks 2/3 with 10 and 11, and so on, matching
// the matrix-unit pairing. A 384 KiB request yields the physically lowest free
// block, i.e. block 0, so only cpus 8 and 9 can reach it at full speed; other
// cores see it about 330x slower. `scratch` therefore returns nullptr unless the
// calling thread runs on cpu 8 or 9, and the caller must fall back to ordinary
// memory.
//
// `bytes` is rounded up to 256 so successive slices stay on a vector-register
// boundary. Returns nullptr if TCM is unavailable for any reason.
int32_t *scratch(size_t bytes) noexcept;

} // namespace pfng::bench::tcm

#endif
