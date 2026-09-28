#ifndef PFNG_KERNEL_COMMON_TCM_ARENA_H
#define PFNG_KERNEL_COMMON_TCM_ARENA_H

#include <cstddef>

namespace pfng::bench::tcm {

// Platform TCM capabilities.
bool supported() noexcept;
size_t capacity() noexcept;
size_t allocation_granule() noexcept;

// Map the platform's complete TCM exactly once and initialise the process-wide
// secondary allocator. Returns false if the platform has TCM but it could not
// be mapped. Platforms without TCM treat initialisation as a successful no-op.
// The mapping remains live until process exit.
bool initialize() noexcept;

// Allocate from the TCM region available to the calling CPU. Returns nullptr
// when TCM is unsupported, unavailable, exhausted, or the CPU is outside the
// platform's TCM-capable set.
//
// A60 CPUs 0-3 share one 512 KiB allocation arena. A100 CPUs 8-15 each allocate
// from their own 384 KiB block within the single 3 MiB mapping.
//
// `alignment` must be a nonzero power of two. No /dev/tcm operation is attempted
// on platforms without TCM.
void *allocate(size_t bytes, size_t alignment = 256) noexcept;

// Return a pointer obtained from allocate(). nullptr is accepted. Invalid or
// already-freed pointers are ignored.
void deallocate(void *pointer) noexcept;

} // namespace pfng::bench::tcm

#endif