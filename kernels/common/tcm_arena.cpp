#include "tcm_arena.h"

#include <cerrno>
#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>

#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace pfng::bench::tcm {
namespace {

constexpr bool is_power_of_two(size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)

constexpr size_t kMaxAllocations = 256;
// Inserting a returned block before coalescing can transiently require one more
// entry than the maximum steady-state number of gaps.
constexpr size_t kMaxFreeBlocks = kMaxAllocations + 2;

struct Block {
    size_t offset = 0;
    size_t size = 0;
};

struct Allocation {
    void *pointer = nullptr;
    Block block;
};

struct Arena {
    uint8_t *base = nullptr;
    size_t bytes = 0;
    std::mutex mutex;
    std::array<Block, kMaxFreeBlocks> free_blocks{};
    std::array<Allocation, kMaxAllocations> allocations{};
    size_t free_count = 0;
    size_t allocation_count = 0;

    void reset(uint8_t *new_base, size_t new_bytes) noexcept {
        base = new_base;
        bytes = new_bytes;
        free_count = new_base == nullptr ? 0 : 1;
        allocation_count = 0;
        if (free_count != 0) {
            free_blocks[0] = {0, new_bytes};
        }
    }
};

uintptr_t align_up(uintptr_t value, size_t alignment) noexcept {
    return (value + alignment - 1) & ~uintptr_t(alignment - 1);
}

uint8_t *map_tcm(size_t bytes) noexcept {
    const int fd = open("/dev/tcm", O_RDWR | O_SYNC);
    if (fd < 0) {
        std::fprintf(stderr, "tcm: open(/dev/tcm) failed: %s; falling back to ordinary memory\n", std::strerror(errno));
        return nullptr;
    }
    void *base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "tcm: mmap(%zu KiB) failed: %s; falling back to ordinary memory\n", bytes >> 10, std::strerror(errno));
        return nullptr;
    }
    return static_cast<uint8_t *>(base);
}

#endif

#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60)

constexpr size_t kCapacity = 512u << 10;
constexpr size_t kGranule = 128u << 10;
constexpr int kFirstCpu = 0;
constexpr int kCpuCount = 4;
constexpr size_t kArenaCount = 1;

#elif defined(PFNG_PLATFORM_SPACEMIT_K3_A100)

constexpr size_t kCapacity = 3u << 20;
constexpr size_t kGranule = 384u << 10;
constexpr int kFirstCpu = 8;
constexpr int kCpuCount = 8;
constexpr size_t kArenaCount = kCpuCount;

#endif

#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)

struct Manager {
    std::once_flag initialize_once;
    uint8_t *base = nullptr;
    bool initialization_succeeded = false;
    std::array<Arena, kArenaCount> arenas;
};

Manager &manager() noexcept {
    static Manager value;
    return value;
}

Arena *arena_for_cpu(int cpu) noexcept {
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60)
    if (cpu < kFirstCpu || cpu >= kFirstCpu + kCpuCount) {
        return nullptr;
    }
    return &manager().arenas[0];
#else
    const int index = cpu - kFirstCpu;
    if (index < 0 || index >= kCpuCount) {
        return nullptr;
    }
    return &manager().arenas[size_t(index)];
#endif
}

void *allocate_from(Arena &arena, size_t bytes, size_t alignment) noexcept {
    std::lock_guard lock(arena.mutex);
    if (arena.base == nullptr || arena.allocation_count == kMaxAllocations) {
        return nullptr;
    }
    for (size_t index = 0; index < arena.free_count; ++index) {
        const Block free = arena.free_blocks[index];
        const uintptr_t address = align_up(reinterpret_cast<uintptr_t>(arena.base + free.offset), alignment);
        const size_t aligned_offset = size_t(address - reinterpret_cast<uintptr_t>(arena.base));
        if (aligned_offset < free.offset || aligned_offset - free.offset > free.size || bytes > free.size - (aligned_offset - free.offset)) {
            continue;
        }

        const size_t prefix = aligned_offset - free.offset;
        const size_t suffix_offset = aligned_offset + bytes;
        const size_t suffix = free.offset + free.size - suffix_offset;
        if (prefix != 0 && suffix != 0 && arena.free_count == kMaxFreeBlocks) {
            return nullptr;
        }
        if (prefix != 0 && suffix != 0) {
            for (size_t move = arena.free_count; move > index + 1; --move) {
                arena.free_blocks[move] = arena.free_blocks[move - 1];
            }
            arena.free_blocks[index] = {free.offset, prefix};
            arena.free_blocks[index + 1] = {suffix_offset, suffix};
            ++arena.free_count;
        } else if (prefix != 0) {
            arena.free_blocks[index] = {free.offset, prefix};
        } else if (suffix != 0) {
            arena.free_blocks[index] = {suffix_offset, suffix};
        } else {
            for (size_t move = index + 1; move < arena.free_count; ++move) {
                arena.free_blocks[move - 1] = arena.free_blocks[move];
            }
            --arena.free_count;
        }

        void *const pointer = arena.base + aligned_offset;
        arena.allocations[arena.allocation_count++] = {pointer, {aligned_offset, bytes}};
        return pointer;
    }
    return nullptr;
}

void insert_free_block(Arena &arena, Block block) noexcept {
    size_t index = 0;
    while (index < arena.free_count && arena.free_blocks[index].offset < block.offset) {
        ++index;
    }
    for (size_t move = arena.free_count; move > index; --move) {
        arena.free_blocks[move] = arena.free_blocks[move - 1];
    }
    arena.free_blocks[index] = block;
    ++arena.free_count;

    if (index > 0 && arena.free_blocks[index - 1].offset + arena.free_blocks[index - 1].size == arena.free_blocks[index].offset) {
        arena.free_blocks[index - 1].size += arena.free_blocks[index].size;
        for (size_t move = index + 1; move < arena.free_count; ++move) {
            arena.free_blocks[move - 1] = arena.free_blocks[move];
        }
        --arena.free_count;
        --index;
    }
    if (index + 1 < arena.free_count && arena.free_blocks[index].offset + arena.free_blocks[index].size == arena.free_blocks[index + 1].offset) {
        arena.free_blocks[index].size += arena.free_blocks[index + 1].size;
        for (size_t move = index + 2; move < arena.free_count; ++move) {
            arena.free_blocks[move - 1] = arena.free_blocks[move];
        }
        --arena.free_count;
    }
}

#endif

} // namespace

bool supported() noexcept {
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    return true;
#else
    return false;
#endif
}

size_t capacity() noexcept {
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    return kCapacity;
#else
    return 0;
#endif
}

size_t allocation_granule() noexcept {
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    return kGranule;
#else
    return 0;
#endif
}

bool initialize() noexcept {
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    Manager &state = manager();
    std::call_once(state.initialize_once, [&state] {
        state.base = map_tcm(kCapacity);
        state.initialization_succeeded = state.base != nullptr;
        if (!state.initialization_succeeded) {
            return;
        }
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60)
        state.arenas[0].reset(state.base, kCapacity);
#else
        for (size_t index = 0; index < kArenaCount; ++index) {
            state.arenas[index].reset(state.base + index * kGranule, kGranule);
        }
#endif
    });
    return state.initialization_succeeded;
#else
    return true;
#endif
}

void *allocate(size_t bytes, size_t alignment) noexcept {
    if (bytes == 0 || !is_power_of_two(alignment)) {
        return nullptr;
    }

#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    Manager &state = manager();
    if (!state.initialization_succeeded) {
        return nullptr;
    }
    Arena *const arena = arena_for_cpu(sched_getcpu());
    if (arena == nullptr || bytes > arena->bytes) {
        return nullptr;
    }
    return allocate_from(*arena, bytes, alignment);
#else
    (void)bytes;
    (void)alignment;
    return nullptr;
#endif
}

void deallocate(void *pointer) noexcept {
    if (pointer == nullptr) {
        return;
    }
#if defined(PFNG_PLATFORM_SPACEMIT_K1_A60) || defined(PFNG_PLATFORM_SPACEMIT_K3_A100)
    Manager &state = manager();
    for (Arena &arena : state.arenas) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
        const uintptr_t begin = reinterpret_cast<uintptr_t>(arena.base);
        if (arena.base == nullptr || address < begin || address >= begin + arena.bytes) {
            continue;
        }
        std::lock_guard lock(arena.mutex);
        for (size_t index = 0; index < arena.allocation_count; ++index) {
            if (arena.allocations[index].pointer != pointer) {
                continue;
            }
            const Block block = arena.allocations[index].block;
            arena.allocations[index] = arena.allocations[--arena.allocation_count];
            if (arena.free_count < kMaxFreeBlocks) {
                insert_free_block(arena, block);
            }
            return;
        }
        return;
    }
#else
    (void)pointer;
#endif
}

} // namespace pfng::bench::tcm