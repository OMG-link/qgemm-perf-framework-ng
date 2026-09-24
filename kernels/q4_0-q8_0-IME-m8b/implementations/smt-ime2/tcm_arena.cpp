#include "tcm_arena.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

namespace pfng::bench::tcm {
namespace {

constexpr size_t kBlockBytes = 384u << 10;  // the driver's allocation granule
constexpr int kOwningCpuFirst = 8;          // block 0 belongs to cpus 8 and 9
constexpr int kOwningCpuLast = 9;

// One mmap per process, performed exactly once. Static initialisation is
// thread-safe, which is what keeps the "at most one request" rule even when
// several OpenMP threads reach this at the same time.
uint8_t *map_once() noexcept {
    const int fd = open("/dev/tcm", O_RDWR | O_SYNC);
    if (fd < 0) {
        std::fprintf(stderr, "tcm: open(/dev/tcm) failed: %s; falling back to ordinary memory\n", std::strerror(errno));
        return nullptr;
    }
    void *base = mmap(nullptr, kBlockBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    // The mapping outlives the descriptor; the descriptor is not needed again.
    close(fd);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "tcm: mmap(%zu KiB) failed: %s; falling back to ordinary memory\n", kBlockBytes >> 10, std::strerror(errno));
        return nullptr;
    }
    return static_cast<uint8_t *>(base);
}

uint8_t *arena() noexcept {
    static uint8_t *const base = map_once();
    return base;
}

} // namespace

int32_t *scratch(size_t bytes) noexcept {
    uint8_t *const base = arena();
    if (base == nullptr) {
        return nullptr;
    }
    const int cpu = sched_getcpu();
    if (cpu < kOwningCpuFirst || cpu > kOwningCpuLast) {
        return nullptr;
    }
    const size_t slice = (bytes + 255) & ~size_t(255);
    const size_t index = size_t(cpu - kOwningCpuFirst);
    if ((index + 1) * slice > kBlockBytes) {
        return nullptr;
    }
    return reinterpret_cast<int32_t *>(base + index * slice);
}

} // namespace pfng::bench::tcm
