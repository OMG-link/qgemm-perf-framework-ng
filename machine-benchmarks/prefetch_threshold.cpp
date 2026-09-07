#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <random>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

extern "C" {
std::uintptr_t prefetch_gap0(const void *, std::size_t, std::size_t, std::size_t);
std::uintptr_t prefetch_gap16(const void *, std::size_t, std::size_t, std::size_t);
std::uintptr_t prefetch_gap64(const void *, std::size_t, std::size_t, std::size_t);
std::uintptr_t prefetch_gap256(const void *, std::size_t, std::size_t, std::size_t);
}

namespace {
constexpr std::size_t kLine = 64;
constexpr std::size_t kRegion = 4096;
constexpr std::size_t kTrashBytes = 96 * 1024 * 1024;

struct Aligned {
    void *data = nullptr;
    explicit Aligned(std::size_t bytes, std::size_t alignment = 4096) {
        if (posix_memalign(&data, alignment, bytes)) data = nullptr;
    }
    ~Aligned() { std::free(data); }
};

struct PerfRead { std::uint64_t value, enabled, running; };

int open_event(std::uint32_t type, std::uint64_t config, int group_fd, bool leader) {
    perf_event_attr attr{};
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = leader;
    attr.pinned = leader;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, group_fd, PERF_FLAG_FD_CLOEXEC));
}

void thrash(void *memory) {
    auto *p = static_cast<volatile unsigned char *>(memory);
    unsigned value = 0;
    for (std::size_t i = 0; i < kTrashBytes; i += kLine) value += p[i];
    asm volatile("" : : "r"(value) : "memory");
}

using Sequence = std::uintptr_t (*)(const void *, std::size_t, std::size_t, std::size_t);
Sequence sequence_for(std::size_t gap) {
    if (gap == 0) return prefetch_gap0;
    if (gap == 16) return prefetch_gap16;
    if (gap == 64) return prefetch_gap64;
    if (gap == 256) return prefetch_gap256;
    return nullptr;
}

bool parse_size(const char *text, std::size_t &value) {
    errno = 0;
    char *end = nullptr;
    const auto parsed = std::strtoull(text, &end, 0);
    if (errno || end == text || *end) return false;
    value = parsed;
    return true;
}
} // namespace

int main(int argc, char **argv) {
    std::size_t trials = 4096, samples = 7, max_train = 12;
    std::vector<std::size_t> gaps{16, 64, 256};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--trials" && i + 1 < argc && parse_size(argv[++i], trials)) {}
        else if (arg == "--samples" && i + 1 < argc && parse_size(argv[++i], samples)) {}
        else if (arg == "--max-train" && i + 1 < argc && parse_size(argv[++i], max_train)) {}
        else if (arg == "--gap" && i + 1 < argc) {
            std::size_t gap = 0;
            if (!parse_size(argv[++i], gap) || !sequence_for(gap)) {
                std::fprintf(stderr, "--gap must be 0, 16, 64, or 256\n");
                return 2;
            }
            gaps.assign(1, gap);
        } else {
            std::fprintf(stderr, "usage: %s [--trials N] [--samples N] [--max-train N] [--gap 0|16|64|256]\n", argv[0]);
            return 2;
        }
    }
    if (!trials || !samples || max_train >= kRegion / kLine) return 2;

    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(0, &cpus);
    if (const int error = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus)) {
        std::fprintf(stderr, "affinity: %s\n", std::strerror(error));
        return 1;
    }

    Aligned regions(trials * kRegion), trash_memory(kTrashBytes);
    if (!regions.data || !trash_memory.data) return 1;
    auto *bytes = static_cast<unsigned char *>(regions.data);
    for (std::size_t i = 0; i < trials * kRegion; ++i) bytes[i] = static_cast<unsigned char>((i * 131 + 17) | 1);
    std::memset(trash_memory.data, 1, kTrashBytes);
    std::vector<void *> region_order(trials);
    for (std::size_t i = 0; i < trials; ++i) region_order[i] = bytes + i * kRegion;
    std::mt19937_64 random(0x6b315f7072656665ULL);
    std::shuffle(region_order.begin(), region_order.end(), random);

    const std::uint64_t l1d_load_miss = PERF_COUNT_HW_CACHE_L1D |
        (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    const int cycles_fd = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, true);
    const int misses_fd = cycles_fd < 0 ? -1 : open_event(PERF_TYPE_HW_CACHE, l1d_load_miss, cycles_fd, false);
    if (cycles_fd < 0 || misses_fd < 0) {
        std::fprintf(stderr, "perf_event_open: %s\n", std::strerror(errno));
        if (cycles_fd >= 0) close(cycles_fd);
        return 3;
    }

    for (const std::size_t gap : gaps) {
        const Sequence run = sequence_for(gap);
        for (std::size_t train = 0; train <= max_train; ++train) {
            std::vector<double> misses;
            misses.reserve(samples);
            for (std::size_t sample = 0; sample < samples; ++sample) {
                thrash(trash_memory.data);
                std::uintptr_t pointer_sink = 0;
                for (void *pointer : region_order) pointer_sink ^= reinterpret_cast<std::uintptr_t>(pointer);
                asm volatile("fence rw,rw" : : "r"(pointer_sink) : "memory");
                if (ioctl(cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 ||
                    ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) return 4;
                const auto sink = run(region_order.data(), trials, train, 0);
                if (ioctl(cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0) return 4;
                PerfRead cycles{}, miss{};
                if (read(cycles_fd, &cycles, sizeof(cycles)) != sizeof(cycles) ||
                    read(misses_fd, &miss, sizeof(miss)) != sizeof(miss)) return 4;
                const double cycle_ratio = cycles.enabled ? static_cast<double>(cycles.running) / cycles.enabled : 0;
                const double miss_ratio = miss.enabled ? static_cast<double>(miss.running) / miss.enabled : 0;
                if (cycle_ratio < .99 || miss_ratio < .99) {
                    std::fprintf(stderr, "counter multiplexed\n");
                    return 4;
                }
                const double per_trial = static_cast<double>(miss.value) / trials;
                misses.push_back(per_trial);
                std::printf("raw gap=%zu train=%zu sample=%zu trials=%zu cycles_per_trial=%.6f l1d_miss_per_trial=%.6f running_pct=%.3f sink=%" PRIuPTR "\n",
                            gap, train, sample, trials, static_cast<double>(cycles.value) / trials,
                            per_trial, 100.0 * std::min(cycle_ratio, miss_ratio), sink);
            }
            std::sort(misses.begin(), misses.end());
            std::printf("summary gap=%zu train=%zu loads=%zu expected_cold_misses=%zu median_l1d_miss_per_trial=%.6f\n",
                        gap, train, train + 1, train + 1, misses[misses.size() / 2]);
        }
    }
    close(misses_fd);
    close(cycles_fd);
    return 0;
}
