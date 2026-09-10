#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

using Kernel = void (*)(std::size_t, const void *, void *, void *);
extern "C" {
void unpack_conflict_full(std::size_t, const void *, void *, void *);
void unpack_conflict_load_2store(std::size_t, const void *, void *, void *);
void unpack_conflict_load_1store(std::size_t, const void *, void *, void *);
void unpack_conflict_load_only(std::size_t, const void *, void *, void *);
void unpack_conflict_store_1(std::size_t, const void *, void *, void *);
void unpack_conflict_store_2_d32(std::size_t, const void *, void *, void *);
void unpack_conflict_store_2_d64(std::size_t, const void *, void *, void *);
void unpack_conflict_store_only(std::size_t, const void *, void *, void *);
}

namespace {
constexpr std::size_t kGroupsPerPass = 1152;
constexpr std::size_t kSrcBytes = kGroupsPerPass * 128;
constexpr std::size_t kDstBytes = kGroupsPerPass * 256;
constexpr std::size_t kOffsetRoom = 64;

struct Buffer {
    void *data{};
    explicit Buffer(std::size_t bytes) {
        if (posix_memalign(&data, 4096, bytes) != 0) data = nullptr;
    }
    ~Buffer() { std::free(data); }
};
struct Case { const char *name; Kernel kernel; };
struct PerfRead { std::uint64_t value, enabled, running; };
struct Counter { std::uint32_t type; std::uint64_t config; const char *name; };

std::uint64_t checksum(const std::uint8_t *p, std::size_t size) {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < size; i += 4096) result = result * 131 + p[i];
    return result * 131 + p[size - 1];
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
int open_counter(const Counter &counter) {
    perf_event_attr attr{};
    attr.type = counter.type;
    attr.size = sizeof(attr);
    attr.config = counter.config;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.pinned = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
}
void touch(std::uint8_t *p, std::size_t size) {
    for (std::size_t i = 0; i < size; i += 4096) p[i] ^= 1;
    p[size - 1] ^= 1;
}
}  // namespace

int main(int argc, char **argv) {
    std::size_t passes = 64;
    int samples = 7;
    std::string_view selected_case;
    int selected_offset = -1;
    Counter counter{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--passes" && ++i < argc) passes = std::strtoull(argv[i], nullptr, 0);
        else if (arg == "--samples" && ++i < argc) samples = std::atoi(argv[i]);
        else if (arg == "--case" && ++i < argc) selected_case = argv[i];
        else if (arg == "--offset" && ++i < argc) selected_offset = std::atoi(argv[i]);
        else if (arg == "--counter" && ++i < argc) {
            const std::string_view value(argv[i]);
            if (value == "cycles") counter = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"};
            else if (value == "instructions") counter = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"};
            else if (value.starts_with("raw:")) {
                counter = {PERF_TYPE_RAW, std::strtoull(argv[i] + 4, nullptr, 0), argv[i]};
            } else {
                std::fprintf(stderr, "unknown counter: %s\n", argv[i]);
                return 2;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--passes N] [--samples N] [--case NAME] [--offset 0|16|32|48] [--counter cycles|instructions|raw:HEX]\n", argv[0]);
            return 2;
        }
    }
    if (passes == 0 || samples < 7) {
        std::fprintf(stderr, "passes must be >0 and samples must be >=7\n");
        return 2;
    }
    constexpr std::array<int, 4> offsets{0, 16, 32, 48};
    if (selected_offset >= 0 && std::find(offsets.begin(), offsets.end(), selected_offset) == offsets.end()) {
        std::fprintf(stderr, "offset must be 0, 16, 32, or 48\n");
        return 2;
    }
    constexpr std::array<Case, 8> cases{{
        {"full", unpack_conflict_full},
        {"load_2store", unpack_conflict_load_2store},
        {"load_1store", unpack_conflict_load_1store},
        {"load_only", unpack_conflict_load_only},
        {"store_1", unpack_conflict_store_1},
        {"store_2_d32", unpack_conflict_store_2_d32},
        {"store_2_d64", unpack_conflict_store_2_d64},
        {"store_2_d128", unpack_conflict_store_only},
    }};
    if (!selected_case.empty() && std::none_of(cases.begin(), cases.end(), [&](const Case &c) { return selected_case == c.name; })) {
        std::fprintf(stderr, "unknown case: %.*s\n", static_cast<int>(selected_case.size()), selected_case.data());
        return 2;
    }

    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(0, &cpus);
    const int affinity = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (affinity != 0) {
        std::fprintf(stderr, "affinity: %s\n", std::strerror(affinity));
        return 1;
    }

    Buffer src(kSrcBytes), dst(kDstBytes + kOffsetRoom), sink(4096);
    if (!src.data || !dst.data || !sink.data) return 1;
    auto *src_bytes = static_cast<std::uint8_t *>(src.data);
    auto *dst_bytes = static_cast<std::uint8_t *>(dst.data);
    for (std::size_t i = 0; i < kSrcBytes; ++i) src_bytes[i] = static_cast<std::uint8_t>(i * 17 + 3);
    std::memset(dst.data, 0, kDstBytes + kOffsetRoom);
    std::memset(sink.data, 0, 4096);
    touch(src_bytes, kSrcBytes);
    touch(dst_bytes, kDstBytes + kOffsetRoom);
    if (mlock(src.data, kSrcBytes) != 0 || mlock(dst.data, kDstBytes + kOffsetRoom) != 0) {
        std::fprintf(stderr, "warning: mlock failed: %s\n", std::strerror(errno));
    }

    const int perf_fd = open_counter(counter);
    if (perf_fd < 0) {
        std::fprintf(stderr, "perf_event_open %s failed: %s\n", counter.name, std::strerror(errno));
        return 3;
    }
    std::printf("meta counter=%s cpu=0 actual_cpu=%d passes=%zu groups_per_pass=%zu samples=%d src_mod64=%zu address_span_src=%zu address_span_dst=%zu timed_region=kernel_only\n",
                counter.name, sched_getcpu(), passes, kGroupsPerPass, samples,
                reinterpret_cast<std::uintptr_t>(src.data) & 63, kSrcBytes, kDstBytes);

    for (const auto &test_case : cases) {
        if (!selected_case.empty() && selected_case != test_case.name) continue;
        std::array<std::vector<double>, 4> normalized;
        for (int sample = 0; sample < samples; ++sample) {
            for (std::size_t order = 0; order < offsets.size(); ++order) {
                const std::size_t index = sample % 2 == 0 ? order : offsets.size() - 1 - order;
                const int offset = offsets[index];
                if (selected_offset >= 0 && selected_offset != offset) continue;
                void *destination = dst_bytes + offset;
                test_case.kernel(1, src.data, destination, sink.data);
                asm volatile("" ::: "memory");
                const auto wall_begin = std::chrono::steady_clock::now();
                if (ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0) != 0 || ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0) != 0) return 1;
                asm volatile("" ::: "memory");
                test_case.kernel(passes, src.data, destination, sink.data);
                asm volatile("" ::: "memory");
                if (ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0) != 0) return 1;
                const auto wall_end = std::chrono::steady_clock::now();
                PerfRead readout{};
                if (read(perf_fd, &readout, sizeof(readout)) != sizeof(readout)) return 1;
                const double running = readout.enabled ? 100.0 * readout.running / readout.enabled : 0.0;
                if (running < 99.0) {
                    std::fprintf(stderr, "running_pct %.6f below 99%%\n", running);
                    return 4;
                }
                const double per_group = static_cast<double>(readout.value) / (passes * kGroupsPerPass);
                normalized[index].push_back(per_group);
                const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin).count();
                std::printf("raw case=%s offset=%d sample=%d value=%" PRIu64 " per_group=%.6f enabled=%" PRIu64 " running=%" PRIu64 " running_pct=%.6f duration_ns=%" PRIu64 " checksum=%" PRIu64 "\n",
                            test_case.name, offset, sample, readout.value, per_group, readout.enabled,
                            readout.running, running, static_cast<std::uint64_t>(duration),
                            checksum(dst_bytes + offset, kDstBytes));
            }
        }
        for (std::size_t index = 0; index < offsets.size(); ++index) {
            if (normalized[index].empty()) continue;
            const auto [lo, hi] = std::minmax_element(normalized[index].begin(), normalized[index].end());
            std::printf("summary case=%s offset=%d median_per_group=%.6f min=%.6f max=%.6f spread_pct=%.3f\n",
                        test_case.name, offsets[index], median(normalized[index]), *lo, *hi,
                        100.0 * (*hi - *lo) / median(normalized[index]));
        }
    }
    close(perf_fd);
    return 0;
}
