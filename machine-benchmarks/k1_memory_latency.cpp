#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <random>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <vector>

extern "C" void *k1_memory_latency_chase(std::size_t iterations, void *head);

namespace {

constexpr std::size_t kLine = 64;
constexpr std::size_t kL1Bytes = 16 * 1024;
constexpr std::size_t kL2Bytes = 256 * 1024;
constexpr std::size_t kDramBytes = 64 * 1024 * 1024;
constexpr std::size_t kL1ScrubBytes = 64 * 1024;
constexpr std::size_t kDramEvictionBytes = 96 * 1024 * 1024;

struct Options {
    std::string mode = "all";
    std::string pmu = "none";
    int cpu = 0;
    std::size_t iterations = 1024 * 1024;
    std::size_t samples = 7;
    std::size_t l1_bytes = kL1Bytes;
    std::size_t l2_bytes = kL2Bytes;
    std::size_t dram_bytes = kDramBytes;
};

struct Buffer {
    explicit Buffer(std::size_t size) : bytes(size) {
        if (posix_memalign(reinterpret_cast<void **>(&data), kLine, bytes) != 0) data = nullptr;
        if (data) std::memset(data, 0, bytes);
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    std::uint8_t *data = nullptr;
    std::size_t bytes = 0;
};

struct EventSpec { const char *name; std::uint32_t type; std::uint64_t config; };
constexpr EventSpec kEvents[] = {
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {"l1d_load_miss", PERF_TYPE_RAW, 0x5},
    {"l2_load_access", PERF_TYPE_RAW, 0xb8},
    {"l2_load_miss", PERF_TYPE_RAW, 0xb9},
};

struct EventValue {
    const char *name = nullptr;
    std::uint64_t value = 0;
    std::uint64_t enabled = 0;
    std::uint64_t running = 0;
};

class Counters {
  public:
    explicit Counters(bool path_events) : requested_(path_events ? std::size(kEvents) : 1) {
        int leader = -1;
        for (std::size_t i = 0; i < requested_; ++i) {
            const auto &event = kEvents[i];
            perf_event_attr attr{};
            attr.type = event.type;
            attr.size = sizeof(attr);
            attr.config = event.config;
            attr.disabled = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            const int fd = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, leader, PERF_FLAG_FD_CLOEXEC));
            if (fd < 0) return;
            if (leader < 0) leader = fd;
            fds_.push_back(fd);
            names_.push_back(event.name);
        }
    }
    ~Counters() { for (int fd : fds_) close(fd); }
    bool valid() const { return fds_.size() == requested_; }
    void start() {
        if (fds_.empty()) return;
        ioctl(fds_.front(), PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds_.front(), PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    std::vector<EventValue> stop() {
        std::vector<EventValue> values;
        if (fds_.empty()) return values;
        ioctl(fds_.front(), PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        struct Raw { std::uint64_t value, enabled, running; } raw{};
        for (std::size_t i = 0; i < fds_.size(); ++i) {
            if (read(fds_[i], &raw, sizeof(raw)) != static_cast<ssize_t>(sizeof(raw))) raw = {};
            values.push_back({names_[i], raw.value, raw.enabled, raw.running});
        }
        return values;
    }
  private:
    std::size_t requested_ = 0;
    std::vector<int> fds_;
    std::vector<const char *> names_;
};

struct Sample { double cycles_per_load = 0; std::uint64_t cycles = 0; std::uintptr_t sink = 0; };
struct Summary { std::string name; double median = 0; double minimum = 0; double maximum = 0; };

bool parse_size(std::string_view text, std::size_t &value, bool zero = false) {
    if (text.empty()) return false;
    std::string input(text);
    char *end = nullptr;
    errno = 0;
    const auto parsed = std::strtoull(input.c_str(), &end, 0);
    if (errno || end == input.c_str() || *end || parsed > SIZE_MAX || (!zero && parsed == 0)) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

void make_ring(Buffer &buffer, std::uint64_t seed) {
    const std::size_t lines = buffer.bytes / kLine;
    std::vector<std::size_t> order(lines);
    for (std::size_t i = 0; i < lines; ++i) order[i] = i;
    std::mt19937_64 random(seed);
    std::shuffle(order.begin(), order.end(), random);
    for (std::size_t i = 0; i < lines; ++i) {
        *reinterpret_cast<void **>(buffer.data + order[i] * kLine) = buffer.data + order[(i + 1) % lines] * kLine;
    }
}

void scan(const Buffer &buffer) {
    volatile const std::uint8_t *data = buffer.data;
    unsigned sum = 0;
    for (std::size_t offset = 0; offset < buffer.bytes; offset += kLine) sum += data[offset];
    asm volatile("" : : "r"(sum) : "memory");
}

void prepare(std::string_view name, Buffer &ring, const Buffer &l1_scrub, const Buffer &dram_eviction) {
    if (name == "l1") {
        k1_memory_latency_chase(ring.bytes / kLine, ring.data);
    } else if (name == "l2") {
        k1_memory_latency_chase(ring.bytes / kLine, ring.data);
        scan(l1_scrub);
    } else {
        scan(dram_eviction);
    }
    asm volatile("" : : : "memory");
}

Sample measure(std::string_view name, Buffer &ring, const Buffer &l1_scrub, const Buffer &dram_eviction,
               const Options &options, std::size_t sample_index) {
    prepare(name, ring, l1_scrub, dram_eviction);
    Counters counters(options.pmu == "path");
    if (!counters.valid()) {
        std::fprintf(stderr, "failed to open complete PMU group: %s\n", std::strerror(errno));
        std::exit(3);
    }
    counters.start();
    void *sink = k1_memory_latency_chase(options.iterations, ring.data);
    auto events = counters.stop();
    const auto cycles = events.front().value;
    std::printf("sample level=%.*s index=%zu bytes=%zu iterations=%zu cycles=%" PRIu64
                " cycles_per_load=%.6f sink=%" PRIuPTR "\n",
                static_cast<int>(name.size()), name.data(), sample_index, ring.bytes, options.iterations, cycles,
                static_cast<double>(cycles) / options.iterations, reinterpret_cast<std::uintptr_t>(sink));
    for (const auto &event : events) {
        const double running = event.enabled ? 100.0 * event.running / event.enabled : 0.0;
        std::printf("pmu level=%.*s index=%zu event=%s value=%" PRIu64 " per_load=%.6f running_pct=%.3f\n",
                    static_cast<int>(name.size()), name.data(), sample_index, event.name, event.value,
                    static_cast<double>(event.value) / options.iterations, running);
        if (running < 99.0) {
            std::fprintf(stderr, "PMU multiplexing exceeded 1%%\n");
            std::exit(4);
        }
    }
    return {static_cast<double>(cycles) / options.iterations, cycles, reinterpret_cast<std::uintptr_t>(sink)};
}

Summary run_level(std::string_view name, Buffer &ring, const Buffer &l1_scrub, const Buffer &dram_eviction,
                  const Options &options) {
    std::vector<double> values;
    values.reserve(options.samples);
    for (std::size_t sample = 0; sample < options.samples; ++sample) {
        values.push_back(measure(name, ring, l1_scrub, dram_eviction, options, sample).cycles_per_load);
    }
    std::sort(values.begin(), values.end());
    Summary result{std::string(name), values[values.size() / 2], values.front(), values.back()};
    std::printf("summary level=%s bytes=%zu samples=%zu median_cycles_per_load=%.6f min=%.6f max=%.6f\n",
                result.name.c_str(), ring.bytes, values.size(), result.median, result.minimum, result.maximum);
    return result;
}

void usage(const char *program) {
    std::printf("Usage: %s [--mode l1|l2|dram|all] [--cpu N] [--iterations N] [--samples N] "
                "[--pmu none|path] [--l1-bytes N] [--l2-bytes N] [--dram-bytes N]\n", program);
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help") { usage(argv[0]); return 0; }
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        if (arg == "--mode") options.mode = argv[++i];
        else if (arg == "--pmu") options.pmu = argv[++i];
        else if (arg == "--cpu") { std::size_t value; if (!parse_size(argv[++i], value, true) || value >= CPU_SETSIZE) return 2; options.cpu = static_cast<int>(value); }
        else if (arg == "--iterations") { if (!parse_size(argv[++i], options.iterations)) return 2; }
        else if (arg == "--samples") { if (!parse_size(argv[++i], options.samples)) return 2; }
        else if (arg == "--l1-bytes") { if (!parse_size(argv[++i], options.l1_bytes)) return 2; }
        else if (arg == "--l2-bytes") { if (!parse_size(argv[++i], options.l2_bytes)) return 2; }
        else if (arg == "--dram-bytes") { if (!parse_size(argv[++i], options.dram_bytes)) return 2; }
        else { usage(argv[0]); return 2; }
    }
    if (options.mode != "l1" && options.mode != "l2" && options.mode != "dram" && options.mode != "all") return 2;
    if (options.pmu != "none" && options.pmu != "path") return 2;
    if (options.samples % 2 == 0) { std::fprintf(stderr, "--samples must be odd\n"); return 2; }
    if ((options.l1_bytes | options.l2_bytes | options.dram_bytes) % kLine != 0) {
        std::fprintf(stderr, "working-set sizes must be multiples of 64 bytes\n"); return 2;
    }
    if (!pin_to(options.cpu) || sched_getcpu() != options.cpu) {
        std::fprintf(stderr, "failed to pin to CPU %d\n", options.cpu); return 1;
    }

    Buffer l1(options.l1_bytes), l2(options.l2_bytes), dram(options.dram_bytes);
    Buffer l1_scrub(kL1ScrubBytes), dram_eviction(kDramEvictionBytes);
    if (!l1.data || !l2.data || !dram.data || !l1_scrub.data || !dram_eviction.data) return 1;
    make_ring(l1, 0x1111); make_ring(l2, 0x2222); make_ring(dram, 0x3333);
    scan(l1_scrub); scan(dram_eviction);

    std::vector<Summary> summaries;
    if (options.mode == "l1" || options.mode == "all") summaries.push_back(run_level("l1", l1, l1_scrub, dram_eviction, options));
    if (options.mode == "l2" || options.mode == "all") summaries.push_back(run_level("l2", l2, l1_scrub, dram_eviction, options));
    if (options.mode == "dram" || options.mode == "all") summaries.push_back(run_level("dram", dram, l1_scrub, dram_eviction, options));
    if (options.mode == "all") {
        const double l1_cycles = summaries[0].median;
        const double l2_cycles = summaries[1].median;
        const double dram_cycles = summaries[2].median;
        std::printf("penalty l2_minus_l1_cycles=%.6f dram_minus_l2_cycles=%.6f dram_minus_l1_cycles=%.6f\n",
                    l2_cycles - l1_cycles, dram_cycles - l2_cycles, dram_cycles - l1_cycles);
    }
    return 0;
}
