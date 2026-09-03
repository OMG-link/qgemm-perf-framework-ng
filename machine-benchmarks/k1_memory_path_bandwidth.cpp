#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <riscv_vector.h>

#include "tcm_mm.h"

extern "C" void k1_rvv_read_m8(std::size_t rounds, const void *data, std::size_t bytes);

namespace {

constexpr std::size_t kLine = 64;
constexpr std::size_t kVectorBytes = 32;
constexpr std::size_t kTcmBytes = 128 * 1024;
constexpr std::size_t kDefaultL1Bytes = 16 * 1024;
constexpr std::size_t kDefaultL2Bytes = 256 * 1024;
constexpr std::size_t kDefaultDramBytes = 64 * 1024 * 1024;
constexpr std::size_t kDefaultTcmBytes = 64 * 1024;
constexpr std::size_t kDefaultRounds = 4096;

struct Options {
    std::string mode = "all";
    std::vector<int> cpus{0, 1, 2, 3};
    std::size_t bytes = 0;
    std::size_t rounds = kDefaultRounds;
    std::string pmu = "none";
    std::size_t load_lmul = 1;
};

struct Counter {
    const char *name;
    std::uint32_t type;
    std::uint64_t config;
};

constexpr Counter kExecutionCounters[] = {
    {"task_clock_ns", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
    {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {"vector_load_inst", PERF_TYPE_RAW, 0x39},
    {"l1d_load_access", PERF_TYPE_RAW, 0x6},
    {"l1d_load_miss", PERF_TYPE_RAW, 0x5},
};

constexpr Counter kPathCounters[] = {
    {"task_clock_ns", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
    {"l1d_prefetch_refill", PERF_TYPE_RAW, 0xae},
    {"l2_load_access", PERF_TYPE_RAW, 0xb8},
    {"l2_load_miss", PERF_TYPE_RAW, 0xb9},
    {"l2_ar_stall_cycle", PERF_TYPE_RAW, 0xbb},
};

struct CounterValue {
    const char *name = nullptr;
    std::uint64_t value = 0;
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
};

struct Buffer {
    explicit Buffer(std::size_t bytes) : bytes(bytes) {
        if (posix_memalign(reinterpret_cast<void **>(&data), kLine, bytes) != 0) data = nullptr;
        if (data) std::memset(data, 1, bytes);
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept : data(other.data), bytes(other.bytes) { other.data = nullptr; }
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
        }
        return *this;
    }
    std::uint8_t *data = nullptr;
    std::size_t bytes;
};

struct Result {
    bool ok = false;
    int requested_cpu = -1;
    int actual_cpu = -1;
    pid_t tid = -1;
    std::uint64_t bytes = 0;
    std::uint64_t checksum = 0;
    std::uint64_t begin_cycle = 0;
    std::uint64_t end_cycle = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::vector<CounterValue> counters;
};

inline std::uint64_t rdcycle() {
    std::uint64_t value;
    asm volatile("rdcycle %0" : "=r"(value) : : "memory");
    return value;
}

class ThreadCounters {
  public:
    explicit ThreadCounters(std::string_view group) {
        if (group == "none") return;
        if (group == "execution") open_group(kExecutionCounters, std::size(kExecutionCounters));
        else if (group == "path") open_group(kPathCounters, std::size(kPathCounters));
    }
    ~ThreadCounters() { for (int fd : fds_) if (fd >= 0) close(fd); }
    bool valid() const { return requested_ == 0 || fds_.size() == requested_; }
    void start() {
        if (fds_.empty()) return;
        ioctl(fds_.front(), PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds_.front(), PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    std::vector<CounterValue> stop() {
        std::vector<CounterValue> values;
        if (fds_.empty()) return values;
        ioctl(fds_.front(), PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
        struct ReadValue { std::uint64_t value, time_enabled, time_running; } raw{};
        for (std::size_t i = 0; i < fds_.size(); ++i) {
            if (read(fds_[i], &raw, sizeof(raw)) != static_cast<ssize_t>(sizeof(raw))) raw = {};
            values.push_back({names_[i], raw.value, raw.time_enabled, raw.time_running});
        }
        return values;
    }

  private:
    void open_group(const Counter *counters, std::size_t count) {
        requested_ = count;
        int leader = -1;
        for (std::size_t i = 0; i < count; ++i) {
            perf_event_attr attr{};
            attr.type = counters[i].type;
            attr.size = sizeof(attr);
            attr.config = counters[i].config;
            attr.disabled = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            const int fd = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, leader, 0));
            if (fd < 0) return;
            if (leader < 0) leader = fd;
            fds_.push_back(fd);
            names_.push_back(counters[i].name);
        }
    }
    std::size_t requested_ = 0;
    std::vector<int> fds_;
    std::vector<const char *> names_;
};

bool parse_number(std::string_view text, std::size_t &value, bool allow_zero = false) {
    if (text.empty()) return false;
    const std::string input(text);
    char *end = nullptr;
    errno = 0;
    const auto number = std::strtoull(input.c_str(), &end, 0);
    if (errno || end == input.c_str() || *end || number > SIZE_MAX || (!allow_zero && number == 0)) return false;
    value = static_cast<std::size_t>(number);
    return true;
}

bool parse_cpus(std::string_view text, std::vector<int> &cpus) {
    cpus.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::size_t cpu = 0;
        if (!parse_number(item, cpu, true) || cpu >= CPU_SETSIZE) return false;
        cpus.push_back(static_cast<int>(cpu));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return !cpus.empty();
}

bool pin_to(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

std::uint64_t rvv_read(const std::uint8_t *data, std::size_t bytes, std::size_t rounds) {
    std::uint64_t checksum = 0;
    for (std::size_t round = 0; round < rounds; ++round) {
        for (std::size_t offset = 0; offset < bytes; offset += 8 * kVectorBytes) {
            vuint8m1_t v0 = __riscv_vle8_v_u8m1(data + offset + 0 * kVectorBytes, 32);
            vuint8m1_t v1 = __riscv_vle8_v_u8m1(data + offset + 1 * kVectorBytes, 32);
            vuint8m1_t v2 = __riscv_vle8_v_u8m1(data + offset + 2 * kVectorBytes, 32);
            vuint8m1_t v3 = __riscv_vle8_v_u8m1(data + offset + 3 * kVectorBytes, 32);
            vuint8m1_t v4 = __riscv_vle8_v_u8m1(data + offset + 4 * kVectorBytes, 32);
            vuint8m1_t v5 = __riscv_vle8_v_u8m1(data + offset + 5 * kVectorBytes, 32);
            vuint8m1_t v6 = __riscv_vle8_v_u8m1(data + offset + 6 * kVectorBytes, 32);
            vuint8m1_t v7 = __riscv_vle8_v_u8m1(data + offset + 7 * kVectorBytes, 32);
            asm volatile("" : : "vr"(v0), "vr"(v1), "vr"(v2), "vr"(v3), "vr"(v4), "vr"(v5), "vr"(v6), "vr"(v7));
        }
        checksum += data[(round * kLine) % bytes];
    }
    return checksum;
}

std::uint64_t rvv_read_m8(const std::uint8_t *data, std::size_t bytes, std::size_t rounds) {
    k1_rvv_read_m8(rounds, data, bytes);
    return data[0] * rounds;
}

std::uint64_t run_read_kernel(const Options &options, const std::uint8_t *data, std::size_t bytes,
                              std::size_t rounds) {
    return options.load_lmul == 8 ? rvv_read_m8(data, bytes, rounds) : rvv_read(data, bytes, rounds);
}

void print_result(std::string_view path, const std::vector<Result> &results, std::size_t rounds) {
    auto earliest = results.front().begin;
    auto latest = results.front().end;
    auto overlap_begin = results.front().begin;
    auto overlap_end = results.front().end;
    std::uint64_t total_bytes = 0;
    std::uint64_t checksum = 0;
    std::uint64_t maximum_worker_cycles = 0;
    double shortest = std::numeric_limits<double>::max();
    double longest = 0.0;
    bool valid = !results.front().counters.empty();
    for (const auto &result : results) {
        earliest = std::min(earliest, result.begin);
        latest = std::max(latest, result.end);
        overlap_begin = std::max(overlap_begin, result.begin);
        overlap_end = std::min(overlap_end, result.end);
        total_bytes += result.bytes;
        checksum ^= result.checksum;
        maximum_worker_cycles = std::max(maximum_worker_cycles, result.end_cycle - result.begin_cycle);
        const double seconds = std::chrono::duration<double>(result.end - result.begin).count();
        shortest = std::min(shortest, seconds);
        longest = std::max(longest, seconds);
        const auto cycles = result.end_cycle - result.begin_cycle;
        std::printf("evidence path=%.*s requested_cpu=%d actual_cpu=%d tid=%d bytes=%llu cycles=%llu B/cycle=%.4f "
                    "seconds=%.9f checksum=%llu\n",
                    static_cast<int>(path.size()), path.data(), result.requested_cpu, result.actual_cpu, result.tid,
                    static_cast<unsigned long long>(result.bytes), static_cast<unsigned long long>(cycles),
                    cycles ? static_cast<double>(result.bytes) / cycles : 0.0, seconds,
                    static_cast<unsigned long long>(result.checksum));
        for (const auto &counter : result.counters) {
            const double running = counter.time_enabled ? 100.0 * counter.time_running / counter.time_enabled : 0.0;
            std::printf("pmu path=%.*s cpu=%d tid=%d event=%s value=%llu time_enabled=%llu time_running=%llu running=%.2f%%\n",
                        static_cast<int>(path.size()), path.data(), result.actual_cpu, result.tid, counter.name,
                        static_cast<unsigned long long>(counter.value), static_cast<unsigned long long>(counter.time_enabled),
                        static_cast<unsigned long long>(counter.time_running), running);
            valid = valid && running >= 99.0;
        }
        if (!result.counters.empty()) {
            const auto task_clock = std::find_if(result.counters.begin(), result.counters.end(), [](const CounterValue &counter) {
                return std::strcmp(counter.name, "task_clock_ns") == 0;
            });
            if (task_clock != result.counters.end()) {
                const double active_seconds = task_clock->value / 1e9;
                const double scheduled_ratio = seconds > 0 ? active_seconds / seconds : 0.0;
                std::printf("schedule cpu=%d tid=%d task_clock_seconds=%.9f wall_seconds=%.9f on_cpu_ratio=%.6f\n",
                            result.actual_cpu, result.tid, active_seconds, seconds, scheduled_ratio);
                valid = valid && scheduled_ratio >= 0.98;
            }
        }
    }
    const double seconds = std::chrono::duration<double>(latest - earliest).count();
    const double overlap = std::max(0.0, std::chrono::duration<double>(overlap_end - overlap_begin).count());
    const double duration_spread = shortest > 0 ? (longest - shortest) / shortest : 1.0;
    const double overlap_ratio = longest > 0 ? overlap / longest : 0.0;
    valid = valid && overlap_ratio >= 0.90;
    const auto aggregate_cycles = maximum_worker_cycles;
    std::printf("aggregate path=%.*s workers=%zu bytes=%llu rounds=%zu cycles=%llu B/cycle=%.4f "
                "makespan_seconds=%.9f overlap_seconds=%.9f checksum=%llu\n",
                static_cast<int>(path.size()), path.data(), results.size(), static_cast<unsigned long long>(total_bytes), rounds,
                static_cast<unsigned long long>(aggregate_cycles),
                aggregate_cycles ? static_cast<double>(total_bytes) / aggregate_cycles : 0.0,
                seconds, overlap, static_cast<unsigned long long>(checksum));
    std::printf("quality path=%.*s pmu=%s duration_spread=%.6f overlap_ratio=%.6f sample_valid=%s\n",
                static_cast<int>(path.size()), path.data(), results.front().counters.empty() ? "absent" : "present",
                duration_spread, overlap_ratio, valid ? "yes" : "no");
}

bool run_buffers(std::string_view path, const Options &options, std::size_t bytes, const std::vector<Buffer *> &buffers) {
    for (const auto *buffer : buffers) if (!buffer || !buffer->data || buffer->bytes < 8 * kVectorBytes) return false;
    std::atomic<std::size_t> ready{0};
    std::vector<Result> results(buffers.size());
    std::vector<std::thread> workers;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        workers.emplace_back([&, i] {
            auto &result = results[i];
            result.requested_cpu = options.cpus[i];
            result.tid = static_cast<pid_t>(syscall(SYS_gettid));
            result.ok = pin_to(result.requested_cpu);
            result.actual_cpu = sched_getcpu();
            if (result.ok) run_read_kernel(options, buffers[i]->data, bytes, 2);
            ThreadCounters counters(options.pmu);
            result.ok = result.ok && counters.valid();
            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) != buffers.size()) asm volatile("" ::: "memory");
            if (result.ok) {
                counters.start();
                result.begin = std::chrono::steady_clock::now();
                result.begin_cycle = rdcycle();
                result.checksum = run_read_kernel(options, buffers[i]->data, bytes, options.rounds);
                result.end_cycle = rdcycle();
                result.end = std::chrono::steady_clock::now();
                result.counters = counters.stop();
                result.bytes = static_cast<std::uint64_t>(bytes) * options.rounds;
            }
        });
    }
    for (auto &worker : workers) worker.join();
    for (const auto &result : results) if (!result.ok || !result.bytes) return false;
    print_result(path, results, options.rounds);
    return true;
}

void usage(const char *program) {
    std::printf("Usage: %s --mode l1|l2|dram|tcm|all [--cpus 0,1,2,3] [--bytes N] [--rounds N] "
                "[--pmu none|execution|path] [--load-lmul 1|8]\n", program);
}

bool run_tcm(const Options &options) {
    if (tcm_init() != 0) {
        std::fprintf(stderr, "tcm_init failed\n");
        return false;
    }
    std::vector<void *> mappings(options.cpus.size(), nullptr);
    std::vector<int> allocation_cpus = options.cpus;
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        std::thread allocator([&, i] {
            if (pin_to(allocation_cpus[i])) mappings[i] = tcm_malloc(kTcmBytes);
        });
        allocator.join();
    }
    bool ok = true;
    bool all_pa_valid = true;
    std::printf("tcm-resource requested_mappings=%zu mapping_size=%zu\n", mappings.size(), kTcmBytes);
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        if (!mappings[i]) { ok = false; continue; }
        const auto pa = reinterpret_cast<std::uintptr_t>(tcm_va_to_pa(mappings[i]));
        const bool valid_pa = pa >= 0xd8000000ull && pa < 0xd8080000ull && (pa % 128) == 0;
        std::printf("tcm-mapping index=%zu cpu=%d va=%p pa=%p valid_pa=%s\n", i, options.cpus[i], mappings[i],
                    reinterpret_cast<void *>(pa), valid_pa ? "yes" : "no");
        all_pa_valid = all_pa_valid && valid_pa;
        std::memset(mappings[i], 1, kTcmBytes);
    }
    std::printf("tcm-identity physical_addresses_valid=%s bank_identity=%s\n", all_pa_valid ? "yes" : "no",
                all_pa_valid ? "available" : "unavailable");
    if (ok) {
        std::atomic<std::size_t> ready{0};
        std::vector<Result> results(mappings.size());
        std::vector<std::thread> workers;
        for (std::size_t i = 0; i < mappings.size(); ++i) workers.emplace_back([&, i] {
            auto &result = results[i];
            result.requested_cpu = options.cpus[i];
            result.tid = static_cast<pid_t>(syscall(SYS_gettid));
            result.ok = pin_to(result.requested_cpu);
            result.actual_cpu = sched_getcpu();
            ThreadCounters counters(options.pmu);
            result.ok = result.ok && counters.valid();
            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) != mappings.size()) asm volatile("" ::: "memory");
            if (result.ok) {
                counters.start();
                result.begin = std::chrono::steady_clock::now();
                result.begin_cycle = rdcycle();
                result.checksum = run_read_kernel(options, static_cast<const std::uint8_t *>(mappings[i]),
                                                  kDefaultTcmBytes, options.rounds);
                result.end_cycle = rdcycle();
                result.end = std::chrono::steady_clock::now();
                result.counters = counters.stop();
                result.bytes = static_cast<std::uint64_t>(kDefaultTcmBytes) * options.rounds;
            }
        });
        for (auto &worker : workers) worker.join();
        for (const auto &result : results) ok = ok && result.ok && result.bytes;
        if (ok) print_result("TCM-register", results, options.rounds);
    }
    for (void *mapping : mappings) if (mapping) tcm_free(mapping);
    tcm_deinit();
    return ok;
}

bool run_mode(std::string_view path, const Options &options, std::size_t default_bytes, std::size_t bytes) {
    const std::size_t size = bytes ? bytes : default_bytes;
    std::vector<Buffer> storage;
    std::vector<Buffer *> buffers;
    storage.reserve(options.cpus.size());
    for (std::size_t i = 0; i < options.cpus.size(); ++i) storage.emplace_back(size);
    for (auto &buffer : storage) buffers.push_back(&buffer);
    return run_buffers(path, options, size, buffers);
}

} // namespace

int main(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help") { usage(argv[0]); return 0; }
        if (i + 1 >= argc) { usage(argv[0]); return 2; }
        if (arg == "--mode") options.mode = argv[++i];
        else if (arg == "--cpus") { if (!parse_cpus(argv[++i], options.cpus)) return 2; }
        else if (arg == "--bytes") { if (!parse_number(argv[++i], options.bytes)) return 2; }
        else if (arg == "--rounds") { if (!parse_number(argv[++i], options.rounds)) return 2; }
        else if (arg == "--pmu") options.pmu = argv[++i];
        else if (arg == "--load-lmul") { if (!parse_number(argv[++i], options.load_lmul)) return 2; }
        else { usage(argv[0]); return 2; }
    }
    if (options.cpus.empty() || options.mode == "") return 2;
    if (options.pmu != "none" && options.pmu != "execution" && options.pmu != "path") return 2;
    if (options.load_lmul != 1 && options.load_lmul != 8) return 2;
    if (options.load_lmul == 8 && options.pmu == "execution") {
        std::fprintf(stderr, "note: K1 vector_load_inst does not count LMUL=m8 vle8.v; use disassembly and kernel cycles\n");
    }
    bool ok = true;
    if (options.mode == "l1" || options.mode == "all") ok = run_mode("L1-register", options, kDefaultL1Bytes, options.bytes) && ok;
    if (options.mode == "l2" || options.mode == "all") {
        const std::size_t default_l2_bytes = options.cpus.size() == 1 ? kDefaultL2Bytes : 96 * 1024;
        ok = run_mode("L2-L1", options, default_l2_bytes, options.bytes) && ok;
    }
    if (options.mode == "dram" || options.mode == "all") ok = run_mode("DRAM-L2", options, kDefaultDramBytes, options.bytes) && ok;
    if (options.mode == "tcm" || options.mode == "all") ok = run_tcm(options) && ok;
    if (options.mode != "l1" && options.mode != "l2" && options.mode != "dram" && options.mode != "tcm" && options.mode != "all") return 2;
    return ok ? 0 : 1;
}