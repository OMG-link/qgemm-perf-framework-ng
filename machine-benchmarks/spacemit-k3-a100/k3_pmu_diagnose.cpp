#include "k3_a100.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

constexpr uint64_t kBytesPerThread = 64ull << 20;
constexpr int kDefaultSamples = 3;
using Kernel = void (*)(const uint8_t *, size_t, uint64_t);

__attribute__((noinline)) void read_scalar(const uint8_t * base, size_t bytes, uint64_t rounds) {
    asm volatile("2:\n\t"
                 "mv t0, %0\n\tadd t1, %0, %1\n\t"
                 "1:\n\t"
                 "ld a0, 0(t0)\n\tld a1, 8(t0)\n\tld a2, 16(t0)\n\tld a3, 24(t0)\n\t"
                 "ld a4, 32(t0)\n\tld a5, 40(t0)\n\tld a6, 48(t0)\n\tld a7, 56(t0)\n\t"
                 "addi t0, t0, 64\n\tbltu t0, t1, 1b\n\t"
                 "addi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(rounds)
                 :
                 : "t0", "t1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "memory");
}

__attribute__((noinline)) void read_rvv_m1(const uint8_t * base, size_t bytes, uint64_t rounds) {
    asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n\t"
                 "2:\n\tmv t1, %0\n\tadd t2, %0, %1\n\t"
                 "1:\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v1, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v2, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v3, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v4, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v5, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v6, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v7, (t1)\n\taddi t1, t1, 128\n\t"
                 "bltu t1, t2, 1b\n\taddi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(rounds)
                 :
                 : "t0", "t1", "t2", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

__attribute__((noinline)) void read_rvv_m8(const uint8_t * base, size_t bytes, uint64_t rounds) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "2:\n\tmv t1, %0\n\tadd t2, %0, %1\n\t"
                 "1:\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 "bltu t1, t2, 1b\n\taddi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(rounds)
                 :
                 : "t0", "t1", "t2", "memory", "v0", "v8", "v16", "v24");
}

class Barrier {
  public:
    explicit Barrier(int target) : target_(target) {}
    void wait() {
        const int generation = generation_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == target_) {
            count_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            return;
        }
        while (generation_.load(std::memory_order_acquire) == generation) asm volatile("" ::: "memory");
    }
  private:
    std::atomic<int> count_{0};
    std::atomic<int> generation_{0};
    int target_;
};

struct Mapping { void * address = MAP_FAILED; size_t size = 0; uint8_t * data = nullptr; bool huge = false; };

bool mapping_uses_thp(const void * address) {
    FILE * file = std::fopen("/proc/self/smaps", "r");
    if (!file) return false;
    const uintptr_t needle = reinterpret_cast<uintptr_t>(address);
    char line[512];
    bool in_vma = false, huge = false;
    while (std::fgets(line, sizeof(line), file)) {
        uintptr_t begin = 0, end = 0;
        if (std::sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &begin, &end) == 2) {
            if (in_vma) break;
            in_vma = needle >= begin && needle < end;
        } else if (in_vma) {
            unsigned long kib = 0;
            if (std::sscanf(line, "AnonHugePages: %lu kB", &kib) == 1 && kib > 0) huge = true;
        }
    }
    std::fclose(file);
    return huge;
}

Mapping allocate_buffer(size_t working_set) {
    constexpr size_t huge_page = 2u << 20;
    const size_t mapped = ((working_set + huge_page - 1) / huge_page) * huge_page;
    Mapping result;
    result.size = mapped;
    result.address = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (result.address == MAP_FAILED) return result;
    result.data = static_cast<uint8_t *>(result.address);
    if (madvise(result.address, mapped, MADV_HUGEPAGE) != 0) {
        munmap(result.address, mapped);
        result.address = MAP_FAILED;
        result.data = nullptr;
        return result;
    }
    for (size_t i = 0; i < mapped; i += 4096) result.data[i] = uint8_t(i >> 12);
    result.huge = mapping_uses_thp(result.data);
    return result;
}

struct Event {
    const char * name;
    uint32_t type;
    uint64_t config;

    constexpr Event(const char * event_name, uint32_t event_type, uint64_t event_config)
        : name(event_name), type(event_type), config(event_config) {}
};
struct RawReading { uint64_t value = 0, time_enabled = 0, time_running = 0; };

class EventGroup {
  public:
    ~EventGroup() { if (candidate_fd_ >= 0) close(candidate_fd_); if (cycle_fd_ >= 0) close(cycle_fd_); }
    bool open(const Event & event) {
        cycle_fd_ = open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, true);
        if (cycle_fd_ < 0) return false;
        candidate_fd_ = open_one(event.type, event.config, cycle_fd_, false);
        return candidate_fd_ >= 0;
    }
    bool start() {
        return ioctl(cycle_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == 0 &&
               ioctl(cycle_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == 0;
    }
    bool stop(RawReading & cycles, RawReading & candidate) {
        if (ioctl(cycle_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) return false;
        return read(cycle_fd_, &cycles, sizeof(cycles)) == ssize_t(sizeof(cycles)) &&
               read(candidate_fd_, &candidate, sizeof(candidate)) == ssize_t(sizeof(candidate));
    }
  private:
    static int open_one(uint32_t type, uint64_t config, int group_fd, bool leader) {
        perf_event_attr attr{};
        attr.type = type;
        attr.size = sizeof(attr);
        attr.config = config;
        attr.disabled = leader;
        attr.pinned = leader;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        return int(syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, PERF_FLAG_FD_CLOEXEC));
    }
    int cycle_fd_ = -1, candidate_fd_ = -1;
};

double ratio(const RawReading & reading) {
    return reading.time_enabled ? double(reading.time_running) / reading.time_enabled : 0.0;
}

struct ThreadResult {
    int requested_cpu = -1, cpu_before = -1, cpu_after = -1, open_errno = 0;
    RawReading cycles, candidate;
    uint64_t known_loads = 0, known_lines = 0;
    bool huge = false, supported = true, ok = false;
};

void worker(int cpu, Kernel kernel, size_t working_set, uint64_t rounds, uint64_t known_loads,
            uint64_t known_lines, const Event & event, Barrier & ready, Barrier & done, ThreadResult & out) {
    out.requested_cpu = cpu;
    out.known_loads = known_loads;
    out.known_lines = known_lines;
    Mapping mapping;
    EventGroup group;
    bool prepared = k3::pin_to_a100(cpu) && k3::verify_cpu(cpu);
    if (prepared) {
        mapping = allocate_buffer(working_set);
        out.huge = mapping.huge;
        prepared = mapping.data && mapping.huge;
    }
    if (prepared) {
        out.supported = group.open(event);
        if (!out.supported) out.open_errno = errno;
        prepared = out.supported;
    }
    if (prepared) kernel(mapping.data, working_set, 16);
    ready.wait();
    if (prepared) {
        out.cpu_before = sched_getcpu();
        const bool started = group.start();
        if (started) kernel(mapping.data, working_set, rounds);
        const bool stopped = started && group.stop(out.cycles, out.candidate);
        out.cpu_after = sched_getcpu();
        out.huge = mapping.huge;
        out.ok = stopped && out.cpu_before == cpu && out.cpu_after == cpu && out.cycles.value > 0 &&
                 ratio(out.cycles) >= 0.99 && ratio(out.candidate) >= 0.99;
    }
    done.wait();
    if (mapping.address != MAP_FAILED) munmap(mapping.address, mapping.size);
}

struct SampleResult {
    std::vector<ThreadResult> threads;
    uint64_t cycles = 0, candidate = 0, known_loads = 0, known_lines = 0;
    bool supported = true, valid = false;
    std::string reject;
};

SampleResult run_sample(Kernel kernel, size_t working_set, uint64_t loads_per_bytes_num,
                        uint64_t loads_per_bytes_den, const std::vector<int> & cpus, const Event & event) {
    SampleResult result;
    result.threads.resize(cpus.size());
    const uint64_t rounds = kBytesPerThread / working_set;
    const uint64_t known_loads = kBytesPerThread * loads_per_bytes_num / loads_per_bytes_den;
    const uint64_t known_lines = kBytesPerThread / 64;
    Barrier ready(int(cpus.size())), done(int(cpus.size()));
    std::vector<std::thread> workers;
    for (size_t i = 0; i < cpus.size(); ++i)
        workers.emplace_back(worker, cpus[i], kernel, working_set, rounds, known_loads, known_lines,
                             std::cref(event), std::ref(ready), std::ref(done), std::ref(result.threads[i]));
    for (auto & thread : workers) thread.join();
    for (const ThreadResult & thread : result.threads) {
        result.supported = result.supported && thread.supported;
        if (!thread.supported) { result.reject = "event_open_failed"; continue; }
        if (!thread.ok && result.reject.empty()) result.reject = "thread_invalid";
        result.cycles += thread.cycles.value;
        result.candidate += thread.candidate.value;
        result.known_loads += thread.known_loads;
        result.known_lines += thread.known_lines;
    }
    result.valid = result.supported && result.reject.empty();
    return result;
}

std::string cpu_label(const std::vector<int> & cpus) {
    std::string label;
    for (size_t i = 0; i < cpus.size(); ++i) { if (i) label += ','; label += std::to_string(cpus[i]); }
    return label;
}

double event_per(uint64_t event, uint64_t denominator) {
    return denominator ? double(event) / denominator : 0.0;
}

struct Mode { const char * name; Kernel kernel; size_t inner_bytes; uint64_t loads_num, loads_den; };
struct Case { const char * level; size_t working_set; std::vector<int> cpus; };

}  // namespace

int main(int argc, char ** argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : kDefaultSamples;
    if (samples < 1) return 2;
    const std::vector<Mode> modes = {
        {"scalar", read_scalar, 64, 8, 64},
        {"m1", read_rvv_m1, 1024, 1, 128},
        {"m8", read_rvv_m8, 4096, 1, 1024},
    };
    static constexpr Event events[] = {
        Event("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
        Event("stalled-cycles-frontend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND),
        Event("stalled-cycles-backend", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND),
        Event("raw-0x2b", PERF_TYPE_RAW, 0x2b), Event("raw-0x39", PERF_TYPE_RAW, 0x39),
        Event("raw-0x5", PERF_TYPE_RAW, 0x5), Event("raw-0x6", PERF_TYPE_RAW, 0x6),
        Event("raw-0xb8", PERF_TYPE_RAW, 0xb8), Event("raw-0xb9", PERF_TYPE_RAW, 0xb9),
        Event("raw-0xad", PERF_TYPE_RAW, 0xad), Event("raw-0xae", PERF_TYPE_RAW, 0xae),
        Event("raw-0xba", PERF_TYPE_RAW, 0xba), Event("raw-0xbb", PERF_TYPE_RAW, 0xbb),
        Event("raw-0x1f", PERF_TYPE_RAW, 0x1f),
    };
    const std::vector<int> one = {8}, cluster = {8, 9, 10, 11};
    const std::vector<Case> cases = {
        {"L1", 16u << 10, one}, {"L1", 16u << 10, cluster},
        {"L2", 128u << 10, one}, {"L2", 128u << 10, cluster},
    };
    std::printf("{\"type\":\"metadata\",\"samples\":%d,\"bytes_per_thread_sample\":%" PRIu64
                ",\"vlen_bits\":1024,\"measurement\":\"pinned cycles leader plus one candidate\","
                "\"minimum_running_ratio\":0.99}\n", samples, kBytesPerThread);
    for (const Mode & mode : modes) for (const Case & test : cases) for (const Event & event : events) {
        if (test.working_set % mode.inner_bytes != 0 || kBytesPerThread % test.working_set != 0) return 3;
        std::vector<double> cycles_values, candidate_values, per_load_values, per_line_values, per_cycle_values;
        const std::string cpus = cpu_label(test.cpus);
        int supported_samples = 0;
        for (int sample = 0; sample < samples; ++sample) {
            SampleResult result = run_sample(mode.kernel, test.working_set, mode.loads_num, mode.loads_den,
                                             test.cpus, event);
            std::printf("{\"type\":\"sample\",\"kernel\":\"%s\",\"level\":\"%s\",\"cpus\":\"%s\","
                        "\"event\":\"%s\",\"event_type\":%u,\"event_config\":%" PRIu64 ","
                        "\"working_set_per_thread\":%zu,\"sample\":%d,\"supported\":%s,\"valid\":%s,"
                        "\"reject\":\"%s\",\"cycles\":%" PRIu64 ",\"candidate_value\":%" PRIu64 ","
                        "\"known_load_instructions\":%" PRIu64 ",\"known_cache_lines\":%" PRIu64 ","
                        "\"event_per_load\":%.12f,\"event_per_line\":%.12f,\"event_per_cycle\":%.12f,\"threads\":[",
                        mode.name, test.level, cpus.c_str(), event.name, event.type, event.config,
                        test.working_set, sample, result.supported ? "true" : "false",
                        result.valid ? "true" : "false", result.reject.c_str(), result.cycles,
                        result.candidate, result.known_loads, result.known_lines,
                        event_per(result.candidate, result.known_loads), event_per(result.candidate, result.known_lines),
                        event_per(result.candidate, result.cycles));
            for (size_t i = 0; i < result.threads.size(); ++i) {
                const ThreadResult & thread = result.threads[i];
                std::printf("%s{\"cpu\":%d,\"cpu_before\":%d,\"cpu_after\":%d,\"supported\":%s,"
                            "\"open_errno\":%d,\"thp\":%s,\"cycles\":%" PRIu64 ","
                            "\"candidate_value\":%" PRIu64 ",\"cycles_running_ratio\":%.9f,"
                            "\"candidate_running_ratio\":%.9f,\"known_load_instructions\":%" PRIu64 ","
                            "\"known_cache_lines\":%" PRIu64 ",\"event_per_load\":%.12f,"
                            "\"event_per_line\":%.12f,\"event_per_cycle\":%.12f}",
                            i ? "," : "", thread.requested_cpu, thread.cpu_before, thread.cpu_after,
                            thread.supported ? "true" : "false", thread.open_errno,
                            thread.huge ? "true" : "false", thread.cycles.value, thread.candidate.value,
                            ratio(thread.cycles), ratio(thread.candidate), thread.known_loads, thread.known_lines,
                            event_per(thread.candidate.value, thread.known_loads),
                            event_per(thread.candidate.value, thread.known_lines),
                            event_per(thread.candidate.value, thread.cycles.value));
            }
            std::printf("]}\n");
            std::fflush(stdout);
            if (result.supported) ++supported_samples;
            if (result.valid) {
                cycles_values.push_back(double(result.cycles));
                candidate_values.push_back(double(result.candidate));
                per_load_values.push_back(event_per(result.candidate, result.known_loads));
                per_line_values.push_back(event_per(result.candidate, result.known_lines));
                per_cycle_values.push_back(event_per(result.candidate, result.cycles));
            }
        }
        std::printf("{\"type\":\"summary\",\"kernel\":\"%s\",\"level\":\"%s\",\"cpus\":\"%s\","
                    "\"event\":\"%s\",\"supported\":%s,\"supported_samples\":%d,\"valid_samples\":%zu,"
                    "\"median_cycles\":%.3f,\"median_candidate_value\":%.3f,"
                    "\"median_event_per_load\":%.12f,\"median_event_per_line\":%.12f,"
                    "\"median_event_per_cycle\":%.12f}\n",
                    mode.name, test.level, cpus.c_str(), event.name,
                    supported_samples == samples ? "true" : "false", supported_samples,
                    candidate_values.size(), k3::median(cycles_values), k3::median(candidate_values),
                    k3::median(per_load_values), k3::median(per_line_values), k3::median(per_cycle_values));
        std::fflush(stdout);
    }
    return 0;
}
