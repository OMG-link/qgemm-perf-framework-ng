// Independent K3/A100 read-bandwidth diagnostic.
//
// Every raw sample is reported. Summaries use all valid samples with median/MAD;
// no best/minimum sample selection is performed.
#include "k3_a100.h"

#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace {

constexpr uint64_t kBytesPerSamplePerThread = 1ull << 30;
constexpr int kDefaultSamples = 9;

using Kernel = void (*)(const uint8_t *, size_t, uint64_t);

// One 64-byte cache line per inner iteration. Eight independent integer
// destination registers ensure that the accounted 64 bytes are all loaded.
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
                 : "t0", "t1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
                   "memory");
}

// VLEN=1024, e8,m1: each load is 128 bytes; rotate through eight destinations.
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
                 : "t0", "t1", "t2", "memory", "v0", "v1", "v2", "v3", "v4", "v5",
                   "v6", "v7");
}

// VLEN=1024, e8,m8: each load is 1024 bytes. LMUL=8 permits exactly
// v0/v8/v16/v24 as non-overlapping destination group starts.
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

struct ThreadResult {
    int requested_cpu = -1, cpu_before = -1, cpu_after = -1;
    uint64_t cycles = 0, instructions = 0, bytes = 0;
    double running_ratio = 0, begin = 0, end = 0;
    bool huge = false, ok = false;
};

void worker(int cpu, Kernel kernel, size_t working_set, uint64_t rounds, Barrier & ready,
            Barrier & done, ThreadResult & out) {
    out.requested_cpu = cpu;
    Mapping mapping;
    k3::CycleCounter counter;
    bool prepared = k3::pin_to_a100(cpu) && k3::verify_cpu(cpu);
    if (prepared) {
        mapping = allocate_buffer(working_set);
        prepared = mapping.data && mapping.huge && counter.open();
    }
    if (prepared) kernel(mapping.data, working_set, 16);
    ready.wait();
    if (prepared) {
        out.cpu_before = sched_getcpu();
        out.begin = k3::wall_seconds();
        const bool started = counter.start();
        if (started) kernel(mapping.data, working_set, rounds);
        k3::CycleCounter::Reading reading{};
        const bool stopped = started && counter.stop(reading);
        out.end = k3::wall_seconds();
        out.cpu_after = sched_getcpu();
        out.cycles = reading.cycles;
        out.instructions = reading.instructions;
        out.running_ratio = reading.running_ratio;
        out.bytes = uint64_t(working_set) * rounds;
        out.huge = mapping.huge;
        out.ok = stopped && out.cpu_before == cpu && out.cpu_after == cpu && out.cycles > 0 &&
                 out.end > out.begin && out.running_ratio >= 0.99;
    }
    done.wait();
    if (mapping.address != MAP_FAILED) munmap(mapping.address, mapping.size);
}

struct SampleResult {
    std::vector<ThreadResult> threads;
    double sum_thread_bpc = 0, aggregate_bpc = 0, bpc_ratio = 0;
    double aggregate_gbs = 0, overlap = 0, frequency_ghz = 0;
    bool valid = false;
    std::string reject;
};

SampleResult run_sample(Kernel kernel, size_t working_set, const std::vector<int> & cpus) {
    SampleResult result;
    result.threads.resize(cpus.size());
    Barrier ready(int(cpus.size())), done(int(cpus.size()));
    const uint64_t rounds = kBytesPerSamplePerThread / working_set;
    std::vector<std::thread> workers;
    for (size_t i = 0; i < cpus.size(); ++i)
        workers.emplace_back(worker, cpus[i], kernel, working_set, rounds, std::ref(ready),
                             std::ref(done), std::ref(result.threads[i]));
    for (auto & thread : workers) thread.join();

    double first_begin = 1e300, last_begin = 0, first_end = 1e300, last_end = 0;
    uint64_t total_bytes = 0;
    std::vector<double> frequencies;
    for (const ThreadResult & thread : result.threads) {
        if (!thread.ok) { result.reject = "thread_invalid"; return result; }
        first_begin = std::min(first_begin, thread.begin);
        last_begin = std::max(last_begin, thread.begin);
        first_end = std::min(first_end, thread.end);
        last_end = std::max(last_end, thread.end);
        total_bytes += thread.bytes;
        result.sum_thread_bpc += double(thread.bytes) / thread.cycles;
        frequencies.push_back(thread.cycles / (thread.end - thread.begin));
    }
    const double span = last_end - first_begin;
    result.overlap = std::max(0.0, first_end - last_begin) / span;
    const double frequency = k3::median(frequencies);
    result.frequency_ghz = frequency / 1e9;
    const auto [minimum, maximum] = std::minmax_element(frequencies.begin(), frequencies.end());
    if (result.overlap < 0.95) { result.reject = "overlap_below_95pct"; return result; }
    if (*maximum / *minimum > 1.02) { result.reject = "frequency_spread_above_2pct"; return result; }
    result.aggregate_gbs = double(total_bytes) / span / 1e9;
    result.aggregate_bpc = double(total_bytes) / (span * frequency);
    result.bpc_ratio = result.aggregate_bpc ? result.sum_thread_bpc / result.aggregate_bpc : 0;
    result.valid = true;
    return result;
}

std::string cpu_label(const std::vector<int> & cpus) {
    std::string label;
    for (size_t i = 0; i < cpus.size(); ++i) { if (i) label += ','; label += std::to_string(cpus[i]); }
    return label;
}

double mad(const std::vector<double> & values) {
    const double center = k3::median(values);
    std::vector<double> deviations;
    for (double value : values) deviations.push_back(std::fabs(value - center));
    return k3::median(deviations);
}

struct Mode { const char * name; Kernel kernel; size_t inner_bytes; };
struct Case { const char * level; size_t working_set; std::vector<int> cpus; };

}  // namespace

int main(int argc, char ** argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : kDefaultSamples;
    if (samples < 1) return 2;
    const std::vector<Mode> modes = {
        {"scalar-ld", read_scalar, 64}, {"rvv-e8-m1", read_rvv_m1, 1024},
        {"rvv-e8-m8", read_rvv_m8, 4096},
    };
    const std::vector<int> one = {8}, cluster = {8, 9, 10, 11};
    const std::vector<int> all = {8, 9, 10, 11, 12, 13, 14, 15};
    const std::vector<Case> cases = {
        {"L1", 16u << 10, one}, {"L1", 16u << 10, cluster}, {"L1", 16u << 10, all},
        {"L2", 128u << 10, one}, {"L2", 128u << 10, cluster}, {"L2", 128u << 10, all},
    };
    std::printf("{\"type\":\"metadata\",\"samples\":%d,\"bytes_per_thread_sample\":%" PRIu64
                ",\"vlen_bits\":1024,\"validity\":\"cpu+thp+pmu99+overlap95+freqspread2\"}\n",
                samples, kBytesPerSamplePerThread);
    for (const Mode & mode : modes) for (const Case & test : cases) {
        if (test.working_set % mode.inner_bytes != 0 ||
            kBytesPerSamplePerThread % test.working_set != 0) return 3;
        std::vector<double> sums, aggregates, ratios, gbss;
        const std::string cpus = cpu_label(test.cpus);
        for (int sample = 0; sample < samples; ++sample) {
            SampleResult result = run_sample(mode.kernel, test.working_set, test.cpus);
            std::printf("{\"type\":\"sample\",\"kernel\":\"%s\",\"level\":\"%s\",\"cpus\":\"%s\"," 
                        "\"working_set_per_thread\":%zu,\"sample\":%d,\"valid\":%s,\"reject\":\"%s\"," 
                        "\"sum_thread_bpc\":%.9f,\"aggregate_bpc\":%.9f,\"sum_to_aggregate_ratio\":%.9f," 
                        "\"aggregate_gbs\":%.9f,\"overlap\":%.9f,\"frequency_ghz\":%.9f,\"threads\":[",
                        mode.name, test.level, cpus.c_str(), test.working_set, sample,
                        result.valid ? "true" : "false", result.reject.c_str(), result.sum_thread_bpc,
                        result.aggregate_bpc, result.bpc_ratio, result.aggregate_gbs, result.overlap,
                        result.frequency_ghz);
            for (size_t i = 0; i < result.threads.size(); ++i) {
                const ThreadResult & thread = result.threads[i];
                std::printf("%s{\"requested_cpu\":%d,\"cpu_before\":%d,\"cpu_after\":%d," 
                            "\"cycles\":%" PRIu64 ",\"instructions\":%" PRIu64 ",\"bytes\":%" PRIu64 "," 
                            "\"running_ratio\":%.9f,\"duration\":%.9f,\"thread_bpc\":%.9f,\"thp\":%s}",
                            i ? "," : "", thread.requested_cpu, thread.cpu_before, thread.cpu_after,
                            thread.cycles, thread.instructions, thread.bytes, thread.running_ratio,
                            thread.end - thread.begin,
                            thread.cycles ? double(thread.bytes) / thread.cycles : 0.0,
                            thread.huge ? "true" : "false");
            }
            std::printf("]}\n"); std::fflush(stdout);
            if (result.valid) {
                sums.push_back(result.sum_thread_bpc); aggregates.push_back(result.aggregate_bpc);
                ratios.push_back(result.bpc_ratio); gbss.push_back(result.aggregate_gbs);
            }
        }
        std::printf("{\"type\":\"summary\",\"kernel\":\"%s\",\"level\":\"%s\",\"cpus\":\"%s\"," 
                    "\"working_set_per_thread\":%zu,\"valid_samples\":%zu," 
                    "\"median_sum_thread_bpc\":%.9f,\"mad_sum_thread_bpc\":%.9f," 
                    "\"median_aggregate_bpc\":%.9f,\"mad_aggregate_bpc\":%.9f," 
                    "\"median_sum_to_aggregate_ratio\":%.9f,\"mad_sum_to_aggregate_ratio\":%.9f," 
                    "\"median_aggregate_gbs\":%.9f,\"mad_aggregate_gbs\":%.9f}\n",
                    mode.name, test.level, cpus.c_str(), test.working_set, aggregates.size(),
                    k3::median(sums), mad(sums), k3::median(aggregates), mad(aggregates),
                    k3::median(ratios), mad(ratios), k3::median(gbss), mad(gbss));
        std::fflush(stdout);
    }
    return 0;
}
