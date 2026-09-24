// SpacemiT K3 / A100 各级存储带宽微基准。
//
// 口径与 docs/2026-09-02-k1-memory-path-bandwidth.md 一致，统一用 B/cycle：
//
//   单核 B/cycle    = 该线程传输的字节数 / 该线程的 CPU cycles
//   聚合 B/cycle    = 所有线程传输的总字节数 / (公共墙钟时间跨度 × 核心频率)
//
// 聚合口径用全芯片公共时间轴上的 cycle，不累加各核 cycle。核心频率由每个
// 线程自己的 cycles/wall-time 求中位数得到（K3 的 governor 是 userspace，
// A100 固定在 1.8 GHz，但这里仍然实测而不假设）。
//
// 访存 kernel 用 LMUL=m8 的 RVV 访问，VLEN=1024 时每条指令搬 1024 B，
// 让前端和地址计算开销可以忽略。
#include "k3_a100.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#include <sys/mman.h>

namespace {

constexpr size_t kBlock = 8192;  // 内层循环每次推进的字节数

// 所有 kernel 都把 `repeat` 遍扫描放进同一段汇编，这样小工作集也不会被
// 每轮的函数调用和 vsetvli 开销污染。
//
// m8 变体每条 vle8.v 搬 1024 B；LMUL=8 只有 v0/v8/v16/v24 四个合法寄存器组，
// 所以 8 条 load 循环复用这四组。
void read_stream_m8(const uint8_t * base, size_t bytes, uint64_t repeat) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\tadd t2, %0, %1\n\t"
                 "1:\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 "bltu t1, t2, 1b\n\t"
                 "addi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(repeat)
                 :
                 : "t0", "t1", "t2", "memory", "v0", "v8", "v16", "v24");
}

// m1 变体每条 vle8.v 搬 128 B，用 8 个独立目标寄存器，每次内层迭代同样
// 推进 1 KiB。用来区分“L1D 命中”和“绕过 L1D 的大粒度流式访问”。
void read_stream_m1(const uint8_t * base, size_t bytes, uint64_t repeat) {
    asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\tadd t2, %0, %1\n\t"
                 "1:\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v1, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v2, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v3, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v4, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v5, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v6, (t1)\n\taddi t1, t1, 128\n\t"
                 "vle8.v v7, (t1)\n\taddi t1, t1, 128\n\t"
                 "bltu t1, t2, 1b\n\t"
                 "addi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(repeat)
                 :
                 : "t0", "t1", "t2", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
}

void write_stream_m8(uint8_t * base, size_t bytes, uint64_t repeat) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "vmv.v.i v0, 1\n\tvmv.v.i v8, 2\n\tvmv.v.i v16, 3\n\tvmv.v.i v24, 4\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\tadd t2, %0, %1\n\t"
                 "1:\n\t"
                 "vse8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vse8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 "bltu t1, t2, 1b\n\t"
                 "addi %2, %2, -1\n\tbnez %2, 2b\n\t"
                 : "+r"(base), "+r"(bytes), "+r"(repeat)
                 :
                 : "t0", "t1", "t2", "memory", "v0", "v8", "v16", "v24");
}

// copy：src 与 dst 各占一半工作集，计入的字节数是读+写之和。
void copy_stream_m8(const uint8_t * src, uint8_t * dst, size_t bytes, uint64_t repeat) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\tmv t3, %1\n\tadd t2, %0, %2\n\t"
                 "1:\n\t"
                 "vle8.v v0, (t1)\n\tvse8.v v0, (t3)\n\t"
                 "addi t1, t1, 1024\n\taddi t3, t3, 1024\n\t"
                 "vle8.v v8, (t1)\n\tvse8.v v8, (t3)\n\t"
                 "addi t1, t1, 1024\n\taddi t3, t3, 1024\n\t"
                 "vle8.v v16, (t1)\n\tvse8.v v16, (t3)\n\t"
                 "addi t1, t1, 1024\n\taddi t3, t3, 1024\n\t"
                 "vle8.v v24, (t1)\n\tvse8.v v24, (t3)\n\t"
                 "addi t1, t1, 1024\n\taddi t3, t3, 1024\n\t"
                 "bltu t1, t2, 1b\n\t"
                 "addi %3, %3, -1\n\tbnez %3, 2b\n\t"
                 : "+r"(src), "+r"(dst), "+r"(bytes), "+r"(repeat)
                 :
                 : "t0", "t1", "t2", "t3", "memory", "v0", "v8", "v16", "v24");
}

enum class Mode { ReadM8, ReadM1, Write, Copy };

struct Sample {
    double   wall_start = 0;
    double   wall_end   = 0;
    uint64_t cycles     = 0;
    uint64_t bytes      = 0;
    double   running    = 0;
    bool     ok         = false;
    int      cpu        = -1;
};

struct SpinBarrier {
    explicit SpinBarrier(int n) : target(n) {}
    void arrive_and_wait() {
        int gen = generation.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == target) {
            count.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == gen) {
                asm volatile("" ::: "memory");
            }
        }
    }
    std::atomic<int> count{0};
    std::atomic<int> generation{0};
    const int        target;
};

uint8_t * alloc_buffer(size_t bytes) {
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    std::memset(p, 0x5a, bytes);
    return static_cast<uint8_t *>(p);
}

void worker(int cpu, Mode mode, size_t working_set, uint64_t rounds, SpinBarrier & barrier,
            Sample & out) {
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        return;
    }
    // copy 模式下工作集一半读一半写。
    const size_t half   = working_set / 2 / kBlock * kBlock;
    const size_t stream = (mode == Mode::Copy) ? half : working_set / kBlock * kBlock;

    uint8_t * buf = alloc_buffer(working_set + 4096);
    if (buf == nullptr) {
        std::fprintf(stderr, "cpu %d: mmap %zu failed\n", cpu, working_set);
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        return;
    }

    k3::CycleCounter counter;
    if (!counter.open()) {
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        return;
    }

    auto run = [&](uint64_t repeat) {
        switch (mode) {
            case Mode::ReadM8:
                read_stream_m8(buf, stream, repeat);
                break;
            case Mode::ReadM1:
                read_stream_m1(buf, stream, repeat);
                break;
            case Mode::Write:
                write_stream_m8(buf, stream, repeat);
                break;
            case Mode::Copy:
                copy_stream_m8(buf, buf + half, half, repeat);
                break;
        }
    };

    // 预热：把工作集带进目标层级，并触发页表填充。
    run(4);

    barrier.arrive_and_wait();
    double t0 = k3::wall_seconds();
    counter.start();
    run(rounds);
    k3::CycleCounter::Reading reading{};
    bool                      ok = counter.stop(reading);
    double                    t1 = k3::wall_seconds();
    barrier.arrive_and_wait();

    out.wall_start = t0;
    out.wall_end   = t1;
    out.cycles     = reading.cycles;
    out.bytes      = uint64_t(mode == Mode::Copy ? 2 * half : stream) * rounds;
    out.running    = reading.running_ratio;
    out.cpu        = sched_getcpu();
    out.ok         = ok && out.cpu == cpu;
    munmap(buf, working_set + 4096);
}

struct Aggregate {
    double per_thread_median = 0;
    double aggregate         = 0;
    double frequency_ghz     = 0;
    double overlap_ratio     = 0;
    double running_min       = 1;
    bool   ok                = false;
};

Aggregate run_point(Mode mode, const std::vector<int> & cpus, size_t working_set, uint64_t rounds) {
    SpinBarrier         barrier(int(cpus.size()));
    std::vector<Sample> samples(cpus.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < cpus.size(); ++i) {
        threads.emplace_back(worker, cpus[i], mode, working_set, rounds, std::ref(barrier),
                             std::ref(samples[i]));
    }
    for (auto & t : threads) {
        t.join();
    }

    Aggregate agg{};
    double    first_start = 1e300, last_end = 0, last_start = 0, first_end = 1e300;
    uint64_t  total_bytes = 0;
    std::vector<double> per_thread, freqs;
    for (const Sample & s : samples) {
        if (!s.ok) {
            return agg;
        }
        first_start = std::min(first_start, s.wall_start);
        last_start  = std::max(last_start, s.wall_start);
        last_end    = std::max(last_end, s.wall_end);
        first_end   = std::min(first_end, s.wall_end);
        total_bytes += s.bytes;
        per_thread.push_back(double(s.bytes) / double(s.cycles));
        freqs.push_back(double(s.cycles) / (s.wall_end - s.wall_start));
        agg.running_min = std::min(agg.running_min, s.running);
    }
    const double span    = last_end - first_start;
    const double overlap = std::max(0.0, first_end - last_start);
    agg.per_thread_median = k3::median(per_thread);
    agg.frequency_ghz     = k3::median(freqs) / 1e9;
    agg.aggregate         = double(total_bytes) / (span * k3::median(freqs));
    agg.overlap_ratio     = span > 0 ? overlap / span : 0;
    agg.ok                = true;
    return agg;
}

const char * mode_name(Mode m) {
    switch (m) {
        case Mode::ReadM8:
            return "rd-m8";
        case Mode::ReadM1:
            return "rd-m1";
        case Mode::Write:
            return "write";
        case Mode::Copy:
            return "copy";
    }
    return "?";
}

}  // namespace

int main(int argc, char ** argv) {
    std::vector<Mode> modes = {Mode::ReadM8, Mode::ReadM1, Mode::Write, Mode::Copy};
    if (argc > 1) {
        std::string m(argv[1]);
        if (m == "rd-m8") {
            modes = {Mode::ReadM8};
        } else if (m == "rd-m1") {
            modes = {Mode::ReadM1};
        } else if (m == "write") {
            modes = {Mode::Write};
        } else if (m == "copy") {
            modes = {Mode::Copy};
        }
    }

    // 工作集：跨越 L1D(64 KiB/核)、L2(1 MiB/4 核簇) 和主存。
    const std::vector<size_t> sizes = {
        16u << 10, 32u << 10, 64u << 10, 128u << 10, 256u << 10,
        512u << 10, 768u << 10, 2u << 20, 64u << 20,
    };
    // 线程组合：单核、同簇 2/4 核、跨簇 8 核。
    const std::vector<std::vector<int>> cpu_sets = {
        {8},            {8, 9},          {8, 10},
        {8, 9, 10, 11}, {8, 10, 12, 14}, {8, 9, 10, 11, 12, 13, 14, 15},
    };

    std::printf("%-6s %8s %10s %14s %14s %8s %8s\n", "mode", "threads", "ws/thread",
                "per-thread B/c", "aggregate B/c", "GHz", "overlap");
    for (Mode mode : modes) {
        for (size_t ws : sizes) {
            for (const std::vector<int> & cpus : cpu_sets) {
                // 每个点至少搬 512 MiB，保证测量区间足够长。
                uint64_t rounds = std::max<uint64_t>((512ull << 20) / ws, 4);
                Aggregate a     = run_point(mode, cpus, ws, rounds);
                if (!a.ok) {
                    std::printf("%-6s %8zu %9zuK %14s %14s %8s %8s\n", mode_name(mode), cpus.size(),
                                ws >> 10, "-", "-", "-", "invalid");
                    continue;
                }
                std::printf("%-6s %8zu %9zuK %14.2f %14.2f %8.3f %7.1f%%%s\n", mode_name(mode),
                            cpus.size(), ws >> 10, a.per_thread_median, a.aggregate,
                            a.frequency_ghz, 100.0 * a.overlap_ratio,
                            a.running_min < 0.99 ? "  [pmu]" : "");
                std::fflush(stdout);
            }
        }
    }
    return 0;
}
