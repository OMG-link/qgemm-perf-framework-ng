// A100 向量 load 带宽的共享结构。
//
// 之前只测了「1 核 / 同簇 4 核 / 每簇 2 核共 4 核 / 全 8 核」，缺了同簇
// 2 核和 3 核，于是无法区分下面这几种模型：
//
//   (a) 每核私有          -> 同簇 N 核聚合 = N × 单核
//   (b) 固定的簇内上限 C  -> 同簇 N 核聚合 = min(N × 单核, C)，且单调不减
//   (c) 按核对共享        -> 同簇 2 核看是否同一对而不同
//
// 实测同簇 4 核只有 57 B/cycle，而每簇 2 核（共 4 核）能到 156（即每簇 78），
// 这既不符合 (a) 也不符合 (b)。这里补全 1/2/3/4 核的扫描来定性。
//
// 关键指标是 **每线程 B/cycle**：它只用该线程自己的 bytes 和自己的 cycles，
// 不依赖任何跨线程的时间对齐，因此不会被聚合口径的误差影响。
//
// load kernel 用零地址运算的 24 基址寄存器版本（单核能取到 60 B/cycle 的
// 那个），工作集 24 KiB，完全驻留每核私有的 64 KiB L1D。
#include "k3_a100.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

#include <sys/mman.h>

namespace {

// 24 个基址寄存器，间距 1024 B，循环体内零地址运算。
#define SETUP24                                                                                    \
    "vsetvli t0, zero, e8, m8, ta, ma\n\t"                                                          \
    "mv   a2, %0\n\t"                                                                               \
    "addi a3, a2, 1024\n\taddi a4, a3, 1024\n\taddi a5, a4, 1024\n\taddi a6, a5, 1024\n\t"          \
    "addi a7, a6, 1024\n\taddi t1, a7, 1024\n\taddi t2, t1, 1024\n\taddi t3, t2, 1024\n\t"          \
    "addi t4, t3, 1024\n\taddi t5, t4, 1024\n\taddi t6, t5, 1024\n\taddi s2, t6, 1024\n\t"          \
    "addi s3, s2, 1024\n\taddi s4, s3, 1024\n\taddi s5, s4, 1024\n\taddi s6, s5, 1024\n\t"          \
    "addi s7, s6, 1024\n\taddi s8, s7, 1024\n\taddi s9, s8, 1024\n\taddi s10, s9, 1024\n\t"         \
    "addi s11, s10, 1024\n\taddi s0, s11, 1024\n\taddi s1, s0, 1024\n\taddi a1, s1, 1024\n\t"

#define BODY24                                                                                     \
    "vl8re8.v v0, (a2)\n\tvl8re8.v v8, (a3)\n\tvl8re8.v v16, (a4)\n\tvl8re8.v v24, (a5)\n\t"        \
    "vl8re8.v v0, (a6)\n\tvl8re8.v v8, (a7)\n\tvl8re8.v v16, (t1)\n\tvl8re8.v v24, (t2)\n\t"        \
    "vl8re8.v v0, (t3)\n\tvl8re8.v v8, (t4)\n\tvl8re8.v v16, (t5)\n\tvl8re8.v v24, (t6)\n\t"        \
    "vl8re8.v v0, (s2)\n\tvl8re8.v v8, (s3)\n\tvl8re8.v v16, (s4)\n\tvl8re8.v v24, (s5)\n\t"        \
    "vl8re8.v v0, (s6)\n\tvl8re8.v v8, (s7)\n\tvl8re8.v v16, (s8)\n\tvl8re8.v v24, (s9)\n\t"        \
    "vl8re8.v v0, (s10)\n\tvl8re8.v v8, (s11)\n\tvl8re8.v v16, (s0)\n\tvl8re8.v v24, (s1)\n\t"

#define CLOBBER24                                                                                  \
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s0",       \
        "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "memory", "v0", "v8",   \
        "v16", "v24"

constexpr int  kLoadsPerPass = 24 * 16;
constexpr size_t kL1Bytes    = 24u << 10;

__attribute__((noinline)) void load_l1(const uint8_t * base, uint64_t rep) {
    asm volatile(SETUP24 "2:\n\t"
                 ".rept 16\n\t" BODY24 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : CLOBBER24);
}

// L2 尺寸（256 KiB）无法零地址运算覆盖，用单指针 + addi 的结构。
// 与 L1D 版本不可直接比绝对值，但同一层级内不同线程数之间可比。
constexpr int    kL2LoadsPerPass = 256;
constexpr size_t kL2Bytes        = 256u << 10;

__attribute__((noinline)) void load_l2(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\t"
                 ".rept 64\n\t"
                 "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"
                 "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "memory", "v0", "v8", "v16", "v24");
}

// 标量对照：每个 cache line 用 8 条 ld 读满，走私有 L1D。
constexpr int kScalarLoadsPerPass = 32 * 64;

__attribute__((noinline)) void load_scalar(const uint8_t * base, uint64_t rep) {
    asm volatile("2:\n\t"
                 "mv t1, %0\n\t"
                 ".rept 64\n\t"
                 "ld a2, 0(t1)\n\tld a3, 8(t1)\n\tld a4, 16(t1)\n\tld a5, 24(t1)\n\t"
                 "ld a6, 32(t1)\n\tld a7, 40(t1)\n\tld t3, 48(t1)\n\tld t4, 56(t1)\n\t"
                 "ld a2, 64(t1)\n\tld a3, 72(t1)\n\tld a4, 80(t1)\n\tld a5, 88(t1)\n\t"
                 "ld a6, 96(t1)\n\tld a7, 104(t1)\n\tld t3, 112(t1)\n\tld t4, 120(t1)\n\t"
                 "ld a2, 128(t1)\n\tld a3, 136(t1)\n\tld a4, 144(t1)\n\tld a5, 152(t1)\n\t"
                 "ld a6, 160(t1)\n\tld a7, 168(t1)\n\tld t3, 176(t1)\n\tld t4, 184(t1)\n\t"
                 "ld a2, 192(t1)\n\tld a3, 200(t1)\n\tld a4, 208(t1)\n\tld a5, 216(t1)\n\t"
                 "ld a6, 224(t1)\n\tld a7, 232(t1)\n\tld t3, 240(t1)\n\tld t4, 248(t1)\n\t"
                 "addi t1, t1, 256\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t1", "a2", "a3", "a4", "a5", "a6", "a7", "t3", "t4", "memory");
}

using Fn = void (*)(const uint8_t *, uint64_t);

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

struct Sample {
    double   t0 = 0, t1 = 0;
    uint64_t cycles = 0, bytes = 0;
    int      cpu = -1;
    bool     ok  = false;
};

uint8_t * alloc_thp(size_t bytes, void ** raw_out, size_t * raw_len) {
    constexpr size_t kHuge = 2u << 20;
    const size_t     len   = ((bytes + kHuge - 1) / kHuge + 1) * kHuge;
    void *           raw =
        mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return nullptr;
    }
    auto addr = (uintptr_t) raw;
    auto base = (uint8_t *) ((addr + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
    madvise(base, len - size_t(base - (uint8_t *) raw), MADV_HUGEPAGE);
    std::memset(base, 0x5a, bytes);
    *raw_out = raw;
    *raw_len = len;
    return base;
}

void worker(int cpu, Fn fn, size_t ws, uint64_t bytes_per_pass, uint64_t rep,
            SpinBarrier & barrier, Sample & out) {
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        return;
    }
    void *   raw     = nullptr;
    size_t   raw_len = 0;
    uint8_t * base   = alloc_thp(ws, &raw, &raw_len);
    k3::CycleCounter counter;
    if (base == nullptr || !counter.open()) {
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
        return;
    }
    fn(base, std::max<uint64_t>(rep / 8, 2));
    barrier.arrive_and_wait();
    double t0 = k3::wall_seconds();
    counter.start();
    fn(base, rep);
    k3::CycleCounter::Reading r{};
    bool                      ok = counter.stop(r);
    double                    t1 = k3::wall_seconds();
    barrier.arrive_and_wait();
    out.t0     = t0;
    out.t1     = t1;
    out.cycles = r.cycles;
    out.bytes  = bytes_per_pass * rep;
    out.cpu    = sched_getcpu();
    out.ok     = ok && out.cpu == cpu && r.running_ratio > 0.99;
    munmap(raw, raw_len);
}

struct Point {
    double per_thread_min = 0;  // 最慢线程
    double per_thread_max = 0;  // 最快线程
    double aggregate      = 0;
    double overlap        = 0;
    bool   ok             = false;
};

// ws 是工作集（决定命中哪一级），bytes_per_pass 是一趟实际读取的字节数。
// 两者不相等：L1D 的 kernel 每趟把 24 KiB 的工作集扫 16 遍。
Point run(Fn fn, size_t ws, uint64_t bytes_per_pass, const std::vector<int> & cpus, int samples) {
    const uint64_t rep = std::max<uint64_t>((256ull << 20) / bytes_per_pass, 4);
    Point          best{};
    for (int s = 0; s < samples; ++s) {
        SpinBarrier              barrier(int(cpus.size()));
        std::vector<Sample>      samp(cpus.size());
        std::vector<std::thread> th;
        for (size_t i = 0; i < cpus.size(); ++i) {
            th.emplace_back(worker, cpus[i], fn, ws, bytes_per_pass, rep,
                            std::ref(barrier), std::ref(samp[i]));
        }
        for (auto & t : th) {
            t.join();
        }
        double   fs = 1e300, le = 0, ls = 0, fe = 1e300;
        uint64_t total = 0;
        std::vector<double> per, freq;
        bool                ok = true;
        for (const Sample & m : samp) {
            if (!m.ok) {
                ok = false;
                break;
            }
            fs = std::min(fs, m.t0);
            ls = std::max(ls, m.t0);
            le = std::max(le, m.t1);
            fe = std::min(fe, m.t1);
            total += m.bytes;
            per.push_back(double(m.bytes) / double(m.cycles));
            freq.push_back(double(m.cycles) / (m.t1 - m.t0));
        }
        if (!ok) {
            continue;
        }
        const double span    = le - fs;
        const double overlap = std::max(0.0, fe - ls) / span;
        const double agg     = double(total) / (span * k3::median(freq));
        if (overlap >= 0.90 && agg > best.aggregate) {
            best.aggregate      = agg;
            best.per_thread_min = *std::min_element(per.begin(), per.end());
            best.per_thread_max = *std::max_element(per.begin(), per.end());
            best.overlap        = overlap;
            best.ok             = true;
        }
    }
    return best;
}

std::string label(const std::vector<int> & cpus) {
    std::string s;
    for (size_t i = 0; i < cpus.size(); ++i) {
        s += (i ? "," : "") + std::to_string(cpus[i]);
    }
    return s;
}

// 每个组合里，各簇（8-11 / 12-15）各有几个活跃核。
std::string cluster_shape(const std::vector<int> & cpus) {
    int c0 = 0, c1 = 0;
    for (int c : cpus) {
        (c <= 11 ? c0 : c1)++;
    }
    return std::to_string(c0) + "+" + std::to_string(c1);
}

const std::vector<std::vector<int>> kSets = {
    {8},
    {8, 9},                            // 同簇，同一对
    {8, 10},                           // 同簇，不同对
    {8, 11},                           // 同簇，不同对
    {9, 11},                           // 同簇，不同对
    {8, 9, 10},                        // 同簇 3 核
    {8, 10, 11},                       // 同簇 3 核
    {8, 9, 10, 11},                    // 同簇 4 核
    {8, 12},                           // 跨簇 1+1
    {8, 9, 12, 13},                    // 跨簇 2+2，各簇同一对
    {8, 10, 12, 14},                   // 跨簇 2+2，各簇不同对
    {8, 9, 10, 12, 13, 14},            // 跨簇 3+3
    {8, 9, 10, 11, 12, 13, 14, 15},    // 全 8 核
};

}  // namespace

int main(int argc, char ** argv) {
    int samples = (argc > 1) ? std::atoi(argv[1]) : 7;

    struct Level {
        const char * name;
        Fn           fn;
        size_t       ws;              // 工作集（决定命中哪一级）
        uint64_t     bytes_per_pass;  // 一趟实际读取的字节数
    };
    const Level levels[] = {
        // 24 × 16 = 384 条 vl8re8.v，每条 1024 B
        {"L1D 24 KiB, 向量 vl8re8.v", load_l1, kL1Bytes, 384ull * 1024},
        // 64 × 4 = 256 条 vle8.v，每条 1024 B，正好扫完 256 KiB
        {"L2 256 KiB, 向量 vle8.v", load_l2, kL2Bytes, 256ull * 1024},
        // 64 × 256 B = 16 KiB
        {"L1D 16 KiB, 标量 ld", load_scalar, 16u << 10, 64ull * 256},
    };

    for (const Level & lv : levels) {
        std::printf("\n## %s\n", lv.name);
        std::printf("%-30s %6s %8s %12s %12s %14s %9s %9s\n", "cpus", "n", "簇分布", "每线程最小",
                    "每线程最大", "聚合 B/cycle", "每簇聚合", "overlap");
        for (const auto & cpus : kSets) {
            Point p = run(lv.fn, lv.ws, lv.bytes_per_pass, cpus, samples);
            if (!p.ok) {
                std::printf("%-30s %6zu %8s %12s\n", label(cpus).c_str(), cpus.size(),
                            cluster_shape(cpus).c_str(), "invalid");
                continue;
            }
            // 活跃簇数：1 或 2
            int clusters = 0;
            bool c0 = false, c1 = false;
            for (int c : cpus) {
                (c <= 11 ? c0 : c1) = true;
            }
            clusters = (c0 ? 1 : 0) + (c1 ? 1 : 0);
            std::printf("%-30s %6zu %8s %12.2f %12.2f %14.2f %9.2f %8.1f%%\n", label(cpus).c_str(),
                        cpus.size(), cluster_shape(cpus).c_str(), p.per_thread_min,
                        p.per_thread_max, p.aggregate, p.aggregate / clusters, 100.0 * p.overlap);
            std::fflush(stdout);
        }
    }
    return 0;
}
