// A100 三条读通路的带宽：L1D→寄存器、L2→L1D、主存→L2。
//
// 直接用「字节数 / cycles」测小工作集会低估 L1D 带宽：每趟扫描到头回绕时
// 有一个固定的重启代价（约 100-130 cycle，推测是流式预取器重新起流），
// 16 KiB 的一趟只有 16 条 load，这个代价占比很大，而 512 KiB 的一趟有 512
// 条 load，几乎看不出来。这会造成 L1D 看起来比 L2 还慢的假象。
//
// 这里改用斜率法：在同一个层级内取多个工作集尺寸，测每趟的 cycles，
// 对 (bytes/趟, cycles/趟) 做线性拟合。
//
//   cycles/趟 = bytes/趟 × (1 / 边际带宽) + 每趟固定开销
//
// 斜率的倒数就是该层级的稳态带宽，截距就是每趟的固定开销。同一层级内
// 各尺寸必须落在同一条直线上，否则说明尺寸跨越了层级边界。
//
// kernel 是 LMUL=m8 的 vle8.v（VLEN=1024 时每条读 1024 B），整趟完全展开，
// 循环体内没有任何分支，只有外层的重复分支。
#include "k3_a100.h"

#include <cmath>
#include <cstring>

#include <atomic>
#include <thread>

#include <sys/mman.h>

namespace {

using ReadFn = void (*)(const uint8_t *, uint64_t);

// 一趟 = NGROUP 组 × 4 条 m8 load × 1024 B。
#define READ_FULL(fname, NGROUP)                                                                   \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                     \
        asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"                                        \
                     "2:\n\t"                                                                      \
                     "mv t1, %0\n\t"                                                               \
                     ".rept " #NGROUP "\n\t"                                                       \
                     "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"                                  \
                     "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"                                  \
                     ".endr\n\t"                                                                   \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                          \
                     : "+r"(base), "+r"(rep)                                                       \
                     :                                                                            \
                     : "t0", "t1", "memory", "v0", "v8", "v16", "v24");                            \
    }

// 主存尺寸无法完全展开，用 8 KiB 粒度的内层循环；该尺寸下每趟固定开销
// 占比可以忽略。
#define READ_LOOP(fname, TOTAL)                                                                    \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                     \
        asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"                                        \
                     "2:\n\t"                                                                      \
                     "mv t1, %0\n\tli t2, " #TOTAL "\n\tadd t2, %0, t2\n\t"                        \
                     "1:\n\t"                                                                      \
                     ".rept 2\n\t"                                                                 \
                     "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"                                  \
                     "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"                                  \
                     ".endr\n\t"                                                                   \
                     "bltu t1, t2, 1b\n\t"                                                         \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                          \
                     : "+r"(base), "+r"(rep)                                                       \
                     :                                                                            \
                     : "t0", "t1", "t2", "memory", "v0", "v8", "v16", "v24");                      \
    }

// 标量对照 kernel：每条 ld 读 8 B，一个 cache line 用 8 条 ld 读满，地址
// 走 ld 的 12 位立即数偏移。用来判断 L1D 是否为标量 load 提供每核私有带宽
// ——如果标量能突破簇内约 56 B/cycle 的上限，说明大 LMUL 的向量 load 走的
// 是另一条（绕过 L1D 的）路径。
// 一趟 = NLINE 组 × 4 line × 64 B。
#define READ_SCALAR(fname, NLINE)                                                                  \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                     \
        asm volatile("2:\n\t"                                                                      \
                     "mv t1, %0\n\t"                                                               \
                     ".rept " #NLINE "\n\t"                                                        \
                     "ld a2, 0(t1)\n\tld a3, 8(t1)\n\tld a4, 16(t1)\n\tld a5, 24(t1)\n\t"          \
                     "ld a6, 32(t1)\n\tld a7, 40(t1)\n\tld t3, 48(t1)\n\tld t4, 56(t1)\n\t"        \
                     "ld a2, 64(t1)\n\tld a3, 72(t1)\n\tld a4, 80(t1)\n\tld a5, 88(t1)\n\t"        \
                     "ld a6, 96(t1)\n\tld a7, 104(t1)\n\tld t3, 112(t1)\n\tld t4, 120(t1)\n\t"     \
                     "ld a2, 128(t1)\n\tld a3, 136(t1)\n\tld a4, 144(t1)\n\tld a5, 152(t1)\n\t"    \
                     "ld a6, 160(t1)\n\tld a7, 168(t1)\n\tld t3, 176(t1)\n\tld t4, 184(t1)\n\t"    \
                     "ld a2, 192(t1)\n\tld a3, 200(t1)\n\tld a4, 208(t1)\n\tld a5, 216(t1)\n\t"    \
                     "ld a6, 224(t1)\n\tld a7, 232(t1)\n\tld t3, 240(t1)\n\tld t4, 248(t1)\n\t"    \
                     "addi t1, t1, 256\n\t"                                                        \
                     ".endr\n\t"                                                                   \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                          \
                     : "+r"(base), "+r"(rep)                                                       \
                     :                                                                             \
                     : "t1", "a2", "a3", "a4", "a5", "a6", "a7", "t3", "t4", "memory");            \
    }

READ_SCALAR(s_16k, 64)   // 64 × 256 B = 16 KiB
READ_SCALAR(s_64k, 256)  // 256 × 256 B = 64 KiB

// L1D 区间。每组是 4 KiB，只取偶数组数（即 8 KiB 的整数倍）：奇数组数的
// 尺寸系统性偏慢约 15%，是循环结构的结构性效应而不是存储层级的性质。
READ_FULL(f_8k, 2)
READ_FULL(f_16k, 4)
READ_FULL(f_24k, 6)
READ_FULL(f_32k, 8)
READ_FULL(f_40k, 10)
READ_FULL(f_48k, 12)
READ_FULL(f_64k, 16)
// L2 区间：> L1D，< 簇内 1 MiB
READ_FULL(f_128k, 32)
READ_FULL(f_192k, 48)
READ_FULL(f_256k, 64)
READ_FULL(f_384k, 96)
READ_FULL(f_512k, 128)
READ_FULL(f_768k, 192)
// 主存区间
READ_LOOP(f_8m, 8388608)
READ_LOOP(f_16m, 16777216)
READ_LOOP(f_32m, 33554432)
READ_LOOP(f_64m, 67108864)

struct Point {
    const char * label;
    ReadFn       fn;
    size_t       bytes;   // 一趟的字节数
    double       cycles;  // 实测每趟 cycles
    double       bpc;     // 朴素 B/cycle
};

struct Level {
    const char *       name;
    std::vector<Point> points;
};

// 最小二乘拟合 cycles = a * bytes + b。
void fit(const std::vector<Point> & pts, double & slope, double & intercept, double & max_resid) {
    double n = double(pts.size()), sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const Point & p : pts) {
        sx += double(p.bytes);
        sy += p.cycles;
        sxx += double(p.bytes) * double(p.bytes);
        sxy += double(p.bytes) * p.cycles;
    }
    slope     = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    intercept = (sy - slope * sx) / n;
    max_resid = 0;
    for (const Point & p : pts) {
        double pred = slope * double(p.bytes) + intercept;
        max_resid   = std::max(max_resid, std::fabs(p.cycles - pred) / pred);
    }
}

// A100 的 L1D 是 64 KiB / 4-way / 256 set / 64 B line，set index 用到物理
// 地址的 bit 12-13，超出 4 KiB 页内偏移。用普通 4 KiB 页时页着色随机，
// 32 KiB 的工作集可能有多页落到同一组，在 4-way 上产生冲突缺失——这正是
// 之前 L1D 区间测量非单调、噪声大的原因。
//
// 因此统一用 2 MiB 透明大页（THP 模式是 madvise），让工作集物理连续，
// set index 均匀分布。
uint8_t * alloc_thp(size_t bytes, void ** raw_out, size_t * raw_len) {
    constexpr size_t kHuge = 2u << 20;
    const size_t     len   = ((bytes + kHuge - 1) / kHuge + 1) * kHuge;
    void *           raw   = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return nullptr;
    }
    auto addr = (uintptr_t) raw;
    auto base = (uint8_t *) ((addr + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
    if (madvise(base, len - size_t(base - (uint8_t *) raw), MADV_HUGEPAGE) != 0) {
        std::fprintf(stderr, "madvise(MADV_HUGEPAGE) failed: %s\n", std::strerror(errno));
    }
    std::memset(base, 0x5a, bytes);
    *raw_out = raw;
    *raw_len = len;
    return base;
}

// 测一个点的每趟 cycles，取多次采样的最小值。
void measure(Point & p, k3::CycleCounter & counter, int samples) {
    void *   raw     = nullptr;
    size_t   raw_len = 0;
    uint8_t * base   = alloc_thp(p.bytes, &raw, &raw_len);
    if (base == nullptr) {
        std::fprintf(stderr, "alloc %zu failed\n", p.bytes);
        return;
    }

    const uint64_t rep = std::max<uint64_t>((256ull << 20) / p.bytes, 4);
    p.fn(base, std::max<uint64_t>(rep / 8, 2));

    double best = 1e300;
    for (int s = 0; s < samples; ++s) {
        if (!counter.start()) {
            break;
        }
        p.fn(base, rep);
        k3::CycleCounter::Reading r{};
        if (!counter.stop(r) || r.running_ratio < 0.99) {
            continue;
        }
        best = std::min(best, double(r.cycles) / double(rep));
    }
    p.cycles = best;
    p.bpc    = double(p.bytes) / best;
    munmap(raw, raw_len);
}

}  // namespace

// ------------------------------------------------------------ 多核共享结构
// 对每条通路测「1 核 / 同簇 4 核 / 每簇 2 核共 4 核 / 全 8 核」，判断该通路
// 是每核私有还是簇内共享。
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

struct MtSample {
    double   t0 = 0, t1 = 0;
    uint64_t cycles = 0, bytes = 0;
    bool     ok = false;
};

void mt_worker(int cpu, ReadFn fn, size_t ws, uint64_t rep, SpinBarrier & barrier, MtSample & out) {
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
    out.bytes  = uint64_t(ws) * rep;
    out.ok     = ok && sched_getcpu() == cpu && r.running_ratio > 0.99;
    munmap(raw, raw_len);
}

void mt_point(const char * level, const char * wslabel, ReadFn fn, size_t ws,
              const std::vector<int> & cpus, int samples) {
    const uint64_t rep = std::max<uint64_t>((256ull << 20) / ws, 4);
    double         best_agg = 0, best_per = 0, best_overlap = 0;
    for (int s = 0; s < samples; ++s) {
        SpinBarrier              barrier(int(cpus.size()));
        std::vector<MtSample>    samp(cpus.size());
        std::vector<std::thread> th;
        for (size_t i = 0; i < cpus.size(); ++i) {
            th.emplace_back(mt_worker, cpus[i], fn, ws, rep, std::ref(barrier), std::ref(samp[i]));
        }
        for (auto & t : th) {
            t.join();
        }
        double   first_start = 1e300, last_end = 0, last_start = 0, first_end = 1e300;
        uint64_t total = 0;
        std::vector<double> per, freq;
        bool                ok = true;
        for (const MtSample & m : samp) {
            ok = ok && m.ok;
            if (!m.ok) {
                break;
            }
            first_start = std::min(first_start, m.t0);
            last_start  = std::max(last_start, m.t0);
            last_end    = std::max(last_end, m.t1);
            first_end   = std::min(first_end, m.t1);
            total += m.bytes;
            per.push_back(double(m.bytes) / double(m.cycles));
            freq.push_back(double(m.cycles) / (m.t1 - m.t0));
        }
        if (!ok) {
            continue;
        }
        double span    = last_end - first_start;
        double overlap = std::max(0.0, first_end - last_start) / span;
        double agg     = double(total) / (span * k3::median(freq));
        if (overlap >= 0.90 && agg > best_agg) {
            best_agg     = agg;
            best_per     = k3::median(per);
            best_overlap = overlap;
        }
    }
    std::printf("%-12s %-9s %-26s %12.2f %14.2f %9.1f%%\n", level, wslabel,
                [&] {
                    static std::string s;
                    s = std::to_string(cpus.size()) + " thr: ";
                    for (size_t i = 0; i < cpus.size(); ++i) {
                        s += (i ? "," : "") + std::to_string(cpus[i]);
                    }
                    return s.c_str();
                }(),
                best_per, best_agg, 100.0 * best_overlap);
    std::fflush(stdout);
}

int main(int argc, char ** argv) {
    int cpu     = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    int samples = (argc > 2) ? std::atoi(argv[2]) : 31;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }

    std::vector<Level> levels = {
        {"L1D -> reg",
         {{"8 KiB", f_8k, 8u << 10, 0, 0},
          {"16 KiB", f_16k, 16u << 10, 0, 0},
          {"24 KiB", f_24k, 24u << 10, 0, 0},
          {"32 KiB", f_32k, 32u << 10, 0, 0},
          {"40 KiB", f_40k, 40u << 10, 0, 0},
          {"48 KiB", f_48k, 48u << 10, 0, 0},
          {"64 KiB", f_64k, 64u << 10, 0, 0}}},
        {"L2 -> L1D",
         {{"128 KiB", f_128k, 128u << 10, 0, 0},
          {"192 KiB", f_192k, 192u << 10, 0, 0},
          {"256 KiB", f_256k, 256u << 10, 0, 0},
          {"384 KiB", f_384k, 384u << 10, 0, 0},
          {"512 KiB", f_512k, 512u << 10, 0, 0},
          {"768 KiB", f_768k, 768u << 10, 0, 0}}},
        {"DRAM -> L2",
         {{"8 MiB", f_8m, 8u << 20, 0, 0},
          {"16 MiB", f_16m, 16u << 20, 0, 0},
          {"32 MiB", f_32m, 32u << 20, 0, 0},
          {"64 MiB", f_64m, 64u << 20, 0, 0}}},
    };

    std::printf("# cpu=%d vlenb=%llu samples=%d  kernel: vle8.v e8,m8 (1024 B/inst)\n", cpu,
                (unsigned long long) k3::vlenb(), samples);

    // 确认 THP 真的生效：分配一块并读 smaps_rollup。
    {
        void *   raw     = nullptr;
        size_t   raw_len = 0;
        uint8_t * b      = alloc_thp(4u << 20, &raw, &raw_len);
        if (b != nullptr) {
            FILE * f = fopen("/proc/self/smaps_rollup", "r");
            char   line[256];
            while (f != nullptr && fgets(line, sizeof(line), f)) {
                if (std::strncmp(line, "AnonHugePages", 13) == 0) {
                    std::printf("# %s", line);
                }
            }
            if (f != nullptr) {
                fclose(f);
            }
            munmap(raw, raw_len);
        }
    }

    for (Level & lv : levels) {
        std::printf("\n## %s\n", lv.name);
        std::printf("%-10s %14s %14s %16s\n", "pass size", "cycles/pass", "naive B/cycle",
                    "pairwise B/cycle");
        double prev_bytes = 0, prev_cycles = 0;
        for (Point & p : lv.points) {
            measure(p, counter, samples);
            double pairwise = 0;
            if (prev_bytes > 0 && p.cycles > prev_cycles) {
                pairwise = (double(p.bytes) - prev_bytes) / (p.cycles - prev_cycles);
            }
            std::printf("%-10s %14.1f %14.2f %16s", p.label, p.cycles, p.bpc,
                        pairwise > 0 ? "" : "-");
            if (pairwise > 0) {
                std::printf("%16.2f", pairwise);
            }
            std::printf("\n");
            std::fflush(stdout);
            prev_bytes  = double(p.bytes);
            prev_cycles = p.cycles;
        }
        double slope, intercept, resid;
        fit(lv.points, slope, intercept, resid);
        std::printf("fit: %.2f B/cycle, 每趟固定开销 %.1f cycle, 最大相对残差 %.2f%%\n",
                    1.0 / slope, intercept, 100.0 * resid);
    }

    // 多核共享结构。L2 用 256 KiB/线程，保证同簇 4 线程合计 1 MiB 正好等于
    // L2 容量而不溢出。
    std::printf("\n## 多核共享结构\n");
    std::printf("%-12s %-9s %-26s %12s %14s %10s\n", "level", "ws/thr", "threads", "per-thr B/c",
                "aggregate B/c", "overlap");
    const std::vector<std::vector<int>> sets = {
        {8}, {8, 9, 10, 11}, {8, 10, 12, 14}, {8, 9, 10, 11, 12, 13, 14, 15},
    };
    int mt_samples = std::max(3, samples / 5);
    for (const auto & s : sets) {
        mt_point("L1D -> reg", "16 KiB", f_16k, 16u << 10, s, mt_samples);
    }
    for (const auto & s : sets) {
        mt_point("L1D -> reg", "64 KiB", f_64k, 64u << 10, s, mt_samples);
    }
    for (const auto & s : sets) {
        mt_point("L2 -> L1D", "256 KiB", f_256k, 256u << 10, s, mt_samples);
    }
    // 标量对照：同样的工作集，走标量 ld。
    for (const auto & s : sets) {
        mt_point("L1D scalar", "16 KiB", s_16k, 16u << 10, s, mt_samples);
    }
    for (const auto & s : sets) {
        mt_point("L1D scalar", "64 KiB", s_64k, 64u << 10, s, mt_samples);
    }
    for (const auto & s : sets) {
        mt_point("DRAM -> L2", "64 MiB", f_64m, 64u << 20, s, mt_samples);
    }
    return 0;
}
