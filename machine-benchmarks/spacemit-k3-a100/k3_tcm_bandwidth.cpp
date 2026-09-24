// K3 A100：TCM → 寄存器带宽（1 / 2 异簇 / 4 同簇 / 8 线程）。
//
// 用法：--span 3072 --samples 7
//   一次 mmap 申请整颗 3 MiB TCM（= 8 个 384 KiB 块），进程内按 384 KiB 切片，
//   每个核读自己归属的那一块。切片顺序 = 物理块升序（驱动 list_sort 保证），
//   第 k 片即第 k 个核的归属块。
//
// 已知限制：多个同时存在的独立 mmap 会触发 /dev/tcm 驱动缺陷（内核 Oops），
// 必须一次 mmap 申请整段再切片；worker 只读已映射内存，不触碰 /dev/tcm。
//
// 测量口径与 k3_load_sharing.cpp / k3_path_bandwidth.cpp 一致：
//   per-thread B/cycle = 该线程自己的字节数 / 该线程的 PMU cycles
//   aggregate B/cycle  = 总字节数 / (公共墙钟窗口 × 实测频率中位数)
// 样本有效条件：请求的 CPU == sched_getcpu()、PMU running/enabled ≥ 99%、
// 公共 timed-region overlap ≥ 90%。
//
// kernel 是 LMUL=m8 的 vle8.v（VLEN=1024 时每条 1024 B），两种结构：
//   full  ：一趟把整个工作集完全展开，每条 load 后跟一条推进指针的 addi
//   zero24：24 个基址寄存器、循环体内零地址运算
// 另有 anon（THP）对照组，用于校验骨架（L1D/L2 数字与文档一致）。
#include "k3_a100.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

namespace {

using Fn = void (*)(const uint8_t *, uint64_t);

// K3 TCM 的分配最小单位：一个块 384 KiB，每簇 4 块 = 1.5 MiB。
constexpr size_t kBlockBytes        = 384u << 10;
constexpr int    kBlocksPerCluster  = 4;
constexpr int    kClusterCount      = 2;
constexpr int    kBlockCount        = kBlocksPerCluster * kClusterCount;
constexpr int    kA100Cpus[kBlockCount] = {8, 9, 10, 11, 12, 13, 14, 15};

#define TCM_VA_TO_PA _IOR('c', 4, int)
struct VaPa {
    void * vaddr;
    void * paddr;
};

// ---------------------------------------------------------------------------
// 读 kernel：一趟 NGROUP 组 × 4 条 m8 load（每条 1024 B），组内每条 load
// 后跟一条 addi 推进指针；整趟完全展开，没有内层分支。
#define READ_FULL(fname, NGROUP)                                                                   \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                     \
        asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"                                        \
                     "2:\n\t"                                                                      \
                     "mv t1, %0\n\t"                                                               \
                     ".rept " #NGROUP "\n\t"                                                       \
                     "vle8.v v0, (t1)\n\taddi t1, t1, 1024\n\t"                                    \
                     "vle8.v v8, (t1)\n\taddi t1, t1, 1024\n\t"                                    \
                     "vle8.v v16, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     "vle8.v v24, (t1)\n\taddi t1, t1, 1024\n\t"                                   \
                     ".endr\n\t"                                                                   \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                          \
                     : "+r"(base), "+r"(rep)                                                       \
                     :                                                                            \
                     : "t0", "t1", "memory", "v0", "v8", "v16", "v24");                            \
    }

READ_FULL(f_48k, 12)    // 12 × 4 KiB = 48 KiB，L1D 对照
READ_FULL(f_384k, 96)   // 96 × 4 KiB = 384 KiB，一个 TCM 块

// 零地址运算：24 个基址寄存器覆盖 24 KiB，循环体内一条地址运算也没有。
// 与 k3_l1d_peak.cpp 的 ZERO_ADDI 变体同构。
#define ZERO_ADDI_SETUP                                                                            \
    "vsetvli t0, zero, e8, m8, ta, ma\n\t"                                                          \
    "mv   a2, %0\n\t"                                                                               \
    "addi a3, a2, 1024\n\taddi a4, a3, 1024\n\taddi a5, a4, 1024\n\taddi a6, a5, 1024\n\t"          \
    "addi a7, a6, 1024\n\taddi t1, a7, 1024\n\taddi t2, t1, 1024\n\taddi t3, t2, 1024\n\t"          \
    "addi t4, t3, 1024\n\taddi t5, t4, 1024\n\taddi t6, t5, 1024\n\taddi s2, t6, 1024\n\t"          \
    "addi s3, s2, 1024\n\taddi s4, s3, 1024\n\taddi s5, s4, 1024\n\taddi s6, s5, 1024\n\t"          \
    "addi s7, s6, 1024\n\taddi s8, s7, 1024\n\taddi s9, s8, 1024\n\taddi s10, s9, 1024\n\t"         \
    "addi s11, s10, 1024\n\taddi s0, s11, 1024\n\taddi s1, s0, 1024\n\taddi a1, s1, 1024\n\t"

#define ZERO_ADDI_BODY                                                                             \
    "vle8.v v0, (a2)\n\tvle8.v v8, (a3)\n\tvle8.v v16, (a4)\n\tvle8.v v24, (a5)\n\t"                \
    "vle8.v v0, (a6)\n\tvle8.v v8, (a7)\n\tvle8.v v16, (t1)\n\tvle8.v v24, (t2)\n\t"                \
    "vle8.v v0, (t3)\n\tvle8.v v8, (t4)\n\tvle8.v v16, (t5)\n\tvle8.v v24, (t6)\n\t"                \
    "vle8.v v0, (s2)\n\tvle8.v v8, (s3)\n\tvle8.v v16, (s4)\n\tvle8.v v24, (s5)\n\t"                \
    "vle8.v v0, (s6)\n\tvle8.v v8, (s7)\n\tvle8.v v16, (s8)\n\tvle8.v v24, (s9)\n\t"                \
    "vle8.v v0, (s10)\n\tvle8.v v8, (s11)\n\tvle8.v v16, (s0)\n\tvle8.v v24, (s1)\n\t"

#define ZERO_ADDI_CLOBBER                                                                          \
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s0", "s1", \
        "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "memory", "v0", "v8", "v16",  \
        "v24"

// 一趟 = 24 KiB（24 条 load），重复 16 次；循环体内零地址运算。
__attribute__((noinline)) void z_24k(const uint8_t * base, uint64_t rep) {
    asm volatile(ZERO_ADDI_SETUP "2:\n\t"
                 ".rept 16\n\t" ZERO_ADDI_BODY ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : ZERO_ADDI_CLOBBER);
}

// 一次测量跨 4 个块（每簇 1.5 MiB）：逐块调用 f_384k。
// 用来区分「TCM 真的这么快」和「其实在读 1 MiB L2」：1.5 MiB 超出 L2 容量。
using Fn4 = void (*)(const uint8_t * const *, uint64_t);

__attribute__((noinline)) void f_cluster4(const uint8_t * const * bases, uint64_t rep) {
    for (int k = 0; k < 4; ++k) {
        f_384k(bases[k], rep);
    }
}

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
    int      want_cpu = -1;
    double   ratio = 0;
    bool     ok  = false;
};

struct Point {
    double per_thread_min = 0;
    double per_thread_max = 0;
    double sum_per_thread  = 0;  // 各线程 B/cycle 之和（不依赖时间对齐）
    double aggregate      = 0;   // 需要公共时间窗
    double overlap        = 0;
    bool   ok             = false;
    bool   overlap_ok     = false;
};

// worker 只读已经映射好的内存，不做任何 mmap/munmap/open。
void worker(int cpu, Fn fn, Fn4 fn4, const std::vector<const uint8_t *> & bases,
            uint64_t bytes_per_pass, uint64_t rep, SpinBarrier & barrier, Sample & out) {
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
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
    const bool multi = (fn4 != nullptr);
    auto       call  = [&](uint64_t r) {
        if (multi) {
            fn4(bases.data(), r);
        } else {
            fn(bases[0], r);
        }
    };
    call(std::max<uint64_t>(rep / 8, 2));  // 预热
    barrier.arrive_and_wait();
    double t0 = k3::wall_seconds();
    counter.start();
    call(rep);
    k3::CycleCounter::Reading r{};
    bool                      stopped = counter.stop(r);
    double                    t1      = k3::wall_seconds();
    barrier.arrive_and_wait();
    out.t0     = t0;
    out.t1     = t1;
    out.cycles = r.cycles;
    out.bytes  = bytes_per_pass * rep;
    out.cpu    = sched_getcpu();
    out.want_cpu = cpu;
    out.ratio  = r.running_ratio;
    out.ok     = stopped && out.cpu == cpu && r.running_ratio > 0.99;
}

// per_worker[i] 是第 i 个 worker 要读的块；单块 kernel 只用第一个。
Point run(Fn fn, Fn4 fn4, const std::vector<std::vector<const uint8_t *>> & per_worker,
          const std::vector<int> & cpus, uint64_t bytes_per_pass, int samples) {
    const uint64_t rep = std::max<uint64_t>((256ull << 20) / bytes_per_pass, 4);
    Point          best{};
    for (int s = 0; s < samples; ++s) {
        SpinBarrier              barrier(int(cpus.size()));
        std::vector<Sample>      samp(cpus.size());
        std::vector<std::thread> th;
        for (size_t i = 0; i < cpus.size(); ++i) {
            th.emplace_back(worker, cpus[i], fn, fn4, std::cref(per_worker[i]), bytes_per_pass, rep,
                            std::ref(barrier), std::ref(samp[i]));
        }
        for (auto & t : th) {
            t.join();
        }
        double              fs = 1e300, le = 0, ls = 0, fe = 1e300;
        uint64_t            total = 0;
        std::vector<double> per, freq;
        bool                ok = true;
        int                 rejected = 0;
        for (const Sample & m : samp) {
            if (!m.ok) {
                ok = false;
                if (++rejected <= 4) {
                    std::printf("    [reject] cpu=%d (want %d) ratio=%.4f cycles=%llu "
                                "t0=%.6f t1=%.6f\n",
                                m.cpu, m.want_cpu, m.ratio, (unsigned long long) m.cycles, m.t0,
                                m.t1);
                }
                continue;
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
            std::fflush(stdout);
            continue;
        }
        const double span    = le - fs;
        const double overlap = std::max(0.0, fe - ls) / span;
        const double agg     = double(total) / (span * k3::median(freq));
        double       sum     = 0;
        for (double v : per) {
            sum += v;
        }
        const double pmin = *std::min_element(per.begin(), per.end());
        const double pmax = *std::max_element(per.begin(), per.end());
        const bool   ov_ok = overlap >= 0.90;
        // 优先选 overlap 合格的样本；全都不合格时退回"各线程速率之和"。
        const bool better = ov_ok ? (!best.overlap_ok || agg > best.aggregate)
                                  : (!best.overlap_ok && sum > best.sum_per_thread);
        if (better) {
            best.aggregate      = agg;
            best.per_thread_min = pmin;
            best.per_thread_max = pmax;
            best.sum_per_thread = sum;
            best.overlap        = overlap;
            best.overlap_ok     = ov_ok;
            best.ok             = true;
        }
        if (!ov_ok && samples > 1) {
            std::printf("    [low-overlap %.1f%%] 每线程 B/cycle:", 100.0 * overlap);
            for (double v : per) {
                std::printf(" %.2f", v);
            }
            std::printf(" | 时长 ms:");
            for (const Sample & m : samp) {
                std::printf(" %.2f", 1e3 * (m.t1 - m.t0));
            }
            std::printf("\n");
            std::fflush(stdout);
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

// 每个组合里两个簇各有几个活跃核。
std::string cluster_shape(const std::vector<int> & cpus) {
    int c0 = 0, c1 = 0;
    for (int c : cpus) {
        (c <= 11 ? c0 : c1)++;
    }
    return std::to_string(c0) + "+" + std::to_string(c1);
}

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

struct TcmBlock {
    uint8_t * base     = nullptr;
    size_t    raw_len  = 0;
    uint64_t  paddr    = 0;
    int       cluster  = -1;
    bool      ok       = false;
};

}  // namespace

int main(int argc, char ** argv) {
    int  samples   = 7;
    int  nblocks   = kBlockCount;
    bool with_anon = true;
    bool share     = false;
    std::vector<int> cpu_override;
    int              req_id  = -1;
    int              span_kb = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--samples" && i + 1 < argc) {
            samples = std::atoi(argv[++i]);
        } else if (a == "--blocks" && i + 1 < argc) {
            nblocks = std::atoi(argv[++i]);
            if (nblocks < 1 || nblocks > kBlockCount) {
                std::fprintf(stderr, "--blocks must be 1..%d\n", kBlockCount);
                return 2;
            }
        } else if (a == "--share") {
            share = true;
        } else if (a == "--span" && i + 1 < argc) {
            // 一次 mmap 申请 span KiB（必须是 384 KiB 的整数倍），内部按
            // 384 KiB 切片给各线程（驱动会用 list_sort 按物理地址排序，
            // 因此第 k 个切片 = 第 k 个物理块）。
            span_kb = std::atoi(argv[++i]);
            if (span_kb <= 0 || span_kb % int(kBlockBytes >> 10) != 0) {
                std::fprintf(stderr, "--span 必须是 %zu KiB 的正整数倍\n", kBlockBytes >> 10);
                return 2;
            }
        } else if (a == "--req-id" && i + 1 < argc) {
            // 试探 offset != 0 的 ABI：把 8 字节请求值放在页对齐的用户内存里，
            // 用它的地址（页对齐指针）作为 mmap 的 offset 传入。
            // 驱动侧对应 _copy_from_user(&req, (void*)(vm_pgoff<<12), 8)。
            req_id = std::atoi(argv[++i]);
        } else if (a == "--cpus" && i + 1 < argc) {
            // 覆盖默认配置集，只测这一个 CPU 列表，例如 --cpus 8,9
            cpu_override.clear();
            const std::string list = argv[++i];
            size_t            pos  = 0;
            while (pos < list.size()) {
                size_t next = list.find(',', pos);
                if (next == std::string::npos) {
                    next = list.size();
                }
                cpu_override.push_back(std::atoi(list.substr(pos, next - pos).c_str()));
                pos = next + 1;
            }
        } else if (a == "--no-anon") {
            with_anon = false;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--samples N] [--blocks N] [--share] [--no-anon]\n", argv[0]);
            return 2;
        }
    }

    // 主线程先进入 A100 核域但先不钉核：之后创建的 worker 才能把 affinity
    // 收窄到 8-15 里的任意一个核。
    if (!k3::enter_a100_domain()) {
        return 1;
    }
    std::printf("# 主线程已进入 A100 域，当前 cpu=%d, vlenb=%llu, samples=%d, 分配块数=%d, TCM 用法=%s\n",
                sched_getcpu(), (unsigned long long) k3::vlenb(), samples, nblocks,
                share ? "多线程共享同批块" : "每线程独立块");

    int fd = open("/dev/tcm", O_RDWR | O_SYNC);
    if (fd < 0) {
        std::fprintf(stderr, "open(/dev/tcm) failed: %s\n", std::strerror(errno));
        return 1;
    }

    // ---- 串行分配：K3 的最小单位是 384 KiB，每簇 4 块 = 1.5 MiB ----
    void * req_page = nullptr;
    off_t  req_off  = 0;
    if (req_id >= 0) {
        req_page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (req_page == MAP_FAILED) {
            std::fprintf(stderr, "req page mmap failed: %s\n", std::strerror(errno));
            return 1;
        }
        *static_cast<volatile uint64_t *>(req_page) = uint64_t(req_id);
        req_off = off_t(uintptr_t(req_page));
        std::printf("# 试探 offset ABI: req_page=%p 值=%d -> mmap offset=%#llx\n", req_page, req_id,
                    static_cast<unsigned long long>(req_off));
    }
    std::vector<TcmBlock> blocks;
    blocks.reserve(kBlockCount);
    void * span_base = nullptr;
    size_t span_len  = 0;
    if (span_kb > 0) {
        // 一次 mmap 拿一大块，内部按 384 KiB 切片（切片的物理块顺序由驱动
        // 的 list_sort 保证是升序）。
        span_len           = size_t(span_kb) << 10;
        const int slices = int(span_len / kBlockBytes);
        if (slices < 1 || slices > kBlockCount) {
            std::fprintf(stderr, "--span 只能是 384..%d KiB\n", int(kBlockCount * kBlockBytes >> 10));
            return 2;
        }
        void * p = mmap(nullptr, span_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, req_off);
        if (p == MAP_FAILED) {
            std::printf("span mmap(%d KiB) FAILED: %s\n", span_kb, std::strerror(errno));
            std::fflush(stdout);
            return 1;
        }
        span_base        = p;
        VaPa va_pa{p, nullptr};
        errno            = 0;
        const int pa_ret = ioctl(fd, TCM_VA_TO_PA, &va_pa);
        std::printf("span mmap 成功: va=%p len=%d KiB slices=%d 首节点paddr=%#llx (VA_TO_PA ret=%d "
                    "errno=%d)\n",
                    p, span_kb, slices, static_cast<unsigned long long>(uint64_t(va_pa.paddr)),
                    pa_ret, errno);
        std::fflush(stdout);
        for (int i = 0; i < slices; ++i) {
            TcmBlock b;
            b.base    = static_cast<uint8_t *>(p) + size_t(i) * kBlockBytes;
            b.raw_len = 0;  // 由 span_base 统一 munmap
            b.ok      = true;
            blocks.push_back(b);
            std::printf("  slice %d: va=%p .. +384 KiB\n", i, static_cast<void *>(b.base));
        }
        std::fflush(stdout);
        nblocks = slices;
    }
    for (int i = 0; span_kb == 0 && i < nblocks; ++i) {
        TcmBlock b;
        void *   p = mmap(nullptr, kBlockBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, req_off);
        if (p == MAP_FAILED) {
            std::printf("block %d: mmap(384 KiB) FAILED: %s\n", i, std::strerror(errno));
            std::fflush(stdout);
            break;
        }
        b.base    = static_cast<uint8_t *>(p);
        b.raw_len = kBlockBytes;
        VaPa va_pa{b.base, nullptr};
        errno            = 0;
        const int pa_ret = ioctl(fd, TCM_VA_TO_PA, &va_pa);
        if (pa_ret == 0 && va_pa.paddr != nullptr) {
            b.paddr = uint64_t(va_pa.paddr);
        }
        b.cluster = (b.paddr >= 0x2000000ull) ? 1 : 0;
        b.ok      = true;
        std::printf("block %d: va=%p VA_TO_PA ret=%d errno=%d paddr=%#llx -> cluster %d\n", i,
                    static_cast<void *>(b.base), pa_ret, errno,
                    static_cast<unsigned long long>(b.paddr), b.cluster);
        std::fflush(stdout);
        blocks.push_back(b);
    }
    if (blocks.size() != size_t(nblocks)) {
        std::fprintf(stderr, "TCM 分配不完整：只拿到 %zu/%d 个 384 KiB 块\n", blocks.size(),
                     nblocks);
        return 1;
    }
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::memset(blocks[i].base, int(0x11 * (i + 1)) & 0xff, kBlockBytes);
    }

    // anon 对照组（THP）；worker 仍然只读。
    void *   a48_raw[kBlockCount]   = {};
    void *   a384_raw[kBlockCount]  = {};
    size_t   a48_len[kBlockCount]   = {};
    size_t   a384_len[kBlockCount]  = {};
    uint8_t * a48[kBlockCount]      = {};
    uint8_t * a384[kBlockCount]     = {};
    if (with_anon) {
        for (int i = 0; i < kBlockCount; ++i) {
            a48[i]  = alloc_thp(48u << 10, &a48_raw[i], &a48_len[i]);
            a384[i] = alloc_thp(kBlockBytes, &a384_raw[i], &a384_len[i]);
            if (a48[i] == nullptr || a384[i] == nullptr) {
                std::fprintf(stderr, "anon alloc failed\n");
                return 1;
            }
        }
    }

    struct Config {
        const char *     name;
        std::vector<int> cpus;
    };
    const std::vector<Config> configs = cpu_override.empty()
                                            ? std::vector<Config>{
                                                  {"1 线程 (cpu8)", {8}},
                                                  {"2 线程 异簇 (8,12)", {8, 12}},
                                                  {"4 线程 同簇 (8-11)", {8, 9, 10, 11}},
                                                  {"8 线程 (8-15)", {8, 9, 10, 11, 12, 13, 14, 15}},
                                              }
                                            : std::vector<Config>{{"自定义 CPU 列表", cpu_override}};

    // which: 0 = TCM 单块, 1 = anon 48 KiB, 2 = anon 384 KiB,
    //        3 = TCM 本簇 4 块(1.5 MiB), 4 = anon 本簇 4 块
    struct Case {
        const char * name;
        Fn           fn;
        Fn4          fn4;
        size_t       bytes_per_pass;
        int          which;
    };
    std::vector<Case> cases = {
        {"TCM 384 KiB 顺序 (vle8.v m8, +addi)", f_384k, nullptr, kBlockBytes, 0},
        // 该 kernel 一趟外层迭代 = 16 次回扫 × 24 KiB = 384 KiB。
        {"TCM 24 KiB 零地址运算 (16 回扫=384 KiB)", z_24k, nullptr, 16 * (24u << 10), 0},
        {"TCM 本簇 1.5 MiB 顺序 (4×384 KiB)", nullptr, f_cluster4, 4 * kBlockBytes, 3},
    };
    if (with_anon) {
        cases.push_back({"anon 48 KiB 顺序 (L1D 对照)", f_48k, nullptr, 48u << 10, 1});
        cases.push_back({"anon 24 KiB 零地址运算 (L1D 对照)", z_24k, nullptr, 16 * (24u << 10), 1});
        cases.push_back({"anon 384 KiB 顺序 (L2 对照)", f_384k, nullptr, kBlockBytes, 2});
        cases.push_back({"anon 1.5 MiB 顺序 (L2 溢出对照)", nullptr, f_cluster4, 4 * kBlockBytes, 4});
    }

    for (const Case & c : cases) {
        std::printf("\n## %s\n", c.name);
        std::printf("%-22s %3s %7s %12s %12s %12s %14s %11s %9s\n", "cpus", "n", "簇分布",
                    "每线程最小", "每线程最大", "线程和B/cyc", "聚合B/cycle", "GB/s@1.79", "overlap");
        for (const Config & cfg : configs) {
            const bool tcm_single  = (c.which == 0);
            const bool tcm_cluster = (c.which == 3);
            // 这个用例需要几块 TCM；anon 用例不依赖 TCM 块数。
            int needed = 0;
            if (tcm_cluster) {
                needed = kBlockCount;  // 本簇 4 块（另一簇的用例需要全部 8 块）
            } else if (tcm_single) {
                needed = share ? 1 : (cfg.cpus.back() - kA100Cpus[0] + 1);
            }
            if (needed > nblocks || (share && tcm_cluster)) {
                std::printf("%-22s %3zu %7s %12s\n", label(cfg.cpus).c_str(), cfg.cpus.size(),
                            cluster_shape(cfg.cpus).c_str(),
                            (share && tcm_cluster) ? "skipped(share 模式不跑多块用例)"
                                                   : "skipped(需要更多块)");
                std::fflush(stdout);
                continue;
            }
            // 每个 worker 要读的块集合。
            std::vector<std::vector<const uint8_t *>> per_worker;
            for (int cpu : cfg.cpus) {
                const int idx = cpu - kA100Cpus[0];
                std::vector<const uint8_t *> one;
                if (c.which == 3) {  // 本簇的 4 个 TCM 块
                    const int first = (cpu <= 11) ? 0 : kBlocksPerCluster;
                    for (int k = 0; k < kBlocksPerCluster; ++k) {
                        one.push_back(blocks[first + k].base);
                    }
                } else if (c.which == 4) {  // 本簇的 4 个 anon 块
                    const int first = (cpu <= 11) ? 0 : kBlocksPerCluster;
                    for (int k = 0; k < kBlocksPerCluster; ++k) {
                        one.push_back(a384[first + k]);
                    }
                } else if (c.which == 0 && share) {  // 所有线程共享同一批 TCM 块
                    one.push_back(blocks[idx % nblocks].base);
                } else {
                    one.push_back(c.which == 0 ? blocks[idx].base
                                               : (c.which == 1 ? a48[idx] : a384[idx]));
                }
                per_worker.push_back(std::move(one));
            }
            Point p = run(c.fn, c.fn4, per_worker, cfg.cpus, c.bytes_per_pass, samples);
            if (!p.ok) {
                std::printf("%-22s %3zu %7s %12s\n", label(cfg.cpus).c_str(), cfg.cpus.size(),
                            cluster_shape(cfg.cpus).c_str(), "invalid");
                std::fflush(stdout);
                continue;
            }
            bool c0 = false, c1 = false;
            for (int x : cfg.cpus) {
                (x <= 11 ? c0 : c1) = true;
            }
            const int clusters = (c0 ? 1 : 0) + (c1 ? 1 : 0);
            char      agg[32];
            if (p.overlap_ok) {
                std::snprintf(agg, sizeof(agg), "%.2f", p.aggregate);
            } else {
                // overlap < 90%：聚合值只作参考，标记 *。
                std::snprintf(agg, sizeof(agg), "%.2f*", p.aggregate);
            }
            std::printf("%-22s %3zu %7s %12.2f %12.2f %12.2f %14s %11.1f %8.1f%%\n",
                        label(cfg.cpus).c_str(), cfg.cpus.size(), cluster_shape(cfg.cpus).c_str(),
                        p.per_thread_min, p.per_thread_max, p.sum_per_thread, agg,
                        p.aggregate * 1.79 / clusters, 100.0 * p.overlap);
            std::fflush(stdout);
        }
    }

    // ---- 串行释放 ----
    if (with_anon) {
        for (int i = 0; i < kBlockCount; ++i) {
            if (a48[i] != nullptr) {
                munmap(a48_raw[i], a48_len[i]);
            }
            if (a384[i] != nullptr) {
                munmap(a384_raw[i], a384_len[i]);
            }
        }
    }
    for (TcmBlock & b : blocks) {
        if (b.raw_len != 0 && munmap(b.base, b.raw_len) != 0) {
            std::fprintf(stderr, "munmap(%p) failed: %s\n", static_cast<void *>(b.base),
                         std::strerror(errno));
        }
    }
    if (span_base != nullptr) {
        if (munmap(span_base, span_len) != 0) {
            std::fprintf(stderr, "munmap(span %p) failed: %s\n", span_base, std::strerror(errno));
        }
    }
    close(fd);
    std::printf("\n# 完成，8 个 384 KiB TCM 块已全部释放\n");
    return 0;
}
