// q4_0-q8_0-IME-RR 的 L76-L95 区域额外开销拆解（microbenchmark 法）。
//
// 背景：docs/boards/spacemit-k3-a100/instruction-throughput.md 给出各指令的
// 稳态吞吐。按“逐指令吞吐相加、标量与 vsetvli 视为与向量流水重叠”的算法，
// L76-L95 的理论下界是
//
//   2 x vl1r.v (128 B)   = 4.96
//   4 x vmv.v.i e32,m2   = 16
//   8 x smt.vmadot i8    = 8
//   ---------------------------
//   合计                 = 28.96 ~ 29 cycle
//
// 而 PC 采样反推的实测值是 45.8 cycle/tile，缺口 16.8 cycle。
//
// 本程序在片上复刻 L76-L95 的指令流顺序、寄存器分配和地址步进，然后用
// “逐项移除”的变体给出各成分的边际实测成本：
//
//   访存开销   full - no_load
//   清零开销   full - no_clear
//   矩阵段     full - no_dot
//   数据依赖   dots_chain - dots_nochain
//   配置开销   full - one_vset
//   源操作数   full - same_w
//   标量/循环  empty
//
// Aq 工作集按三种 ring 分别测量，因为真实 kernel 对每个 N tile 会流式读完
// 720 KiB 的 packed A（每 15 KiB 换一个 K block），既不是纯 L1D 常驻，也不是
// 完全冷启动。
//
// 用法：k3_rr_l76_overhead [cpu]
#include "k3_a100.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using Body = void (*)(const uint8_t *a, uint64_t iters, uint64_t mask);

constexpr size_t kABytes = (2u << 20) + 4096;

alignas(256) uint8_t g_a[kABytes];
alignas(256) uint8_t g_w[1024];

// 与真实 kernel 相同的 W 与 A 寄存器预置。循环外一次性装入，循环内不再触碰。
#define W_SETUP                                                                                    \
    "vsetvli t0, zero, e8, m1, ta, ma\n\t"                                                         \
    "vl1r.v v16, (%[w])\n\t"                                                                       \
    "addi t0, %[w], 0x80\n\t"                                                                      \
    "vl1r.v v17, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x100\n\t"                                                                     \
    "vl1r.v v18, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x180\n\t"                                                                     \
    "vl1r.v v19, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x200\n\t"                                                                     \
    "vl1r.v v21, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x280\n\t"                                                                     \
    "vl1r.v v22, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x300\n\t"                                                                     \
    "vl1r.v v25, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x380\n\t"                                                                     \
    "vl1r.v v26, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x400\n\t"                                                                     \
    "vl1r.v v28, (t0)\n\t"                                                                         \
    "addi t0, %[w], 0x480\n\t"                                                                     \
    "vl1r.v v29, (t0)\n\t"

// Aq 两条 128 B 整寄存器 load，与 kernel +0x106/+0x10e 一致。
#define L_FULL                                                                                     \
    "vl1r.v v16, (t2)\n\t"                                                                         \
    "addi t3, t2, 0x80\n\t"                                                                        \
    "vl1r.v v17, (t3)\n\t"
#define L_NONE ""

// Q 累加器清零，与 kernel +0x112..+0x122 一致。
#define C_FULL                                                                                     \
    "vsetvli t3, zero, e32, m2, ta, ma\n\t"                                                        \
    "vmv.v.i v14, 0\n\t"                                                                           \
    "vmv.v.i v12, 0\n\t"                                                                           \
    "vmv.v.i v10, 0\n\t"                                                                           \
    "vmv.v.i v8,  0\n\t"
#define C_NONE ""

#define VS8 "vsetvli t3, zero, e8, m1, tu, ma\n\t"

// 真实调度：4 条 low dot，再 4 条 high dot；每条链在本区域内被更新两次。
#define D_CHAIN                                                                                    \
    VS8                                                                                            \
    "smt.vmadot v14, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v12, v16, v19, i8\n\t"                                                             \
    "smt.vmadot v10, v16, v21, i8\n\t"                                                             \
    "smt.vmadot v8,  v16, v22, i8\n\t"                                                             \
    VS8                                                                                            \
    "smt.vmadot v14, v17, v25, i8\n\t"                                                             \
    "smt.vmadot v12, v17, v26, i8\n\t"                                                             \
    "smt.vmadot v10, v17, v28, i8\n\t"                                                             \
    "smt.vmadot v8,  v17, v29, i8\n\t"

// 去掉累加器 RAW 依赖：第二组写入 4 个独立累加器。
#define D_NOCHAIN                                                                                  \
    VS8                                                                                            \
    "smt.vmadot v14, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v12, v16, v19, i8\n\t"                                                             \
    "smt.vmadot v10, v16, v21, i8\n\t"                                                             \
    "smt.vmadot v8,  v16, v22, i8\n\t"                                                             \
    VS8                                                                                            \
    "smt.vmadot v2,  v17, v25, i8\n\t"                                                             \
    "smt.vmadot v4,  v17, v26, i8\n\t"                                                             \
    "smt.vmadot v6,  v17, v28, i8\n\t"                                                             \
    "smt.vmadot v30, v17, v29, i8\n\t"

// 去掉第二条重复的 vsetvli（kernel +0x13a 与 +0x126 的 vtype 相同）。
#define D_ONEVSET                                                                                  \
    VS8                                                                                            \
    "smt.vmadot v14, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v12, v16, v19, i8\n\t"                                                             \
    "smt.vmadot v10, v16, v21, i8\n\t"                                                             \
    "smt.vmadot v8,  v16, v22, i8\n\t"                                                             \
    "smt.vmadot v14, v17, v25, i8\n\t"                                                             \
    "smt.vmadot v12, v17, v26, i8\n\t"                                                             \
    "smt.vmadot v10, v17, v28, i8\n\t"                                                             \
    "smt.vmadot v8,  v17, v29, i8\n\t"

// 8 条 dot 共用同一个 W 源寄存器，检验源读端口的压力。
#define D_SAMEW                                                                                    \
    VS8                                                                                            \
    "smt.vmadot v14, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v12, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v10, v16, v18, i8\n\t"                                                             \
    "smt.vmadot v8,  v16, v18, i8\n\t"                                                             \
    VS8                                                                                            \
    "smt.vmadot v14, v17, v18, i8\n\t"                                                             \
    "smt.vmadot v12, v17, v18, i8\n\t"                                                             \
    "smt.vmadot v10, v17, v18, i8\n\t"                                                             \
    "smt.vmadot v8,  v17, v18, i8\n\t"

// 进入 L76-L95 时的 vtype 状态。真实 kernel 的上一区域以 `vsetvli e16,m1` +
// `vslidedown.vi` 结束，因此 +0x112 的 `vsetvli e32,m2` 是一次真实的
// e16->e32 切换；复刻件里若沿用 e8,m1 则切换代价不同。
#define PREP_NONE ""
#define PREP_E8 "vsetvli t3, zero, e8, m1, ta, ma\n\t" "vslidedown.vi v30, v30, 1\n\t"
#define PREP_E16 "vsetvli t3, zero, e16, m1, ta, ma\n\t" "vslidedown.vi v30, v30, 1\n\t"

// 一次内层迭代 = L76-L95 的完整指令流；地址按 256 B 步进，用 mask 在 ring 内环绕。
#define STREAM_P(fname, PREP, LOADS, CLEARS, DOTS)                                                 \
    __attribute__((noinline)) void fname(const uint8_t *a, uint64_t iters, uint64_t mask) {        \
        asm volatile(W_SETUP                                                                       \
                     "li t1, 0\n\t"                                                                \
                     "1:\n\t" PREP                                                                 \
                     "add t2, %[a], t1\n\t" LOADS CLEARS DOTS "addi t1, t1, 256\n\t"               \
                     "and t1, t1, %[mask]\n\t"                                                     \
                     "addi %[n], %[n], -1\n\t"                                                     \
                     "bnez %[n], 1b\n\t"                                                           \
                     : [n] "+r"(iters)                                                             \
                     : [a] "r"(a), [mask] "r"(mask), [w] "r"(g_w)                                  \
                     : "t0", "t1", "t2", "t3", "memory", "v0", "v1", "v2", "v3", "v4", "v5",       \
                       "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",     \
                       "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",       \
                       "v27", "v28", "v29", "v30", "v31");                                         \
    }
#define STREAM(fname, LOADS, CLEARS, DOTS) STREAM_P(fname, PREP_NONE, LOADS, CLEARS, DOTS)

STREAM(k_full, L_FULL, C_FULL, D_CHAIN)
STREAM_P(k_full_prep_e8, PREP_E8, L_FULL, C_FULL, D_CHAIN)
STREAM_P(k_full_prep_e16, PREP_E16, L_FULL, C_FULL, D_CHAIN)
STREAM(k_no_load, L_NONE, C_FULL, D_CHAIN)
STREAM(k_no_clear, L_FULL, C_NONE, D_CHAIN)
STREAM(k_no_dot, L_FULL, C_FULL, "")
STREAM(k_no_chain, L_FULL, C_FULL, D_NOCHAIN)
STREAM(k_one_vset, L_FULL, C_FULL, D_ONEVSET)
STREAM(k_same_w, L_FULL, C_FULL, D_SAMEW)
STREAM(k_loads_only, L_FULL, C_NONE, "")
STREAM(k_clears_only, L_NONE, C_FULL, "")
STREAM(k_dots_chain, L_NONE, C_NONE, D_CHAIN)
STREAM(k_dots_nochain, L_NONE, C_NONE, D_NOCHAIN)
STREAM(k_empty, L_NONE, C_NONE, "")

struct Case {
    const char * name;
    Body         fn;
};

const Case kCases[] = {
    {"full",          k_full},
    {"full_prep_e8",  k_full_prep_e8},
    {"full_prep_e16", k_full_prep_e16},
    {"no_load",       k_no_load},
    {"no_clear",      k_no_clear},
    {"no_dot",        k_no_dot},
    {"no_chain",      k_no_chain},
    {"one_vset",      k_one_vset},
    {"same_w",        k_same_w},
    {"loads_only",    k_loads_only},
    {"clears_only",   k_clears_only},
    {"dots_chain",    k_dots_chain},
    {"dots_nochain",  k_dots_nochain},
    {"empty",         k_empty},
};

// 访存特征对照：每迭代 1/2/4/8 条相邻 vl1r.v，指针步进 = 条数 × 128 B。
// 用来判断整寄存器 load 的成本是“每条指令固定开销”还是“每字节带宽”，
// 以及 2 条相邻 load 是否因为 WAW（写同一目标寄存器）而串行化。
#define VC_ALL                                                                                     \
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "memory", "v0", "v1", "v2", "v3", "v4", "v5",   \
        "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",     \
        "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"

#define LDK(fname, STMTS, STRIDE)                                                                  \
    __attribute__((noinline)) void fname(const uint8_t *a, uint64_t iters, uint64_t mask) {        \
        asm volatile("vsetvli t0, zero, e8, m1, ta, ma\n\t"                                        \
                     "li t1, 0\n\t"                                                                \
                     "1:\n\t"                                                                      \
                     "add t2, %[a], t1\n\t" STMTS "addi t1, t1, " STRIDE "\n\t"                    \
                     "and t1, t1, %[mask]\n\t"                                                     \
                     "addi %[n], %[n], -1\n\t"                                                     \
                     "bnez %[n], 1b\n\t"                                                           \
                     : [n] "+r"(iters)                                                             \
                     : [a] "r"(a), [mask] "r"(mask)                                                \
                     : VC_ALL);                                                                    \
    }

LDK(k_ld1, "vl1r.v v16, (t2)\n\t", "128")
LDK(k_ld2, "vl1r.v v16, (t2)\n\t"                                                                  \
           "addi t3, t2, 0x80\n\t"                                                                 \
           "vl1r.v v17, (t3)\n\t",                                                                 \
    "256")
LDK(k_ld4, "vl1r.v v16, (t2)\n\t"                                                                  \
           "addi t3, t2, 0x80\n\t"                                                                 \
           "vl1r.v v17, (t3)\n\t"                                                                  \
           "addi t4, t2, 0x100\n\t"                                                                \
           "vl1r.v v2, (t4)\n\t"                                                                   \
           "addi t5, t2, 0x180\n\t"                                                                \
           "vl1r.v v3, (t5)\n\t",                                                                  \
    "512")
LDK(k_ld8, "vl1r.v v16, (t2)\n\t"                                                                  \
           "addi t3, t2, 0x80\n\t"                                                                 \
           "vl1r.v v17, (t3)\n\t"                                                                  \
           "addi t4, t2, 0x100\n\t"                                                                \
           "vl1r.v v2, (t4)\n\t"                                                                   \
           "addi t5, t2, 0x180\n\t"                                                                \
           "vl1r.v v3, (t5)\n\t"                                                                   \
           "addi t6, t2, 0x200\n\t"                                                                \
           "vl1r.v v4, (t6)\n\t"                                                                   \
           "addi t3, t2, 0x280\n\t"                                                                \
           "vl1r.v v5, (t3)\n\t"                                                                   \
           "addi t4, t2, 0x300\n\t"                                                                \
           "vl1r.v v6, (t4)\n\t"                                                                   \
           "addi t5, t2, 0x380\n\t"                                                                \
           "vl1r.v v7, (t5)\n\t",                                                                  \
    "1024")
// 同一目标寄存器的两条 load 不推进地址：检验 WAW 串行化假设。
LDK(k_ld2_same, "vl1r.v v16, (t2)\n\t"                                                             \
                "vl1r.v v17, (t2)\n\t",                                                            \
    "0")

const Case kLdCases[] = {
    {"ld1  (128 B)",    k_ld1},
    {"ld2  (256 B)",    k_ld2},
    {"ld4  (512 B)",    k_ld4},
    {"ld8  (1024 B)",   k_ld8},
    {"ld2_same_addr",   k_ld2_same},
};

struct Ring {
    const char * name;
    size_t       bytes;
};

// packed A 的真实工作集：每个 (N tile, K block) 调用的 15 KiB，以及一个 N tile
// 跨 48 个 K block 的 720 KiB 流。
const Ring kRings[] = {
    {"16 KiB (L1D hot)", 16u << 10},
    {"768 KiB (L2 满)", 768u << 10},
    {"2 MiB (L2 溢出)", 2u << 20},
};

}  // namespace

int main(int argc, char ** argv) {
    const int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }
    std::memset(g_a, 0x11, sizeof(g_a));
    std::memset(g_w, 0x22, sizeof(g_w));

    constexpr uint64_t kTargetCycles = 60'000'000;
    constexpr int      kSamples      = 7;

    std::printf("# L76-L95 instruction-stream replica, VLEN=%llu bits, cpu=%d\n",
                (unsigned long long) k3::vlenb() * 8, cpu);
    std::printf("# theoretical floor (throughput sum) = 2*2.48 + 4*4 + 8*1 = 28.96 cycle\n\n");

    auto run_case = [&](const Case & c, uint64_t mask) {
        const uint64_t iters = kTargetCycles / 30 + 1;
        c.fn(g_a, iters / 8 + 1, mask);  // 预热
        double   best     = 1e18;
        uint64_t best_ret = 0;
        for (int s = 0; s < kSamples; ++s) {
            counter.start();
            c.fn(g_a, iters, mask);
            k3::CycleCounter::Reading r{};
            if (!counter.stop(r) || r.running_ratio < 0.99) {
                continue;
            }
            const double per = double(r.cycles) / double(iters);
            if (per < best) {
                best     = per;
                best_ret = r.instructions;
            }
        }
        std::printf("%-14s %14.3f %18.3f\n", c.name, best, double(best_ret) / double(iters));
        std::fflush(stdout);
    };

    for (const Ring & ring : kRings) {
        std::printf("## ring = %s\n", ring.name);
        std::printf("%-14s %14s %18s\n", "variant", "cycle/iter", "instructions/iter");
        for (const Case & c : kCases) {
            run_case(c, ring.bytes - 1);
        }
        std::printf("\n");
    }

    std::printf("## 访存特征（16 KiB ring，L1D 常驻）\n");
    std::printf("%-14s %14s %18s\n", "variant", "cycle/iter", "instructions/iter");
    for (const Case & c : kLdCases) {
        run_case(c, (16u << 10) - 1);
    }
    std::printf("\n");
    return 0;
}
