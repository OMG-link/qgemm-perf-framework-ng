// 验证 A100 的 L1D→寄存器峰值是否达到官方宣称的 512 bit/cycle（64 B/cycle）。
//
// k3_path_bandwidth 测到 55–60 B/cycle，比 64 低 6%–14%。怀疑点是那里的
// 循环体每条向量 load 都跟一条 addi 推进指针：
//
//     vle8.v v0, (t1)        # 1024 B，理论 16 cycle
//     addi   t1, t1, 1024    # 如果不能与向量 load 并行发射，就多 1 cycle
//
// 若 addi 不能与向量 load 重叠，每 4 KiB 的代价是 4×(16+1) = 68 而不是 64
// cycle，恰好给出 1024×4/68 = 60.2 B/cycle。
//
// RVV 的单位步长 load 没有立即数偏移，只能用基址寄存器。所以这里把整个
// 工作集的地址预先装进多个通用寄存器，循环体内 **一条地址运算都没有**，
// 只有向量 load 和最外层的一条分支。
//
// 指标直接用 cycles/load：m8 load 搬 1024 B，若 L1D 通路是 64 B/cycle，
// 应当收敛到 16.000。
#include "k3_a100.h"

#include <cmath>
#include <cstring>

#include <sys/mman.h>

namespace {

using Fn = void (*)(const uint8_t *, uint64_t);

// ---------------------------------------------------------------------------
// 变体 A：每条 load 跟一条依赖的 addi（k3_path_bandwidth 用的结构）
// 一趟 = 16 条 load = 16 KiB
__attribute__((noinline)) void a_addi_dep(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\t"
                 ".rept 4\n\t"
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

// 变体 B：4 个轮转指针，addi 与它所推进的 load 之间隔开，放松依赖。
__attribute__((noinline)) void b_addi_rotated(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "li t5, 4096\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\t"
                 "addi t2, t1, 1024\n\taddi t3, t2, 1024\n\taddi t4, t3, 1024\n\t"
                 ".rept 4\n\t"
                 "vle8.v v0, (t1)\n\tvle8.v v8, (t2)\n\t"
                 "vle8.v v16, (t3)\n\tvle8.v v24, (t4)\n\t"
                 "add t1, t1, t5\n\tadd t2, t2, t5\n\t"
                 "add t3, t3, t5\n\tadd t4, t4, t5\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "t2", "t3", "t4", "t5", "memory", "v0", "v8", "v16", "v24");
}

// ---------------------------------------------------------------------------
// 变体 C：循环体内零地址运算。16 个基址寄存器覆盖 16 KiB，整趟里把这 16 条
// load 重复 REPT 次，外层只有 addi + bnez 两条标量指令。
#define ZERO_ADDI_BODY                                                                             \
    "vle8.v v0, (a2)\n\tvle8.v v8, (a3)\n\tvle8.v v16, (a4)\n\tvle8.v v24, (a5)\n\t"                \
    "vle8.v v0, (a6)\n\tvle8.v v8, (a7)\n\tvle8.v v16, (t1)\n\tvle8.v v24, (t2)\n\t"                \
    "vle8.v v0, (t3)\n\tvle8.v v8, (t4)\n\tvle8.v v16, (t5)\n\tvle8.v v24, (t6)\n\t"                \
    "vle8.v v0, (s2)\n\tvle8.v v8, (s3)\n\tvle8.v v16, (s4)\n\tvle8.v v24, (s5)\n\t"

// addi 的立即数上限是 2047，所以基址寄存器用 1024 步长链式构造。
// 这段在外层循环之外，只执行一次，不计入稳态吞吐。
#define ZERO_ADDI_SETUP                                                                            \
    "vsetvli t0, zero, e8, m8, ta, ma\n\t"                                                          \
    "mv   a2, %0\n\taddi a3, a2, 1024\n\taddi a4, a3, 1024\n\taddi a5, a4, 1024\n\t"                \
    "addi a6, a5, 1024\n\taddi a7, a6, 1024\n\taddi t1, a7, 1024\n\taddi t2, t1, 1024\n\t"          \
    "addi t3, t2, 1024\n\taddi t4, t3, 1024\n\taddi t5, t4, 1024\n\taddi t6, t5, 1024\n\t"          \
    "addi s2, t6, 1024\n\taddi s3, s2, 1024\n\taddi s4, s3, 1024\n\taddi s5, s4, 1024\n\t"

#define ZERO_ADDI_CLOBBER                                                                          \
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3",       \
        "s4", "s5", "memory", "v0", "v8", "v16", "v24"

#define DEFINE_ZERO_ADDI(fname, REPT)                                                              \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                      \
        asm volatile(ZERO_ADDI_SETUP                                                                \
                     "2:\n\t"                                                                       \
                     ".rept " #REPT "\n\t" ZERO_ADDI_BODY ".endr\n\t"                                \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                           \
                     : "+r"(base), "+r"(rep)                                                        \
                     :                                                                              \
                     : ZERO_ADDI_CLOBBER);                                                          \
    }

DEFINE_ZERO_ADDI(c_zero_addi_r1, 1)    // 16 条 load / 趟
DEFINE_ZERO_ADDI(c_zero_addi_r4, 4)    // 64 条 load / 趟
DEFINE_ZERO_ADDI(c_zero_addi_r16, 16)  // 256 条 load / 趟
DEFINE_ZERO_ADDI(c_zero_addi_r64, 64)  // 1024 条 load / 趟

// 变体 D：零地址运算，但只用 4 个基址寄存器（工作集 4 KiB），检查工作集
// 大小对峰值有没有影响。
#define SMALL_BODY                                                                                 \
    "vle8.v v0, (a2)\n\tvle8.v v8, (a3)\n\tvle8.v v16, (a4)\n\tvle8.v v24, (a5)\n\t"

__attribute__((noinline)) void d_zero_addi_4k(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "mv a2, %0\n\taddi a3, a2, 1024\n\taddi a4, a3, 1024\n\taddi a5, a4, 1024\n\t"
                 "2:\n\t"
                 ".rept 64\n\t" SMALL_BODY ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "a2", "a3", "a4", "a5", "memory", "v0", "v8", "v16", "v24");
}

// 变体 E：零地址运算 + 每条 load 写不同的目标寄存器组（v0/v8/v16/v24 轮转
// 已经是 4 组，这里改成只用 1 组，检查目标寄存器写端口是否是瓶颈）。
__attribute__((noinline)) void e_single_dest(const uint8_t * base, uint64_t rep) {
    asm volatile(ZERO_ADDI_SETUP
                 "2:\n\t"
                 ".rept 16\n\t"
                 "vle8.v v0, (a2)\n\tvle8.v v0, (a3)\n\tvle8.v v0, (a4)\n\tvle8.v v0, (a5)\n\t"
                 "vle8.v v0, (a6)\n\tvle8.v v0, (a7)\n\tvle8.v v0, (t1)\n\tvle8.v v0, (t2)\n\t"
                 "vle8.v v0, (t3)\n\tvle8.v v0, (t4)\n\tvle8.v v0, (t5)\n\tvle8.v v0, (t6)\n\t"
                 "vle8.v v0, (s2)\n\tvle8.v v0, (s3)\n\tvle8.v v0, (s4)\n\tvle8.v v0, (s5)\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : ZERO_ADDI_CLOBBER);
}

// 变体 F：零地址运算，用 m4（512 B/条）和 m2（256 B/条），检查峰值是不是
// 只有 m8 才能达到。
__attribute__((noinline)) void f_m4(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m4, ta, ma\n\t"
                 "mv   a2, %0\n\taddi a3, a2, 512\n\taddi a4, a3, 512\n\taddi a5, a4, 512\n\t"
                 "addi a6, a5, 512\n\taddi a7, a6, 512\n\taddi t1, a7, 512\n\taddi t2, t1, 512\n\t"
                 "2:\n\t"
                 ".rept 32\n\t"
                 "vle8.v v0, (a2)\n\tvle8.v v4, (a3)\n\tvle8.v v8, (a4)\n\tvle8.v v12, (a5)\n\t"
                 "vle8.v v16, (a6)\n\tvle8.v v20, (a7)\n\tvle8.v v24, (t1)\n\tvle8.v v28, (t2)\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "t2", "a2", "a3", "a4", "a5", "a6", "a7", "memory", "v0", "v4",
                   "v8", "v12", "v16", "v20", "v24", "v28");
}

__attribute__((noinline)) void f_m2(const uint8_t * base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e8, m2, ta, ma\n\t"
                 "mv   a2, %0\n\taddi a3, a2, 256\n\taddi a4, a3, 256\n\taddi a5, a4, 256\n\t"
                 "addi a6, a5, 256\n\taddi a7, a6, 256\n\taddi t1, a7, 256\n\taddi t2, t1, 256\n\t"
                 "2:\n\t"
                 ".rept 32\n\t"
                 "vle8.v v0, (a2)\n\tvle8.v v2, (a3)\n\tvle8.v v4, (a4)\n\tvle8.v v6, (a5)\n\t"
                 "vle8.v v8, (a6)\n\tvle8.v v10, (a7)\n\tvle8.v v12, (t1)\n\tvle8.v v14, (t2)\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "t2", "a2", "a3", "a4", "a5", "a6", "a7", "memory", "v0", "v2",
                   "v4", "v6", "v8", "v10", "v12", "v14");
}

// ---------------------------------------------------------------------------
// 变体 G：把基址寄存器数量推到 24（24 KiB 工作集），检查「不同地址个数」
// 是不是还在限制速率。s0/s1 也拿来用。
#define G_SETUP                                                                                    \
    "vsetvli t0, zero, e8, m8, ta, ma\n\t"                                                          \
    "mv   a2, %0\n\t"                                                                               \
    "addi a3, a2, 1024\n\taddi a4, a3, 1024\n\taddi a5, a4, 1024\n\taddi a6, a5, 1024\n\t"          \
    "addi a7, a6, 1024\n\taddi t1, a7, 1024\n\taddi t2, t1, 1024\n\taddi t3, t2, 1024\n\t"          \
    "addi t4, t3, 1024\n\taddi t5, t4, 1024\n\taddi t6, t5, 1024\n\taddi s2, t6, 1024\n\t"          \
    "addi s3, s2, 1024\n\taddi s4, s3, 1024\n\taddi s5, s4, 1024\n\taddi s6, s5, 1024\n\t"          \
    "addi s7, s6, 1024\n\taddi s8, s7, 1024\n\taddi s9, s8, 1024\n\taddi s10, s9, 1024\n\t"         \
    "addi s11, s10, 1024\n\taddi s0, s11, 1024\n\taddi s1, s0, 1024\n\taddi a1, s1, 1024\n\t"

#define G_BODY                                                                                     \
    "vle8.v v0, (a2)\n\tvle8.v v8, (a3)\n\tvle8.v v16, (a4)\n\tvle8.v v24, (a5)\n\t"                \
    "vle8.v v0, (a6)\n\tvle8.v v8, (a7)\n\tvle8.v v16, (t1)\n\tvle8.v v24, (t2)\n\t"                \
    "vle8.v v0, (t3)\n\tvle8.v v8, (t4)\n\tvle8.v v16, (t5)\n\tvle8.v v24, (t6)\n\t"                \
    "vle8.v v0, (s2)\n\tvle8.v v8, (s3)\n\tvle8.v v16, (s4)\n\tvle8.v v24, (s5)\n\t"                \
    "vle8.v v0, (s6)\n\tvle8.v v8, (s7)\n\tvle8.v v16, (s8)\n\tvle8.v v24, (s9)\n\t"                \
    "vle8.v v0, (s10)\n\tvle8.v v8, (s11)\n\tvle8.v v16, (s0)\n\tvle8.v v24, (s1)\n\t"

#define G_CLOBBER                                                                                  \
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s0",       \
        "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "memory", "v0", "v8",   \
        "v16", "v24"

__attribute__((noinline)) void g_24addr(const uint8_t * base, uint64_t rep) {
    asm volatile(G_SETUP
                 "2:\n\t"
                 ".rept 16\n\t" G_BODY ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : G_CLOBBER);
}

// 变体 H：同样 24 个地址，但发射顺序打乱成 4 KiB 跨步，检查连续两条 load
// 落在相邻 cache set 上是否造成冲突。
__attribute__((noinline)) void h_24addr_strided(const uint8_t * base, uint64_t rep) {
    asm volatile(G_SETUP
                 "2:\n\t"
                 ".rept 16\n\t"
                 "vle8.v v0, (a2)\n\tvle8.v v8, (a6)\n\tvle8.v v16, (t3)\n\tvle8.v v24, (s2)\n\t"
                 "vle8.v v0, (s6)\n\tvle8.v v8, (s10)\n\tvle8.v v16, (a3)\n\tvle8.v v24, (a7)\n\t"
                 "vle8.v v0, (t4)\n\tvle8.v v8, (s3)\n\tvle8.v v16, (s7)\n\tvle8.v v24, (s11)\n\t"
                 "vle8.v v0, (a4)\n\tvle8.v v8, (t1)\n\tvle8.v v16, (t5)\n\tvle8.v v24, (s4)\n\t"
                 "vle8.v v0, (s8)\n\tvle8.v v8, (s0)\n\tvle8.v v16, (a5)\n\tvle8.v v24, (t2)\n\t"
                 "vle8.v v0, (t6)\n\tvle8.v v8, (s5)\n\tvle8.v v16, (s9)\n\tvle8.v v24, (s1)\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : G_CLOBBER);
}

// 变体 I：24 个地址，8 个不同的目标寄存器组（m4），检查目标寄存器组数量。
__attribute__((noinline)) void i_24addr_m4dest(const uint8_t * base, uint64_t rep) {
    asm volatile(G_SETUP
                 "2:\n\t"
                 ".rept 16\n\t"
                 "vle8.v v0, (a2)\n\tvle8.v v8, (a3)\n\tvle8.v v16, (a4)\n\tvle8.v v24, (a5)\n\t"
                 "vle8.v v0, (a6)\n\tvle8.v v8, (a7)\n\tvle8.v v16, (t1)\n\tvle8.v v24, (t2)\n\t"
                 "vle8.v v0, (t3)\n\tvle8.v v8, (t4)\n\tvle8.v v16, (t5)\n\tvle8.v v24, (t6)\n\t"
                 "vle8.v v0, (s2)\n\tvle8.v v8, (s3)\n\tvle8.v v16, (s4)\n\tvle8.v v24, (s5)\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : G_CLOBBER);
}

// ---------------------------------------------------------------------------
// LMUL 扫描：统一用 24 个基址寄存器、零地址运算、同样的展开结构，只改
// LMUL 和地址间距。用来判断「每条指令的固定开销」和「每个 cache line 的
// 边际代价」各是多少。
//
// VLEN=1024 时每条 load 读 LMUL × 128 B，即 LMUL × 2 个 64 B line。
#define LMUL_SETUP(VSET, STEP)                                                                     \
    VSET                                                                                            \
    "mv   a2, %0\n\t"                                                                               \
    "addi a3, a2, " STEP "\n\taddi a4, a3, " STEP "\n\taddi a5, a4, " STEP "\n\t"                   \
    "addi a6, a5, " STEP "\n\taddi a7, a6, " STEP "\n\taddi t1, a7, " STEP "\n\t"                   \
    "addi t2, t1, " STEP "\n\taddi t3, t2, " STEP "\n\taddi t4, t3, " STEP "\n\t"                   \
    "addi t5, t4, " STEP "\n\taddi t6, t5, " STEP "\n\taddi s2, t6, " STEP "\n\t"                   \
    "addi s3, s2, " STEP "\n\taddi s4, s3, " STEP "\n\taddi s5, s4, " STEP "\n\t"                   \
    "addi s6, s5, " STEP "\n\taddi s7, s6, " STEP "\n\taddi s8, s7, " STEP "\n\t"                   \
    "addi s9, s8, " STEP "\n\taddi s10, s9, " STEP "\n\taddi s11, s10, " STEP "\n\t"                \
    "addi s0, s11, " STEP "\n\taddi s1, s0, " STEP "\n\taddi a1, s1, " STEP "\n\t"

// 24 条 load，每条一个地址寄存器。目标寄存器数量按 LMUL 能提供的最大值：
// m1 有 24 个可用（v0-v23），m2 有 12，m4 有 8，m8 只有 4。小 LMUL 的 load
// 很快，如果目标寄存器数量不够，WAW 会把测量变成写回延迟而不是吞吐。
#define ADDR24                                                                                     \
    "a2", "a3", "a4", "a5", "a6", "a7", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3", "s4",       \
        "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s0", "s1"

#define L24(OP, D00, D01, D02, D03, D04, D05, D06, D07, D08, D09, D10, D11, D12, D13, D14, D15,     \
            D16, D17, D18, D19, D20, D21, D22, D23)                                                \
    OP " " D00 ", (a2)\n\t" OP " " D01 ", (a3)\n\t" OP " " D02 ", (a4)\n\t"                         \
    OP " " D03 ", (a5)\n\t" OP " " D04 ", (a6)\n\t" OP " " D05 ", (a7)\n\t"                         \
    OP " " D06 ", (t1)\n\t" OP " " D07 ", (t2)\n\t" OP " " D08 ", (t3)\n\t"                         \
    OP " " D09 ", (t4)\n\t" OP " " D10 ", (t5)\n\t" OP " " D11 ", (t6)\n\t"                         \
    OP " " D12 ", (s2)\n\t" OP " " D13 ", (s3)\n\t" OP " " D14 ", (s4)\n\t"                         \
    OP " " D15 ", (s5)\n\t" OP " " D16 ", (s6)\n\t" OP " " D17 ", (s7)\n\t"                         \
    OP " " D18 ", (s8)\n\t" OP " " D19 ", (s9)\n\t" OP " " D20 ", (s10)\n\t"                        \
    OP " " D21 ", (s11)\n\t" OP " " D22 ", (s0)\n\t" OP " " D23 ", (s1)\n\t"

// m1：24 个不同目标寄存器
#define DESTS_M1                                                                                   \
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",         \
        "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
// m2：12 个（v0,v2,...,v22），每个用两次
#define DESTS_M2                                                                                   \
    "v0", "v2", "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22", "v0", "v2",      \
        "v4", "v6", "v8", "v10", "v12", "v14", "v16", "v18", "v20", "v22"
// m4：8 个（v0,v4,...,v28），每个用三次
#define DESTS_M4                                                                                   \
    "v0", "v4", "v8", "v12", "v16", "v20", "v24", "v28", "v0", "v4", "v8", "v12", "v16", "v20",     \
        "v24", "v28", "v0", "v4", "v8", "v12", "v16", "v20", "v24", "v28"
// m8：只有 4 个（v0,v8,v16,v24），每个用六次
#define DESTS_M8                                                                                   \
    "v0", "v8", "v16", "v24", "v0", "v8", "v16", "v24", "v0", "v8", "v16", "v24", "v0", "v8",       \
        "v16", "v24", "v0", "v8", "v16", "v24", "v0", "v8", "v16", "v24"

#define DEFINE_LMUL(fname, VSET, STEP, OP, DESTS)                                                  \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                      \
        asm volatile(LMUL_SETUP(VSET, STEP) "2:\n\t"                                                \
                     ".rept 16\n\t" L24(OP, DESTS) ".endr\n\t"                                      \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                           \
                     : "+r"(base), "+r"(rep)                                                        \
                     :                                                                              \
                     : G_CLOBBER, "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11",     \
                       "v12", "v13", "v14", "v15", "v17", "v18", "v19", "v20", "v21", "v22",        \
                       "v23", "v28", "v29", "v30", "v31");                                          \
    }

// 更长的 pass，确认 17.00 cycle/load 是地板而不是残留开销。
#define DEFINE_LMUL_REPT(fname, VSET, STEP, OP, REPT, DESTS)                                        \
    __attribute__((noinline)) void fname(const uint8_t * base, uint64_t rep) {                      \
        asm volatile(LMUL_SETUP(VSET, STEP) "2:\n\t"                                                \
                     ".rept " #REPT "\n\t" L24(OP, DESTS) ".endr\n\t"                                \
                     "addi %1, %1, -1\n\tbnez %1, 2b\n\t"                                           \
                     : "+r"(base), "+r"(rep)                                                        \
                     :                                                                              \
                     : G_CLOBBER, "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v9", "v10", "v11",     \
                       "v12", "v13", "v14", "v15", "v17", "v18", "v19", "v20", "v21", "v22",        \
                       "v23", "v28", "v29", "v30", "v31");                                          \
    }

DEFINE_LMUL_REPT(l_m8_r64, "vsetvli t0, zero, e8, m8, ta, ma\n\t", "1024", "vl8re8.v", 64, DESTS_M8)
DEFINE_LMUL_REPT(l_m8_r128, "vsetvli t0, zero, e8, m8, ta, ma\n\t", "1024", "vl8re8.v", 128,
                 DESTS_M8)
// 元素宽度对单位步长 load 不应有影响，用 e64 交叉验证。
DEFINE_LMUL_REPT(l_m8_e64, "vsetvli t0, zero, e64, m8, ta, ma\n\t", "1024", "vle64.v", 64, DESTS_M8)

DEFINE_LMUL(l_m1, "vsetvli t0, zero, e8, m1, ta, ma\n\t", "128", "vle8.v", DESTS_M1)
DEFINE_LMUL(l_m2, "vsetvli t0, zero, e8, m2, ta, ma\n\t", "256", "vle8.v", DESTS_M2)
DEFINE_LMUL(l_m4, "vsetvli t0, zero, e8, m4, ta, ma\n\t", "512", "vle8.v", DESTS_M4)
DEFINE_LMUL(l_m8, "vsetvli t0, zero, e8, m8, ta, ma\n\t", "1024", "vle8.v", DESTS_M8)
// 整寄存器组 load，不依赖 vtype/vl
DEFINE_LMUL(l_vl1re8, "vsetvli t0, zero, e8, m1, ta, ma\n\t", "128", "vl1re8.v", DESTS_M1)
DEFINE_LMUL(l_vl4re8, "vsetvli t0, zero, e8, m4, ta, ma\n\t", "512", "vl4re8.v", DESTS_M4)
DEFINE_LMUL(l_vl8re8, "vsetvli t0, zero, e8, m8, ta, ma\n\t", "1024", "vl8re8.v", DESTS_M8)

struct Case {
    const char * name;
    Fn           fn;
    int          loads_per_pass;
    int          bytes_per_load;
    size_t       working_set;
    int          addi_per_load;  // 循环体内每条 load 平均伴随的标量指令数
};

const Case kCases[] = {
    {"A  addi/load, 依赖", a_addi_dep, 16, 1024, 16u << 10, 1},
    {"B  addi/load, 轮转指针", b_addi_rotated, 16, 1024, 16u << 10, 1},
    {"C  零 addi, 16 load/趟", c_zero_addi_r1, 16, 1024, 16u << 10, 0},
    {"C  零 addi, 64 load/趟", c_zero_addi_r4, 64, 1024, 16u << 10, 0},
    {"C  零 addi, 256 load/趟", c_zero_addi_r16, 256, 1024, 16u << 10, 0},
    {"C  零 addi, 1024 load/趟", c_zero_addi_r64, 1024, 1024, 16u << 10, 0},
    {"D  零 addi, 4 KiB 工作集", d_zero_addi_4k, 256, 1024, 4u << 10, 0},
    {"E  零 addi, 单目标寄存器", e_single_dest, 256, 1024, 16u << 10, 0},
    {"F  零 addi, m4 (512 B)", f_m4, 256, 512, 4u << 10, 0},
    {"F  零 addi, m2 (256 B)", f_m2, 256, 256, 2u << 10, 0},
    {"G  零 addi, 24 地址顺序", g_24addr, 384, 1024, 24u << 10, 0},
    {"H  零 addi, 24 地址跨步", h_24addr_strided, 384, 1024, 24u << 10, 0},
    {"I  零 addi, 16/24 地址子集", i_24addr_m4dest, 256, 1024, 24u << 10, 0},
    {"L  vle8.v m1 (128 B), 24 addr", l_m1, 384, 128, 3u << 10, 0},
    {"L  vle8.v m2 (256 B), 24 addr", l_m2, 384, 256, 6u << 10, 0},
    {"L  vle8.v m4 (512 B), 24 addr", l_m4, 384, 512, 12u << 10, 0},
    {"L  vle8.v m8 (1024 B), 24 addr", l_m8, 384, 1024, 24u << 10, 0},
    {"L  vl1re8.v (128 B), 24 addr", l_vl1re8, 384, 128, 3u << 10, 0},
    {"L  vl4re8.v (512 B), 24 addr", l_vl4re8, 384, 512, 12u << 10, 0},
    {"L  vl8re8.v (1024 B), 24 addr", l_vl8re8, 384, 1024, 24u << 10, 0},
    {"M  vl8re8.v, 1536 load/趟", l_m8_r64, 1536, 1024, 24u << 10, 0},
    {"M  vl8re8.v, 3072 load/趟", l_m8_r128, 3072, 1024, 24u << 10, 0},
    {"M  vle64.v m8, 1536 load/趟", l_m8_e64, 1536, 1024, 24u << 10, 0},
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

}  // namespace

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

    void *   raw     = nullptr;
    size_t   raw_len = 0;
    uint8_t * base   = alloc_thp(64u << 10, &raw, &raw_len);
    if (base == nullptr) {
        return 1;
    }

    std::printf("# cpu=%d vlenb=%llu samples=%d  官方 L1D 宣称 64 B/cycle\n", cpu,
                (unsigned long long) k3::vlenb(), samples);
    std::printf("%-28s %8s %10s %12s %10s %10s\n", "variant", "ws", "cyc/load", "cyc per 64B",
                "B/cycle", "% of 64");

    for (const Case & c : kCases) {
        const uint64_t bytes_per_pass = uint64_t(c.loads_per_pass) * uint64_t(c.bytes_per_load);
        const uint64_t rep = std::max<uint64_t>((512ull << 20) / bytes_per_pass, 4);

        c.fn(base, std::max<uint64_t>(rep / 8, 2));

        double   best_cycles = 1e300;
        uint64_t retired     = 0;
        for (int s = 0; s < samples; ++s) {
            if (!counter.start()) {
                return 1;
            }
            c.fn(base, rep);
            k3::CycleCounter::Reading r{};
            if (!counter.stop(r) || r.running_ratio < 0.99) {
                continue;
            }
            if (double(r.cycles) < best_cycles) {
                best_cycles = double(r.cycles);
                retired     = r.instructions;
            }
        }
        const double total_loads = double(rep) * double(c.loads_per_pass);
        const double cyc_per_load = best_cycles / total_loads;
        const double bpc = double(c.bytes_per_load) / cyc_per_load;
        std::printf("%-28s %6zuK %10.4f %12.4f %10.2f %9.1f%%   (retired/load %.3f)\n", c.name,
                    c.working_set >> 10, cyc_per_load, cyc_per_load * 64.0 / c.bytes_per_load, bpc,
                    100.0 * bpc / 64.0, double(retired) / total_loads);
        std::fflush(stdout);
    }
    munmap(raw, raw_len);
    return 0;
}
