// SpacemiT K3 / A100 指令吞吐微基准。
//
// 每个用例是一段手写汇编循环：循环体里放若干条互相无依赖的目标指令，
// 再用 .rept 展开到足够长，使 addi/bnez 的前端开销可以忽略。cycles 由
// perf_event_open 采集（K3 用户态读不了 cycle CSR），吞吐定义为
// cycles / 目标指令条数，不是依赖延迟。
//
// 每个用例都用两个不同的展开长度各测一遍。只有两者收敛到同一个速率时，
// 结果才被认为是稳定的硬件吞吐。
#include "k3_a100.h"

#include <csetjmp>
#include <csignal>
#include <cmath>

namespace {

// 汇编器接受但硬件不实现的编码会触发 SIGILL；用 sigsetjmp 把这种用例
// 标记为 unsupported 而不是让整个程序挂掉。
sigjmp_buf g_illegal_jmp;
volatile sig_atomic_t g_expect_illegal = 0;

void sigill_handler(int) {
    if (g_expect_illegal) {
        siglongjmp(g_illegal_jmp, 1);
    }
    _exit(132);
}

// L1 驻留的数据缓冲区，供 load/store 用例使用。
alignas(4096) uint8_t g_buffer[16384];

struct Result {
    double cycles_per_inst = 0.0;
    double insts_per_cycle = 0.0;
    double running_ratio   = 0.0;
    uint64_t retired       = 0;
};

using KernelFn = void (*)(uint64_t iters, void * buf);

struct Case {
    const char * mnemonic;
    const char * config;
    KernelFn     kernel;
    int          insts_per_iter_a;
    KernelFn     kernel_b;
    int          insts_per_iter_b;
};

// 一次测量：运行 kernel，返回 cycles/inst。
Result measure(k3::CycleCounter & counter, KernelFn kernel, uint64_t iters, int insts_per_iter) {
    // 预热，把数据和指令都带进 cache。
    kernel(std::max<uint64_t>(iters / 64, 1), g_buffer);

    Result res{};
    if (!counter.start()) {
        std::fprintf(stderr, "counter start failed\n");
        return res;
    }
    kernel(iters, g_buffer);
    k3::CycleCounter::Reading r{};
    if (!counter.stop(r)) {
        std::fprintf(stderr, "counter stop failed\n");
        return res;
    }
    const double total = double(iters) * double(insts_per_iter);
    res.cycles_per_inst = double(r.cycles) / total;
    res.insts_per_cycle = total / double(r.cycles);
    res.running_ratio   = r.running_ratio;
    res.retired         = r.instructions;
    return res;
}

// 循环体：BLOCK 内含 NBLOCK 条互不依赖的目标指令，再 .rept REPT 次。
#define DEFINE_KERNEL(fname, SETUP, BLOCK, REPT)                                                   \
    __attribute__((noinline)) void fname(uint64_t iters, void * buf) {                             \
        register void * addr asm("a1") = buf;                                                      \
        asm volatile(INIT_VREGS SETUP                                                              \
                     "1:\n\t"                                                                      \
                     ".rept " #REPT "\n\t" BLOCK ".endr\n\t"                                       \
                     "addi %0, %0, -1\n\t"                                                         \
                     "bnez %0, 1b\n\t"                                                             \
                     : "+r"(iters)                                                                 \
                     : "r"(addr)                                                                   \
                     : "t0", "t1", "t2", "a2", "a3", "a4", "a5", "a6", "a7", "fa0", "memory",                                                             \
                       "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",   \
                       "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",       \
                       "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");      \
    }

// 用一个宏同时生成两个展开长度的版本。
#define DEFINE_CASE(id, SETUP, BLOCK, NBLOCK)                                                      \
    DEFINE_KERNEL(id##_a, SETUP, BLOCK, 4)                                                         \
    DEFINE_KERNEL(id##_b, SETUP, BLOCK, 16)                                                        \
    constexpr int id##_n = (NBLOCK);

// 把所有向量寄存器和 fa0 初始化为 1.0f 的位模式，避免 NaN/非规格化数
// 影响浮点用例的吞吐，同时给掩码用例一个固定的非平凡掩码。
#define INIT_VREGS                                                                                 \
    "li t0, 0x3f800000\n\t"                                                                        \
    "fmv.w.x fa0, t0\n\t"                                                                          \
    "vsetvli t1, zero, e32, m8, ta, ma\n\t"                                                        \
    "vmv.v.x v0, t0\n\t"                                                                           \
    "vmv.v.x v8, t0\n\t"                                                                           \
    "vmv.v.x v16, t0\n\t"                                                                           \
    "vmv.v.x v24, t0\n\t"

// ---------------------------------------------------------------- vtype setup
#define VSET_E8M1   "vsetvli t0, zero, e8, m1, tu, ma\n\t"
#define VSET_E8M2   "vsetvli t0, zero, e8, m2, tu, ma\n\t"
#define VSET_E8M8   "vsetvli t0, zero, e8, m8, tu, ma\n\t"
#define VSET_E16M1  "vsetvli t0, zero, e16, m1, tu, ma\n\t"
#define VSET_E32M1  "vsetvli t0, zero, e32, m1, tu, ma\n\t"
#define VSET_E32M2  "vsetvli t0, zero, e32, m2, tu, ma\n\t"

// ------------------------------------------------------------- 标准 RVV 用例
DEFINE_CASE(vadd_vi, VSET_E8M1,
            "vadd.vi v8, v16, 1\n\tvadd.vi v9, v17, 1\n\tvadd.vi v10, v18, 1\n\t"
            "vadd.vi v11, v19, 1\n\tvadd.vi v12, v20, 1\n\tvadd.vi v13, v21, 1\n\t"
            "vadd.vi v14, v22, 1\n\tvadd.vi v15, v23, 1\n\t", 8)

DEFINE_CASE(vand_vi, VSET_E8M1,
            "vand.vi v8, v16, 1\n\tvand.vi v9, v17, 1\n\tvand.vi v10, v18, 1\n\t"
            "vand.vi v11, v19, 1\n\tvand.vi v12, v20, 1\n\tvand.vi v13, v21, 1\n\t"
            "vand.vi v14, v22, 1\n\tvand.vi v15, v23, 1\n\t", 8)

DEFINE_CASE(vsrl_vi, VSET_E8M1,
            "vsrl.vi v8, v16, 1\n\tvsrl.vi v9, v17, 1\n\tvsrl.vi v10, v18, 1\n\t"
            "vsrl.vi v11, v19, 1\n\tvsrl.vi v12, v20, 1\n\tvsrl.vi v13, v21, 1\n\t"
            "vsrl.vi v14, v22, 1\n\tvsrl.vi v15, v23, 1\n\t", 8)

DEFINE_CASE(vadd_vv_m8, VSET_E8M8,
            "vadd.vv v8, v16, v24\n\tvadd.vv v0, v16, v24\n\t", 2)

DEFINE_CASE(vfmacc_vv_m2, VSET_E32M2,
            "vfmacc.vv v8, v16, v18\n\tvfmacc.vv v10, v20, v22\n\t"
            "vfmacc.vv v12, v24, v26\n\tvfmacc.vv v14, v28, v30\n\t", 4)

DEFINE_CASE(vfmacc_vv_m1, VSET_E32M1,
            "vfmacc.vv v8, v16, v17\n\tvfmacc.vv v9, v18, v19\n\t"
            "vfmacc.vv v10, v20, v21\n\tvfmacc.vv v11, v22, v23\n\t"
            "vfmacc.vv v12, v24, v25\n\tvfmacc.vv v13, v26, v27\n\t"
            "vfmacc.vv v14, v28, v29\n\tvfmacc.vv v15, v30, v31\n\t", 8)

DEFINE_CASE(vfmul_vf_m2, VSET_E32M2,
            "vfmul.vf v8, v16, fa0\n\tvfmul.vf v10, v18, fa0\n\t"
            "vfmul.vf v12, v20, fa0\n\tvfmul.vf v14, v22, fa0\n\t", 4)

DEFINE_CASE(vfcvt_f_x_v_m2, VSET_E32M2,
            "vfcvt.f.x.v v8, v16\n\tvfcvt.f.x.v v10, v18\n\t"
            "vfcvt.f.x.v v12, v20\n\tvfcvt.f.x.v v14, v22\n\t", 4)

DEFINE_CASE(vfwcvt_f_f_v, VSET_E16M1,
            "vfwcvt.f.f.v v8, v16\n\tvfwcvt.f.f.v v10, v17\n\t"
            "vfwcvt.f.f.v v12, v18\n\tvfwcvt.f.f.v v14, v19\n\t", 4)

// 访存用例给每条指令一个独立的地址，避免同地址串行化。a2 是 a1 的别名，
// 用来给 8 条 128 B 访问铺开 1 KiB 的地址空间（仍然完全驻留 L1D）。
#define ADDR8(OP, SUF)                                                                             \
    OP " v8, " SUF "(a1)\n\t" OP " v9, " SUF "(a2)\n\t" OP " v10, " SUF "(a3)\n\t"                  \
    OP " v11, " SUF "(a4)\n\t" OP " v12, " SUF "(a5)\n\t" OP " v13, " SUF "(a6)\n\t"                \
    OP " v14, " SUF "(a7)\n\t" OP " v15, " SUF "(t2)\n\t"

// 建立 8 个相隔 128 B 的地址寄存器。
#define INIT_ADDRS                                                                                 \
    "addi a2, a1, 128\n\taddi a3, a1, 256\n\taddi a4, a1, 384\n\t"                                 \
    "addi a5, a1, 512\n\taddi a6, a1, 640\n\taddi a7, a1, 768\n\taddi t2, a1, 896\n\t"

DEFINE_CASE(vle8_v_m1, INIT_ADDRS VSET_E8M1, ADDR8("vle8.v", ""), 8)

DEFINE_CASE(vle8_v_m8, INIT_ADDRS VSET_E8M8,
            "vle8.v v8, (a1)\n\tvle8.v v16, (a5)\n\t", 2)

DEFINE_CASE(vl1re16_v, INIT_ADDRS VSET_E8M1, ADDR8("vl1re16.v", ""), 8)

DEFINE_CASE(vl2re32_v, INIT_ADDRS VSET_E8M1,
            "vl2re32.v v8, (a1)\n\tvl2re32.v v10, (a3)\n\tvl2re32.v v12, (a5)\n\t"
            "vl2re32.v v14, (a7)\n\t", 4)

DEFINE_CASE(vl8re8_v, INIT_ADDRS VSET_E8M1,
            "vl8re8.v v8, (a1)\n\tvl8re8.v v16, (a5)\n\t", 2)

DEFINE_CASE(vmv2r_v, VSET_E8M2,
            "vmv2r.v v8, v16\n\tvmv2r.v v10, v18\n\tvmv2r.v v12, v20\n\tvmv2r.v v14, v22\n\t", 4)

DEFINE_CASE(vse32_v_masked, INIT_ADDRS VSET_E32M1,
            "vse32.v v8, (a1), v0.t\n\tvse32.v v9, (a3), v0.t\n\t"
            "vse32.v v10, (a5), v0.t\n\tvse32.v v11, (a7), v0.t\n\t", 4)

DEFINE_CASE(vse8_v_m1, INIT_ADDRS VSET_E8M1,
            "vse8.v v8, (a1)\n\tvse8.v v9, (a3)\n\tvse8.v v10, (a5)\n\tvse8.v v11, (a7)\n\t", 4)

// --------------------------------------------------- XSMTVDotII 整数 dot 用例
// vd 必须是偶数寄存器（M2 累加器组），且不能与源寄存器重叠。
//
// smt.vmadot 的依赖延迟是 8 个 cycle（见 k3_madot_rate），所以循环体里必须
// 放足够多的独立累加器才能测到吞吐而不是延迟。这里统一用 13 个累加器
// v2..v26，源寄存器固定为 v28/v30（偶数，slide 与 fp 形式也接受），
// 掩码留给 v0。
#define MADOT_DESTS(OP, TAIL)                                                                      \
    OP " v2, " TAIL "\n\t" OP " v4, " TAIL "\n\t" OP " v6, " TAIL "\n\t"                            \
    OP " v8, " TAIL "\n\t" OP " v10, " TAIL "\n\t" OP " v12, " TAIL "\n\t"                          \
    OP " v14, " TAIL "\n\t" OP " v16, " TAIL "\n\t" OP " v18, " TAIL "\n\t"                         \
    OP " v20, " TAIL "\n\t" OP " v22, " TAIL "\n\t" OP " v24, " TAIL "\n\t"                         \
    OP " v26, " TAIL "\n\t"

#define MADOT_BLOCK(OP, TY) MADOT_DESTS(OP, "v28, v30, " TY)

DEFINE_CASE(vmadot_i8,    VSET_E8M1, MADOT_BLOCK("smt.vmadot", "i8"), 13)
DEFINE_CASE(vmadot_i4,    VSET_E8M1, MADOT_BLOCK("smt.vmadot", "i4"), 13)
DEFINE_CASE(vmadotu_i8,   VSET_E8M1, MADOT_BLOCK("smt.vmadotu", "i8"), 13)
DEFINE_CASE(vmadotu_i4,   VSET_E8M1, MADOT_BLOCK("smt.vmadotu", "i4"), 13)
DEFINE_CASE(vmadotsu_i8,  VSET_E8M1, MADOT_BLOCK("smt.vmadotsu", "i8"), 13)
DEFINE_CASE(vmadotsu_i4,  VSET_E8M1, MADOT_BLOCK("smt.vmadotsu", "i4"), 13)
DEFINE_CASE(vmadotus_i8,  VSET_E8M1, MADOT_BLOCK("smt.vmadotus", "i8"), 13)
DEFINE_CASE(vmadotus_i4,  VSET_E8M1, MADOT_BLOCK("smt.vmadotus", "i4"), 13)

// slide 形式（vmadot1/2/3）只支持 i8，且 vs1/vs2 也必须是偶数寄存器。
#define MADOT_SLIDE_BLOCK(OP) MADOT_DESTS(OP, "v28, v30")

DEFINE_CASE(vmadot1_i8, VSET_E8M1, MADOT_SLIDE_BLOCK("smt.vmadot1"), 13)
DEFINE_CASE(vmadot2_i8, VSET_E8M1, MADOT_SLIDE_BLOCK("smt.vmadot2"), 13)
DEFINE_CASE(vmadot3_i8, VSET_E8M1, MADOT_SLIDE_BLOCK("smt.vmadot3"), 13)

// .hp / .sp 形式带 vmask（只能是 v0 或 v1）和一个立即数，源寄存器不能是掩码寄存器。
#define MADOT_MASKED_BLOCK(OP, IMM, TY) MADOT_DESTS(OP, "v28, v30, v0, " IMM ", " TY)

DEFINE_CASE(vmadot_hp_i8, VSET_E8M1, MADOT_MASKED_BLOCK("smt.vmadot.hp", "2", "i8"), 13)
DEFINE_CASE(vmadot_hp_i4, VSET_E8M1, MADOT_MASKED_BLOCK("smt.vmadot.hp", "2", "i4"), 13)
DEFINE_CASE(vmadot_sp_i8, VSET_E8M1, MADOT_MASKED_BLOCK("smt.vmadot.sp", "1", "i8"), 13)
DEFINE_CASE(vmadot_sp_i4, VSET_E8M1, MADOT_MASKED_BLOCK("smt.vmadot.sp", "1", "i4"), 13)

// ------------------------------------------- XSMTVDotII 浮点 widening dot 用例
// vd/vs1/vs2 都必须是偶数寄存器且互不重叠。
#define FWMADOT_BLOCK(OP, TY) MADOT_DESTS(OP, "v28, v30, " TY)

DEFINE_CASE(vfwmadot_fp16,   VSET_E16M1, FWMADOT_BLOCK("smt.vfwmadot", "fp16"), 13)
DEFINE_CASE(vfwmadot_bfp16,  VSET_E16M1, FWMADOT_BLOCK("smt.vfwmadot", "bfp16"), 13)
DEFINE_CASE(vfwmadot1_fp16,  VSET_E16M1, FWMADOT_BLOCK("smt.vfwmadot1", "fp16"), 13)
DEFINE_CASE(vfwmadot2_fp16,  VSET_E16M1, FWMADOT_BLOCK("smt.vfwmadot2", "fp16"), 13)
DEFINE_CASE(vfwmadot3_fp16,  VSET_E16M1, FWMADOT_BLOCK("smt.vfwmadot3", "fp16"), 13)

// ------------------------------------------------ XSMTVDotII pack/unpack 用例
#define PACK_BLOCK(OP) MADOT_DESTS(OP, "v28, v30, 1")

DEFINE_CASE(vpack_vv, VSET_E8M1, PACK_BLOCK("smt.vpack.vv"), 13)
DEFINE_CASE(vupack_vv, VSET_E8M1, PACK_BLOCK("smt.vupack.vv"), 13)
DEFINE_CASE(vnpack_vv, VSET_E8M1, PACK_BLOCK("smt.vnpack.vv"), 13)
DEFINE_CASE(vnpack4_vv, VSET_E8M1, PACK_BLOCK("smt.vnpack4.vv"), 13)
DEFINE_CASE(vnspack_vv, VSET_E8M1, PACK_BLOCK("smt.vnspack.vv"), 13)
DEFINE_CASE(vnspack4_vv, VSET_E8M1, PACK_BLOCK("smt.vnspack4.vv"), 13)

#define CASE(id, mnemonic, config) \
    Case { mnemonic, config, id##_a, id##_n * 4, id##_b, id##_n * 16 }

const Case kCases[] = {
    CASE(vadd_vi, "vadd.vi", "e8,m1"),
    CASE(vand_vi, "vand.vi", "e8,m1"),
    CASE(vsrl_vi, "vsrl.vi", "e8,m1"),
    CASE(vadd_vv_m8, "vadd.vv", "e8,m8"),
    CASE(vfmacc_vv_m1, "vfmacc.vv", "e32,m1"),
    CASE(vfmacc_vv_m2, "vfmacc.vv", "e32,m2"),
    CASE(vfmul_vf_m2, "vfmul.vf", "e32,m2"),
    CASE(vfcvt_f_x_v_m2, "vfcvt.f.x.v", "e32,m2"),
    CASE(vfwcvt_f_f_v, "vfwcvt.f.f.v", "e16,m1 to e32,m2"),
    CASE(vle8_v_m1, "vle8.v", "e8,m1"),
    CASE(vle8_v_m8, "vle8.v", "e8,m8"),
    CASE(vl1re16_v, "vl1re16.v", "one 1024-bit register"),
    CASE(vl2re32_v, "vl2re32.v", "two registers / 2048 bits"),
    CASE(vl8re8_v, "vl8re8.v", "eight registers / 8192 bits"),
    CASE(vmv2r_v, "vmv2r.v", "two registers / 2048 bits"),
    CASE(vse8_v_m1, "vse8.v", "e8,m1"),
    CASE(vse32_v_masked, "vse32.v, masked", "e32,m1"),
    CASE(vmadot_i8, "smt.vmadot", "e8,m1; i8; M2 accumulator"),
    CASE(vmadot_i4, "smt.vmadot", "e8,m1; i4; M2 accumulator"),
    CASE(vmadotu_i8, "smt.vmadotu", "e8,m1; i8; M2 accumulator"),
    CASE(vmadotu_i4, "smt.vmadotu", "e8,m1; i4; M2 accumulator"),
    CASE(vmadotsu_i8, "smt.vmadotsu", "e8,m1; i8; M2 accumulator"),
    CASE(vmadotsu_i4, "smt.vmadotsu", "e8,m1; i4; M2 accumulator"),
    CASE(vmadotus_i8, "smt.vmadotus", "e8,m1; i8; M2 accumulator"),
    CASE(vmadotus_i4, "smt.vmadotus", "e8,m1; i4; M2 accumulator"),
    CASE(vmadot1_i8, "smt.vmadot1", "e8,m1; i8; M2 accumulator"),
    CASE(vmadot2_i8, "smt.vmadot2", "e8,m1; i8; M2 accumulator"),
    CASE(vmadot3_i8, "smt.vmadot3", "e8,m1; i8; M2 accumulator"),
    CASE(vmadot_hp_i8, "smt.vmadot.hp", "e8,m1; i8; imm 2; mask v0"),
    CASE(vmadot_hp_i4, "smt.vmadot.hp", "e8,m1; i4; imm 2; mask v0"),
    CASE(vmadot_sp_i8, "smt.vmadot.sp", "e8,m1; i8; imm 1; mask v0"),
    CASE(vmadot_sp_i4, "smt.vmadot.sp", "e8,m1; i4; imm 1; mask v0"),
    CASE(vfwmadot_fp16, "smt.vfwmadot", "e16,m1; fp16"),
    CASE(vfwmadot_bfp16, "smt.vfwmadot", "e16,m1; bfp16"),
    CASE(vfwmadot1_fp16, "smt.vfwmadot1", "e16,m1; fp16"),
    CASE(vfwmadot2_fp16, "smt.vfwmadot2", "e16,m1; fp16"),
    CASE(vfwmadot3_fp16, "smt.vfwmadot3", "e16,m1; fp16"),
    CASE(vpack_vv, "smt.vpack.vv", "e8,m1; imm 1"),
    CASE(vupack_vv, "smt.vupack.vv", "e8,m1; imm 1"),
    CASE(vnpack_vv, "smt.vnpack.vv", "e8,m1; imm 1"),
    CASE(vnpack4_vv, "smt.vnpack4.vv", "e8,m1; imm 1"),
    CASE(vnspack_vv, "smt.vnspack.vv", "e8,m1; imm 1"),
    CASE(vnspack4_vv, "smt.vnspack4.vv", "e8,m1; imm 1"),
};

}  // namespace

int main(int argc, char ** argv) {
    int      cpu     = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    int      samples = (argc > 2) ? std::atoi(argv[2]) : 5;
    uint64_t target        = 20'000'000ULL;   // 标定用的指令条数基准
    uint64_t budget_cycles = 400'000'000ULL;  // 每次正式测量的 cycle 预算

    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }

    for (size_t i = 0; i < sizeof(g_buffer); ++i) {
        g_buffer[i] = uint8_t(i * 31 + 7);
    }

    std::printf("# cpu=%d vlenb=%llu samples=%d\n", cpu, (unsigned long long) k3::vlenb(), samples);
    std::printf("%-18s %-34s %10s %10s %10s %8s\n", "mnemonic", "config", "cyc/inst(A)", "cyc/inst(B)",
                "inst/cyc", "spread");

    struct sigaction sa{};
    sa.sa_handler = sigill_handler;
    sa.sa_flags   = SA_NODEFER;
    sigaction(SIGILL, &sa, nullptr);

    for (const Case & c : kCases) {
        // 先用极短的一次调用探测该编码是否被硬件实现。
        g_expect_illegal = 1;
        if (sigsetjmp(g_illegal_jmp, 1) != 0) {
            g_expect_illegal = 0;
            std::printf("%-18s %-34s %10s %10s %10s %8s\n", c.mnemonic, c.config, "-", "-", "-",
                        "SIGILL");
            std::fflush(stdout);
            continue;
        }
        c.kernel(1, g_buffer);
        g_expect_illegal = 0;

        // 标定一次，按固定的 cycle 预算选择迭代次数，让慢指令不会拖长整轮测试。
        auto pick_iters = [&](KernelFn k, int per_iter) {
            uint64_t probe = std::max<uint64_t>(target / 64 / uint64_t(per_iter), 1);
            Result   r     = measure(counter, k, probe, per_iter);
            double   cpi   = r.cycles_per_inst > 0 ? r.cycles_per_inst : 1.0;
            double   want  = double(budget_cycles) / (cpi * double(per_iter));
            return std::max<uint64_t>(uint64_t(want), 1);
        };
        const uint64_t iters_a = pick_iters(c.kernel, c.insts_per_iter_a);
        const uint64_t iters_b = pick_iters(c.kernel_b, c.insts_per_iter_b);

        std::vector<double> a, b;
        double              ratio_min = 1.0;
        for (int s = 0; s < samples; ++s) {
            Result ra = measure(counter, c.kernel, iters_a, c.insts_per_iter_a);
            Result rb = measure(counter, c.kernel_b, iters_b, c.insts_per_iter_b);
            a.push_back(ra.cycles_per_inst);
            b.push_back(rb.cycles_per_inst);
            ratio_min = std::min({ratio_min, ra.running_ratio, rb.running_ratio});
        }
        // 取最小值：吞吐测量中任何干扰只会让 cycles 变多。
        double ma     = k3::minimum(a);
        double mb     = k3::minimum(b);
        double spread = std::fabs(ma - mb) / std::max(ma, mb);
        std::printf("%-18s %-34s %10.4f %10.4f %10.4f %7.2f%%%s\n", c.mnemonic, c.config, ma, mb, 1.0 / mb,
                    100.0 * spread, ratio_min < 0.99 ? "  [pmu-multiplexed]" : "");
        std::fflush(stdout);
    }
    return 0;
}
