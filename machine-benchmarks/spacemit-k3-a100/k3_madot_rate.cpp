// smt.vmadot 系列吞吐的定向验证。
//
// 主表里 smt.vmadot 的吞吐落在 10/9 cycle/inst 这个非整数值上，单靠一个
// 寄存器分配方案无法判定它是硬件速率还是某种冲突。这里扫描独立累加器
// 个数（1/2/4/8/12）和不同的源寄存器分配，看速率是否收敛。
#include "k3_a100.h"

#include <cmath>

namespace {

using KernelFn = void (*)(uint64_t iters);

#define MADOT_KERNEL(fname, BLOCK, REPT)                                                           \
    __attribute__((noinline)) void fname(uint64_t iters) {                                         \
        asm volatile("li t0, 0x01010101\n\t"                                                       \
                     "vsetvli t1, zero, e32, m8, ta, ma\n\t"                                       \
                     "vmv.v.x v0, t0\n\tvmv.v.x v8, t0\n\t"                                        \
                     "vmv.v.x v16, t0\n\tvmv.v.x v24, t0\n\t"                                      \
                     "vsetvli t1, zero, e8, m1, tu, ma\n\t"                                        \
                     "1:\n\t"                                                                      \
                     ".rept " #REPT "\n\t" BLOCK ".endr\n\t"                                       \
                     "addi %0, %0, -1\n\t"                                                         \
                     "bnez %0, 1b\n\t"                                                             \
                     : "+r"(iters)                                                                 \
                     :                                                                             \
                     : "t0", "t1", "memory",                                                       \
                       "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",   \
                       "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",       \
                       "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");      \
    }

#define D1  "smt.vmadot v16, v2, v3\n\t"
#define D2  D1 "smt.vmadot v18, v4, v5\n\t"
#define D4  D2 "smt.vmadot v20, v6, v7\n\tsmt.vmadot v22, v8, v9\n\t"
#define D8  D4 "smt.vmadot v24, v2, v5\n\tsmt.vmadot v26, v4, v7\n\t" \
               "smt.vmadot v28, v6, v9\n\tsmt.vmadot v30, v8, v3\n\t"
#define D12 D8 "smt.vmadot v10, v2, v7\n\tsmt.vmadot v12, v4, v9\n\t" \
               "smt.vmadot v14, v6, v3\n\tsmt.vmadot v0, v8, v5\n\t"

// 所有源寄存器相同、只有累加器不同的变体，用来分离源读端口的影响。
#define S8 "smt.vmadot v16, v2, v3\n\tsmt.vmadot v18, v2, v3\n\t" \
           "smt.vmadot v20, v2, v3\n\tsmt.vmadot v22, v2, v3\n\t" \
           "smt.vmadot v24, v2, v3\n\tsmt.vmadot v26, v2, v3\n\t" \
           "smt.vmadot v28, v2, v3\n\tsmt.vmadot v30, v2, v3\n\t"

MADOT_KERNEL(k_d1_r8,   D1,  8)
MADOT_KERNEL(k_d1_r32,  D1,  32)
MADOT_KERNEL(k_d2_r8,   D2,  8)
MADOT_KERNEL(k_d2_r32,  D2,  32)
MADOT_KERNEL(k_d4_r8,   D4,  8)
MADOT_KERNEL(k_d4_r32,  D4,  32)
MADOT_KERNEL(k_d8_r4,   D8,  4)
MADOT_KERNEL(k_d8_r16,  D8,  16)
MADOT_KERNEL(k_d8_r64,  D8,  64)
MADOT_KERNEL(k_d12_r4,  D12, 4)
MADOT_KERNEL(k_d12_r16, D12, 16)
MADOT_KERNEL(k_s8_r4,   S8,  4)
MADOT_KERNEL(k_s8_r16,  S8,  16)

// 混入独立的 vadd.vi，测 matrix 单元与普通 VALU 是否能并行发射。
#define MIX "smt.vmadot v16, v2, v3\n\tvadd.vi v10, v11, 1\n\t" \
            "smt.vmadot v18, v4, v5\n\tvadd.vi v12, v13, 1\n\t" \
            "smt.vmadot v20, v6, v7\n\tvadd.vi v14, v15, 1\n\t" \
            "smt.vmadot v22, v8, v9\n\tvadd.vi v1, v0, 1\n\t"
MADOT_KERNEL(k_mix_r16, MIX, 16)

struct Case {
    const char * name;
    KernelFn     fn;
    int          insts_per_iter;  // 只数 smt.vmadot
};

const Case kCases[] = {
    {"1 acc,  8 rept", k_d1_r8, 1 * 8},
    {"1 acc, 32 rept", k_d1_r32, 1 * 32},
    {"2 acc,  8 rept", k_d2_r8, 2 * 8},
    {"2 acc, 32 rept", k_d2_r32, 2 * 32},
    {"4 acc,  8 rept", k_d4_r8, 4 * 8},
    {"4 acc, 32 rept", k_d4_r32, 4 * 32},
    {"8 acc,  4 rept", k_d8_r4, 8 * 4},
    {"8 acc, 16 rept", k_d8_r16, 8 * 16},
    {"8 acc, 64 rept", k_d8_r64, 8 * 64},
    {"12 acc,  4 rept", k_d12_r4, 12 * 4},
    {"12 acc, 16 rept", k_d12_r16, 12 * 16},
    {"8 acc shared src,  4 rept", k_s8_r4, 8 * 4},
    {"8 acc shared src, 16 rept", k_s8_r16, 8 * 16},
    {"4 acc + 4 vadd.vi, 16 rept", k_mix_r16, 4 * 16},
};

}  // namespace

int main(int argc, char ** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }
    std::printf("# smt.vmadot i8, e8/m1, VLEN=%llu bits, cpu=%d\n", (unsigned long long) k3::vlenb() * 8, cpu);
    std::printf("%-30s %12s %12s %14s\n", "variant", "cyc/vmadot", "vmadot/cyc", "retired/inst");
    for (const Case & c : kCases) {
        const uint64_t iters = 400'000'000ULL / uint64_t(c.insts_per_iter) / 2;
        c.fn(iters / 32 + 1);
        double   best     = 1e18;
        uint64_t best_ret = 0;
        for (int s = 0; s < 5; ++s) {
            counter.start();
            c.fn(iters);
            k3::CycleCounter::Reading r{};
            counter.stop(r);
            double cpi = double(r.cycles) / (double(iters) * c.insts_per_iter);
            if (cpi < best) {
                best     = cpi;
                best_ret = r.instructions;
            }
        }
        std::printf("%-30s %12.4f %12.4f %14.4f\n", c.name, best, 1.0 / best,
                    double(best_ret) / (double(iters) * c.insts_per_iter));
        std::fflush(stdout);
    }
    return 0;
}
