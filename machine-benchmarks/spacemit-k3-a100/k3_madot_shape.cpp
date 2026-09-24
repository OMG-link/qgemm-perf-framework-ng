// 用全 1 输入测出 smt.vmadot 每条指令实际完成的 MAC 数量。
//
// 把累加器清零、两个源寄存器填成全 1（i8 时每字节 1，i4 时每 nibble 1），
// 执行一条 dot 指令，再把 M2 累加器写回内存。每个非零输出元素的值就等于
// 该输出点的归约深度 K，非零元素个数就是输出点数，两者相乘即每条指令的
// MAC 数。
#include "k3_a100.h"

#include <cstring>

namespace {

alignas(4096) int32_t g_out[256];

// 执行一条指令并回写累加器。fill 是填进 vs1/vs2 每个字节的值。
template <typename Emit>
void run_one(uint8_t fill, int32_t * out, Emit emit) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "vmv.v.i v0, 0\n\tvmv.v.i v8, 0\n\tvmv.v.i v16, 0\n\tvmv.v.i v24, 0\n\t"
                 "vsetvli t0, zero, e8, m1, tu, ma\n\t"
                 "vmv.v.x v28, %0\n\tvmv.v.x v29, %0\n\t"
                 "vmv.v.x v30, %0\n\tvmv.v.x v31, %0\n\t"
                 :
                 : "r"((uint64_t) fill)
                 : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                   "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                   "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    emit();
    asm volatile("vsetvli t0, zero, e32, m2, ta, ma\n\t"
                 "vse32.v v2, (%0)\n\t"
                 :
                 : "r"(out)
                 : "t0", "memory");
}

// 与 run_one 相同，但掩码寄存器 v0 置全 1。
template <typename Emit>
void run_one_masked(uint8_t fill, int32_t * out, Emit emit) {
    asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                 "vmv.v.i v0, 0\n\tvmv.v.i v8, 0\n\tvmv.v.i v16, 0\n\tvmv.v.i v24, 0\n\t"
                 "vsetvli t0, zero, e8, m1, tu, ma\n\t"
                 "vmv.v.i v0, -1\n\tvmv.v.i v1, -1\n\t"
                 "vmv.v.x v28, %0\n\tvmv.v.x v29, %0\n\t"
                 "vmv.v.x v30, %0\n\tvmv.v.x v31, %0\n\t"
                 :
                 : "r"((uint64_t) fill)
                 : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                   "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                   "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    emit();
    asm volatile("vsetvli t0, zero, e32, m2, ta, ma\n\t"
                 "vse32.v v2, (%0)\n\t"
                 :
                 : "r"(out)
                 : "t0", "memory");
}

void report(const char * name, int32_t * out, int count) {
    int     nonzero = 0;
    int32_t first   = 0;
    bool    uniform = true;
    for (int i = 0; i < count; ++i) {
        if (out[i] != 0) {
            if (nonzero == 0) {
                first = out[i];
            } else if (out[i] != first) {
                uniform = false;
            }
            ++nonzero;
        }
    }
    std::printf("%-24s outputs=%3d  value=%6d  uniform=%-3s  MAC/inst=%6d\n", name, nonzero, first,
                uniform ? "yes" : "no", nonzero * (first > 0 ? first : 0));
    std::printf("  first 16 int32: ");
    for (int i = 0; i < 16 && i < count; ++i) {
        std::printf("%d ", out[i]);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    const int lanes = int(k3::vlenb() * 2 / sizeof(int32_t));  // M2 组的 int32 元素数
    std::printf("# VLEN=%llu bits, M2 accumulator holds %d int32 lanes\n",
                (unsigned long long) k3::vlenb() * 8, lanes);

    // i8: 每字节填 1。
    for (int i = 0; i < lanes; ++i) {
        g_out[i] = -1;
    }
    run_one(0x01, g_out, [] { asm volatile("smt.vmadot v2, v28, v28, i8" ::: "v2", "v3"); });
    report("smt.vmadot i8", g_out, lanes);

    // i4: 每 nibble 填 1 -> 每字节 0x11。
    run_one(0x11, g_out, [] { asm volatile("smt.vmadot v2, v28, v28, i4" ::: "v2", "v3"); });
    report("smt.vmadot i4", g_out, lanes);

    run_one(0x01, g_out, [] { asm volatile("smt.vmadot1 v2, v28, v30, i8" ::: "v2", "v3"); });
    report("smt.vmadot1 i8", g_out, lanes);
    run_one(0x01, g_out, [] { asm volatile("smt.vmadot2 v2, v28, v30, i8" ::: "v2", "v3"); });
    report("smt.vmadot2 i8", g_out, lanes);
    run_one(0x01, g_out, [] { asm volatile("smt.vmadot3 v2, v28, v30, i8" ::: "v2", "v3"); });
    report("smt.vmadot3 i8", g_out, lanes);

    // .hp / .sp 带掩码，掩码全 1 时再看输出。
#define HP_CASE(IMM)                                                                               \
    run_one_masked(0x01, g_out,                                                                    \
                   [] { asm volatile("smt.vmadot.hp v2, v28, v30, v0, " #IMM ", i8" ::: "v2", "v3"); }); \
    report("smt.vmadot.hp i8 imm" #IMM, g_out, lanes);
    HP_CASE(0) HP_CASE(1) HP_CASE(2) HP_CASE(3)
    HP_CASE(4) HP_CASE(5) HP_CASE(6) HP_CASE(7)
#define SP_CASE(IMM)                                                                               \
    run_one_masked(0x01, g_out,                                                                    \
                   [] { asm volatile("smt.vmadot.sp v2, v28, v30, v0, " #IMM ", i8" ::: "v2", "v3"); }); \
    report("smt.vmadot.sp i8 imm" #IMM, g_out, lanes);
    SP_CASE(0) SP_CASE(1) SP_CASE(2) SP_CASE(3)

    // .hp 用 fp16 1.0 作为输入再看一次，判断它是不是 fp16 累加。
    for (int imm = 0; imm < 8; ++imm) {
        (void) imm;
    }
    run_one_masked(0x00, g_out, [] {
        asm volatile("li t1, 0x3c003c00\n\t"
                     "vsetvli t0, zero, e32, m1, tu, ma\n\t"
                     "vmv.v.x v28, t1\n\tvmv.v.x v29, t1\n\t"
                     "vmv.v.x v30, t1\n\tvmv.v.x v31, t1\n\t"
                     "vsetvli t0, zero, e8, m1, tu, ma\n\t"
                     "smt.vmadot.hp v2, v28, v30, v0, 0, i8\n\t" ::
                         : "t0", "t1", "v2", "v3", "v28", "v29", "v30", "v31");
    });
    std::printf("%-24s first 16 raw u16: ", "smt.vmadot.hp fp16 in");
    for (int i = 0; i < 8; ++i) {
        uint16_t lo = uint16_t(uint32_t(g_out[i]) & 0xffff);
        uint16_t hi = uint16_t(uint32_t(g_out[i]) >> 16);
        std::printf("%04x %04x ", lo, hi);
    }
    std::printf("\n");

    // fp16 widening dot：累加器是 fp32。
    run_one(0x00, g_out, [] {
        asm volatile("vsetvli t0, zero, e16, m1, tu, ma\n\t"
                     "li t1, 0x3c003c00\n\t"  // 两个 fp16 1.0
                     "vsetvli t0, zero, e32, m1, tu, ma\n\t"
                     "vmv.v.x v28, t1\n\tvmv.v.x v29, t1\n\t"
                     "vmv.v.x v30, t1\n\tvmv.v.x v31, t1\n\t"
                     "vsetvli t0, zero, e16, m1, tu, ma\n\t"
                     "smt.vfwmadot v2, v28, v30, fp16\n\t" ::
                         : "t0", "t1", "v2", "v3", "v28", "v29", "v30", "v31");
    });
    std::printf("%-24s first 16 fp32: ", "smt.vfwmadot fp16");
    for (int i = 0; i < 16; ++i) {
        float f;
        std::memcpy(&f, &g_out[i], 4);
        std::printf("%g ", f);
    }
    std::printf("\n");
    return 0;
}
