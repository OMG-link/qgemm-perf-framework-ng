// 确定 A100 上 smt.vmadot i8 三个操作数的元素排布。
//
// 已知形状是 A(8x16) x B(16x8) = C(8x8)（见 k3_madot_shape）。本程序用
// one-hot 输入把三个排布逐位测出来，并检验下面这组假设：
//
//   vs1(A)  : byte index = m * 16 + k      （每行 16 个 K 连续）
//   vs2(B)  : byte index = n * 16 + k      （每列 16 个 K 连续）
//   vd(C)   : int32 lane  = m * 8  + n      （行主序）
//
// 最后再跑一遍随机小矩阵与标量参考结果对比。
#include "k3_a100.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>
#include <smt_vector.h>

namespace {

constexpr int kM = 8;
constexpr int kN = 8;
constexpr int kK = 16;
constexpr int kRegBytes = 128;  // VLEN = 1024 bit
constexpr int kLanes = 64;      // M2 累加器的 int32 元素数

alignas(128) int8_t g_a[kRegBytes];
alignas(128) int8_t g_b[kRegBytes];
alignas(256) int32_t g_c[kLanes];

void dot(const int8_t * a, const int8_t * b, int32_t * c) {
    const size_t vl8 = __riscv_vsetvlmax_e8m1();
    vint8m1_t va = __riscv_vle8_v_i8m1(a, vl8);
    vint8m1_t vb = __riscv_vle8_v_i8m1(b, vl8);
    vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, kLanes);
    // 第 4 个参数 3 选 i8 模式，第 5 个是 vl。
    acc = __riscv_smt_vmadot_i32m2(acc, va, vb, 3, vl8);
    __riscv_vse32_v_i32m2(c, acc, kLanes);
}

int failures = 0;

void fail(const char * what, int i, int j) {
    if (failures < 20) {
        std::printf("  FAIL %s (i=%d, j=%d)\n", what, i, j);
    }
    ++failures;
}

}  // namespace

int main(int argc, char ** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    std::printf("# VLEN=%llu bits\n", (unsigned long long) k3::vlenb() * 8);

    // 1. B 全 1，A one-hot：非零 lane 集合暴露该字节属于哪一行 m。
    std::printf("[1] vs1 row mapping (expect m = byte / 16)\n");
    std::memset(g_b, 1, sizeof(g_b));
    for (int i = 0; i < kRegBytes; ++i) {
        std::memset(g_a, 0, sizeof(g_a));
        g_a[i] = 1;
        dot(g_a, g_b, g_c);
        const int expect_m = i / kK;
        for (int lane = 0; lane < kLanes; ++lane) {
            const int want = (lane / kN == expect_m) ? 1 : 0;
            if (g_c[lane] != want) {
                fail("vs1 row mapping", i, lane);
                break;
            }
        }
    }
    std::printf("  %s\n", failures == 0 ? "ok" : "mismatch");

    // 2. A 全 1，B one-hot：非零 lane 集合暴露该字节属于哪一列 n。
    const int before_b = failures;
    std::printf("[2] vs2 column mapping (expect n = byte / 16)\n");
    std::memset(g_a, 1, sizeof(g_a));
    for (int j = 0; j < kRegBytes; ++j) {
        std::memset(g_b, 0, sizeof(g_b));
        g_b[j] = 1;
        dot(g_a, g_b, g_c);
        const int expect_n = j / kK;
        for (int lane = 0; lane < kLanes; ++lane) {
            const int want = (lane % kN == expect_n) ? 1 : 0;
            if (g_c[lane] != want) {
                fail("vs2 column mapping", j, lane);
                break;
            }
        }
    }
    std::printf("  %s\n", failures == before_b ? "ok" : "mismatch");

    // 3. 两边都 one-hot：只有 k 相同时才有贡献，落点唯一。
    const int before_k = failures;
    std::printf("[3] shared-K position and vd lane order\n");
    for (int i = 0; i < kRegBytes; ++i) {
        for (int j = 0; j < kRegBytes; ++j) {
            std::memset(g_a, 0, sizeof(g_a));
            std::memset(g_b, 0, sizeof(g_b));
            g_a[i] = 1;
            g_b[j] = 1;
            dot(g_a, g_b, g_c);
            const bool same_k = (i % kK) == (j % kK);
            const int hit = (i / kK) * kN + (j / kK);
            for (int lane = 0; lane < kLanes; ++lane) {
                const int want = (same_k && lane == hit) ? 1 : 0;
                if (g_c[lane] != want) {
                    fail("one-hot product", i, j);
                    goto next_pair;
                }
            }
        next_pair:;
        }
    }
    std::printf("  %s\n", failures == before_k ? "ok" : "mismatch");

    // 4. 随机矩阵与标量参考对比。
    {
        const int before = failures;
        std::printf("[4] random 8x16x8 against scalar reference\n");
        int8_t A[kM][kK];
        int8_t B[kK][kN];
        std::srand(1);
        for (int trial = 0; trial < 64; ++trial) {
            for (int m = 0; m < kM; ++m)
                for (int k = 0; k < kK; ++k) A[m][k] = int8_t(std::rand() % 255 - 127);
            for (int k = 0; k < kK; ++k)
                for (int n = 0; n < kN; ++n) B[k][n] = int8_t(std::rand() % 255 - 127);
            for (int m = 0; m < kM; ++m)
                for (int k = 0; k < kK; ++k) g_a[m * kK + k] = A[m][k];
            for (int n = 0; n < kN; ++n)
                for (int k = 0; k < kK; ++k) g_b[n * kK + k] = B[k][n];
            dot(g_a, g_b, g_c);
            for (int m = 0; m < kM; ++m) {
                for (int n = 0; n < kN; ++n) {
                    int32_t reference = 0;
                    for (int k = 0; k < kK; ++k) reference += int32_t(A[m][k]) * int32_t(B[k][n]);
                    if (g_c[m * kN + n] != reference) {
                        fail("random gemm", m, n);
                        trial = 1 << 20;
                        m = kM;
                        break;
                    }
                }
            }
        }
        std::printf("  %s\n", failures == before ? "ok" : "mismatch");
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL HYPOTHESES HOLD" : "HYPOTHESIS REJECTED",
                failures);
    return failures == 0 ? 0 : 1;
}
