#include "kernel.h"

#include <riscv_vector.h>
#include <smt_vector.h>

#include <algorithm>
#include <cstring>

#include "rvv_vl.h"
#include "types.h"

using namespace ime::m4n32b;

namespace {

constexpr int kZeroPoint = -8;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

__attribute__((noinline)) void store_half(vint32m2_t acc0, vint32m2_t acc1, vint32m2_t acc2,
                       vint32m2_t acc3, int32_t *destination) {
    alignas(64) int32_t lanes[4][16];
    const size_t vl = ime::rvv::vl(16, 2);
    __riscv_vse32_v_i32m2(lanes[0], acc0, vl);
    __riscv_vse32_v_i32m2(lanes[1], acc1, vl);
    __riscv_vse32_v_i32m2(lanes[2], acc2, vl);
    __riscv_vse32_v_i32m2(lanes[3], acc3, vl);
    for (size_t group = 0; group < 4; ++group) {
        for (size_t lane = 0; lane < 16; ++lane) {
            const size_t row = lane / 4;
            const size_t column = 4 * group + lane % 4;
            destination[row * NR + column] = lanes[group][lane];
        }
    }
}

inline vint8m1_t unpack_low(vuint8m1_t packed, size_t vl) {
    const vuint8m1_t q = __riscv_vand_vx_u8m1(packed, 15, vl);
    return __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(q), kZeroPoint, vl);
}

inline vint8m1_t unpack_high(vuint8m1_t packed, size_t vl) {
    const vuint8m1_t q = __riscv_vsrl_vx_u8m1(packed, 4, vl);
    return __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(q), kZeroPoint, vl);
}

__attribute__((noinline)) void reduce_batch(
    const block_q8_0_m4 *a, const block_q4_0_n32 *b,
    const int32_t components[ReductionBatch][MR][NR], float fp_acc[MR][NR],
    size_t first, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        const size_t block = first + index;
        const vfloat16m2_t bscale_h =
            __riscv_vle16_v_f16m2(b[block].d, ime::rvv::vl(16, 2));
        const vfloat32m4_t bscale =
            __riscv_vfwcvt_f_f_v_f32m4(bscale_h, ime::rvv::vl(16, 2));
        for (size_t row = 0; row < MR; ++row) {
            const vint32m4_t ci = __riscv_vle32_v_i32m4(
                components[index][row], ime::rvv::vl(32, 4));
            const vfloat32m4_t cf =
                __riscv_vfcvt_f_x_v_f32m4(ci, ime::rvv::vl(32, 4));
            vfloat32m4_t sum = __riscv_vle32_v_f32m4(fp_acc[row], ime::rvv::vl(32, 4));
            const float as = static_cast<float>(a[block].d[row]);
            const vfloat32m4_t scale =
                __riscv_vfmul_vf_f32m4(bscale, as, ime::rvv::vl(32, 4));
            sum = __riscv_vfmacc_vv_f32m4(sum, cf, scale, ime::rvv::vl(32, 4));
            __riscv_vse32_v_f32m4(fp_acc[row], sum, ime::rvv::vl(32, 4));
        }
    }
}

} // namespace

void q4_0_q8_0_ime_m4n32b(const uint8_t *GGML_RESTRICT quant_a,
                           const uint8_t *GGML_RESTRICT quant_b,
                           float *GGML_RESTRICT output, size_t block_count_k,
                           size_t ldc, bool accumulate) {
    const auto *a = reinterpret_cast<const block_q8_0_m4 *>(quant_a);
    const auto *b = reinterpret_cast<const block_q4_0_n32 *>(quant_b);

    alignas(64) float fp_acc[MR][NR];
    for (size_t row = 0; row < MR; ++row) {
        if (accumulate)
            std::memcpy(fp_acc[row], output + row * ldc, NR * sizeof(float));
        else
            std::fill_n(fp_acc[row], NR, 0.0f);
    }

    alignas(64) int32_t components[ReductionBatch][MR][NR];
    size_t reduction_count = 0;

    for (size_t bk = 0; bk < block_count_k; ++bk) {
        for (size_t half = 0; half < 2; ++half) {
            vint32m2_t acc0 = __riscv_vmv_v_x_i32m2(0, ime::rvv::vl(16, 2));
            vint32m2_t acc1 = __riscv_vmv_v_x_i32m2(0, ime::rvv::vl(16, 2));
            vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, ime::rvv::vl(16, 2));
            vint32m2_t acc3 = __riscv_vmv_v_x_i32m2(0, ime::rvv::vl(16, 2));

#pragma clang loop unroll(disable)
            for (size_t plane = 0; plane < 2; ++plane) {
                const size_t vl8 = __riscv_vsetvlmax_e8m1();
                const vint8m1_t av0 = __riscv_vle8_v_i8m1(a[bk].qs + plane * 64, vl8);
                const vint8m1_t av1 = __riscv_vle8_v_i8m1(a[bk].qs + plane * 64 + 32, vl8);
                const size_t group = half * 4;
                const vuint8m1_t p0 = __riscv_vle8_v_u8m1(b[bk].qs[plane][group + 0], vl8);
                const vuint8m1_t p1 = __riscv_vle8_v_u8m1(b[bk].qs[plane][group + 1], vl8);
                const vuint8m1_t p2 = __riscv_vle8_v_u8m1(b[bk].qs[plane][group + 2], vl8);
                const vuint8m1_t p3 = __riscv_vle8_v_u8m1(b[bk].qs[plane][group + 3], vl8);

                const vint8m1_t lo0 = unpack_low(p0, vl8);
                const vint8m1_t lo1 = unpack_low(p1, vl8);
                const vint8m1_t lo2 = unpack_low(p2, vl8);
                const vint8m1_t lo3 = unpack_low(p3, vl8);
                const vint8m1_t hi0 = unpack_high(p0, vl8);
                const vint8m1_t hi1 = unpack_high(p1, vl8);
                const vint8m1_t hi2 = unpack_high(p2, vl8);
                const vint8m1_t hi3 = unpack_high(p3, vl8);

                acc0 = __riscv_smt_vmadot_i32m2(acc0, av0, lo0, kVmadotMode, kVmadotSignedness);
                acc1 = __riscv_smt_vmadot_i32m2(acc1, av0, lo1, kVmadotMode, kVmadotSignedness);
                acc2 = __riscv_smt_vmadot_i32m2(acc2, av0, lo2, kVmadotMode, kVmadotSignedness);
                acc3 = __riscv_smt_vmadot_i32m2(acc3, av0, lo3, kVmadotMode, kVmadotSignedness);
                acc0 = __riscv_smt_vmadot_i32m2(acc0, av1, hi0, kVmadotMode, kVmadotSignedness);
                acc1 = __riscv_smt_vmadot_i32m2(acc1, av1, hi1, kVmadotMode, kVmadotSignedness);
                acc2 = __riscv_smt_vmadot_i32m2(acc2, av1, hi2, kVmadotMode, kVmadotSignedness);
                acc3 = __riscv_smt_vmadot_i32m2(acc3, av1, hi3, kVmadotMode, kVmadotSignedness);
            }
            store_half(acc0, acc1, acc2, acc3,
                       &components[reduction_count][0][half * 16]);
        }

        ++reduction_count;
        if (reduction_count == ReductionBatch || bk + 1 == block_count_k) {
            reduce_batch(a, b, components, fp_acc, bk + 1 - reduction_count,
                         reduction_count);
            reduction_count = 0;
        }
    }

    for (size_t row = 0; row < MR; ++row)
        std::memcpy(output + row * ldc, fp_acc[row], NR * sizeof(float));
}
