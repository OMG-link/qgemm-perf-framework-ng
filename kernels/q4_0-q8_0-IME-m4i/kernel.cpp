#include <cstddef>
#include <riscv_vector.h>
#include <smt_vector.h>

#include "kernel.h"
#include "types.h"

constexpr size_t kBlockLength = QK8_0;
constexpr size_t kOutputN = 16;

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(const uint8_t *GGML_RESTRICT QuantA, const uint8_t *GGML_RESTRICT QuantBData, float *GGML_RESTRICT acc, size_t BlockCountK) {

    const size_t INNER = kBlockLength / 16;

    auto A = (const block_q8_0_ime_m4 *)QuantA;
    auto B = (const block_q4_0_ime_n16 *)QuantBData;

    vfloat32m2_t acc_0, acc_1, acc_2, acc_3;

    {
        size_t vl = __riscv_vsetvlmax_e32m2();
        acc_0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc_1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc_2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc_3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    }

    for (size_t bk = 0; bk < BlockCountK; ++bk) {
        auto &curr_B = B[bk];

        float a_scale0, a_scale1, a_scale2, a_scale3;
        {
            a_scale0 = A[bk].d[0];
            a_scale1 = A[bk].d[1];
            a_scale2 = A[bk].d[2];
            a_scale3 = A[bk].d[3];
        }

        vint32m2_t inner_acc0, inner_acc1, inner_acc2, inner_acc3;
        {
            size_t vl = __riscv_vsetvlmax_e32m2();
            inner_acc0 = __riscv_vmv_v_x_i32m2(0, vl);
            inner_acc1 = __riscv_vmv_v_x_i32m2(0, vl);
            inner_acc2 = __riscv_vmv_v_x_i32m2(0, vl);
            inner_acc3 = __riscv_vmv_v_x_i32m2(0, vl);
        }
        vfloat32m1_t A_scale_01 = __riscv_vfmv_v_f_f32m1(a_scale1, 8);
        vfloat32m1_t A_scale_23 = __riscv_vfmv_v_f_f32m1(a_scale3, 8);
        A_scale_01 = __riscv_vfmv_v_f_f32m1_tu(A_scale_01, a_scale0, 4);
        A_scale_23 = __riscv_vfmv_v_f_f32m1_tu(A_scale_23, a_scale2, 4);
        vfloat32m2_t A_scale = __riscv_vcreate_v_f32m1_f32m2(A_scale_01, A_scale_23);

        #pragma clang loop unroll(disable)
        for (size_t inner = 0; inner < INNER; ++inner) {

            size_t vl8 = __riscv_vsetvlmax_e8m1();
            vuint8m1_t B_data1 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 0 * 32], vl8);
            vuint8m1_t B_data2 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 1 * 32], vl8);
            vuint8m1_t B_data3 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 2 * 32], vl8);
            vuint8m1_t B_data4 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 3 * 32], vl8);

            vuint8m1_t B_lo1 = __riscv_vand_vx_u8m1(B_data1, 15, vl8);
            vuint8m1_t B_lo2 = __riscv_vand_vx_u8m1(B_data2, 15, vl8);
            vuint8m1_t B_lo3 = __riscv_vand_vx_u8m1(B_data3, 15, vl8);
            vuint8m1_t B_lo4 = __riscv_vand_vx_u8m1(B_data4, 15, vl8);

            vuint8m1_t B_hi1 = __riscv_vsrl_vx_u8m1(B_data1, 4, vl8);
            vuint8m1_t B_hi2 = __riscv_vsrl_vx_u8m1(B_data2, 4, vl8);
            vuint8m1_t B_hi3 = __riscv_vsrl_vx_u8m1(B_data3, 4, vl8);
            vuint8m1_t B_hi4 = __riscv_vsrl_vx_u8m1(B_data4, 4, vl8);

            vint8m1_t B_lo1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo1), -8, vl8);
            vint8m1_t B_lo2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo2), -8, vl8);
            vint8m1_t B_lo3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo3), -8, vl8);
            vint8m1_t B_lo4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo4), -8, vl8);
            vint8m1_t B_hi1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi1), -8, vl8);
            vint8m1_t B_hi2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi2), -8, vl8);
            vint8m1_t B_hi3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi3), -8, vl8);
            vint8m1_t B_hi4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi4), -8, vl8);

            vint8m1_t A1 = __riscv_vle8_v_i8m1(&A[bk].qs[inner * 64], vl8);
            vint8m1_t A2 = __riscv_vle8_v_i8m1(&A[bk].qs[inner * 64 + 32], vl8);

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, B_lo1i, 3, 0);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, B_lo2i, 3, 0);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, B_lo3i, 3, 0);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, B_lo4i, 3, 0);

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, B_hi1i, 3, 0);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, B_hi2i, 3, 0);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, B_hi3i, 3, 0);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, B_hi4i, 3, 0);
        }

        size_t vl_mask1 = __riscv_vsetvlmax_e16mf2();
        vuint16mf2_t idx_mask1 = __riscv_vid_v_u16mf2(vl_mask1);
        vbool32_t mask16 = __riscv_vmsgtu_vx_u16mf2_b32(idx_mask1, 3, vl_mask1);

        size_t vl_f16_mf4 = __riscv_vsetvlmax_e16mf4();
        size_t vl_f16_mf2 = __riscv_vsetvlmax_e16mf2();

        vfloat16mf2_t s0 = __riscv_vle16_v_f16mf2((const _Float16 *)curr_B.d + 0, vl_f16_mf4);
        vfloat16mf2_t s1 = __riscv_vle16_v_f16mf2((const _Float16 *)curr_B.d + 4, vl_f16_mf4);
        vfloat16mf2_t s2 = __riscv_vle16_v_f16mf2((const _Float16 *)curr_B.d + 8, vl_f16_mf4);
        vfloat16mf2_t s3 = __riscv_vle16_v_f16mf2((const _Float16 *)curr_B.d + 12, vl_f16_mf4);

        s0 = __riscv_vle16_v_f16mf2_mu(mask16, s0, (const _Float16 *)curr_B.d + -4, vl_f16_mf2);
        s1 = __riscv_vle16_v_f16mf2_mu(mask16, s1, (const _Float16 *)curr_B.d + 0, vl_f16_mf2);
        s2 = __riscv_vle16_v_f16mf2_mu(mask16, s2, (const _Float16 *)curr_B.d + 4, vl_f16_mf2);
        s3 = __riscv_vle16_v_f16mf2_mu(mask16, s3, (const _Float16 *)curr_B.d + 8, vl_f16_mf2);

        vfloat32m1_t scale_lo0 = __riscv_vfwcvt_f_f_v_f32m1(s0, vl_f16_mf2);
        vfloat32m1_t scale_lo1 = __riscv_vfwcvt_f_f_v_f32m1(s1, vl_f16_mf2);
        vfloat32m1_t scale_lo2 = __riscv_vfwcvt_f_f_v_f32m1(s2, vl_f16_mf2);
        vfloat32m1_t scale_lo3 = __riscv_vfwcvt_f_f_v_f32m1(s3, vl_f16_mf2);

        const size_t vl32 = __riscv_vsetvlmax_e32m1();

        auto scale_0 = __riscv_vcreate_v_f32m1_f32m2(scale_lo0, scale_lo0);
        scale_0 = __riscv_vfmul_vv_f32m2(scale_0, A_scale, 16);
        auto scale_1 = __riscv_vcreate_v_f32m1_f32m2(scale_lo1, scale_lo1);
        scale_1 = __riscv_vfmul_vv_f32m2(scale_1, A_scale, 16);
        auto scale_2 = __riscv_vcreate_v_f32m1_f32m2(scale_lo2, scale_lo2);
        scale_2 = __riscv_vfmul_vv_f32m2(scale_2, A_scale, 16);
        auto scale_3 = __riscv_vcreate_v_f32m1_f32m2(scale_lo3, scale_lo3);
        scale_3 = __riscv_vfmul_vv_f32m2(scale_3, A_scale, 16);
        size_t vl_m2 = __riscv_vsetvlmax_e32m2();
        vfloat32m2_t inner_f0 = __riscv_vfcvt_f_x_v_f32m2(inner_acc0, vl_m2);
        vfloat32m2_t inner_f1 = __riscv_vfcvt_f_x_v_f32m2(inner_acc1, vl_m2);
        vfloat32m2_t inner_f2 = __riscv_vfcvt_f_x_v_f32m2(inner_acc2, vl_m2);
        vfloat32m2_t inner_f3 = __riscv_vfcvt_f_x_v_f32m2(inner_acc3, vl_m2);

        acc_0 = __riscv_vfmacc_vv_f32m2(acc_0, inner_f0, scale_0, vl_m2);
        acc_1 = __riscv_vfmacc_vv_f32m2(acc_1, inner_f1, scale_1, vl_m2);
        acc_2 = __riscv_vfmacc_vv_f32m2(acc_2, inner_f2, scale_2, vl_m2);
        acc_3 = __riscv_vfmacc_vv_f32m2(acc_3, inner_f3, scale_3, vl_m2);
    }

    vfloat32m8_t acc_tile = __riscv_vcreate_v_f32m2_f32m8(acc_0, acc_1, acc_2, acc_3);

    // The tile result is added to the caller's acc tile one 4-float group at a
    // time: the low 4 lanes of each unpack chunk hold an even row's group, its
    // high 4 lanes the following odd row's group. Plain 4-lane loads, adds and
    // stores keep the epilogue independent.
    auto accumulate_unpack = [](vfloat32m8_t acc_tile, float *dst, size_t row_stride) {
        float *row0 = dst;
        float *row1 = row0 + row_stride;
        float *row2 = row1 + row_stride;
        float *row3 = row2 + row_stride;

        size_t vl_mf2 = __riscv_vsetvlmax_e32mf2();
        size_t vl_m1 = __riscv_vsetvlmax_e32m1();

        vfloat32m1_t acc_0 = __riscv_vget_v_f32m8_f32m1(acc_tile, 0);
        vfloat32m1_t acc_1 = __riscv_vget_v_f32m8_f32m1(acc_tile, 1);
        vfloat32m1_t acc_2 = __riscv_vget_v_f32m8_f32m1(acc_tile, 2);
        vfloat32m1_t acc_3 = __riscv_vget_v_f32m8_f32m1(acc_tile, 3);
        vfloat32m1_t acc_4 = __riscv_vget_v_f32m8_f32m1(acc_tile, 4);
        vfloat32m1_t acc_5 = __riscv_vget_v_f32m8_f32m1(acc_tile, 5);
        vfloat32m1_t acc_6 = __riscv_vget_v_f32m8_f32m1(acc_tile, 6);
        vfloat32m1_t acc_7 = __riscv_vget_v_f32m8_f32m1(acc_tile, 7);

        auto group_low = [&](vfloat32m1_t value) {
            return __riscv_vlmul_trunc_v_f32m1_f32mf2(value);
        };
        auto group_high = [&](vfloat32m1_t value) {
            return __riscv_vlmul_trunc_v_f32m1_f32mf2(__riscv_vslidedown_vx_f32m1(value, 4, vl_m1));
        };
        auto accumulate = [&](vfloat32mf2_t value, float *where) {
            vfloat32mf2_t previous = __riscv_vle32_v_f32mf2(where, vl_mf2);
            __riscv_vse32_v_f32mf2(where, __riscv_vfadd_vv_f32mf2(value, previous, vl_mf2), vl_mf2);
        };

        accumulate(group_low(acc_0), row0);
        row0 += 4;
        accumulate(group_high(acc_0), row1);
        row1 += 4;
        accumulate(group_low(acc_1), row2);
        row2 += 4;
        accumulate(group_high(acc_1), row3);
        row3 += 4;
        accumulate(group_low(acc_2), row0);
        row0 += 4;
        accumulate(group_high(acc_2), row1);
        row1 += 4;
        accumulate(group_low(acc_3), row2);
        row2 += 4;
        accumulate(group_high(acc_3), row3);
        row3 += 4;
        accumulate(group_low(acc_4), row0);
        row0 += 4;
        accumulate(group_high(acc_4), row1);
        row1 += 4;
        accumulate(group_low(acc_5), row2);
        row2 += 4;
        accumulate(group_high(acc_5), row3);
        row3 += 4;
        accumulate(group_low(acc_6), row0);
        accumulate(group_high(acc_6), row1);
        accumulate(group_low(acc_7), row2);
        accumulate(group_high(acc_7), row3);
    };
    accumulate_unpack(acc_tile, acc, kOutputN);
}
