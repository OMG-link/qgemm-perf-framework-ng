#include <cstdint>
#include <riscv_vector.h>
#include <cstddef>
#include <smt_vector.h>

#include "ime1.h"


// 8x16 寄存器分块，延迟规约
void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(size_t            BlkLen,
                                                       const uint8_t *   QuantA,
                                                       const uint8_t *   QuantBData,
                                                       float *           C,
                                                       size_t            CountN,
                                                       size_t            BlockCountK,
                                                       const size_t      ldc) {

    using BlockA = block_q8_0x8_scale32;
    using BlockB = block_q4_0x16;
    
    size_t       LDC   = ldc * sizeof(float);
    const size_t INNER = 2;
    float        tmp[8 * 16];

    auto A = (const BlockA *)QuantA;
    auto B = (const BlockB *)QuantBData;

    for (size_t n = 0; n < CountN; n += 16) {
        size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
        auto curr_B_row = &B[n / 16 * BlockCountK];
        float * CPtr = C + n;
        if (NBLKS < 16) {
            CPtr = tmp;
            LDC  = 16 * sizeof(float);
        }

        size_t         ldc_bytes = LDC;

        float acc[8 * 16] = {};
        constexpr size_t red_comp_num = 8;
        float reduction_component_0[4 * 16 * red_comp_num];
        float reduction_component_1[4 * 16 * red_comp_num];
        uint16_t B_scales[16 * red_comp_num];
        float A_scales03[4 * red_comp_num];
        float A_scales47[4 * red_comp_num];
        
        constexpr size_t rs = sizeof(reduction_component_0) * 2 + sizeof(B_scales) + sizeof(A_scales47) * 2;
        constexpr size_t ts = rs + red_comp_num * (sizeof(BlockA) + sizeof(BlockB));
        constexpr size_t ss = (sizeof(BlockA) + sizeof(BlockB) + (8 * 16 + 8) * sizeof(float) + 16 * sizeof(uint16_t));
        constexpr float in = 32 * 1024. / ss;
        // reduction component count
        int rcc = 0;
        for (size_t bk = 0; bk < BlockCountK; ) {
            auto& curr_B = curr_B_row[bk];

            // 内循环累加器
            vint32m2_t inner_acc0 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc1 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc2 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc3 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc4 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc5 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc6 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc7 = __riscv_vmv_v_x_i32m2(0, 16);


            // INNER 循环
            for (size_t inner = 0; inner < INNER; ++inner) {
                // 加载 B 数据（4 组 8 位数据，每组 32 字节）
                size_t vl8 = __riscv_vsetvlmax_e8m1();
                vuint8m1_t B_data1 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 0 * 32], vl8);
                vuint8m1_t B_data2 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 1 * 32], vl8);
                vuint8m1_t B_data3 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 2 * 32], vl8);
                vuint8m1_t B_data4 = __riscv_vle8_v_u8m1((const uint8_t *)&curr_B.qs[inner * 128 + 3 * 32], vl8);


                // 解包 4‑bit：低 4 位 (v2‑v5)，高 4 位 (v6‑v9)
                vuint8m1_t B_lo1 = __riscv_vand_vx_u8m1(B_data1, 15, vl8);
                vuint8m1_t B_lo2 = __riscv_vand_vx_u8m1(B_data2, 15, vl8);
                vuint8m1_t B_lo3 = __riscv_vand_vx_u8m1(B_data3, 15, vl8);
                vuint8m1_t B_lo4 = __riscv_vand_vx_u8m1(B_data4, 15, vl8);

                vuint8m1_t B_hi1 = __riscv_vsrl_vx_u8m1(B_data1, 4, vl8);
                vuint8m1_t B_hi2 = __riscv_vsrl_vx_u8m1(B_data2, 4, vl8);
                vuint8m1_t B_hi3 = __riscv_vsrl_vx_u8m1(B_data3, 4, vl8);
                vuint8m1_t B_hi4 = __riscv_vsrl_vx_u8m1(B_data4, 4, vl8);

                // 减 8（零点）
                vint8m1_t B_lo1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo1), -8, vl8);
                vint8m1_t B_lo2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo2), -8, vl8);
                vint8m1_t B_lo3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo3), -8, vl8);
                vint8m1_t B_lo4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_lo4), -8, vl8);
                vint8m1_t B_hi1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi1), -8, vl8);
                vint8m1_t B_hi2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi2), -8, vl8);
                vint8m1_t B_hi3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi3), -8, vl8);
                vint8m1_t B_hi4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(B_hi4), -8, vl8);

                // 加载 A 数据（两组 32 个 int8）
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

                // 加载 A 矩阵 M 方向的第二块
                A1 = __riscv_vle8_v_i8m1(&A[bk].qs[128 + inner * 64], vl8);
                A2 = __riscv_vle8_v_i8m1(&A[bk].qs[128 + inner * 64 + 32], vl8);


                inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A1, B_lo1i, 3, 0);
                inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A1, B_lo2i, 3, 0);
                inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A1, B_lo3i, 3, 0);
                inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A1, B_lo4i, 3, 0);
                
                inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A2, B_hi1i, 3, 0);
                inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A2, B_hi2i, 3, 0);
                inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A2, B_hi3i, 3, 0);
                inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A2, B_hi4i, 3, 0);
            }
            size_t vl_m8 = __riscv_vsetvlmax_e32m8();
            
            vint32m8_t acc_m8_0 = __riscv_vcreate_v_i32m2_i32m8(inner_acc0, inner_acc1, inner_acc2, inner_acc3);
            vint32m8_t acc_m8_1 = __riscv_vcreate_v_i32m2_i32m8(inner_acc4, inner_acc5, inner_acc6, inner_acc7);
            vfloat32m8_t acc_f_m8_0 = __riscv_vfcvt_f_x_v_f32m8(acc_m8_0, vl_m8);
            vfloat32m8_t acc_f_m8_1 = __riscv_vfcvt_f_x_v_f32m8(acc_m8_1, vl_m8);
            auto store_unpack = [] (const vfloat32m8_t facc, float *C_start) {
                const size_t vl_m8 = __riscv_vsetvlmax_e32m8();
                float *a1 = C_start;
                float *a2 = a1 + 16;
                float *a3 = a2 + 16;
                float *a4 = a3 + 16;
                float *a2_off = a2 - 4;  // addi a2, a2, -16 => -4 floats
                float *a4_off = a4 - 4;
                
                auto make_high4_mask_e32mf2 = []() -> vbool32_t {
                    size_t vl = __riscv_vsetvlmax_e32m1();
                    vuint32m1_t idx = __riscv_vid_v_u32m1(vl);
                    return __riscv_vmsgtu_vx_u32m1_b32(idx, 3, vl);
                };
                vbool32_t mask32_4 = make_high4_mask_e32mf2();

                size_t vl_mf2 = __riscv_vsetvlmax_e32mf2();
                size_t vl_m1  = __riscv_vsetvlmax_e32m1();
                
                vfloat32m1_t acc_0 = __riscv_vget_v_f32m8_f32m1(facc, 0);
                vfloat32m1_t acc_1 = __riscv_vget_v_f32m8_f32m1(facc, 1);
                vfloat32m1_t acc_2 = __riscv_vget_v_f32m8_f32m1(facc, 2);
                vfloat32m1_t acc_3 = __riscv_vget_v_f32m8_f32m1(facc, 3);
                vfloat32m1_t acc_4 = __riscv_vget_v_f32m8_f32m1(facc, 4);
                vfloat32m1_t acc_5 = __riscv_vget_v_f32m8_f32m1(facc, 5);
                vfloat32m1_t acc_6 = __riscv_vget_v_f32m8_f32m1(facc, 6);
                vfloat32m1_t acc_7 = __riscv_vget_v_f32m8_f32m1(facc, 7);
                
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_0), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_1), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_2), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_3), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_4), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_5), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_6), vl_mf2);
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_7), vl_mf2);
                
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_0, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_1, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_2, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_3, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_4, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_5, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_6, vl_m1);
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_7, vl_m1);
            };
            store_unpack(acc_f_m8_0, reduction_component_0 + rcc * 4 * 16);
            store_unpack(acc_f_m8_1, reduction_component_1 + rcc * 4 * 16);
            // __riscv_vse32_v_f32m8(reduction_component_0 + rcc * 4 * 16, acc_f_m8_0, vl(32, 8));
            // __riscv_vse32_v_f32m8(reduction_component_1 + rcc * 4 * 16, acc_f_m8_1, vl(32, 8));
            memcpy(B_scales + rcc * 16, curr_B.d, 16 * sizeof(uint16_t));
            memcpy(A_scales03 + rcc * 4, A[bk].d, 4 * sizeof(float));
            memcpy(A_scales47 + rcc * 4, A[bk].d + 4, 4 * sizeof(float));


            
            bk++;
            rcc++;
            if (rcc == red_comp_num or bk == BlockCountK) {
                auto acc_4x16 = [&] (float *comps, float *acc, float *a_scales, uint16_t *b_scales) {
                    vfloat32m8_t acc_r = __riscv_vle32_v_f32m8(acc, vl(32, 8));
                    for (int rci = 0; rci < rcc; rci++) {
                        vfloat32m8_t component = __riscv_vle32_v_f32m8(comps + rci * 4 * 16, vl(32, 8));
                        vfloat16m1_t bsh = __riscv_vle16_v_f16m1((_Float16 *)(b_scales + rci * 16), vl(16, 1));
                        float as0 = a_scales[rci * 4 + 0];
                        float as1 = a_scales[rci * 4 + 1];
                        float as2 = a_scales[rci * 4 + 2];
                        float as3 = a_scales[rci * 4 + 3];
                        vfloat32m2_t bs = __riscv_vfwcvt_f_f_v_f32m2(bsh, vl(16, 1));
                        // vfloat32m2_t comp_0 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 0), as0, vl(32, 2));
                        // vfloat32m2_t comp_1 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 1), as1, vl(32, 2));
                        // vfloat32m2_t comp_2 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 2), as2, vl(32, 2));
                        // vfloat32m2_t comp_3 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 3), as3, vl(32, 2));
                        // vfloat32m2_t acc_0 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 0), comp_0, bs, vl(32, 2));
                        // vfloat32m2_t acc_1 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 1), comp_1, bs, vl(32, 2));
                        // vfloat32m2_t acc_2 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 2), comp_2, bs, vl(32, 2));
                        // vfloat32m2_t acc_3 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 3), comp_3, bs, vl(32, 2));
                        vfloat32m2_t abscale_0 = __riscv_vfmul_vf_f32m2(bs, as0, vl(32, 2));
                        vfloat32m2_t abscale_1 = __riscv_vfmul_vf_f32m2(bs, as1, vl(32, 2));
                        vfloat32m2_t abscale_2 = __riscv_vfmul_vf_f32m2(bs, as2, vl(32, 2));
                        vfloat32m2_t abscale_3 = __riscv_vfmul_vf_f32m2(bs, as3, vl(32, 2));
                        vfloat32m2_t acc_0 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 0), __riscv_vget_v_f32m8_f32m2(component, 0), abscale_0, vl(32, 2));
                        vfloat32m2_t acc_1 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 1), __riscv_vget_v_f32m8_f32m2(component, 1), abscale_1, vl(32, 2));
                        vfloat32m2_t acc_2 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 2), __riscv_vget_v_f32m8_f32m2(component, 2), abscale_2, vl(32, 2));
                        vfloat32m2_t acc_3 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc_r, 3), __riscv_vget_v_f32m8_f32m2(component, 3), abscale_3, vl(32, 2));
                        acc_r = __riscv_vcreate_v_f32m2_f32m8(acc_0, acc_1, acc_2, acc_3);
                    }
                    __riscv_vse32_v_f32m8(acc, acc_r, vl(32, 8));
                };
                acc_4x16(reduction_component_0, acc         , A_scales03, B_scales);
                acc_4x16(reduction_component_1, acc + 4 * 16, A_scales47, B_scales);
                rcc = 0;
            }
        }
        memcpy(CPtr + 0 * ldc, acc + 0 * 16, 16 * sizeof(float));
        memcpy(CPtr + 1 * ldc, acc + 1 * 16, 16 * sizeof(float));
        memcpy(CPtr + 2 * ldc, acc + 2 * 16, 16 * sizeof(float));
        memcpy(CPtr + 3 * ldc, acc + 3 * 16, 16 * sizeof(float));
        memcpy(CPtr + 4 * ldc, acc + 4 * 16, 16 * sizeof(float));
        memcpy(CPtr + 5 * ldc, acc + 5 * 16, 16 * sizeof(float));
        memcpy(CPtr + 6 * ldc, acc + 6 * 16, 16 * sizeof(float));
        memcpy(CPtr + 7 * ldc, acc + 7 * 16, 16 * sizeof(float));

    }
    
}