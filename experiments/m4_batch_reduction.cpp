#include <cstring>
#include <riscv_vector.h>
#include <cstddef>
#include <smt_vector.h>

#include "ime1.h"

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed(size_t            BlkLen,
                                                const uint8_t * GGML_RESTRICT QuantA,
                                                const uint8_t * GGML_RESTRICT QuantBData,
                                                float *           GGML_RESTRICT C,
                                                size_t            CountN,
                                                size_t            BlockCountK,
                                                const size_t      ldc) {


    using BlockA = block_q8_0x4_scale32;
    using BlockB = block_q4_0x16;
    size_t       LDC   = ldc * sizeof(float);
    const size_t INNER = BlkLen / 16;
    float        tmp[4 * 16];

    auto A = (const block_q8_0x4_scale32 *)QuantA;
    auto B = (const block_q4_0x16 *)QuantBData;
    constexpr size_t s = sizeof(block_q8_0x4_scale32) + sizeof(block_q4_0x16);
    for (size_t n = 0; n < CountN; n += 16) {
        size_t      NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
        // std::byte * QuantBDataPtr = (std::byte *) QuantBData +
        //                             n * BlockCountK * BlkLen / 2 +
        //                             n * BlockCountK * sizeof(uint16_t); // fp16
        auto curr_B_row = &B[n / 16 * BlockCountK];
        float * CPtr = C + n;
        if (NBLKS < 16) {
            CPtr = tmp;
            LDC  = 16 * sizeof(float);
        }

        // 累加器
        vfloat32m8_t acc;

        constexpr size_t red_comp_num = 32;
        float reduction_components[4 * 16 * red_comp_num];
        uint16_t B_scales[16 * red_comp_num];
        float A_scales[4 * red_comp_num];

        constexpr size_t rs = sizeof(reduction_components) + sizeof(B_scales) + sizeof(A_scales);
        constexpr size_t ts = rs + red_comp_num * (sizeof(BlockA) + sizeof(BlockB));
        constexpr size_t ss = (sizeof(BlockA) + sizeof(BlockB) + (4 * 16 + 4) * sizeof(float) + 16 * sizeof(uint16_t));
        constexpr float in = 32 * 1024. / ss;

        // 初始化为 0
        {
            size_t vl = __riscv_vsetvlmax_e32m8();
            acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        }

        int rcc = 0;
        for (size_t bk = 0; bk < BlockCountK; ) {
            auto& curr_B = curr_B_row[bk];
            // ----- 加载 A 的 block scale（4 个 float） -----
            float a_scale0, a_scale1, a_scale2, a_scale3;
            {
                a_scale0 = A[bk].d[0];
                a_scale1 = A[bk].d[1];
                a_scale2 = A[bk].d[2];
                a_scale3 = A[bk].d[3];
            }



            // 内循环累加器（int32 m8，对应 v16, v18, v20, v22 共 4 个）
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
            vint32m2_t row_sum_A;
            vint8m1_t all_8 = __riscv_vmv_v_x_i8m1(8, 32);

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
                // row_sum_A = __riscv_smt_vmadot_i32m2(row_sum_A, A1, all_8, 3, 0);
                
                asm volatile("# vmadot begin:");
                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, B_lo1i, 3, 0);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, B_lo2i, 3, 0);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, B_lo3i, 3, 0);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, B_lo4i, 3, 0);
                
                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, B_hi1i, 3, 0);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, B_hi2i, 3, 0);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, B_hi3i, 3, 0);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, B_hi4i, 3, 0);
                asm volatile("# vmadot end:");
                // row_sum_A = __riscv_smt_vmadot_i32m2(row_sum_A, A2, all_8, 3, 0);
            }
            
            vint32m8_t inner_acc_full =  __riscv_vcreate_v_i32m2_i32m8(
                                                    inner_acc0, inner_acc1, inner_acc2, inner_acc3);

            size_t vl_m8 = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t inner_acc_f = __riscv_vfcvt_f_x_v_f32m8(inner_acc_full, vl_m8);

            auto store_unpack = [] (vfloat32m8_t acc, float *dst, size_t row_stride){
                float *a1 = dst;
                float *a2 = a1 + row_stride;
                float *a3 = a2 + row_stride;
                float *a4 = a3 + row_stride;
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
                
                vfloat32m1_t acc_0 = __riscv_vget_v_f32m8_f32m1(acc, 0);
                vfloat32m1_t acc_1 = __riscv_vget_v_f32m8_f32m1(acc, 1);
                vfloat32m1_t acc_2 = __riscv_vget_v_f32m8_f32m1(acc, 2);
                vfloat32m1_t acc_3 = __riscv_vget_v_f32m8_f32m1(acc, 3);
                vfloat32m1_t acc_4 = __riscv_vget_v_f32m8_f32m1(acc, 4);
                vfloat32m1_t acc_5 = __riscv_vget_v_f32m8_f32m1(acc, 5);
                vfloat32m1_t acc_6 = __riscv_vget_v_f32m8_f32m1(acc, 6);
                vfloat32m1_t acc_7 = __riscv_vget_v_f32m8_f32m1(acc, 7);
                
                // 低半段（无掩码）存储
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_0), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_1), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_2), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_3), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_4), vl_mf2); a1 += 4;
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_5), vl_mf2); a3 += 4;
                __riscv_vse32_v_f32mf2(a1, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_6), vl_mf2);
                __riscv_vse32_v_f32mf2(a3, __riscv_vlmul_trunc_v_f32m1_f32mf2(acc_7), vl_mf2);
                
                // 高半段（掩码）存储
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_0, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_1, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_2, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_3, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_4, vl_m1); a2_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_5, vl_m1); a4_off += 4;
                __riscv_vse32_v_f32m1_m(mask32_4, a2_off, acc_6, vl_m1);
                __riscv_vse32_v_f32m1_m(mask32_4, a4_off, acc_7, vl_m1);
            };
            auto unpack = [] (vfloat32m8_t acc) -> vfloat32m8_t {
                vfloat32m1_t acc_0 = __riscv_vget_v_f32m8_f32m1(acc, 0);
                vfloat32m1_t acc_1 = __riscv_vget_v_f32m8_f32m1(acc, 1);
                vfloat32m1_t acc_2 = __riscv_vget_v_f32m8_f32m1(acc, 2);
                vfloat32m1_t acc_3 = __riscv_vget_v_f32m8_f32m1(acc, 3);
                vfloat32m1_t acc_4 = __riscv_vget_v_f32m8_f32m1(acc, 4);
                vfloat32m1_t acc_5 = __riscv_vget_v_f32m8_f32m1(acc, 5);
                vfloat32m1_t acc_6 = __riscv_vget_v_f32m8_f32m1(acc, 6);
                vfloat32m1_t acc_7 = __riscv_vget_v_f32m8_f32m1(acc, 7);
                vfloat32m1_t unpacked_0, unpacked_1, unpacked_2, unpacked_3, unpacked_4, unpacked_5, unpacked_6, unpacked_7;
                const size_t vl = __riscv_vsetvlmax_e32m1();
                unpacked_0 =  __riscv_vmv_v_v_f32m1(acc_0, vl);
                unpacked_2 =  __riscv_vmv_v_v_f32m1(acc_2, vl);
                unpacked_4 =  __riscv_vmv_v_v_f32m1(acc_1, vl);
                unpacked_6 =  __riscv_vmv_v_v_f32m1(acc_3, vl);
                unpacked_1 =  __riscv_vmv_v_v_f32m1(acc_4, vl);
                unpacked_3 =  __riscv_vmv_v_v_f32m1(acc_6, vl);
                unpacked_5 =  __riscv_vmv_v_v_f32m1(acc_5, vl);
                unpacked_7 =  __riscv_vmv_v_v_f32m1(acc_7, vl);
                
                unpacked_0 =  __riscv_vslideup_vx_f32m1(unpacked_0, acc_2, 4, vl);
                unpacked_4 =  __riscv_vslideup_vx_f32m1(unpacked_4, acc_3, 4, vl);
                unpacked_1 =  __riscv_vslideup_vx_f32m1(unpacked_1, acc_6, 4, vl);
                unpacked_5 =  __riscv_vslideup_vx_f32m1(unpacked_5, acc_7, 4, vl);
                
                const size_t vl_mf2 = 4;
                unpacked_2 = __riscv_vslidedown_vx_f32m1_tu(unpacked_2, acc_0, 4, vl_mf2);
                unpacked_6 = __riscv_vslidedown_vx_f32m1_tu(unpacked_6, acc_1, 4, vl_mf2);
                unpacked_3 = __riscv_vslidedown_vx_f32m1_tu(unpacked_3, acc_4, 4, vl_mf2);
                unpacked_7 = __riscv_vslidedown_vx_f32m1_tu(unpacked_7, acc_5, 4, vl_mf2);
                return __riscv_vcreate_v_f32m1_f32m8(
                    unpacked_0, unpacked_1, unpacked_2, unpacked_3, unpacked_4, unpacked_5, unpacked_6, unpacked_7);
            };
            // inner_acc_f = unpack(inner_acc_f);
            store_unpack(inner_acc_f, reduction_components + rcc * 4 * 16, 16);
            memcpy(A_scales + rcc * 4, A[bk].d, 4 * sizeof(float));
            memcpy(B_scales + rcc * 16, curr_B.d, 16 * sizeof(uint16_t));

            bk++;
            rcc++;

            // inner_acc_f = __riscv_vle32_v_f32m8(unpack_buffer, vl(32, 8));
            if (rcc == red_comp_num or bk == BlockCountK) {
                for (int rci = 0; rci < rcc; rci++) {
                    vfloat32m8_t component = __riscv_vle32_v_f32m8(reduction_components + rci * 4 * 16, vl(32, 8));
                    vfloat16m1_t bsh = __riscv_vle16_v_f16m1((_Float16 *)(B_scales + rci * 16), vl(16, 1));
                    float as0 = A_scales[rci * 4 + 0];
                    float as1 = A_scales[rci * 4 + 1];
                    float as2 = A_scales[rci * 4 + 2];
                    float as3 = A_scales[rci * 4 + 3];
                    vfloat32m2_t bs = __riscv_vfwcvt_f_f_v_f32m2(bsh, vl(16, 1));
                    vfloat32m2_t comp_0 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 0), as0, vl(32, 2));
                    vfloat32m2_t comp_1 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 1), as1, vl(32, 2));
                    vfloat32m2_t comp_2 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 2), as2, vl(32, 2));
                    vfloat32m2_t comp_3 = __riscv_vfmul_vf_f32m2(__riscv_vget_v_f32m8_f32m2(component, 3), as3, vl(32, 2));
                    vfloat32m2_t acc_0 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc, 0), comp_0, bs, vl(32, 2));
                    vfloat32m2_t acc_1 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc, 1), comp_1, bs, vl(32, 2));
                    vfloat32m2_t acc_2 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc, 2), comp_2, bs, vl(32, 2));
                    vfloat32m2_t acc_3 = __riscv_vfmacc_vv_f32m2(__riscv_vget_v_f32m8_f32m2(acc, 3), comp_3, bs, vl(32, 2));
                    acc = __riscv_vcreate_v_f32m2_f32m8(acc_0, acc_1, acc_2, acc_3);
                }
                rcc = 0;
            }

        }

        __riscv_vse32_v_f32m2(CPtr + 0 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 0), vl(32, 2));
        __riscv_vse32_v_f32m2(CPtr + 1 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 1), vl(32, 2));
        __riscv_vse32_v_f32m2(CPtr + 2 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 2), vl(32, 2));
        __riscv_vse32_v_f32m2(CPtr + 3 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 3), vl(32, 2));
        
    }

}