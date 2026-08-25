#include <riscv_vector.h>
#include <cstddef>
#include <smt_vector.h>

#include "ime1.h"

constexpr size_t kMr = 4;
constexpr size_t kNr = 16;
constexpr size_t kOutputM = kMr;
constexpr size_t kOutputN = kNr;
constexpr size_t kStepKPerIter = 16;
constexpr size_t kBlockLength = QK8_0;
constexpr size_t kBytesPerMIV = 32;
constexpr size_t kBBytesPerKIter = kMr * kBytesPerMIV;
constexpr size_t kABytesPerKIter = 2 * kBytesPerMIV;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr size_t kRedBatchSize = 32;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed(const uint8_t * GGML_RESTRICT baseA,
                                                const uint8_t * GGML_RESTRICT baseW,
                                                float *           GGML_RESTRICT baseC,
                                                size_t            CountN,
                                                size_t            BlockCountK,
                                                const size_t      ldc) {
    const size_t numKIter = kBlockLength / kStepKPerIter;

    auto baseBlockA = (const block_q8_0x4_scale32 *)baseA;
    auto baseBlockW = (const block_q4_0x16 *)baseW;
    for (size_t n = 0; n < CountN; n += kOutputN) {
        auto rowBlockW = &baseBlockW[n / kOutputN * BlockCountK];
        float * rowC = baseC + n;

        vfloat32m8_t acc;

        float reduceInputBuffer[kOutputM * kOutputN * kRedBatchSize];

        {
            size_t vl = __riscv_vsetvlmax_e32m8();
            acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        }

        for (size_t bk = 0; bk < BlockCountK; ) {
            auto& blockW = rowBlockW[bk];

            // 内循环累加器（int32 m8，对应 v16, v18, v20, v22 共 4 个）
            vint32m2_t inner_acc0, inner_acc1, inner_acc2, inner_acc3;
            {
                size_t vl = __riscv_vsetvlmax_e32m2();
                inner_acc0 = __riscv_vmv_v_x_i32m2(0, vl);
                inner_acc1 = __riscv_vmv_v_x_i32m2(0, vl);
                inner_acc2 = __riscv_vmv_v_x_i32m2(0, vl);
                inner_acc3 = __riscv_vmv_v_x_i32m2(0, vl);
            }
            // INNER 循环
            for (size_t kIter = 0; kIter < numKIter; ++kIter) {
                size_t vl8 = __riscv_vsetvlmax_e8m1();
                vuint8m1_t bPacked1 = __riscv_vle8_v_u8m1((const uint8_t *)&blockW.qs[kIter * kBBytesPerKIter + 0 * kBytesPerMIV], vl8);
                vuint8m1_t bPacked2 = __riscv_vle8_v_u8m1((const uint8_t *)&blockW.qs[kIter * kBBytesPerKIter + 1 * kBytesPerMIV], vl8);
                vuint8m1_t bPacked3 = __riscv_vle8_v_u8m1((const uint8_t *)&blockW.qs[kIter * kBBytesPerKIter + 2 * kBytesPerMIV], vl8);
                vuint8m1_t bPacked4 = __riscv_vle8_v_u8m1((const uint8_t *)&blockW.qs[kIter * kBBytesPerKIter + 3 * kBytesPerMIV], vl8);


                // 解包 4‑bit：低 4 位 (v2‑v5)，高 4 位 (v6‑v9)
                vuint8m1_t bLo1 = __riscv_vand_vx_u8m1(bPacked1, 15, vl8);
                vuint8m1_t bLo2 = __riscv_vand_vx_u8m1(bPacked2, 15, vl8);
                vuint8m1_t bLo3 = __riscv_vand_vx_u8m1(bPacked3, 15, vl8);
                vuint8m1_t bLo4 = __riscv_vand_vx_u8m1(bPacked4, 15, vl8);

                vuint8m1_t bHi1 = __riscv_vsrl_vx_u8m1(bPacked1, 4, vl8);
                vuint8m1_t bHi2 = __riscv_vsrl_vx_u8m1(bPacked2, 4, vl8);
                vuint8m1_t bHi3 = __riscv_vsrl_vx_u8m1(bPacked3, 4, vl8);
                vuint8m1_t bHi4 = __riscv_vsrl_vx_u8m1(bPacked4, 4, vl8);

                // 减 8（零点）
                vint8m1_t bLo1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo1), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo2), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo3), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo4), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi1), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi2), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi3), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi4), kQuantizationZeroPoint, vl8);

                vint8m1_t A1 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[kIter * kABytesPerKIter], vl8);
                vint8m1_t A2 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[kIter * kABytesPerKIter + kBytesPerMIV], vl8);
                
                asm volatile("# vmadot begin:");
                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, bLo1i, kVmadotMode, kVmadotSignedness);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, bLo2i, kVmadotMode, kVmadotSignedness);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, bLo3i, kVmadotMode, kVmadotSignedness);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, bLo4i, kVmadotMode, kVmadotSignedness);
                
                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, bHi1i, kVmadotMode, kVmadotSignedness);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, bHi2i, kVmadotMode, kVmadotSignedness);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, bHi3i, kVmadotMode, kVmadotSignedness);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, bHi4i, kVmadotMode, kVmadotSignedness);
                asm volatile("# vmadot end:");
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
            store_unpack(inner_acc_f, reduceInputBuffer + (bk % kRedBatchSize) * kOutputM * kOutputN, kOutputN);

            bk++;

            if (bk % kRedBatchSize == 0 or bk == BlockCountK) {
                const size_t reduction_count =
                    bk % kRedBatchSize == 0 ? kRedBatchSize : bk % kRedBatchSize;
                const size_t reduction_block_start = bk - reduction_count;
                for (size_t rci = 0; rci < reduction_count; rci++) {
                    const size_t reduction_block = reduction_block_start + rci;
                    vfloat32m8_t component = __riscv_vle32_v_f32m8(reduceInputBuffer + rci * kOutputM * kOutputN, vl(32, 8));
                    vfloat16m1_t bsh = __riscv_vle16_v_f16m1(
                        (_Float16 *)rowBlockW[reduction_block].d, vl(16, 1));
                    float as0 = baseBlockA[reduction_block].d[0];
                    float as1 = baseBlockA[reduction_block].d[1];
                    float as2 = baseBlockA[reduction_block].d[2];
                    float as3 = baseBlockA[reduction_block].d[3];
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
            }
        }

        __riscv_vse32_v_f32m2(rowC + 0 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 0), vl(32, 2));
        __riscv_vse32_v_f32m2(rowC + 1 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 1), vl(32, 2));
        __riscv_vse32_v_f32m2(rowC + 2 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 2), vl(32, 2));
        __riscv_vse32_v_f32m2(rowC + 3 * ldc, __riscv_vget_v_f32m8_f32m2(acc, 3), vl(32, 2));
        
    }

}