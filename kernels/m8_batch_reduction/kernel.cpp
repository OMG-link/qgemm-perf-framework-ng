#include <cstddef>
#include <cstring>
#include <riscv_vector.h>
#include <smt_vector.h>

#include "ime1.h"

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;
constexpr size_t kOutputM = kMr;
constexpr size_t kOutputN = kNr;
constexpr size_t kStepKPerIter = 16;
constexpr size_t kBlockLength = QK8_0;
constexpr size_t kBytesPerMIV = 32;
constexpr size_t kBBytesPerKIter = 4 * kBytesPerMIV;
constexpr size_t kABytesPerKIter = 2 * kBytesPerMIV;
constexpr size_t kASecondBlockOffset = 2 * kABytesPerKIter;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr size_t kRedBatchSize = 8;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(const uint8_t *GGML_RESTRICT baseA, const uint8_t *GGML_RESTRICT baseWQs,
                                                                const uint16_t *GGML_RESTRICT baseWScales, float *GGML_RESTRICT baseC, size_t CountN,
                                                                size_t BlockCountK, const size_t ldc) {

    const size_t numKIter = kBlockLength / kStepKPerIter;

    auto baseBlockA = (const block_q8_0x8_scale32 *)baseA;
    for (size_t n = 0; n < CountN; n += kOutputN) {
        float *rowC = baseC + n;

        float acc[kOutputM * kOutputN] = {};
        int32_t reduction_component_0[4 * kOutputN * kRedBatchSize];
        int32_t reduction_component_1[4 * kOutputN * kRedBatchSize];
        size_t reductionCount = 0;
        for (size_t bk = 0; bk < BlockCountK;) {
            const size_t blockWIndex = (n / kOutputN) * BlockCountK + bk;

            vint32m2_t inner_acc0 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc1 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc2 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc3 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc4 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc5 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc6 = __riscv_vmv_v_x_i32m2(0, 16);
            vint32m2_t inner_acc7 = __riscv_vmv_v_x_i32m2(0, 16);

#pragma clang loop unroll(disable)
            for (size_t inner = 0; inner < numKIter; ++inner) {

                size_t vl8 = __riscv_vsetvlmax_e8m1();
                const uint8_t *blockWQs = baseWQs + blockWIndex * 16 * (QK4_0 / 2);
                vuint8m1_t bPacked1 = __riscv_vle8_v_u8m1(blockWQs + inner * kBBytesPerKIter + 0 * kBytesPerMIV, vl8);
                vuint8m1_t bPacked2 = __riscv_vle8_v_u8m1(blockWQs + inner * kBBytesPerKIter + 1 * kBytesPerMIV, vl8);
                vuint8m1_t bPacked3 = __riscv_vle8_v_u8m1(blockWQs + inner * kBBytesPerKIter + 2 * kBytesPerMIV, vl8);
                vuint8m1_t bPacked4 = __riscv_vle8_v_u8m1(blockWQs + inner * kBBytesPerKIter + 3 * kBytesPerMIV, vl8);

                vuint8m1_t bLo1 = __riscv_vand_vx_u8m1(bPacked1, 15, vl8);
                vuint8m1_t bLo2 = __riscv_vand_vx_u8m1(bPacked2, 15, vl8);
                vuint8m1_t bLo3 = __riscv_vand_vx_u8m1(bPacked3, 15, vl8);
                vuint8m1_t bLo4 = __riscv_vand_vx_u8m1(bPacked4, 15, vl8);

                vuint8m1_t bHi1 = __riscv_vsrl_vx_u8m1(bPacked1, 4, vl8);
                vuint8m1_t bHi2 = __riscv_vsrl_vx_u8m1(bPacked2, 4, vl8);
                vuint8m1_t bHi3 = __riscv_vsrl_vx_u8m1(bPacked3, 4, vl8);
                vuint8m1_t bHi4 = __riscv_vsrl_vx_u8m1(bPacked4, 4, vl8);

                vint8m1_t bLo1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo1), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo2), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo3), kQuantizationZeroPoint, vl8);
                vint8m1_t bLo4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bLo4), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi1i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi1), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi2i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi2), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi3i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi3), kQuantizationZeroPoint, vl8);
                vint8m1_t bHi4i = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(bHi4), kQuantizationZeroPoint, vl8);

                vint8m1_t A1 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[inner * kABytesPerKIter], vl8);
                vint8m1_t A2 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[inner * kABytesPerKIter + kBytesPerMIV], vl8);

                vint8m1_t A3 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[kASecondBlockOffset + inner * kABytesPerKIter], vl8);
                vint8m1_t A4 = __riscv_vle8_v_i8m1(&baseBlockA[bk].qs[kASecondBlockOffset + inner * kABytesPerKIter + kBytesPerMIV], vl8);

                asm volatile("" : : "vr"(bLo1i), "vr"(bLo2i), "vr"(bLo3i), "vr"(bLo4i), "vr"(bHi1i), "vr"(bHi2i), "vr"(bHi3i), "vr"(bHi4i), "vr"(A1), "vr"(A2), "vr"(A3), "vr"(A4));

                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, bLo1i, kVmadotMode, kVmadotSignedness);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, bLo2i, kVmadotMode, kVmadotSignedness);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, bLo3i, kVmadotMode, kVmadotSignedness);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, bLo4i, kVmadotMode, kVmadotSignedness);

                inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, bHi1i, kVmadotMode, kVmadotSignedness);
                inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, bHi2i, kVmadotMode, kVmadotSignedness);
                inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, bHi3i, kVmadotMode, kVmadotSignedness);
                inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, bHi4i, kVmadotMode, kVmadotSignedness);

                inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A3, bLo1i, kVmadotMode, kVmadotSignedness);
                inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A3, bLo2i, kVmadotMode, kVmadotSignedness);
                inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A3, bLo3i, kVmadotMode, kVmadotSignedness);
                inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A3, bLo4i, kVmadotMode, kVmadotSignedness);

                inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A4, bHi1i, kVmadotMode, kVmadotSignedness);
                inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A4, bHi2i, kVmadotMode, kVmadotSignedness);
                inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A4, bHi3i, kVmadotMode, kVmadotSignedness);
                inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A4, bHi4i, kVmadotMode, kVmadotSignedness);
            }
            const size_t vl_m2 = __riscv_vsetvlmax_e32m2();
            vint32m8_t acc_m8_0 = __riscv_vcreate_v_i32m2_i32m8(inner_acc0, inner_acc1, inner_acc2, inner_acc3);
            vint32m8_t acc_m8_1 = __riscv_vcreate_v_i32m2_i32m8(inner_acc4, inner_acc5, inner_acc6, inner_acc7);
            auto store_unpack = [](const vint32m8_t iacc, int32_t *C_start) {
                const size_t vl_m8 = __riscv_vsetvlmax_e32m8();
                int32_t *a1 = C_start;
                int32_t *a2 = a1 + 16;
                int32_t *a3 = a2 + 16;
                int32_t *a4 = a3 + 16;
                int32_t *a2_off = a2 - 4;
                int32_t *a4_off = a4 - 4;

                auto make_high4_mask_e32mf2 = []() -> vbool32_t {
                    size_t vl = __riscv_vsetvlmax_e32m1();
                    vuint32m1_t idx = __riscv_vid_v_u32m1(vl);
                    return __riscv_vmsgtu_vx_u32m1_b32(idx, 3, vl);
                };
                vbool32_t mask32_4 = make_high4_mask_e32mf2();

                asm volatile("" : : "vr"(mask32_4));

                size_t vl_mf2 = __riscv_vsetvlmax_e32mf2();
                size_t vl_m1 = __riscv_vsetvlmax_e32m1();

                vint32m1_t acc_0 = __riscv_vget_v_i32m8_i32m1(iacc, 0);
                vint32m1_t acc_1 = __riscv_vget_v_i32m8_i32m1(iacc, 1);
                vint32m1_t acc_2 = __riscv_vget_v_i32m8_i32m1(iacc, 2);
                vint32m1_t acc_3 = __riscv_vget_v_i32m8_i32m1(iacc, 3);
                vint32m1_t acc_4 = __riscv_vget_v_i32m8_i32m1(iacc, 4);
                vint32m1_t acc_5 = __riscv_vget_v_i32m8_i32m1(iacc, 5);
                vint32m1_t acc_6 = __riscv_vget_v_i32m8_i32m1(iacc, 6);
                vint32m1_t acc_7 = __riscv_vget_v_i32m8_i32m1(iacc, 7);

                __riscv_vse32_v_i32mf2(a1, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_0), vl_mf2);
                a1 += 4;
                __riscv_vse32_v_i32mf2(a3, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_1), vl_mf2);
                a3 += 4;
                __riscv_vse32_v_i32mf2(a1, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_2), vl_mf2);
                a1 += 4;
                __riscv_vse32_v_i32mf2(a3, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_3), vl_mf2);
                a3 += 4;
                __riscv_vse32_v_i32mf2(a1, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_4), vl_mf2);
                a1 += 4;
                __riscv_vse32_v_i32mf2(a3, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_5), vl_mf2);
                a3 += 4;
                __riscv_vse32_v_i32mf2(a1, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_6), vl_mf2);
                __riscv_vse32_v_i32mf2(a3, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_7), vl_mf2);

                asm volatile("" : : "vr"(acc_0), "vr"(acc_1), "vr"(acc_2), "vr"(acc_3), "vr"(acc_4), "vr"(acc_5), "vr"(acc_6), "vr"(acc_7));

                __riscv_vse32_v_i32m1_m(mask32_4, a2_off, acc_0, vl_m1);
                a2_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a4_off, acc_1, vl_m1);
                a4_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a2_off, acc_2, vl_m1);
                a2_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a4_off, acc_3, vl_m1);
                a4_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a2_off, acc_4, vl_m1);
                a2_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a4_off, acc_5, vl_m1);
                a4_off += 4;
                __riscv_vse32_v_i32m1_m(mask32_4, a2_off, acc_6, vl_m1);
                __riscv_vse32_v_i32m1_m(mask32_4, a4_off, acc_7, vl_m1);
            };
            store_unpack(acc_m8_0, reduction_component_0 + reductionCount * 4 * kOutputN);
            store_unpack(acc_m8_1, reduction_component_1 + reductionCount * 4 * kOutputN);

            bk++;
            reductionCount++;
            if (reductionCount == kRedBatchSize or bk == BlockCountK) {
                const size_t reductionBlockStart = bk - reductionCount;
                auto reduce_4x16 = [&](const int32_t *components, float *accumulator, size_t aScaleOffset) {
                    vfloat32m2_t acc_0 = __riscv_vle32_v_f32m2(accumulator + 0 * kOutputN, vl(32, 2));
                    vfloat32m2_t acc_1 = __riscv_vle32_v_f32m2(accumulator + 1 * kOutputN, vl(32, 2));
                    vfloat32m2_t acc_2 = __riscv_vle32_v_f32m2(accumulator + 2 * kOutputN, vl(32, 2));
                    vfloat32m2_t acc_3 = __riscv_vle32_v_f32m2(accumulator + 3 * kOutputN, vl(32, 2));
                    for (size_t reductionIndex = 0; reductionIndex < reductionCount; ++reductionIndex) {
                        const size_t reductionBlock = reductionBlockStart + reductionIndex;
                        const int32_t *component = components + reductionIndex * 4 * kOutputN;
                        vint32m2_t component_0_i = __riscv_vle32_v_i32m2(component + 0 * kOutputN, vl(32, 2));
                        vint32m2_t component_1_i = __riscv_vle32_v_i32m2(component + 1 * kOutputN, vl(32, 2));
                        vint32m2_t component_2_i = __riscv_vle32_v_i32m2(component + 2 * kOutputN, vl(32, 2));
                        vint32m2_t component_3_i = __riscv_vle32_v_i32m2(component + 3 * kOutputN, vl(32, 2));
                        vfloat16m1_t bsh = __riscv_vle16_v_f16m1(
                            (_Float16 *)(baseWScales + (n / kOutputN) * BlockCountK * 16 + reductionBlock * 16), vl(16, 1));
                        float as0 = baseBlockA[reductionBlock].d[aScaleOffset + 0];
                        float as1 = baseBlockA[reductionBlock].d[aScaleOffset + 1];
                        float as2 = baseBlockA[reductionBlock].d[aScaleOffset + 2];
                        float as3 = baseBlockA[reductionBlock].d[aScaleOffset + 3];
                        vfloat32m2_t bs = __riscv_vfwcvt_f_f_v_f32m2(bsh, vl(16, 1));
                        vfloat32m2_t component_0 = __riscv_vfcvt_f_x_v_f32m2(component_0_i, vl(32, 2));
                        vfloat32m2_t component_1 = __riscv_vfcvt_f_x_v_f32m2(component_1_i, vl(32, 2));
                        vfloat32m2_t component_2 = __riscv_vfcvt_f_x_v_f32m2(component_2_i, vl(32, 2));
                        vfloat32m2_t component_3 = __riscv_vfcvt_f_x_v_f32m2(component_3_i, vl(32, 2));

                        vfloat32m2_t abscale_0 = __riscv_vfmul_vf_f32m2(bs, as0, vl(32, 2));
                        vfloat32m2_t abscale_1 = __riscv_vfmul_vf_f32m2(bs, as1, vl(32, 2));
                        vfloat32m2_t abscale_2 = __riscv_vfmul_vf_f32m2(bs, as2, vl(32, 2));
                        vfloat32m2_t abscale_3 = __riscv_vfmul_vf_f32m2(bs, as3, vl(32, 2));
                        acc_0 = __riscv_vfmacc_vv_f32m2(acc_0, component_0, abscale_0, vl(32, 2));
                        acc_1 = __riscv_vfmacc_vv_f32m2(acc_1, component_1, abscale_1, vl(32, 2));
                        acc_2 = __riscv_vfmacc_vv_f32m2(acc_2, component_2, abscale_2, vl(32, 2));
                        acc_3 = __riscv_vfmacc_vv_f32m2(acc_3, component_3, abscale_3, vl(32, 2));
                    }
                    __riscv_vse32_v_f32m2(accumulator + 0 * kOutputN, acc_0, vl(32, 2));
                    __riscv_vse32_v_f32m2(accumulator + 1 * kOutputN, acc_1, vl(32, 2));
                    __riscv_vse32_v_f32m2(accumulator + 2 * kOutputN, acc_2, vl(32, 2));
                    __riscv_vse32_v_f32m2(accumulator + 3 * kOutputN, acc_3, vl(32, 2));
                };
                reduce_4x16(reduction_component_0, acc, 0);
                reduce_4x16(reduction_component_1, acc + 4 * kOutputN, 4);
                reductionCount = 0;
            }
        }
        for (size_t row = 0; row < kOutputM; ++row) {
            std::memcpy(rowC + row * ldc, acc + row * kOutputN, kOutputN * sizeof(float));
        }
    }
}
