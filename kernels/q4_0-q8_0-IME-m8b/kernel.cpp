#include <cstddef>
#include <cstring>
#include <riscv_vector.h>
#include <smt_vector.h>

#include "kernel.h"
#include "l1d_load_miss_probe.h"
#include "rvv_vl.h"

#define REPEAT_8(operation, ...)                                                                                                                                                                       \
    operation(0, __VA_ARGS__) operation(1, __VA_ARGS__) operation(2, __VA_ARGS__) operation(3, __VA_ARGS__) operation(4, __VA_ARGS__) operation(5, __VA_ARGS__) operation(6, __VA_ARGS__)              \
        operation(7, __VA_ARGS__)

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;
constexpr size_t kOutputM = kMr;
constexpr size_t kOutputN = kNr;
constexpr size_t kStepKPerIter = 16;
constexpr size_t kBlockLength = QK8_0;
constexpr size_t kBytesPerMIV = 32;
constexpr size_t kBBytesPerKIter = 4 * kBytesPerMIV;
constexpr size_t kABytesPerKIter = 4 * kBytesPerMIV;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr size_t kRedBatchSize = 8;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(const int8_t *GGML_RESTRICT baseAQs, const float *GGML_RESTRICT baseAScales, const uint8_t *GGML_RESTRICT baseWQs,
                                                                const uint16_t *GGML_RESTRICT baseWScales, float *GGML_RESTRICT baseC, size_t CountN, size_t BlockCountK, const size_t ldc) {

    const size_t numKIter = kBlockLength / kStepKPerIter;
    for (size_t n = 0; n < CountN; n += kOutputN) {
        float *rowC = baseC + n;

        float acc[kOutputM * kOutputN] = {};
        int32_t reduction_components[kOutputM * kOutputN * kRedBatchSize];
        size_t reductionCount = 0;
        for (size_t bk = 0; bk < BlockCountK;) {
            const size_t blockWIndex = (n / kOutputN) * BlockCountK + bk;
            const int8_t *blockAQs = baseAQs + bk * kMr * QK8_0;

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
                IME_L1D_PROBE_VLE8_M1(vuint8m1_t, bPacked1, blockWQs + inner * kBBytesPerKIter + 0 * kBytesPerMIV, ime_l1d_probe_m8b_w_main_0);
                IME_L1D_PROBE_VLE8_M1(vuint8m1_t, bPacked2, blockWQs + inner * kBBytesPerKIter + 1 * kBytesPerMIV, ime_l1d_probe_m8b_w_main_1);
                IME_L1D_PROBE_VLE8_M1(vuint8m1_t, bPacked3, blockWQs + inner * kBBytesPerKIter + 2 * kBytesPerMIV, ime_l1d_probe_m8b_w_main_2);
                IME_L1D_PROBE_VLE8_M1(vuint8m1_t, bPacked4, blockWQs + inner * kBBytesPerKIter + 3 * kBytesPerMIV, ime_l1d_probe_m8b_w_main_3);

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

                const int8_t *aInner = blockAQs + inner * kABytesPerKIter;
                IME_L1D_PROBE_VLE8_M1(vint8m1_t, A1, aInner, ime_l1d_probe_m8b_a_main_0);
                IME_L1D_PROBE_VLE8_M1(vint8m1_t, A2, aInner + kBytesPerMIV, ime_l1d_probe_m8b_a_main_1);
                IME_L1D_PROBE_VLE8_M1(vint8m1_t, A3, aInner + 2 * kBytesPerMIV, ime_l1d_probe_m8b_a_main_2);
                IME_L1D_PROBE_VLE8_M1(vint8m1_t, A4, aInner + 3 * kBytesPerMIV, ime_l1d_probe_m8b_a_main_3);

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
            int32_t *component = reduction_components + reductionCount * kOutputM * kOutputN;
            size_t vl_m1 = __riscv_vsetvlmax_e32m1();
            vuint32m1_t idx = __riscv_vid_v_u32m1(vl_m1);
            vbool32_t mask32_4 = __riscv_vmsgtu_vx_u32m1_b32(idx, 3, vl_m1);
            asm volatile("" : : "vr"(mask32_4));

            size_t vl_mf2 = __riscv_vsetvlmax_e32mf2();

#define SPLIT_ACCUMULATOR(index, unused)                                                                                                                                                               \
    vint32m1_t acc_low##index = __riscv_vget_v_i32m2_i32m1(inner_acc##index, 0);                                                                                                                       \
    vint32m1_t acc_high##index = __riscv_vget_v_i32m2_i32m1(inner_acc##index, 1);
#define STORE_LOW(index, unused)                                                                                                                                                                       \
    __riscv_vse32_v_i32mf2(component + (index / 4) * 4 * kOutputN + (index % 4) * 4, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_low##index), vl_mf2);                                                      \
    __riscv_vse32_v_i32mf2(component + (index / 4) * 4 * kOutputN + 2 * kOutputN + (index % 4) * 4, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_high##index), vl_mf2);
#define STORE_HIGH(index, unused)                                                                                                                                                                      \
    __riscv_vse32_v_i32m1_m(mask32_4, component + (index / 4) * 4 * kOutputN + (index % 4) * 4 + 12, acc_low##index, vl_m1);                                                                           \
    __riscv_vse32_v_i32m1_m(mask32_4, component + (index / 4) * 4 * kOutputN + 2 * kOutputN + (index % 4) * 4 + 12, acc_high##index, vl_m1);
#define KEEP_LIVE(index, unused) , "vr"(acc_low##index), "vr"(acc_high##index)

            REPEAT_8(SPLIT_ACCUMULATOR, _)
            REPEAT_8(STORE_LOW, _)

            asm volatile("" : : "r"(component)REPEAT_8(KEEP_LIVE, _));

            REPEAT_8(STORE_HIGH, _)

#undef KEEP_LIVE
#undef STORE_HIGH
#undef STORE_LOW
#undef SPLIT_ACCUMULATOR

            bk++;
            reductionCount++;
            if (reductionCount == kRedBatchSize or bk == BlockCountK) {
                const size_t reductionBlockStart = bk - reductionCount;
#define LOAD_REDUCTION_ACCUMULATOR(index, unused)                                                                                                                                                      \
    IME_L1D_PROBE_VLE32_M2(vfloat32m2_t, reduction_acc##index, acc + index * kOutputN, ime::rvv::vl(32, 2), IME_L1D_PROBE_CAT(ime_l1d_probe_m8b_accumulator_, index));
#define REDUCE_ROW(index, unused)                                                                                                                                                                      \
    {                                                                                                                                                                                                  \
        IME_L1D_PROBE_VLE32_M2(vint32m2_t, component_i, component + index * kOutputN, ime::rvv::vl(32, 2), IME_L1D_PROBE_CAT(ime_l1d_probe_m8b_component_, index));                                    \
        vfloat32m2_t component_f = __riscv_vfcvt_f_x_v_f32m2(component_i, ime::rvv::vl(32, 2));                                                                                                        \
        IME_L1D_PROBE_FLW(a_scale, baseAScales + reductionBlock * kMr + index, IME_L1D_PROBE_CAT(ime_l1d_probe_m8b_a_scale_, index));                                                                  \
        vfloat32m2_t abscale = __riscv_vfmul_vf_f32m2(bs, a_scale, ime::rvv::vl(32, 2));                                                                                                               \
        reduction_acc##index = __riscv_vfmacc_vv_f32m2(reduction_acc##index, component_f, abscale, ime::rvv::vl(32, 2));                                                                               \
    }
#define REDUCTION_BARRIER(index0, index1, index2, index3)                                                                                                                                              \
    asm volatile("" : "+vr"(reduction_acc##index0), "+vr"(reduction_acc##index1), "+vr"(reduction_acc##index2), "+vr"(reduction_acc##index3) : : "memory");
#define REDUCE_ROW_QUAD(index0, index1, index2, index3)                                                                                                                                                \
    REDUCE_ROW(index0, _)                                                                                                                                                                              \
    REDUCE_ROW(index1, _)                                                                                                                                                                              \
    REDUCE_ROW(index2, _)                                                                                                                                                                              \
    REDUCE_ROW(index3, _)                                                                                                                                                                              \
    REDUCTION_BARRIER(index0, index1, index2, index3)
#define STORE_REDUCTION_ACCUMULATOR(index, unused) __riscv_vse32_v_f32m2(acc + index * kOutputN, reduction_acc##index, ime::rvv::vl(32, 2));

                REPEAT_8(LOAD_REDUCTION_ACCUMULATOR, _)

                for (size_t reductionIndex = 0; reductionIndex < reductionCount; ++reductionIndex) {
                    const size_t reductionBlock = reductionBlockStart + reductionIndex;
                    const int32_t *component = reduction_components + reductionIndex * kOutputM * kOutputN;
                    IME_L1D_PROBE_VLE16_M1(vfloat16m1_t, bsh, (_Float16 *)(baseWScales + (n / kOutputN) * BlockCountK * kOutputN + reductionBlock * kOutputN), ime::rvv::vl(16, 1),
                                           ime_l1d_probe_m8b_w_scale_0);
                    vfloat32m2_t bs = __riscv_vfwcvt_f_f_v_f32m2(bsh, ime::rvv::vl(16, 1));

                    REDUCE_ROW_QUAD(0, 1, 2, 3)
                    REDUCE_ROW_QUAD(4, 5, 6, 7)
                }

                REPEAT_8(STORE_REDUCTION_ACCUMULATOR, _)

#undef STORE_REDUCTION_ACCUMULATOR
#undef REDUCE_ROW_QUAD
#undef REDUCTION_BARRIER
#undef REDUCE_ROW
#undef LOAD_REDUCTION_ACCUMULATOR
                reductionCount = 0;
            }
        }
        for (size_t row = 0; row < kOutputM; ++row) {
            std::memcpy(rowC + row * ldc, acc + row * kOutputN, kOutputN * sizeof(float));
        }
    }
}
