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
constexpr size_t kPackedBBytesPerKIter = 4 * kBytesPerMIV;
constexpr size_t kUnpackedBBytesPerKIter = 8 * kBytesPerMIV;
constexpr size_t kPackedBBytesPerBlock = 16 * (QK4_0 / 2);
constexpr size_t kUnpackedBBytesPerBlock = 16 * QK4_0;
constexpr size_t kABytesPerKIter = 4 * kBytesPerMIV;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr size_t kRedBatchSize = 8;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

[[gnu::noinline]] void unpack_q4_0_ime_m8b_cache_blocking_rvv(const uint8_t *GGML_RESTRICT packed_b_qs, int8_t *GGML_RESTRICT unpacked_b_qs, size_t block_count) {
    const size_t vl8 = __riscv_vsetvl_e8m4(kPackedBBytesPerKIter);
    for (size_t block = 0; block < block_count; ++block) {
        const uint8_t *packed_block = packed_b_qs + block * kPackedBBytesPerBlock;
        int8_t *unpacked_block = unpacked_b_qs + block * kUnpackedBBytesPerBlock;
        for (size_t inner = 0; inner < kBlockLength / kStepKPerIter; ++inner) {
            const uint8_t *packed_inner = packed_block + inner * kPackedBBytesPerKIter;
            int8_t *unpacked_inner = unpacked_block + inner * kUnpackedBBytesPerKIter;
            const vuint8m4_t packed = __riscv_vle8_v_u8m4(packed_inner, vl8);
            const vuint8m4_t lo = __riscv_vand_vx_u8m4(packed, 15, vl8);
            const vuint8m4_t hi = __riscv_vsrl_vx_u8m4(packed, 4, vl8);
            __riscv_vse8_v_i8m4(unpacked_inner, __riscv_vadd_vx_i8m4(__riscv_vreinterpret_v_u8m4_i8m4(lo), kQuantizationZeroPoint, vl8), vl8);
            __riscv_vse8_v_i8m4(unpacked_inner + kPackedBBytesPerKIter, __riscv_vadd_vx_i8m4(__riscv_vreinterpret_v_u8m4_i8m4(hi), kQuantizationZeroPoint, vl8), vl8);
        }
    }
}

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack_CacheBlocking(const int8_t *GGML_RESTRICT baseAQs, const float *GGML_RESTRICT baseAScales,
                                                                                           const int8_t *GGML_RESTRICT baseWQs, const uint16_t *GGML_RESTRICT baseWScales, float *GGML_RESTRICT baseC,
                                                                                           size_t BlockCountK, const size_t ldc, bool firstKc) {

    const size_t numKIter = kBlockLength / kStepKPerIter;

    float *rowC = baseC;
    float acc[kOutputM * kOutputN];
    bool firstReductionFlush = true;

    int32_t reduction_components[kOutputM * kOutputN * kRedBatchSize];
    size_t reductionCount = 0;
    for (size_t bk = 0; bk < BlockCountK;) {
        const size_t blockWIndex = bk;

        vint32m2_t inner_acc0 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc1 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc2 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc3 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc4 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc5 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc6 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t inner_acc7 = __riscv_vmv_v_x_i32m2(0, 16);

        asm volatile("" : "+vr"(inner_acc0), "+vr"(inner_acc1), "+vr"(inner_acc2), "+vr"(inner_acc3), "+vr"(inner_acc4), "+vr"(inner_acc5), "+vr"(inner_acc6), "+vr"(inner_acc7));

        // The probe macros emit globally visible symbol ranges, so this loop must
        // stay rolled: unrolling would duplicate every probe label.
#pragma clang loop unroll(disable)
        for (size_t inner = 0; inner < numKIter; ++inner) {

            const size_t vl8 = __riscv_vsetvlmax_e8m1();
            const int8_t *blockWQs = baseWQs + blockWIndex * kUnpackedBBytesPerBlock;
            const int8_t *unpackedInner = blockWQs + inner * kUnpackedBBytesPerKIter;
            IME_L1D_PROBE_VLE8_M8(vint8m8_t, unpackedB, unpackedInner, __riscv_vsetvlmax_e8m8(), ime_l1d_probe_m8b_cb_w_main_0);
            vint8m1_t bLo1i = __riscv_vget_v_i8m8_i8m1(unpackedB, 0);
            vint8m1_t bLo2i = __riscv_vget_v_i8m8_i8m1(unpackedB, 1);
            vint8m1_t bLo3i = __riscv_vget_v_i8m8_i8m1(unpackedB, 2);
            vint8m1_t bLo4i = __riscv_vget_v_i8m8_i8m1(unpackedB, 3);
            vint8m1_t bHi1i = __riscv_vget_v_i8m8_i8m1(unpackedB, 4);
            vint8m1_t bHi2i = __riscv_vget_v_i8m8_i8m1(unpackedB, 5);
            vint8m1_t bHi3i = __riscv_vget_v_i8m8_i8m1(unpackedB, 6);
            vint8m1_t bHi4i = __riscv_vget_v_i8m8_i8m1(unpackedB, 7);

            const int8_t *aInner = baseAQs + bk * kMr * QK8_0 + inner * kABytesPerKIter;
            IME_L1D_PROBE_VLE8_M1(vint8m1_t, A1, aInner, ime_l1d_probe_m8b_cb_a_main_0);
            IME_L1D_PROBE_VLE8_M1(vint8m1_t, A2, aInner + kBytesPerMIV, ime_l1d_probe_m8b_cb_a_main_1);
            IME_L1D_PROBE_VLE8_M1(vint8m1_t, A3, aInner + 2 * kBytesPerMIV, ime_l1d_probe_m8b_cb_a_main_2);
            IME_L1D_PROBE_VLE8_M1(vint8m1_t, A4, aInner + 3 * kBytesPerMIV, ime_l1d_probe_m8b_cb_a_main_3);

            asm volatile("" : : "vr"(bLo1i), "vr"(bLo2i), "vr"(bLo3i), "vr"(bLo4i), "vr"(bHi1i), "vr"(bHi2i), "vr"(bHi3i), "vr"(bHi4i), "vr"(A1), "vr"(A2), "vr"(A3), "vr"(A4));

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, bLo1i, kVmadotMode, kVmadotSignedness);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, bLo2i, kVmadotMode, kVmadotSignedness);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, bLo3i, kVmadotMode, kVmadotSignedness);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, bLo4i, kVmadotMode, kVmadotSignedness);

            inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A3, bLo1i, kVmadotMode, kVmadotSignedness);
            inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A3, bLo2i, kVmadotMode, kVmadotSignedness);
            inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A3, bLo3i, kVmadotMode, kVmadotSignedness);
            inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A3, bLo4i, kVmadotMode, kVmadotSignedness);

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, bHi1i, kVmadotMode, kVmadotSignedness);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, bHi2i, kVmadotMode, kVmadotSignedness);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, bHi3i, kVmadotMode, kVmadotSignedness);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, bHi4i, kVmadotMode, kVmadotSignedness);

            inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A4, bHi1i, kVmadotMode, kVmadotSignedness);
            inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A4, bHi2i, kVmadotMode, kVmadotSignedness);
            inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A4, bHi3i, kVmadotMode, kVmadotSignedness);
            inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A4, bHi4i, kVmadotMode, kVmadotSignedness);

            asm volatile("" : "+vr"(inner_acc0), "+vr"(inner_acc1), "+vr"(inner_acc2), "+vr"(inner_acc3), "+vr"(inner_acc4), "+vr"(inner_acc5), "+vr"(inner_acc6), "+vr"(inner_acc7));
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
            const bool finalReductionFlush = bk == BlockCountK;
#define DECLARE_REDUCTION_ACCUMULATOR(index, unused) vfloat32m2_t reduction_acc##index;
#define ZERO_REDUCTION_ACCUMULATOR(index, unused) reduction_acc##index = __riscv_vfmv_v_f_f32m2(0.0f, ime::rvv::vl(32, 2));
#define LOAD_8_REDUCTION_ACCUMULATORS(address, stride, symbol_prefix)                                                                                                                                  \
    asm volatile(IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 0), "vl2re32.v %0, (%8)") IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 1), "vl2re32.v %1, (%9)")                             \
                     IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 2), "vl2re32.v %2, (%10)") IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 3), "vl2re32.v %3, (%11)")                       \
                         IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 4), "vl2re32.v %4, (%12)") IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 5), "vl2re32.v %5, (%13)")                   \
                             IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 6), "vl2re32.v %6, (%14)") IME_L1D_PROBE_ASM(IME_L1D_PROBE_CAT(symbol_prefix, 7), "vl2re32.v %7, (%15)")               \
                 : "=&vr"(reduction_acc0), "=&vr"(reduction_acc1), "=&vr"(reduction_acc2), "=&vr"(reduction_acc3), "=&vr"(reduction_acc4), "=&vr"(reduction_acc5), "=&vr"(reduction_acc6),             \
                   "=&vr"(reduction_acc7)                                                                                                                                                              \
                 : "r"((address) + 0 * (stride)), "r"((address) + 1 * (stride)), "r"((address) + 2 * (stride)), "r"((address) + 3 * (stride)), "r"((address) + 4 * (stride)),                          \
                   "r"((address) + 5 * (stride)), "r"((address) + 6 * (stride)), "r"((address) + 7 * (stride))                                                                                         \
                 : "memory");
#define REDUCE_ROW(index, unused)                                                                                                                                                                      \
    {                                                                                                                                                                                                  \
        IME_L1D_PROBE_VLE32_M2(vint32m2_t, component_i, component + index * kOutputN, ime::rvv::vl(32, 2), IME_L1D_PROBE_CAT(ime_l1d_probe_m8b_cb_component_, index));                                 \
        vfloat32m2_t component_f = __riscv_vfcvt_f_x_v_f32m2(component_i, ime::rvv::vl(32, 2));                                                                                                        \
        IME_L1D_PROBE_FLW(a_scale, baseAScales + reductionBlock * kMr + index, IME_L1D_PROBE_CAT(ime_l1d_probe_m8b_cb_a_scale_, index));                                                               \
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
#define STORE_REDUCTION_ACCUMULATOR(index, unused)                                                                                                                                                     \
    if (finalReductionFlush)                                                                                                                                                                           \
        __riscv_vse32_v_f32m2(rowC + index * ldc, reduction_acc##index, ime::rvv::vl(32, 2));                                                                                                          \
    else                                                                                                                                                                                               \
        __riscv_vse32_v_f32m2(acc + index * kOutputN, reduction_acc##index, ime::rvv::vl(32, 2));

            REPEAT_8(DECLARE_REDUCTION_ACCUMULATOR, _)
            if (firstReductionFlush) {
                if (firstKc) {
                    REPEAT_8(ZERO_REDUCTION_ACCUMULATOR, _)
                } else {
                    LOAD_8_REDUCTION_ACCUMULATORS(rowC, ldc, ime_l1d_probe_m8b_cb_c_accumulator_)
                }
            } else {
                LOAD_8_REDUCTION_ACCUMULATORS(acc, kOutputN, ime_l1d_probe_m8b_cb_local_accumulator_)
            }

            for (size_t reductionIndex = 0; reductionIndex < reductionCount; ++reductionIndex) {
                const size_t reductionBlock = reductionBlockStart + reductionIndex;
                const int32_t *component = reduction_components + reductionIndex * kOutputM * kOutputN;
                IME_L1D_PROBE_VLE16_M1(vfloat16m1_t, bsh, (_Float16 *)(baseWScales + reductionBlock * kOutputN), ime::rvv::vl(16, 1), ime_l1d_probe_m8b_cb_w_scale_0);
                vfloat32m2_t bs = __riscv_vfwcvt_f_f_v_f32m2(bsh, ime::rvv::vl(16, 1));

                REDUCE_ROW_QUAD(0, 1, 2, 3)
                REDUCE_ROW_QUAD(4, 5, 6, 7)
            }

            REPEAT_8(STORE_REDUCTION_ACCUMULATOR, _)
            firstReductionFlush = false;

#undef STORE_REDUCTION_ACCUMULATOR
#undef REDUCE_ROW_QUAD
#undef REDUCTION_BARRIER
#undef REDUCE_ROW
#undef LOAD_8_REDUCTION_ACCUMULATORS
#undef ZERO_REDUCTION_ACCUMULATOR
#undef DECLARE_REDUCTION_ACCUMULATOR
            reductionCount = 0;
        }
    }
}
