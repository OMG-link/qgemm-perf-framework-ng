#include <cstddef>
#include <riscv_vector.h>
#include <smt_vector.h>

#include "kernel.h"

#define REPEAT_8(operation, ...)                                                                                                                                                                       \
    operation(0, __VA_ARGS__) operation(1, __VA_ARGS__) operation(2, __VA_ARGS__) operation(3, __VA_ARGS__) operation(4, __VA_ARGS__) operation(5, __VA_ARGS__) operation(6, __VA_ARGS__)               \
        operation(7, __VA_ARGS__)
#define REPEAT_16(operation, ...)                                                                                                                                                                      \
    REPEAT_8(operation, __VA_ARGS__) operation(8, __VA_ARGS__) operation(9, __VA_ARGS__) operation(10, __VA_ARGS__) operation(11, __VA_ARGS__) operation(12, __VA_ARGS__) operation(13, __VA_ARGS__)    \
        operation(14, __VA_ARGS__) operation(15, __VA_ARGS__)

constexpr size_t kMr = 16;
constexpr size_t kNr = 32;
constexpr size_t kOutputM = kMr;
constexpr size_t kOutputN = kNr;
constexpr size_t kBlockLength = QK8_0;
// One 1024-bit vector register. The A60 kernel calls this 32.
constexpr size_t kBytesPerMIV = 128;
// One packed B register covers K k in its low nibble and K k + 16 in its high
// nibble, so a single iteration spans the whole QK8_0 block and numKIter is 1.
// The A60 kernel needs two iterations of 16.
constexpr size_t kStepKPerIter = 32;
constexpr size_t kBBytesPerKIter = 4 * kBytesPerMIV;
constexpr size_t kABytesPerKIter = 4 * kBytesPerMIV;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr size_t kRedBatchSize = 8;
constexpr int kVmadotMode = 3;
// The trailing intrinsic operand is vl. The matrix unit ignores it, but it
// still drives the vsetvli in front of the dot.
constexpr size_t kVmadotVl = 128;

// VLEN is fixed at 1024 bit on the A100, so every vl is a compile-time
// constant. kernels/common/rvv_vl.h hardcodes 256 and is unusable here.
constexpr size_t kVl32m2 = 64;   // one int32 accumulator, an 8 x 8 output block
constexpr size_t kVl32m1 = 32;   // one output row of kOutputN int32 or float
constexpr size_t kVl16mf2 = 32;  // kNr fp16 column scales

void SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_IME2_BatchRed(const int8_t *GGML_RESTRICT baseAQs, const float *GGML_RESTRICT baseAScales, const uint8_t *GGML_RESTRICT baseWQs,
                                                         const uint16_t *GGML_RESTRICT baseWScales, float *GGML_RESTRICT acc, size_t BlockCountK,
                                                         int32_t *GGML_RESTRICT reduction_components) {

    static_assert(kIme2ReductionComponentCount == kOutputM * kOutputN * kRedBatchSize);
    const size_t numKIter = kBlockLength / kStepKPerIter;
    size_t reductionCount = 0;
    for (size_t bk = 0; bk < BlockCountK;) {
        const int8_t *blockAQs = baseAQs + bk * kMr * QK8_0;

        vint32m2_t inner_acc0 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc1 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc2 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc3 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc4 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc5 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc6 = __riscv_vmv_v_x_i32m2(0, kVl32m2);
        vint32m2_t inner_acc7 = __riscv_vmv_v_x_i32m2(0, kVl32m2);

#pragma clang loop unroll(disable)
        for (size_t inner = 0; inner < numKIter; ++inner) {

            size_t vl8 = __riscv_vsetvlmax_e8m1();
            const uint8_t *blockWQs = baseWQs + bk * kNr * (QK4_0 / 2);
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

            const int8_t *aInner = blockAQs + inner * kABytesPerKIter;
            vint8m1_t A1 = __riscv_vle8_v_i8m1(aInner, vl8);
            vint8m1_t A2 = __riscv_vle8_v_i8m1(aInner + kBytesPerMIV, vl8);
            vint8m1_t A3 = __riscv_vle8_v_i8m1(aInner + 2 * kBytesPerMIV, vl8);
            vint8m1_t A4 = __riscv_vle8_v_i8m1(aInner + 3 * kBytesPerMIV, vl8);

            asm volatile("" : : "vr"(bLo1i), "vr"(bLo2i), "vr"(bLo3i), "vr"(bLo4i), "vr"(bHi1i), "vr"(bHi2i), "vr"(bHi3i), "vr"(bHi4i), "vr"(A1), "vr"(A2), "vr"(A3), "vr"(A4));

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A1, bLo1i, kVmadotMode, kVmadotVl);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A1, bLo2i, kVmadotMode, kVmadotVl);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A1, bLo3i, kVmadotMode, kVmadotVl);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A1, bLo4i, kVmadotMode, kVmadotVl);

            inner_acc0 = __riscv_smt_vmadot_i32m2(inner_acc0, A2, bHi1i, kVmadotMode, kVmadotVl);
            inner_acc1 = __riscv_smt_vmadot_i32m2(inner_acc1, A2, bHi2i, kVmadotMode, kVmadotVl);
            inner_acc2 = __riscv_smt_vmadot_i32m2(inner_acc2, A2, bHi3i, kVmadotMode, kVmadotVl);
            inner_acc3 = __riscv_smt_vmadot_i32m2(inner_acc3, A2, bHi4i, kVmadotMode, kVmadotVl);

            inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A3, bLo1i, kVmadotMode, kVmadotVl);
            inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A3, bLo2i, kVmadotMode, kVmadotVl);
            inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A3, bLo3i, kVmadotMode, kVmadotVl);
            inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A3, bLo4i, kVmadotMode, kVmadotVl);

            inner_acc4 = __riscv_smt_vmadot_i32m2(inner_acc4, A4, bHi1i, kVmadotMode, kVmadotVl);
            inner_acc5 = __riscv_smt_vmadot_i32m2(inner_acc5, A4, bHi2i, kVmadotMode, kVmadotVl);
            inner_acc6 = __riscv_smt_vmadot_i32m2(inner_acc6, A4, bHi3i, kVmadotMode, kVmadotVl);
            inner_acc7 = __riscv_smt_vmadot_i32m2(inner_acc7, A4, bHi4i, kVmadotMode, kVmadotVl);
        }

        // Scheduling boundary between the dots and the writeback. Without it the
        // scheduler hoists each accumulator's chunk-0 store up next to that
        // accumulator's last dot, so the two phases interleave and no address
        // range separates them. Keeping all eight accumulators live here pins
        // every store below the last dot.
#define KEEP_ACCUMULATORS_LIVE(index, unused) , "vr"(inner_acc##index)
        asm volatile("" : : "i"(0)REPEAT_8(KEEP_ACCUMULATORS_LIVE, _));
#undef KEEP_ACCUMULATORS_LIVE

        int32_t *component = reduction_components + reductionCount * kOutputM * kOutputN;

        // Accumulator 4r + g holds an 8 x 8 block at lane m * 8 + n, and it has
        // to land in a row-major kOutputM x kOutputN tile. Each i32m1 half of
        // the accumulator is 32 lanes, i.e. four output rows of 8. Chunk c of a
        // half is lanes 8c..8c+7 and must go to row c, so the store base is
        // pulled back by 8c elements and the mask plus vl select exactly that
        // lane window: chunk 0 needs no mask (an mf4 store is exactly 8 lanes),
        // chunks 1..3 are masked m1 stores. The A60 kernel does the same thing
        // with one mf2 store plus one masked m1 store, because there a half is
        // only two rows of 4.
        size_t vl_m1 = __riscv_vsetvlmax_e32m1();
        vuint32m1_t idx = __riscv_vid_v_u32m1(vl_m1);
        vbool32_t mask32_8 = __riscv_vmsgtu_vx_u32m1_b32(idx, 7, vl_m1);
        vbool32_t mask32_16 = __riscv_vmsgtu_vx_u32m1_b32(idx, 15, vl_m1);
        vbool32_t mask32_24 = __riscv_vmsgtu_vx_u32m1_b32(idx, 23, vl_m1);
        asm volatile("" : : "vr"(mask32_8), "vr"(mask32_16), "vr"(mask32_24));

        // One chunk is 8 elements. e32 cannot go below mf2 (LMUL >= SEW/ELEN),
        // and mf2 is 16 lanes at VLEN 1024, so chunk 0 is an mf2 store with an
        // explicit vl of 8 rather than a vlmax store as on the A60.
        size_t vl_chunk = 8;

#define SPLIT_ACCUMULATOR(index, unused)                                                                                                                                                               \
    vint32m1_t acc_low##index = __riscv_vget_v_i32m2_i32m1(inner_acc##index, 0);                                                                                                                       \
    vint32m1_t acc_high##index = __riscv_vget_v_i32m2_i32m1(inner_acc##index, 1);
#define TILE_BASE(index) ((index / 4) * 8 * kOutputN + (index % 4) * 8)
#define STORE_LOW(index, unused)                                                                                                                                                                       \
    __riscv_vse32_v_i32mf2(component + TILE_BASE(index), __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_low##index), vl_chunk);                                                                                 \
    __riscv_vse32_v_i32mf2(component + TILE_BASE(index) + 4 * kOutputN, __riscv_vlmul_trunc_v_i32m1_i32mf2(acc_high##index), vl_chunk);
#define STORE_HIGH(index, chunk, mask, vl)                                                                                                                                                             \
    __riscv_vse32_v_i32m1_m(mask, component + TILE_BASE(index) + chunk * (kOutputN - 8), acc_low##index, vl);                                                                                          \
    __riscv_vse32_v_i32m1_m(mask, component + TILE_BASE(index) + 4 * kOutputN + chunk * (kOutputN - 8), acc_high##index, vl);
#define KEEP_LIVE(index, unused) , "vr"(acc_low##index), "vr"(acc_high##index)

        REPEAT_8(SPLIT_ACCUMULATOR, _)
        REPEAT_8(STORE_LOW, _)

        asm volatile("" : : "r"(component)REPEAT_8(KEEP_LIVE, _));

        REPEAT_8(STORE_HIGH, 1, mask32_8, 16)
        REPEAT_8(STORE_HIGH, 2, mask32_16, 24)
        REPEAT_8(STORE_HIGH, 3, mask32_24, 32)

#undef KEEP_LIVE
#undef STORE_HIGH
#undef STORE_LOW
#undef TILE_BASE
#undef SPLIT_ACCUMULATOR

        bk++;
        reductionCount++;
        if (reductionCount == kRedBatchSize or bk == BlockCountK) {
            const size_t reductionBlockStart = bk - reductionCount;
            // One output row is kOutputN = 32 floats, which is exactly one
            // f32m1 register at VLEN 1024, so there are kOutputM = 16 of them.
            // The A60 kernel holds 8 rows of 16 floats in f32m2; both are 16
            // vector registers.
#define LOAD_REDUCTION_ACCUMULATOR(index, unused) vfloat32m1_t reduction_acc##index = __riscv_vle32_v_f32m1(acc + index * kOutputN, kVl32m1);
#define REDUCE_ROW(index, unused)                                                                                                                                                                      \
    {                                                                                                                                                                                                  \
        vint32m1_t component_i = __riscv_vle32_v_i32m1(component + index * kOutputN, kVl32m1);                                                                                                          \
        vfloat32m1_t component_f = __riscv_vfcvt_f_x_v_f32m1(component_i, kVl32m1);                                                                                                                     \
        vfloat32m1_t abscale = __riscv_vfmul_vf_f32m1(bs, baseAScales[reductionBlock * kMr + index], kVl32m1);                                                                                          \
        reduction_acc##index = __riscv_vfmacc_vv_f32m1(reduction_acc##index, component_f, abscale, kVl32m1);                                                                                            \
    }
#define REDUCTION_BARRIER(index0, index1, index2, index3)                                                                                                                                              \
    asm volatile("" : "+vr"(reduction_acc##index0), "+vr"(reduction_acc##index1), "+vr"(reduction_acc##index2), "+vr"(reduction_acc##index3) : : "memory");
#define REDUCE_ROW_QUAD(index0, index1, index2, index3)                                                                                                                                                \
    REDUCE_ROW(index0, _)                                                                                                                                                                              \
    REDUCE_ROW(index1, _)                                                                                                                                                                              \
    REDUCE_ROW(index2, _)                                                                                                                                                                              \
    REDUCE_ROW(index3, _)                                                                                                                                                                              \
    REDUCTION_BARRIER(index0, index1, index2, index3)
#define STORE_REDUCTION_ACCUMULATOR(index, unused) __riscv_vse32_v_f32m1(acc + index * kOutputN, reduction_acc##index, kVl32m1);

            REPEAT_16(LOAD_REDUCTION_ACCUMULATOR, _)

            for (size_t reductionIndex = 0; reductionIndex < reductionCount; ++reductionIndex) {
                const size_t reductionBlock = reductionBlockStart + reductionIndex;
                const int32_t *component = reduction_components + reductionIndex * kOutputM * kOutputN;
                vfloat16mf2_t bsh = __riscv_vle16_v_f16mf2((const _Float16 *)(baseWScales + reductionBlock * kOutputN), kVl16mf2);
                vfloat32m1_t bs = __riscv_vfwcvt_f_f_v_f32m1(bsh, kVl16mf2);

                REDUCE_ROW_QUAD(0, 1, 2, 3)
                REDUCE_ROW_QUAD(4, 5, 6, 7)
                REDUCE_ROW_QUAD(8, 9, 10, 11)
                REDUCE_ROW_QUAD(12, 13, 14, 15)
            }

            REPEAT_16(STORE_REDUCTION_ACCUMULATOR, _)

#undef STORE_REDUCTION_ACCUMULATOR
#undef REDUCE_ROW_QUAD
#undef REDUCTION_BARRIER
#undef REDUCE_ROW
#undef LOAD_REDUCTION_ACCUMULATOR
            reductionCount = 0;
        }
    }
}
