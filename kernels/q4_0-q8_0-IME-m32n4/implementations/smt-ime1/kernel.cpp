#include "kernel.h"

#include <riscv_vector.h>
#include <smt_vector.h>

namespace {

constexpr size_t kMr = 32;
constexpr size_t kNr = 4;
constexpr size_t kBytesPerVector = 32;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr int kVmadotMode = 3;
constexpr int kVmadotSignedness = 0;

} // namespace

void q4_0_q8_0_ime_m32n4(const block_q8_0_ime_m32 *a, const block_q4_0_ime_n4 *b, float *c, size_t block_count_k, size_t ldc) {
    int32_t components[kMr * kNr * 8];

    for (size_t block = 0; block < block_count_k; ++block) {
        vint32m2_t accumulator0 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator1 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator2 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator3 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator4 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator5 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator6 = __riscv_vmv_v_x_i32m2(0, 16);
        vint32m2_t accumulator7 = __riscv_vmv_v_x_i32m2(0, 16);

        for (size_t inner = 0; inner < 2; ++inner) {
            const size_t vl8 = __riscv_vsetvlmax_e8m1();
            const vuint8m1_t packed = __riscv_vle8_v_u8m1(b[block].qs + inner * kBytesPerVector, vl8);
            const vuint8m1_t low = __riscv_vand_vx_u8m1(packed, 15, vl8);
            const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl8);
            const vint8m1_t weights_low = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(low), kQuantizationZeroPoint, vl8);
            const vint8m1_t weights_high = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(high), kQuantizationZeroPoint, vl8);

#define ACCUMULATE_GROUP(index)                                                                                                                                                                        \
    {                                                                                                                                                                                                  \
        const int8_t *activation = a[block].qs + index * 4 * QK8_0 + inner * 2 * kBytesPerVector;                                                                                                      \
        const vint8m1_t activation_low = __riscv_vle8_v_i8m1(activation, vl8);                                                                                                                         \
        const vint8m1_t activation_high = __riscv_vle8_v_i8m1(activation + kBytesPerVector, vl8);                                                                                                      \
        accumulator##index = __riscv_smt_vmadot_i32m2(accumulator##index, activation_low, weights_low, kVmadotMode, kVmadotSignedness);                                                                \
        accumulator##index = __riscv_smt_vmadot_i32m2(accumulator##index, activation_high, weights_high, kVmadotMode, kVmadotSignedness);                                                              \
    }
            ACCUMULATE_GROUP(0)
            ACCUMULATE_GROUP(1)
            ACCUMULATE_GROUP(2)
            ACCUMULATE_GROUP(3)
            ACCUMULATE_GROUP(4)
            ACCUMULATE_GROUP(5)
            ACCUMULATE_GROUP(6)
            ACCUMULATE_GROUP(7)
#undef ACCUMULATE_GROUP
        }

        int32_t *block_components = components + block * kMr * kNr;
#define STORE_GROUP(index) __riscv_vse32_v_i32m2(block_components + index * 16, accumulator##index, 16)
        STORE_GROUP(0);
        STORE_GROUP(1);
        STORE_GROUP(2);
        STORE_GROUP(3);
        STORE_GROUP(4);
        STORE_GROUP(5);
        STORE_GROUP(6);
        STORE_GROUP(7);
#undef STORE_GROUP
    }

    const size_t vl = __riscv_vsetvl_e32m1(kNr);
    for (size_t row = 0; row < kMr; ++row) {
        vfloat32m1_t output = __riscv_vle32_v_f32m1(c + row * ldc, vl);
        for (size_t block = 0; block < block_count_k; ++block) {
            const vfloat16mf2_t weight_scale = __riscv_vle16_v_f16mf2(b[block].d, vl);
            const vint32m1_t component_i32 = __riscv_vle32_v_i32m1(components + block * kMr * kNr + row * kNr, vl);
            const vfloat32m1_t component = __riscv_vfcvt_f_x_v_f32m1(component_i32, vl);
            const vfloat32m1_t scale = __riscv_vfwmul_vf_f32m1(weight_scale, a[block].d[row], vl);
            output = __riscv_vfmacc_vv_f32m1(output, component, scale, vl);
        }
        __riscv_vse32_v_f32m1(c + row * ldc, output, vl);
    }
}
