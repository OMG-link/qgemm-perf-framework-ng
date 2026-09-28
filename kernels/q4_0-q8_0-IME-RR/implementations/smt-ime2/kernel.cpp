#include "kernel.h"

#include <riscv_vector.h>
#include <smt_vector.h>

namespace {

constexpr size_t kRegisterBytes = 128;
constexpr size_t kVl8m1 = 128;
constexpr size_t kVl16m1 = 64;
constexpr size_t kVl32m2 = 64;
constexpr int8_t kQuantizationZeroPoint = -8;
constexpr int kVmadotMode = 3;

inline void clear_i32_accumulators(vint32m2_t &q0, vint32m2_t &q1, vint32m2_t &q2, vint32m2_t &q3) {
    size_t vl;
    asm volatile("vsetvli %4, zero, e32, m2, ta, ma\n\t"
                 "vmv.v.i %0, 0\n\t"
                 "vmv.v.i %1, 0\n\t"
                 "vmv.v.i %2, 0\n\t"
                 "vmv.v.i %3, 0"
                 : "=vr"(q0), "=vr"(q1), "=vr"(q2), "=vr"(q3), "=r"(vl));
}

inline void clear_f32_accumulators(vfloat32m2_t &s0, vfloat32m2_t &s1, vfloat32m2_t &s2, vfloat32m2_t &s3) {
    size_t vl;
    asm volatile("vsetvli %4, zero, e32, m2, ta, ma\n\t"
                 "vmv.v.i %0, 0\n\t"
                 "vmv.v.i %1, 0\n\t"
                 "vmv.v.i %2, 0\n\t"
                 "vmv.v.i %3, 0"
                 : "=vr"(s0), "=vr"(s1), "=vr"(s2), "=vr"(s3), "=r"(vl));
}

} // namespace

void q4_0_q8_0_ime_rr_w_stationary(const int8_t *GGML_RESTRICT a_qs,
                                   const uint16_t *GGML_RESTRICT a_scales,
                                   const uint8_t *GGML_RESTRICT w_qs,
                                   const uint16_t *GGML_RESTRICT w_scales,
                                   float *GGML_RESTRICT packed_c,
                                   size_t m_tile_count) {
    vuint8m1_t packed0 = __riscv_vle8_v_u8m1(w_qs + 0 * kRegisterBytes, kVl8m1);
    vuint8m1_t packed1 = __riscv_vle8_v_u8m1(w_qs + 1 * kRegisterBytes, kVl8m1);
    vuint8m1_t packed2 = __riscv_vle8_v_u8m1(w_qs + 2 * kRegisterBytes, kVl8m1);
    vuint8m1_t packed3 = __riscv_vle8_v_u8m1(w_qs + 3 * kRegisterBytes, kVl8m1);

    vint8m1_t wlo0 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed0, 15, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t wlo1 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed1, 15, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t wlo2 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed2, 15, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t wlo3 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(packed3, 15, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t whi0 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed0, 4, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t whi1 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed1, 4, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t whi2 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed2, 4, kVl8m1)), kQuantizationZeroPoint, kVl8m1);
    vint8m1_t whi3 = __riscv_vadd_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(packed3, 4, kVl8m1)), kQuantizationZeroPoint, kVl8m1);

    vuint16m1_t scale_lanes = __riscv_vid_v_u16m1(kVl16m1);
    vbool16_t scale_mask = __riscv_vmseq_vx_u16m1_b16(__riscv_vand_vx_u16m1(scale_lanes, 7, kVl16m1), 0, kVl16m1);
    vfloat16m1_t zero_scale = __riscv_vfmv_v_f_f16m1(0.0f, kVl16m1);
    const auto *weight_scales = reinterpret_cast<const _Float16 *>(w_scales);
    vfloat16m1_t ws0 = __riscv_vle16_v_f16m1_tumu(scale_mask, zero_scale, weight_scales + 0 * kImeRrScaleMatrixElements, kVl16m1);
    vfloat16m1_t ws1 = __riscv_vle16_v_f16m1_tumu(scale_mask, zero_scale, weight_scales + 1 * kImeRrScaleMatrixElements, kVl16m1);
    vfloat16m1_t ws2 = __riscv_vle16_v_f16m1_tumu(scale_mask, zero_scale, weight_scales + 2 * kImeRrScaleMatrixElements, kVl16m1);
    vfloat16m1_t ws3 = __riscv_vle16_v_f16m1_tumu(scale_mask, zero_scale, weight_scales + 3 * kImeRrScaleMatrixElements, kVl16m1);

    size_t tile_m = 0;
    size_t scale_group = 0;
    while (tile_m < m_tile_count) {
        const auto *group_a_scales = reinterpret_cast<const _Float16 *>(a_scales + scale_group * kImeRrScaleMatrixElements);
        vfloat16m1_t as = __riscv_vle16_v_f16m1(group_a_scales, kVl16m1);
        const size_t remaining = m_tile_count - tile_m;
        const size_t group_tiles = remaining < kImeRrActivationTilesPerScaleLoad ? remaining : kImeRrActivationTilesPerScaleLoad;

#pragma clang loop unroll(disable)
        for (size_t tile_in_group = 0; tile_in_group < group_tiles; ++tile_in_group, ++tile_m) {
            asm volatile("" : : "vr"(wlo0), "vr"(wlo1), "vr"(wlo2), "vr"(wlo3),
                                  "vr"(whi0), "vr"(whi1), "vr"(whi2), "vr"(whi3),
                                  "vr"(ws0), "vr"(ws1), "vr"(ws2), "vr"(ws3));

            const int8_t *tile_a_qs = a_qs + tile_m * kImeRrMr * QK8_0;
            vint8m1_t aq0 = __riscv_vle8_v_i8m1(tile_a_qs + 0 * kRegisterBytes, kVl8m1);
            vint8m1_t aq1 = __riscv_vle8_v_i8m1(tile_a_qs + 1 * kRegisterBytes, kVl8m1);

            vint32m2_t q0, q1, q2, q3;
            clear_i32_accumulators(q0, q1, q2, q3);

            q0 = __riscv_smt_vmadot_i32m2(q0, aq0, wlo0, kVmadotMode, kVl8m1);
            q1 = __riscv_smt_vmadot_i32m2(q1, aq0, wlo1, kVmadotMode, kVl8m1);
            q2 = __riscv_smt_vmadot_i32m2(q2, aq0, wlo2, kVmadotMode, kVl8m1);
            q3 = __riscv_smt_vmadot_i32m2(q3, aq0, wlo3, kVmadotMode, kVl8m1);
            asm volatile("" : "+vr"(q0), "+vr"(q1), "+vr"(q2), "+vr"(q3));
            q0 = __riscv_smt_vmadot_i32m2(q0, aq1, whi0, kVmadotMode, kVl8m1);
            q1 = __riscv_smt_vmadot_i32m2(q1, aq1, whi1, kVmadotMode, kVl8m1);
            q2 = __riscv_smt_vmadot_i32m2(q2, aq1, whi2, kVmadotMode, kVl8m1);
            q3 = __riscv_smt_vmadot_i32m2(q3, aq1, whi3, kVmadotMode, kVl8m1);
            asm volatile("" : "+vr"(q0), "+vr"(q1), "+vr"(q2), "+vr"(q3));

            float *tile_c = packed_c + tile_m * kImeRrPackedCElements;

            vfloat32m2_t s0, s1, s2, s3;
            clear_f32_accumulators(s0, s1, s2, s3);
            s0 = __riscv_smt_vfwmadot(s0, as, ws0, 0, kVl16m1);
            s1 = __riscv_smt_vfwmadot(s1, as, ws1, 0, kVl16m1);
            s2 = __riscv_smt_vfwmadot(s2, as, ws2, 0, kVl16m1);
            s3 = __riscv_smt_vfwmadot(s3, as, ws3, 0, kVl16m1);
            asm volatile("" : "+vr"(s0), "+vr"(s1), "+vr"(s2), "+vr"(s3));

#define REDUCE_GROUP(index)                                                                                                                                                                            \
            {                                                                                                                                                                                              \
                vfloat32m2_t quant = __riscv_vfcvt_f_x_v_f32m2(q##index, kVl32m2);                                                                                                                         \
                vfloat32m2_t c = __riscv_vle32_v_f32m2(tile_c + index * 64, kVl32m2);                                                                                                                      \
                c = __riscv_vfmacc_vv_f32m2(c, quant, s##index, kVl32m2);                                                                                                                                  \
                __riscv_vse32_v_f32m2(tile_c + index * 64, c, kVl32m2);                                                                                                                                    \
            }
            REDUCE_GROUP(0)
            REDUCE_GROUP(1)
            REDUCE_GROUP(2)
            REDUCE_GROUP(3)
#undef REDUCE_GROUP
            as = __riscv_vslidedown_vx_f16m1(as, 1, kVl16m1);
        }
        ++scale_group;
    }
}