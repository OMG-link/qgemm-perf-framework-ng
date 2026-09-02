// Derived from llama.cpp commit 314e729347defd9851e857f78084160c5786a7d8,
// ggml/src/ggml-cpu/arch/riscv/quants.c. The RVV vl256 algorithm is kept intact.
#include <cassert>
#include <cstdint>
#include <cstring>
#include <riscv_vector.h>

#include "ggml.h"
#include "tables.h"

#define NOINLINE __attribute__((__noinline__))

static NOINLINE void ggml_vec_dot_iq2_xxs_q8_K_vl256(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const auto * GGML_RESTRICT x = static_cast<const block_iq2_xxs *>(vx);
    const auto * GGML_RESTRICT y = static_cast<const block_q8_K *>(vy);

    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;
    const uint64_t * grid64  = (const uint64_t *)iq2xxs_grid;

    uint32_t shift_constants[4] = {0, 7, 14, 21};
    vuint32mf2_t v_shifts = __riscv_vle32_v_u32mf2(shift_constants, 4);

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const float combined_scale = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t  * GGML_RESTRICT q2_ptr = (const uint8_t *) x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        float sum = 0.0f;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8, 32); q8 += 32;
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8, 32); q8 += 32;

            vuint8mf8_t v_raw_q2_1 = __riscv_vle8_v_u8mf8(q2_ptr, 4);
            vuint8mf8_t v_raw_q2_2 = __riscv_vle8_v_u8mf8(q2_ptr + 8, 4);

            vuint16mf4_t vidx_q2_1 = __riscv_vwcvtu_x_x_v_u16mf4(v_raw_q2_1, 4);
            vuint16mf4_t vidx_q2_2 = __riscv_vwcvtu_x_x_v_u16mf4(v_raw_q2_2, 4);

            vidx_q2_1 = __riscv_vsll_vx_u16mf4(vidx_q2_1, 3, 4);
            vidx_q2_2 = __riscv_vsll_vx_u16mf4(vidx_q2_2, 3, 4);

            uint32_t s_packed_1, s_packed_2;
            memcpy(&s_packed_1, q2_ptr + 4, 4);
            memcpy(&s_packed_2, q2_ptr + 12, 4);

            vuint32mf2_t v_s_1 = __riscv_vmv_v_x_u32mf2(s_packed_1, 4);
            vuint32mf2_t v_s_2 = __riscv_vmv_v_x_u32mf2(s_packed_2, 4);

            v_s_1 = __riscv_vsrl_vv_u32mf2(v_s_1, v_shifts, 4);
            v_s_2 = __riscv_vsrl_vv_u32mf2(v_s_2, v_shifts, 4);

            v_s_1 = __riscv_vand_vx_u32mf2(v_s_1, 127, 4);
            v_s_2 = __riscv_vand_vx_u32mf2(v_s_2, 127, 4);

            // Narrow u32 -> u16 (vncvt) and Scale by 8 to get byte offsets
            vuint16mf4_t vidx_s2_1 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(v_s_1, 4), 3, 4);
            vuint16mf4_t vidx_s2_2 = __riscv_vsll_vx_u16mf4(__riscv_vncvt_x_x_w_u16mf4(v_s_2, 4), 3, 4);

            // Load q2 values from lookup grid
            vuint64m1_t vq2_64_1 = __riscv_vluxei16_v_u64m1(grid64, vidx_q2_1, 4);
            vuint64m1_t vq2_64_2 = __riscv_vluxei16_v_u64m1(grid64, vidx_q2_2, 4);
            vint8m1_t q2_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vq2_64_1));
            vint8m1_t q2_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vq2_64_2));

            // Load sign values
            vuint64m1_t vs2_64_1 = __riscv_vluxei16_v_u64m1(signs64, vidx_s2_1, 4);
            vuint64m1_t vs2_64_2 = __riscv_vluxei16_v_u64m1(signs64, vidx_s2_2, 4);
            vint8m1_t s2_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vs2_64_1));
            vint8m1_t s2_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(vs2_64_2));

            // Apply signs to q8
            vint8m1_t q8s_1 = __riscv_vmul_vv_i8m1(q8_1, s2_1, 32);
            vint8m1_t q8s_2 = __riscv_vmul_vv_i8m1(q8_2, s2_2, 32);

            // multiplying q2 with q8
            vint16m2_t dot1 = __riscv_vwmul_vv_i16m2(q8s_1, q2_1, 32);
            vint16m2_t dot2 = __riscv_vwmul_vv_i16m2(q8s_2, q2_2, 32);

            vint32m1_t zero_vec = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t sumv1 = __riscv_vwredsum_vs_i16m2_i32m1(dot1, zero_vec, 32);
            vint32m1_t sumv2 = __riscv_vwredsum_vs_i16m2_i32m1(dot2, zero_vec, 32);
            int32_t scalar_sum1 = __riscv_vmv_x_s_i32m1_i32(sumv1);
            int32_t scalar_sum2 = __riscv_vmv_x_s_i32m1_i32(sumv2);
            int16_t scale1 = 2 * ((s_packed_1 >> 28) & 0xF) + 1;
            int16_t scale2 = 2 * ((s_packed_2 >> 28) & 0xF) + 1;

            sum += scalar_sum1 * scale1 + scalar_sum2 * scale2;
            q2_ptr += 16;
        }
        sumf += sum * combined_scale;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                                const void * GGML_RESTRICT vx, size_t bx,
                                const void * GGML_RESTRICT vy, size_t by, int nrc) {
    // This framework targets SpacemiT X60 with a fixed 256-bit VLEN.
    assert(__riscv_vlenb() * 8 == 256);
    ggml_vec_dot_iq2_xxs_q8_K_vl256(n, s, bs, vx, bx, vy, by, nrc);
}
