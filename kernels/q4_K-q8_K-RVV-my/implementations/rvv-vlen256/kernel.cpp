#include "types.h"
#include <riscv_vector.h>
// #include "riscv_subt.h"

void ggml_gemm_q4_K_8x32_q8_K_my(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, const void *GGML_RESTRICT vy, int m, int n) {
    const int MR = 12;
    const int NR = 32;

    assert(__riscv_vlenb() >= NR);
    assert(k % QK_K == 0);
    assert(m % MR == 0);
    assert(n % NR == 0);

    auto blocks_w = (const block_q4_K_rvv_n32 *)vx;
    auto blocks_a = (const block_q8_K_rvv_m12 *)vy;

#define CURRENT_MR 12
#define REPEAT_MR(fun) _REPEAT_HELPER1(CURRENT_MR, fun)
#define _REPEAT_HELPER1(MR, fun) _REPEAT_HELPER2(MR, fun)
#define _REPEAT_HELPER2(MR, fun) REPEAT_##MR(fun)
#define REPEAT_8(fun) fun(0) fun(1) fun(2) fun(3) fun(4) fun(5) fun(6) fun(7)
#define REPEAT_12(fun) fun(0) fun(1) fun(2) fun(3) fun(4) fun(5) fun(6) fun(7) fun(8) fun(9) fun(10) fun(11)
#define OP_DEF_SUM_SUBBLOCK(i) vint16m2_t sum_subblock##i;
#define OP_ACCUMULATE_SUBBLOCK_FIRST_ROUND(i)                                                                                                                                                          \
    {                                                                                                                                                                                                  \
        int8_t a = block_a.qs[(i_qsk * kQKSubBlock + i_k) * MR + i];                                                                                                                                         \
        sum_subblock##i = __riscv_vwmul_vx_i16m2(w, a, NR);                                                                                                                                            \
    }
#define OP_ACCUMULATE_SUBBLOCK_NEXT_ROUNDS(i)                                                                                                                                                          \
    {                                                                                                                                                                                                  \
        int8_t a = block_a.qs[(i_qsk * kQKSubBlock + i_k) * MR + i];                                                                                                                                         \
        sum_subblock##i = __riscv_vwmacc_vx_i16m2(sum_subblock##i, a, w, NR);                                                                                                                          \
    }
#define OP_ACCUMULATE_BLOCK_FIRST_ROUND(i)                                                                                                                                                             \
    {                                                                                                                                                                                                  \
        vint32m4_t sum_block_row;                                                                                                                                                                      \
        sum_block_row = __riscv_vwmul_vv_i32m4(sum_subblock##i, w_scale, NR);                                                                                                                          \
        __riscv_vse32_v_i32m4(sum_block + NR * i, sum_block_row, NR);                                                                                                                                  \
    }
#define OP_ACCUMULATE_BLOCK_NEXT_ROUNDS(i)                                                                                                                                                             \
    {                                                                                                                                                                                                  \
        vint32m4_t sum_block_row = __riscv_vle32_v_i32m4(sum_block + NR * i, NR);                                                                                                                      \
        sum_block_row = __riscv_vwmacc_vv_i32m4(sum_block_row, sum_subblock##i, w_scale, NR);                                                                                                          \
        __riscv_vse32_v_i32m4(sum_block + NR * i, sum_block_row, NR);                                                                                                                                  \
    }
#define OP_LOAD_MIN_SUBBLOCK(i)                                                                                                                                                                        \
    vint16m2_t w_min##i;                                                                                                                                                                               \
    {                                                                                                                                                                                                  \
        vint8m1_t w_min_i8 = __riscv_vle8_v_i8m1(&block_w.mins[i * NR], NR);                                                                                                                           \
        w_min##i = __riscv_vsext_vf2_i16m2(w_min_i8, NR);                                                                                                                                              \
    }

    assert(MR == CURRENT_MR);

    for (int i_mr = 0; i_mr < m / MR; i_mr++) {
        for (int i_nr = 0; i_nr < n / NR; i_nr++) {
            float sum[MR * NR];
            for (int i_qk = 0; i_qk < k / QK_K; i_qk++) { // K: block
                auto &block_a = blocks_a[i_mr * (k / QK_K) + i_qk];
                auto &block_w = blocks_w[i_nr * (k / QK_K) + i_qk];
                int32_t sum_block[MR * NR] = {0};
                for (int i_qsk = 0; i_qsk < QK_K / kQKSubBlock; i_qsk++) { // K: subblock
                    REPEAT_MR(OP_DEF_SUM_SUBBLOCK)
#pragma GCC unroll 16
                    for (int i_2k = 0; i_2k < kQKSubBlock / 2; i_2k++) { // K: 2 indices
                        vuint8m1_t wx2 = __riscv_vle8_v_u8m1(&block_w.qs[(i_qsk * (kQKSubBlock / 2) + i_2k) * NR], NR);
                        {
                            vint8m1_t w = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(wx2, 0x0F, NR));
                            int i_k = i_2k * 2;
                            if (i_2k == 0) {
                                REPEAT_MR(OP_ACCUMULATE_SUBBLOCK_FIRST_ROUND)
                            } else {
                                REPEAT_MR(OP_ACCUMULATE_SUBBLOCK_NEXT_ROUNDS)
                            }
                        }
                        {
                            int i_k = i_2k * 2 + 1;
                            vint8m1_t w = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(wx2, 4, NR));
                            REPEAT_MR(OP_ACCUMULATE_SUBBLOCK_NEXT_ROUNDS)
                        }
                    }

                    asm volatile(""
                                 :
                                 : "vr"(sum_subblock0), "vr"(sum_subblock1), "vr"(sum_subblock2), "vr"(sum_subblock3), "vr"(sum_subblock4), "vr"(sum_subblock5), "vr"(sum_subblock6),
                                   "vr"(sum_subblock7), "vr"(sum_subblock8), "vr"(sum_subblock9), "vr"(sum_subblock10), "vr"(sum_subblock11));

                    vint8m1_t w_scale_i8 = __riscv_vle8_v_i8m1(&block_w.scales[i_qsk * NR], NR);
                    vint16m2_t w_scale = __riscv_vsext_vf2_i16m2(w_scale_i8, NR);

                    // if (i_qsk == 0) {
                    //     REPEAT_MR(OP_ACCUMULATE_BLOCK_FIRST_ROUND)
                    // } else {
                    REPEAT_MR(OP_ACCUMULATE_BLOCK_NEXT_ROUNDS)
                    // }
                }

                vfloat16m2_t w_d_f16 = __riscv_vle16_v_f16m2(block_w.d, NR);
                vfloat32m4_t w_d = __riscv_vfwcvt_f_f_v_f32m4(w_d_f16, NR);

                for (int i_m = 0; i_m < MR; i_m++) {
                    vint32m4_t sum_block_row_i32 = __riscv_vle32_v_i32m4(sum_block + i_m * NR, NR);
                    vfloat32m4_t sum_block_row = __riscv_vfcvt_f_x_v_f32m4(sum_block_row_i32, NR);
                    vfloat32m4_t d = __riscv_vfmul_vf_f32m4(w_d, block_a.d[i_m], NR);
                    vfloat32m4_t sum_row;
                    if (i_qk == 0) {
                        sum_row = __riscv_vfmul_vv_f32m4(sum_block_row, d, NR);
                    } else {
                        sum_row = __riscv_vle32_v_f32m4(&sum[i_m * NR], NR);
                        sum_row = __riscv_vfmacc_vv_f32m4(sum_row, sum_block_row, d, NR);
                    }
                    __riscv_vse32_v_f32m4(&sum[i_m * NR], sum_row, NR);
                }
            }

            float sum_min[MR * NR];
            for (int64_t i_qk = 0; i_qk < k / QK_K; i_qk++) {
                auto &block_a = blocks_a[i_mr * (k / QK_K) + i_qk];
                auto &block_w = blocks_w[i_nr * (k / QK_K) + i_qk];
                REPEAT_8(OP_LOAD_MIN_SUBBLOCK)
                vfloat16m2_t w_dmin_f16 = __riscv_vle16_v_f16m2(block_w.dmin, NR);
                vfloat32m4_t w_dmin = __riscv_vfwcvt_f_f_v_f32m4(w_dmin_f16, NR);
                for (int i_m = 0; i_m < MR; i_m++) {
                    vint32m4_t sum_min_block_row = __riscv_vwmul_vx_i32m4(w_min0, block_a.bsums[i_m + 0 * MR], NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 1 * MR], w_min1, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 2 * MR], w_min2, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 3 * MR], w_min3, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 4 * MR], w_min4, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 5 * MR], w_min5, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 6 * MR], w_min6, NR);
                    sum_min_block_row = __riscv_vwmacc_vx_i32m4(sum_min_block_row, block_a.bsums[i_m + 7 * MR], w_min7, NR);

                    vfloat32m4_t sum_min_block_row_f32 = __riscv_vfcvt_f_x_v_f32m4(sum_min_block_row, NR);
                    vfloat32m4_t dmin = __riscv_vfmul_vf_f32m4(w_dmin, block_a.d[i_m], NR);
                    vfloat32m4_t sum_min_row;
                    if (i_qk == 0) {
                        sum_min_row = __riscv_vfmul_vv_f32m4(sum_min_block_row_f32, dmin, NR);
                    } else {
                        sum_min_row = __riscv_vle32_v_f32m4(sum_min + i_m * NR, NR);
                        sum_min_row = __riscv_vfmacc_vv_f32m4(sum_min_row, sum_min_block_row_f32, dmin, NR);
                    }
                    __riscv_vse32_v_f32m4(sum_min + i_m * NR, sum_min_row, NR);
                }
            }

            for (int i_m = 0; i_m < MR; i_m++) {
                vfloat32m4_t sum_row = __riscv_vle32_v_f32m4(sum + i_m * NR, NR);
                vfloat32m4_t sum_min_row = __riscv_vle32_v_f32m4(sum_min + i_m * NR, NR);
                vfloat32m4_t s_row = __riscv_vfsub_vv_f32m4(sum_row, sum_min_row, NR);
                __riscv_vse32_v_f32m4(&s[(i_mr * MR + i_m) * bs + i_nr * NR], s_row, NR);
            }
        }
    }
}