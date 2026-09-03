#include "gemm.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

#include "ggml_min.h"

#ifdef USE_TCM
#include "fine_tcm_mm.h"
#endif

extern int64_t selected_cycles;
[[maybe_unused]] static inline uint64_t rdcycle() {
    uint64_t cycle;
    asm volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}
[[maybe_unused]] static int64_t selected_cycles_start;
[[maybe_unused]] static inline void start_select() { selected_cycles_start = rdcycle(); }
[[maybe_unused]] static inline void end_select() { selected_cycles += rdcycle() - selected_cycles_start; }

namespace gemm::link {

namespace outer {

constexpr int QK_SK = 32;

constexpr int NR = 32;
constexpr int KC = 16384;

#define decompress_iq2_subblock(w_q, w_scale, w_ptr)                                                                                                                                                   \
    vint8m1_t w_q;                                                                                                                                                                                     \
    int8_t w_scale;                                                                                                                                                                                    \
    {                                                                                                                                                                                                  \
        assert(__riscv_vlenb() * 8 >= QK_SK * 8);                                                                                                                                                      \
        uint32_t shift_constants[4] = {0, 7, 14, 21};                                                                                                                                                  \
        vuint32mf2_t v_shifts = __riscv_vle32_v_u32mf2(shift_constants, 4);                                                                                                                            \
        vuint8mf8_t w_raw_lo = __riscv_vle8_v_u8mf8(reinterpret_cast<const uint8_t *>(w_ptr), 4);                                                                                                      \
        vuint16mf4_t w_q_offset = __riscv_vwmulu_vx_u16mf4(w_raw_lo, 8, 4);                                                                                                                            \
        uint32_t w_raw_hi = *reinterpret_cast<const uint32_t *>(reinterpret_cast<const uint8_t *>(w_ptr) + 4);                                                                                         \
        vuint32mf2_t w_s_offset = __riscv_vmv_v_x_u32mf2(w_raw_hi, 4);                                                                                                                                 \
        w_s_offset = __riscv_vsrl_vv_u32mf2(w_s_offset, v_shifts, 4);                                                                                                                                  \
        w_s_offset = __riscv_vand_vx_u32mf2(w_s_offset, 127, 4);                                                                                                                                       \
        vuint64m1_t w_q_u64 = __riscv_vluxei16_v_u64m1(iq2xxs_grid, w_q_offset, 4);                                                                                                                    \
        vint8m1_t w_q_i8 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u64m1_u8m1(w_q_u64));                                                                                               \
        vuint8mf8_t w_s_u8 = __riscv_vluxei32_v_u8mf8(ksigns_iq2xs, w_s_offset, 4);                                                                                                                    \
        vbool8_t w_s_b32 = __riscv_vreinterpret_v_u8m1_b8(__riscv_vlmul_ext_v_u8mf8_u8m1(w_s_u8));                                                                                                     \
        vint8m1_t w_qn_i8 = __riscv_vneg_v_i8m1(w_q_i8, 32);                                                                                                                                           \
        w_q = __riscv_vmerge_vvm_i8m1(w_q_i8, w_qn_i8, w_s_b32, 32);                                                                                                                                   \
        w_scale = 2 * ((w_raw_hi >> 28) & 0xF) + 1;                                                                                                                                                    \
    }

/**
 * in:      [NR, KC/QK_K] x [QK_K]
 * w_qs:    [KC/QK_K] x [QK_K, NR]
 * w_scale: [KC/QK_K] x [QK_K/QK_SK, NR]
 * w_d:     [KC/QK_K] x [NR]
 */
static void decompress_iq2(const block_iq2_xxs *in, size_t bin, int8_t *w_qs, int16_t *w_scale, float32_t *w_d, int k) {
    for (int i_qk = 0; i_qk < k / QK_K; i_qk++) {
        for (int i_qsk = 0; i_qsk < QK_K / QK_SK; i_qsk++) {
            for (int i_t = 0; i_t < NR / 4; i_t++) {
                // Decompress subblock
                decompress_iq2_subblock(w_q_0, w_scale_0, &in[(i_t * 4 + 0) * bin + i_qk].qs[i_qsk * 4]);
                decompress_iq2_subblock(w_q_1, w_scale_1, &in[(i_t * 4 + 1) * bin + i_qk].qs[i_qsk * 4]);
                decompress_iq2_subblock(w_q_2, w_scale_2, &in[(i_t * 4 + 2) * bin + i_qk].qs[i_qsk * 4]);
                decompress_iq2_subblock(w_q_3, w_scale_3, &in[(i_t * 4 + 3) * bin + i_qk].qs[i_qsk * 4]);
                // Writeback
                vint8m1x4_t w_qx4 = __riscv_vcreate_v_i8m1x4(w_q_0, w_q_1, w_q_2, w_q_3);
                __riscv_vssseg4e8_v_i8m1x4(&w_qs[(i_qk * QK_K + i_qsk * QK_SK) * NR + i_t * 4], NR, w_qx4, QK_SK);
                w_scale[(i_qk * (QK_K / QK_SK) + i_qsk) * NR + i_t * 4 + 0] = w_scale_0;
                w_scale[(i_qk * (QK_K / QK_SK) + i_qsk) * NR + i_t * 4 + 1] = w_scale_1;
                w_scale[(i_qk * (QK_K / QK_SK) + i_qsk) * NR + i_t * 4 + 2] = w_scale_2;
                w_scale[(i_qk * (QK_K / QK_SK) + i_qsk) * NR + i_t * 4 + 3] = w_scale_3;
            }
        }
        for (int i_n = 0; i_n < NR; i_n++) {
            w_d[i_qk * NR + i_n] = in[i_n * bin + i_qk].d;
        }
    }
}

static constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

/**
 * in: [M, K/QK] x [QK]
 * out: [M/MR, K/QK] x [QK, MR]
 */
void pack_a_iq2_xxs_br(int m, int k, const block_q8_K *in, block_q8_Kx<MR> *out) {
    for (int i_m = 0; i_m < m; i_m++) {
        for (int i_qk = 0; i_qk < k / QK_K; i_qk++) {
            auto block_in = &in[i_m * k / QK_K + i_qk];
            auto block_out = &out[i_m / MR * k / QK_K + i_qk];
            block_out->d[i_m % MR] = block_in->d;
            for (int i_k = 0; i_k < QK_K; i_k++) {
                block_out->qs[i_k * MR + i_m % MR] = block_in->qs[i_k];
            }
        }
    }
}

/**
 * vx(W): [N, K/QK_K] x [QK_K]
 * vy(A): [M/MR, K/QK_K] x [QK_K, MR]
 */
void ggml_gemm_iq2_xxs_q8_K_br(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx, const void *GGML_RESTRICT vy, size_t by, int m, int n) {
    assert(__riscv_vlenb() == NR);
    assert(k % QK_K == 0);
    assert(n % NR == 0);
    assert(m % MR == 0);

    auto w_raw = reinterpret_cast<const block_iq2_xxs *>(vx);
    auto a_raw = reinterpret_cast<const block_q8_Kx<MR> *>(vy);

// Macros to avoid duplicated code when unrolling MR
#define CURRENT_MR 12
#define REPEAT_MR(fun) _REPEAT_HELPER1(CURRENT_MR, fun)
#define _REPEAT_HELPER1(MR, fun) _REPEAT_HELPER2(MR, fun)
#define _REPEAT_HELPER2(MR, fun) REPEAT_##MR(fun)
#define REPEAT_12(fun) fun(0) fun(1) fun(2) fun(3) fun(4) fun(5) fun(6) fun(7) fun(8) fun(9) fun(10) fun(11)
#define OP_DEF_SUM(i) vint16m2_t sum_subblock##i;
#define OP_ACCUMULATE_SUBBLOCK_FIRST_ROUND(i)                                                                                                                                                          \
    {                                                                                                                                                                                                  \
        int8_t a = block_a.qs[(i_qsk * QK_SK + i_k) * MR + i];                                                                                                                                         \
        sum_subblock##i = __riscv_vwmul_vx_i16m2(w, a, NR);                                                                                                                                            \
    }
#define OP_ACCUMULATE_SUBBLOCK_NEXT_ROUNDS(i)                                                                                                                                                          \
    {                                                                                                                                                                                                  \
        int8_t a = block_a.qs[(i_qsk * QK_SK + i_k) * MR + i];                                                                                                                                         \
        sum_subblock##i = __riscv_vwmacc_vx_i16m2(sum_subblock##i, a, w, NR);                                                                                                                          \
    }
#define OP_SAVE_SUBBLOCK(i) __riscv_vse16_v_i16m2(&sum_subblock[(i * (QK_K / QK_SK) + i_qsk) * NR], sum_subblock##i, NR);
#define OP_ACCUMULATE_BLOCK_FIRST_ROUND(i)                                                                                                                                                             \
    {                                                                                                                                                                                                  \
        vint32m4_t sum_block_row;                                                                                                                                                                      \
        sum_block_row = __riscv_vwmul_vv_i32m4(sum_subblock##i, w_scale, NR);                                                                                                                          \
        __riscv_vse32_v_i32m4(&sum_block[i * NR], sum_block_row, NR);                                                                                                                                  \
    }
#define OP_ACCUMULATE_BLOCK_NEXT_ROUNDS(i)                                                                                                                                                             \
    {                                                                                                                                                                                                  \
        vint32m4_t sum_block_row = __riscv_vle32_v_i32m4(&sum_block[i * NR], NR);                                                                                                                      \
        sum_block_row = __riscv_vwmacc_vv_i32m4(sum_block_row, sum_subblock##i, w_scale, NR);                                                                                                          \
        __riscv_vse32_v_i32m4(&sum_block[i * NR], sum_block_row, NR);                                                                                                                                  \
    }
    assert(CURRENT_MR == MR);

    int8_t w_qs[KC * NR];
#ifdef USE_TCM
    auto w_scale = (int16_t *)fine_mm_malloc((KC / QK_SK) * NR * sizeof(int16_t));
    auto w_d = (float32_t *)fine_mm_malloc((KC / QK_K) * NR * sizeof(float32_t));
    auto sum = (float32_t *)fine_mm_malloc(MR * NR * sizeof(float32_t));
    auto sum_block = (int32_t *)fine_mm_malloc(MR * NR * sizeof(int32_t));
    auto sum_subblock = (int16_t *)fine_mm_malloc(MR * (QK_K / QK_SK) * NR * sizeof(int16_t));
#else
    int16_t w_scale[(KC / QK_SK) * NR];
    float32_t w_d[(KC / QK_K) * NR];
    float32_t sum[MR * NR];
    int32_t sum_block[MR * NR];
    int16_t sum_subblock[MR * (QK_K / QK_SK) * NR];
#endif

    for (int i_nr = 0; i_nr < n / NR; i_nr++) {
        for (int i_kc = 0; i_kc < div_ceil(k, KC); i_kc++) {
            int n_kc = std::min(k - i_kc * KC, KC);
            decompress_iq2(&w_raw[(i_nr * NR) * bx + i_kc * (KC / QK_K)], bx, w_qs, w_scale, w_d, n_kc);
            for (int i_mr = 0; i_mr < m / MR; i_mr++) {
                for (int i_qk = 0; i_qk < n_kc / QK_K; i_qk++) {
                    const auto &block_a = a_raw[i_mr * by + (i_kc * (KC / QK_K) + i_qk)];
                    const auto *block_w_qs = &w_qs[(i_qk * QK_K) * NR];
                    const auto *block_w_scale = &w_scale[(i_qk * (QK_K / QK_SK)) * NR];
                    const auto *block_w_d = &w_d[i_qk * NR];
                    for (int i_qsk = 0; i_qsk < QK_K / QK_SK; i_qsk++) {
                        REPEAT_MR(OP_DEF_SUM)
#pragma unroll
                        for (int i_k = 0; i_k < QK_SK; i_k++) {
                            vint8m1_t w = __riscv_vle8_v_i8m1(&block_w_qs[(i_qsk * QK_SK + i_k) * NR], NR);
                            if (i_k == 0) {
                                REPEAT_MR(OP_ACCUMULATE_SUBBLOCK_FIRST_ROUND)
                            } else {
                                REPEAT_MR(OP_ACCUMULATE_SUBBLOCK_NEXT_ROUNDS)
                            }
                        }
                        REPEAT_MR(OP_SAVE_SUBBLOCK);
                    }
                    vfloat32m4_t w_d = __riscv_vle32_v_f32m4(block_w_d, NR);

                    for (int i_m = 0; i_m < MR; i_m++) {
                        vint32m4_t sum_block_row;
#pragma unroll
                        for (int i_qsk = 0; i_qsk < QK_K / QK_SK; i_qsk++) {
                            vint16m2_t w_scale = __riscv_vle16_v_i16m2(&block_w_scale[i_qsk * NR], NR);
                            vint16m2_t sum_subblock_row = __riscv_vle16_v_i16m2(&sum_subblock[(i_m * (QK_K / QK_SK) + i_qsk) * NR], NR);
                            if (i_qsk == 0) {
                                sum_block_row = __riscv_vwmul_vv_i32m4(sum_subblock_row, w_scale, NR);
                            } else {
                                sum_block_row = __riscv_vwmacc_vv_i32m4(sum_block_row, sum_subblock_row, w_scale, NR);
                            }
                        }
                        __riscv_vse32_v_i32m4(&sum_block[i_m * NR], sum_block_row, NR);
                    }

#pragma unroll
                    for (int i_m = 0; i_m < MR; i_m++) {
                        float32_t a_d = block_a.d[i_m];
                        vfloat32m4_t d = __riscv_vfmul_vf_f32m4(w_d, a_d, NR);
                        vint32m4_t sum_block_row_i32 = __riscv_vle32_v_i32m4(&sum_block[i_m * NR], NR);
                        vfloat32m4_t sum_block_row = __riscv_vfcvt_f_x_v_f32m4(sum_block_row_i32, NR);
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
                for (int i_m = 0; i_m < MR; i_m++) {
                    vfloat32m4_t sum_row = __riscv_vle32_v_f32m4(&sum[i_m * NR], NR);
                    vfloat32m4_t c_row;
                    if (i_kc == 0) {
                        c_row = __riscv_vfmul_vf_f32m4(sum_row, 0.125f, NR);
                    } else {
                        c_row = __riscv_vle32_v_f32m4(&s[(i_mr * MR + i_m) * bs + (i_nr * NR)], NR);
                        c_row = __riscv_vfmacc_vf_f32m4(c_row, 0.125f, sum_row, NR);
                    }
                    __riscv_vse32_v_f32m4(&s[(i_mr * MR + i_m) * bs + (i_nr * NR)], c_row, NR);
                }
            }
        }
    }

#ifdef USE_TCM
    fine_mm_free(w_scale);
    fine_mm_free(w_d);
    fine_mm_free(sum);
    fine_mm_free(sum_block);
    fine_mm_free(sum_subblock);
#endif
}

} // namespace outer

} // namespace gemm::link