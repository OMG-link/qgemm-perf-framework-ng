#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio> // for GGML_ASSERT
#include <stdexcept>
#include <thread>

#include "wrapper.h"
#include "ime1_dispatch.h"

void sqnbitgemm_spacemit_ime_i8i4(const size_t blk_len, const size_t gemm_k, const qnbitgemm_spacemit_ime_args *gemm_args, void *const per_gemm_ws, const size_t m_start, const size_t m_count,
                                  const size_t n_start, const size_t n_count) {
    constexpr size_t scale_stride = sizeof(uint16_t);
    constexpr size_t blk_bitwidth = 4;

    const size_t k_blks = ime_div_round_up(gemm_k, blk_len);

    const size_t lda = k_blks * ime_q8_block_size(blk_len);
    const size_t ldc = gemm_args->ldc;
    const size_t ldb = k_blks * (blk_len * blk_bitwidth / 8);
    const std::byte *quant_a_ptr = static_cast<const std::byte *>(per_gemm_ws) + m_start * lda;

    const size_t zero_point_stride = gemm_args->quant_b_zp != nullptr ? sizeof(uint8_t) : 0;
    const size_t packed_b_stride = ldb + k_blks * (scale_stride + zero_point_stride);
    const std::byte *packed_quant_b_data = gemm_args->packed_quant_b_data + n_start * packed_b_stride;

    float *c_ptr = gemm_args->c_ptr + m_start * ldc + n_start;

    size_t count_n = 0;
    const size_t compute_block_count_n = m_count == 1 ? n_count : 16;
    for (size_t n = 0; n < n_count; n += count_n) {
        count_n = std::min(n_count - n, compute_block_count_n);

        const std::byte *a_row = quant_a_ptr;
        const std::byte *b_col = packed_quant_b_data + n * packed_b_stride;
        const std::byte *b_col_zp = (zero_point_stride != 0) ? b_col : nullptr;
        float *c_blk = c_ptr + n;

        int32_t rows_remaining = m_count;

        while (rows_remaining > 0) {
            const auto rows_handled =
                sqnbitgemm_spacemit_ime::ime1::gemm_kernel_i8i4(blk_len, a_row, b_col, nullptr, b_col_zp, c_blk, rows_remaining, count_n, gemm_k, k_blks, ldc, nullptr, scale_stride);

            c_blk += rows_handled * ldc;
            a_row += rows_handled * lda;

            rows_remaining -= rows_handled;
        }
    }
}