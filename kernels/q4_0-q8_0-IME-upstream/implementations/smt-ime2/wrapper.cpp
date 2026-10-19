#include "wrapper.h"

#include <algorithm>

#include "ime2_dispatch.h"

namespace {

// NB_COLS of the IME2 i8-i4 microkernels.
constexpr size_t kColumnsPerBlock = 32;

} // namespace

void sqnbitgemm_spacemit_ime_ime2_i8i4(size_t          block_length,
                                       size_t          block_count_k,
                                       const uint8_t * quant_a,
                                       size_t          row_stride_a,
                                       const uint8_t * quant_b,
                                       size_t          col_stride_b,
                                       float *         output,
                                       size_t          ldc,
                                       size_t          row_count,
                                       size_t          column_count) {
    const uint8_t *b_block = quant_b;
    for (size_t column = 0; column < column_count; column += kColumnsPerBlock, b_block += kColumnsPerBlock * col_stride_b) {
        const size_t columns_in_block = std::min(column_count - column, kColumnsPerBlock);

        const uint8_t *a_row = quant_a;
        float *        c_block = output + column;
        size_t         rows_remaining = row_count;

        while (rows_remaining > 0) {
            const size_t rows_handled = spacemit_kernels::ime2::gemm_kernel_i8i4(
                block_length, a_row, b_block, nullptr, c_block, rows_remaining, columns_in_block,
                block_count_k, ldc);
            c_block += rows_handled * ldc;
            a_row += rows_handled * row_stride_a;
            rows_remaining -= rows_handled;
        }
    }
}
