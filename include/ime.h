#ifndef IME_LLAMA_IME_H
#define IME_LLAMA_IME_H

#include <cassert>
#include <cstddef>
#include <cstdint>

struct qnbitgemm_spacemit_ime_args {
    const float *a_ptr = nullptr;
    size_t lda = 0;
    const std::byte *packed_quant_b_data = nullptr;
    const float *quant_b_scale = nullptr;
    const void *quant_b_zp = nullptr;
    const float *quant_b_blksum = nullptr;
    const float *bias = nullptr;
    float *c_ptr = nullptr;
    size_t ldc = 0;
};

constexpr size_t ime_div_round_up(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

constexpr size_t ime_q8_block_size(size_t block_length) {
    const size_t block_size = sizeof(float) + block_length * sizeof(int8_t);
    assert(block_size % alignof(float) == 0);
    return block_size;
}

void sqnbitgemm_spacemit_ime_i8i4(size_t block_length, size_t gemm_k,
                                  const qnbitgemm_spacemit_ime_args *gemm_args,
                                  void *per_gemm_workspace, size_t m_start,
                                  size_t m_count, size_t n_start, size_t n_count);

#endif