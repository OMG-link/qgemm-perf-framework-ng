#ifndef IME_BENCH_Q4_0_COMMON_H
#define IME_BENCH_Q4_0_COMMON_H

#include <cstring>

#include "kernel_registry.h"
#include "ime1.h"
#include "llama_reference.h"

namespace ime::bench::adapters {

struct Q4_0M4N16ImePackedData {
    std::vector<std::byte> packed_a_m4;
    std::vector<block_q4_0x16> packed_b;
};

template <size_t MRows, class State>
bool initialize_q4_0_ime(const BenchmarkRequest &request,
                          const BenchmarkInput &generic_input,
                          State &state) {
    static_assert(MRows == 4 || MRows == 8);
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input) return false;
    state.m_value = request.m;
    state.n_value = request.n;
    state.k_value = request.k;
    state.blocks_k_value = request.k / QK8_0;
    state.output_value.resize(request.m * request.n);

    if constexpr (MRows == 4) {
        const size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
        state.q4.packed_a_m4.resize((request.m / 4) * state.blocks_k() * m4_block_size);
        std::vector<block_q8_0> reference_q8(4 * state.blocks_k());
        for (size_t tile_m = 0; tile_m < request.m; tile_m += 4) {
            auto *destination = state.q4.packed_a_m4.data() +
                                (tile_m / 4) * state.blocks_k() * m4_block_size;
            sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(
                QK8_0, input->activation.data() + tile_m * request.k, request.k, destination);
            for (size_t row = 0; row < 4; ++row) {
                llama_reference::quantize_row_q8_0(
                    input->activation.data() + (tile_m + row) * request.k,
                    reference_q8.data() + row * state.blocks_k(), request.k);
            }
            for (size_t block = 0; block < state.blocks_k(); ++block) {
                auto *scales = reinterpret_cast<float *>(destination + block * m4_block_size);
                for (size_t row = 0; row < 4; ++row)
                    scales[row] = GGML_FP16_TO_FP32(reference_q8[row * state.blocks_k() + block].d);
            }
        }
    } else {
        state.packed_a_m8.resize((request.m / 8) * state.blocks_k());
        for (size_t tile_m = 0; tile_m < request.m; tile_m += 8) {
            constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
            std::vector<std::byte> packed_four_rows(state.blocks_k() * m4_block_size);
            for (size_t half = 0; half < 2; ++half) {
                auto *packed = packed_four_rows.data();
                sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(
                    QK8_0, input->activation.data() + (tile_m + half * 4) * request.k,
                    request.k, packed);
                for (size_t block = 0; block < state.blocks_k(); ++block) {
                    const auto *source = packed + block * m4_block_size;
                    auto &destination = state.packed_a_m8[(tile_m / 8) * state.blocks_k() + block];
                    std::memcpy(destination.d + half * 4, source, 4 * sizeof(float));
                    std::memcpy(destination.qs + half * 4 * QK8_0,
                                source + 4 * sizeof(float), 4 * QK8_0);
                }
            }
        }
    }

    auto pack_q4_n16 = [](const block_q4_0 *rows) {
        block_q4_0x16 output{};
        for (size_t row = 0; row < 16; ++row) output.d[row] = rows[row].d;
        for (size_t row = 0; row < 16; ++row) {
            for (size_t j = 0; j < QK4_0 / 4; ++j) {
                output.qs[row * QK4_0 / 4 + j] =
                    (rows[row].qs[j] & 0x0f) |
                    static_cast<uint8_t>((rows[row].qs[j + QK4_0 / 4] & 0x0f) << 4);
                output.qs[4 * QK4_0 + row * QK4_0 / 4 + j] =
                    static_cast<uint8_t>((rows[row].qs[j] & 0xf0) >> 4) |
                    (rows[row].qs[j + QK4_0 / 4] & 0xf0);
            }
        }
        return output;
    };
    if constexpr (MRows == 4) {
        state.q4.packed_b.resize((request.n / 16) * state.blocks_k());
        for (size_t tile_n = 0; tile_n < request.n; tile_n += 16) {
            for (size_t block = 0; block < state.blocks_k(); ++block) {
                block_q4_0 rows[16];
                for (size_t row = 0; row < 16; ++row)
                    rows[row] = input->weight[(tile_n + row) * state.blocks_k() + block];
                state.q4.packed_b[(tile_n / 16) * state.blocks_k() + block] = pack_q4_n16(rows);
            }
        }
    } else {
        state.packed_b_qs.resize((request.n / 16) * state.blocks_k() * 16 * (QK4_0 / 2));
        state.packed_b_scales.resize((request.n / 16) * state.blocks_k() * 16);
        for (size_t tile_n = 0; tile_n < request.n; tile_n += 16) {
            for (size_t block = 0; block < state.blocks_k(); ++block) {
                block_q4_0 rows[16];
                for (size_t row = 0; row < 16; ++row)
                    rows[row] = input->weight[(tile_n + row) * state.blocks_k() + block];
                const size_t index = (tile_n / 16) * state.blocks_k() + block;
                const auto packed = pack_q4_n16(rows);
                std::memcpy(state.packed_b_qs.data() + index * 16 * (QK4_0 / 2), packed.qs,
                            16 * (QK4_0 / 2));
                std::memcpy(state.packed_b_scales.data() + index * 16, packed.d,
                            16 * sizeof(uint16_t));
            }
        }
    }
    return true;
}

} // namespace ime::bench::adapters

#endif