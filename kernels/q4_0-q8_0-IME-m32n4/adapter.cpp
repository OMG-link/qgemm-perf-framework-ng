#include "adapter_common.h"
#include "ime1_dispatch.h"
#include "llama_reference.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMc = 32;
constexpr size_t kNc = 512;
constexpr size_t kKcBlocks = 256 / QK8_0;

struct M32N4State : CommonState {
    std::vector<block_q8_0_ime_m32> packed_a;
    std::vector<block_q4_0_ime_n4> packed_b;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input)
        return {nullptr, "expected Q4_0/Q8_0 input"};

    auto state = std::make_unique<M32N4State>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK8_0;
    state->output_value.resize(request.m * request.n);
    state->packed_a.resize((request.m / kMc) * state->blocks_k());
    state->packed_b.resize((request.n / 4) * state->blocks_k());

    constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    std::vector<std::byte> packed_four_rows(state->blocks_k() * m4_block_size);
    std::vector<block_q8_0> reference_q8(kMc * state->blocks_k());
    for (size_t tile_m = 0; tile_m < request.m; tile_m += kMc) {
        for (size_t row = 0; row < kMc; ++row) {
            llama_reference::quantize_row_q8_0(input->activation.data() + (tile_m + row) * request.k, reference_q8.data() + row * state->blocks_k(), request.k);
        }
        for (size_t group = 0; group < kMc / 4; ++group) {
            sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(QK8_0, input->activation.data() + (tile_m + group * 4) * request.k, request.k, packed_four_rows.data());
            for (size_t block = 0; block < state->blocks_k(); ++block) {
                const std::byte *source = packed_four_rows.data() + block * m4_block_size;
                auto &destination = state->packed_a[(tile_m / kMc) * state->blocks_k() + block];
                for (size_t row = 0; row < 4; ++row) {
                    const auto &scale = reference_q8[(group * 4 + row) * state->blocks_k() + block].d;
                    std::memcpy(&destination.d[group * 4 + row], &scale, sizeof(scale));
                }
                std::memcpy(destination.qs + group * 4 * QK8_0, source + 4 * sizeof(float), 4 * QK8_0);
            }
        }
    }

    for (size_t tile_n = 0; tile_n < request.n; tile_n += 4) {
        for (size_t block = 0; block < state->blocks_k(); ++block) {
            auto &destination = state->packed_b[(tile_n / 4) * state->blocks_k() + block];
            for (size_t column = 0; column < 4; ++column) {
                const block_q4_0 &source = input->weight[(tile_n + column) * state->blocks_k() + block];
                std::memcpy(&destination.d[column], &source.d, sizeof(source.d));
                for (size_t j = 0; j < QK4_0 / 4; ++j) {
                    destination.qs[column * 8 + j] = (source.qs[j] & 0x0f) | static_cast<uint8_t>((source.qs[j + 8] & 0x0f) << 4);
                    destination.qs[32 + column * 8 + j] = static_cast<uint8_t>((source.qs[j] & 0xf0) >> 4) | (source.qs[j + 8] & 0xf0);
                }
            }
        }
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M32N4State *>(opaque);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::fill(state.output().begin(), state.output().end(), 0.0f);
        for (size_t block_k = 0; block_k < state.blocks_k(); block_k += kKcBlocks) {
            const size_t kc_blocks = std::min(kKcBlocks, state.blocks_k() - block_k);
            for (size_t block_n = 0; block_n < state.n(); block_n += kNc) {
                const size_t nc = std::min(kNc, state.n() - block_n);
                for (size_t tile_m = 0; tile_m < state.m(); tile_m += kMc) {
                    const auto *a = state.packed_a.data() + (tile_m / kMc) * state.blocks_k() + block_k;
                    for (size_t tile_n = block_n; tile_n < block_n + nc; tile_n += 4) {
                        const auto *b = state.packed_b.data() + (tile_n / 4) * state.blocks_k() + block_k;
                        float *c = state.output().data() + tile_m * state.n() + tile_n;
                        q4_0_q8_0_ime_m32n4(a, b, c, kc_blocks, state.n());
                    }
                }
            }
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<M32N4State *>(state); }

} // namespace

void register_q4_0_ime_m32n4() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m32n4",
        .name = "Q4_0 x Q8_0 IME M32xN4 KC256 NC512",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<32, 4, 32>, prepare, reset_common, run, export_common, checksum_common, destroy},
    });
}

} // namespace ime::bench::adapters
