#include "adapter_common.h"
#include "ime1_dispatch.h"
#include "kernel.h"
#include "llama_reference.h"
#include "types.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace ime::bench::adapters {
namespace {

using namespace ime::m4n32b;

struct State : CommonState {
    std::vector<block_q8_0_m4> packed_a;
    std::vector<block_q4_0_n32> packed_b;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input) return {nullptr, "expected Q4_0/Q8_0 input"};

    auto state = std::make_unique<State>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK8_0;
    state->output_value.resize(request.m * request.n);

    const size_t m_tiles = request.m / MR;
    const size_t n_tiles = request.n / NR;
    state->packed_a.resize(m_tiles * state->blocks_k());
    state->packed_b.resize(n_tiles * state->blocks_k());

    constexpr size_t upstream_block_bytes = MR * (sizeof(float) + QK8_0);
    std::vector<std::byte> upstream_a(state->blocks_k() * upstream_block_bytes);
    std::vector<block_q8_0> reference_a(MR * state->blocks_k());
    for (size_t tile_m = 0; tile_m < request.m; tile_m += MR) {
        sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(
            QK8_0, input->activation.data() + tile_m * request.k, request.k,
            upstream_a.data());
        for (size_t row = 0; row < MR; ++row) {
            llama_reference::quantize_row_q8_0(
                input->activation.data() + (tile_m + row) * request.k,
                reference_a.data() + row * state->blocks_k(), request.k);
        }
        for (size_t block = 0; block < state->blocks_k(); ++block) {
            auto &destination = state->packed_a[(tile_m / MR) * state->blocks_k() + block];
            for (size_t row = 0; row < MR; ++row)
                destination.d[row] = static_cast<_Float16>(GGML_FP16_TO_FP32(
                    reference_a[row * state->blocks_k() + block].d));
            std::memcpy(destination.qs,
                        upstream_a.data() + block * upstream_block_bytes + MR * sizeof(float),
                        sizeof(destination.qs));
        }
    }

    for (size_t tile_n = 0; tile_n < request.n; tile_n += NR) {
        for (size_t block = 0; block < state->blocks_k(); ++block) {
            auto &destination = state->packed_b[(tile_n / NR) * state->blocks_k() + block];
            for (size_t column = 0; column < NR; ++column) {
                const auto &source = input->weight[(tile_n + column) * state->blocks_k() + block];
                destination.d[column] = static_cast<_Float16>(GGML_FP16_TO_FP32(source.d));
                const size_t group = column / 4;
                const size_t lane_column = column % 4;
                for (size_t j = 0; j < QK4_0 / 4; ++j) {
                    destination.qs[0][group][lane_column * 8 + j] =
                        (source.qs[j] & 0x0f) |
                        static_cast<uint8_t>((source.qs[j + 8] & 0x0f) << 4);
                    destination.qs[1][group][lane_column * 8 + j] =
                        static_cast<uint8_t>((source.qs[j] & 0xf0) >> 4) |
                        (source.qs[j + 8] & 0xf0);
                }
            }
        }
    }

    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        // Required blocking order: KC512 -> MC128 -> N32 -> M4.
        for (size_t kc = 0; kc < state.k(); kc += KC) {
            const size_t kc_blocks = std::min(KC, state.k() - kc) / QK8_0;
            const size_t block_start = kc / QK8_0;
            for (size_t mc = 0; mc < state.m(); mc += MC) {
                const size_t mc_end = std::min(mc + MC, state.m());
                for (size_t n = 0; n < state.n(); n += NR) {
                    const auto *b = state.packed_b.data() +
                                    (n / NR) * state.blocks_k() + block_start;
                    for (size_t m = mc; m < mc_end; m += MR) {
                        const auto *a = state.packed_a.data() +
                                        (m / MR) * state.blocks_k() + block_start;
                        q4_0_q8_0_ime_m4n32b(
                            reinterpret_cast<const uint8_t *>(a),
                            reinterpret_cast<const uint8_t *>(b),
                            state.output().data() + m * state.n() + n,
                            kc_blocks, state.n(), kc != 0);
                    }
                }
            }
        }
    }
}

} // namespace

void register_q4_0_ime_m4n32b() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m4n32b",
        .name = "Q4_0 x Q8_0 IME M4xN32 batch reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<MR, NR, QK8_0>, prepare, reset_common, run,
                      export_common, checksum_common, destroy_common},
    });
}

} // namespace ime::bench::adapters
