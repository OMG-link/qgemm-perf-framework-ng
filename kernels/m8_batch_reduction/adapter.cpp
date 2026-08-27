#include "adapter_common.h"

#include <cstring>
#include <memory>

#include "ime1.h"

namespace ime::bench::adapters {
namespace {

struct M8State : CommonState {
    std::vector<block_q8_0x8_scale32> packed_a_m8;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto common_result = prepare_common(request, input);
    if (!common_result.state) return common_result;
    std::unique_ptr<CommonState> common(static_cast<CommonState *>(common_result.state));
    auto state = std::make_unique<M8State>();
    static_cast<CommonState &>(*state) = std::move(*common);
    state->packed_a_m8.resize((state->m / 8) * state->blocks_k);
    state->packed_b_qs.resize((state->n / 16) * state->blocks_k * 16 * (QK4_0 / 2));
    state->packed_b_scales.resize((state->n / 16) * state->blocks_k * 16);
    constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    for (size_t tile_m = 0; tile_m < state->m; tile_m += 8) {
        for (size_t block = 0; block < state->blocks_k; ++block) {
            const auto *low = state->packed_a_m4.data() + (tile_m / 4) * state->blocks_k * m4_block_size + block * m4_block_size;
            const auto *high = low + state->blocks_k * m4_block_size;
            auto &destination = state->packed_a_m8[(tile_m / 8) * state->blocks_k + block];
            std::memcpy(destination.d, low, 4 * sizeof(float));
            std::memcpy(destination.d + 4, high, 4 * sizeof(float));
            std::memcpy(destination.qs, low + 4 * sizeof(float), 4 * QK8_0);
            std::memcpy(destination.qs + 4 * QK8_0, high + 4 * sizeof(float), 4 * QK8_0);
        }
    }
    for (size_t tile_n = 0; tile_n < state->n; tile_n += 16) {
        for (size_t block = 0; block < state->blocks_k; ++block) {
            const auto &source = state->packed_b[(tile_n / 16) * state->blocks_k + block];
            const size_t index = (tile_n / 16) * state->blocks_k + block;
            std::memcpy(state->packed_b_qs.data() + index * 16 * (QK4_0 / 2), source.qs,
                        16 * (QK4_0 / 2));
            std::memcpy(state->packed_b_scales.data() + index * 16, source.d, 16 * sizeof(uint16_t));
        }
    }
    state->packed_a_m4.clear();
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8State *>(opaque);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t tile_m = 0; tile_m < state.m; tile_m += 8) {
            const auto *a = reinterpret_cast<const uint8_t *>(
                state.packed_a_m8.data() + (tile_m / 8) * state.blocks_k);
            SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(
                a, state.packed_b_qs.data(), state.packed_b_scales.data(),
                state.output.data() + tile_m * state.n, state.n, state.blocks_k, state.n);
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<M8State *>(state); }

} // namespace

void register_m8_batch_reduction() {
    register_kernel({
        .id = "m8-batch-reduction",
        .name = "M8 delayed batch reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_m8_shape, prepare, reset_common, run,
                      export_common, checksum_common, destroy},
    });
}

} // namespace ime::bench::adapters