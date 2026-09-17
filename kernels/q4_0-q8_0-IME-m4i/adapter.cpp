#include "adapter_common.h"
#include "q4_0_common.h"

#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {
constexpr size_t kMr = 4;
constexpr size_t kNr = 16;
constexpr size_t kTileFloats = kMr * kNr;

struct State : CommonState {
    size_t threads = 1;
    Q4_0M4N16ImePackedData q4;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<State>();
    if (!initialize_q4_0_ime<4>(request, input, *state)) return {nullptr, "expected Q4_0/Q8_0 input"};
    state->threads = request.threads;
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    constexpr size_t m4_block_size = kMr * (sizeof(float) + QK8_0);
    const size_t blocks_k = state.blocks_k();
    const auto *packed_b = reinterpret_cast<const uint8_t *>(state.q4.packed_b.data());
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += kMr) {
            const auto *a = reinterpret_cast<const uint8_t *>(state.q4.packed_a_m4.data() + (tile_m / kMr) * blocks_k * m4_block_size);
            float *output_rows = state.output().data() + tile_m * state.n();
            for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
                const auto *b = packed_b + (tile_n / kNr) * blocks_k * sizeof(block_q4_0_ime_n16);

                float tile[kTileFloats] = {};
                SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(a, b, tile, blocks_k);
                c_unpack_ime_m4_n16(tile, output_rows + tile_n, state.n());
            }
        }
    }
}

}

void register_q4_0_ime_m4i() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m4i",
        .name = "Q4_0 x Q8_0 IME M4 immediate reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<4, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy_common},
    });
}

}
