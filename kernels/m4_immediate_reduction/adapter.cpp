#include "adapter_common.h"
#include "q4_0_common.h"

#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {
struct State : CommonState { Q4_0M4N16ImePackedData q4; };

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<State>();
    if (!initialize_q4_0_ime<4>(request, input, *state)) return {nullptr, "expected Q4_0/Q8_0 input"};
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += 4) {
            const auto *a = reinterpret_cast<const uint8_t *>(state.q4.packed_a_m4.data() + (tile_m / 4) * state.blocks_k() * m4_block_size);
            SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(a, reinterpret_cast<const uint8_t *>(state.q4.packed_b.data()), state.output().data() + tile_m * state.n(), state.n(), state.blocks_k(),
                                                              state.n());
        }
    }
}

}

void register_m4_immediate_reduction() {
    register_kernel({
        .id = "m4-immediate-reduction",
        .name = "M4 immediate reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<4, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy_common},
    });
}

}
