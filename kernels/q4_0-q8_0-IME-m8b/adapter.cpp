#include "adapter_common.h"
#include "q4_0_common.h"

#include <cstring>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

struct M8State : CommonState {
    std::vector<block_q8_0_ime_m8> packed_a_m8;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8State>();
    if (!initialize_q4_0_ime<8>(request, input, *state)) return {nullptr, "expected Q4_0/Q8_0 input"};
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8State *>(opaque);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += 8) {
            const auto *a = reinterpret_cast<const uint8_t *>(
                state.packed_a_m8.data() + (tile_m / 8) * state.blocks_k());
            SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(
                a, state.packed_b_qs.data(), state.packed_b_scales.data(),
                state.output().data() + tile_m * state.n(), state.n(), state.blocks_k(), state.n());
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<M8State *>(state); }

} // namespace

void register_q4_0_ime_m8b() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m8b",
        .name = "Q4_0 x Q8_0 IME M8 batch reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<8, 16, 32>, prepare, reset_common, run,
                      export_common, checksum_common, destroy},
    });
}

} // namespace ime::bench::adapters