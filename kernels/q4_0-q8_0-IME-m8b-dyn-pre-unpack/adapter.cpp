#include "adapter_common.h"
#include "q4_0_common.h"

#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;

struct M8DynPreUnpackState : CommonState {
    std::vector<block_q8_0_ime_m8> packed_a_m8;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8DynPreUnpackState>();
    if (!initialize_q4_0_ime<8>(request, input, *state)) {
        return {nullptr, "expected Q4_0/Q8_0 input"};
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8DynPreUnpackState *>(opaque);
    const size_t total_b_blocks = (state.n() / kNr) * state.blocks_k();

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::unique_ptr<int8_t[]> unpacked_b_qs{new int8_t[state.n() * state.k()]};
        unpack_q4_0_ime_m8b_rvv(
            state.packed_b_qs.data(), unpacked_b_qs.get(), total_b_blocks);
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += kMr) {
            const auto *a = reinterpret_cast<const uint8_t *>(
                state.packed_a_m8.data() + (tile_m / kMr) * state.blocks_k());
            SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack(
                a, unpacked_b_qs.get(), state.packed_b_scales.data(),
                state.output().data() + tile_m * state.n(), state.n(), state.blocks_k(), state.n());
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<M8DynPreUnpackState *>(state); }

} // namespace

void register_q4_0_ime_m8b_dyn_pre_unpack() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m8b-dyn-pre-unpack",
        .name = "Q4_0 x Q8_0 IME M8 batch reduction dynamic pre-unpack",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<8, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy},
    });
}

} // namespace ime::bench::adapters
