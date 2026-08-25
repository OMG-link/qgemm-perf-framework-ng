#include "adapter_common.h"

#include "ime1.h"

namespace ime::bench::adapters {
namespace {

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<CommonState *>(opaque);
    constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t tile_m = 0; tile_m < state.m; tile_m += 4) {
            const auto *a = reinterpret_cast<const uint8_t *>(
                state.packed_a_m4.data() + (tile_m / 4) * state.blocks_k * m4_block_size);
            SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_BatchRed(
                QK8_0, a, reinterpret_cast<const uint8_t *>(state.packed_b.data()),
                state.output.data() + tile_m * state.n, state.n, state.blocks_k, state.n);
        }
    }
}

} // namespace

void register_m4_batch_reduction() {
    register_kernel({
        .id = "m4-batch-reduction",
        .name = "M4 delayed batch reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_m4_shape, prepare_common, reset_common, run,
                      export_common, checksum_common, destroy_common},
    });
}

} // namespace ime::bench::adapters