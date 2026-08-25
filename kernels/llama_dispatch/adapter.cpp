#include "adapter_common.h"

#include "ime1.h"

namespace ime::bench::adapters {
namespace {

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<CommonState *>(opaque);
    constexpr size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t tile_m = 0; tile_m < state.m; tile_m += 4) {
            const auto *a = state.packed_a_m4.data() + (tile_m / 4) * state.blocks_k * m4_block_size;
            for (size_t tile_n = 0; tile_n < state.n; tile_n += 16) {
                const auto *b = reinterpret_cast<const std::byte *>(
                    state.packed_b.data() + (tile_n / 16) * state.blocks_k);
                sqnbitgemm_spacemit_ime::ime1::gemm_kernel_i8i4(
                    QK8_0, a, b, nullptr, nullptr,
                    state.output.data() + tile_m * state.n + tile_n,
                    4, 16, state.k, state.blocks_k, state.n, nullptr, sizeof(uint16_t));
            }
        }
    }
}

} // namespace

void register_llama_dispatch() {
    register_kernel({
        .id = "llama-dispatch",
        .name = "llama.cpp IME dispatcher",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_m4_shape, prepare_common, reset_common, run,
                      export_common, checksum_common, destroy_common},
    });
}

} // namespace ime::bench::adapters