#include "adapter_common.h"
#include "q4_0_common.h"

#include <algorithm>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;
constexpr size_t kBlocksPerPanel = 16;
constexpr size_t kAChunkBudgetBytes = 128 * 1024;
constexpr size_t kUnpackedBytesPerBlock = kNr * QK4_0;
constexpr size_t kPanelBytes = kBlocksPerPanel * kUnpackedBytesPerBlock;
constexpr size_t kPackedARowOrder[kMr] = {0, 1, 4, 5, 2, 3, 6, 7};
static_assert(kPanelBytes == 8 * 1024);
static_assert(sizeof(block_q8_0_ime_m8) == 288);

struct M8DynPreUnpackCacheBlockingState : CommonState {
    std::vector<block_q8_0_ime_m8> packed_a_m8;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8DynPreUnpackCacheBlockingState>();
    if (!initialize_q4_0_ime<8>(request, input, *state))
        return {nullptr, "expected Q4_0/Q8_0 input"};

    const size_t blocks_k = state->blocks_k();
    const size_t tiles_m = state->m() / kMr;
    auto m_major_a = std::move(state->packed_a_m8);
    state->packed_a_m8.resize(m_major_a.size());
    for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
        const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
        const size_t panel_base = block_k * tiles_m;
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            for (size_t local_block = 0; local_block < local_blocks; ++local_block) {
                const auto &source = m_major_a[tile_m * blocks_k + block_k + local_block];
                auto &destination = state->packed_a_m8[panel_base + tile_m * local_blocks + local_block];
                std::copy_n(source.d, kMr, destination.d);
                for (size_t packed_row = 0; packed_row < kMr; ++packed_row) {
                    std::copy_n(source.qs + kPackedARowOrder[packed_row] * QK8_0, QK8_0,
                                destination.qs + packed_row * QK8_0);
                }
            }
        }
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8DynPreUnpackCacheBlockingState *>(opaque);
    const size_t blocks_k = state.blocks_k();

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::unique_ptr<int8_t[]> unpacked_b{new int8_t[state.n() * state.k()]};

        for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
            const size_t n_block_base = (tile_n / kNr) * blocks_k;
            for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
                const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
                unpack_q4_0_ime_m8b_cache_blocking_rvv(state.packed_b_qs.data() + (n_block_base + block_k) * kNr * (QK4_0 / 2),
                                                        unpacked_b.get() + (n_block_base + block_k) * kUnpackedBytesPerBlock, local_blocks);
            }
        }

        const size_t tiles_m = state.m() / kMr;
        for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
            const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
            const size_t panel_base = block_k * tiles_m;
            const size_t a_bytes_per_tile = local_blocks * sizeof(block_q8_0_ime_m8);
            const size_t chunk_tiles = std::max<size_t>(1, std::min(tiles_m, kAChunkBudgetBytes / a_bytes_per_tile));

            for (size_t chunk_begin = 0; chunk_begin < tiles_m; chunk_begin += chunk_tiles) {
                const size_t chunk_end = std::min(tiles_m, chunk_begin + chunk_tiles);
                for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
                    const size_t n_block_base = (tile_n / kNr) * blocks_k;
                    const int8_t *unpacked_panel = unpacked_b.get() + (n_block_base + block_k) * kUnpackedBytesPerBlock;
                    const uint16_t *panel_scales = state.packed_b_scales.data() + (n_block_base + block_k) * kNr;

                    for (size_t tile_m = chunk_begin; tile_m < chunk_end; ++tile_m) {
                        const auto *a = reinterpret_cast<const uint8_t *>(state.packed_a_m8.data() + panel_base + tile_m * local_blocks);
                        SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack_CacheBlocking(
                            a, unpacked_panel, panel_scales, state.output().data() + tile_m * kMr * state.n() + tile_n,
                            local_blocks, state.n(), block_k == 0);
                    }
                }
            }
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<M8DynPreUnpackCacheBlockingState *>(state); }

} // namespace

void register_q4_0_ime_m8b_dyn_pre_unpack_cache_blocking() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking",
        .name = "Q4_0 x Q8_0 IME M8 dynamic pre-unpack cache blocking",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<8, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy},
    });
}

} // namespace ime::bench::adapters
