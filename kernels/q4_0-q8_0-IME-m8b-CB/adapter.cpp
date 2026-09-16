#include "adapter_common.h"
#include "q4_0_common.h"

#include <algorithm>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;
// L1D budget: one K block contributes a packed B block (kNr * (QK4_0/2) =
// 256 B) plus its scales (32 B) and an activation block (kMr * QK8_0 = 256 B)
// plus its scales (32 B). kBlocksPerPanel = 32 therefore keeps the resident
// panel pair at 32 * 576 B = 18 KiB, leaving about a third of the 32 KiB L1D
// for the microkernel scratch; 64 blocks would need 36 KiB and evict the panel
// the chunk is supposed to reuse.
constexpr size_t kBlocksPerPanel = 32;
constexpr size_t kAChunkBudgetBytes = 128 * 1024;
constexpr size_t kAQuantBytesPerBlock = kMr * QK8_0;
constexpr size_t kAScalesPerBlock = kMr;
constexpr size_t kAScalesBytesPerBlock = kAScalesPerBlock * sizeof(float);
constexpr size_t kPackedBQBytesPerBlock = kNr * (QK4_0 / 2);
constexpr size_t kPackedAChunkOrder[kMr] = {0, 1, 4, 5, 2, 3, 6, 7};
constexpr size_t kTileFloats = kMr * kNr;

static_assert(kAQuantBytesPerBlock == 256);
static_assert(sizeof(block_q8_0_ime_m8) == kAQuantBytesPerBlock + kAScalesBytesPerBlock);

// Cache-blocked M8 batch reduction.
//
// The plain m8b kernel walks the whole packed B for every M tile, so the packed
// B stream is re-read once per M tile (m/8 passes) and dominates the cluster's
// shared L2/DRAM fill path. This variant splits K into panels and walks an
// activation chunk inside each panel, so one K panel of packed B is reused by
// every M tile of the chunk from L1D while the activation stream is served by
// the shared L2. The C tile is accumulated in a packed buffer across K panels.
struct M8CacheBlockingState : CommonState {
    size_t threads = 1;
    std::vector<block_q8_0_ime_m8> packed_a_m8; // staging: M-major [M tile][K block]
    std::vector<int8_t> packed_a_qs;            // [K panel][M tile][local block][kMr * QK8_0]
    std::vector<float> packed_a_scales;         // [K panel][M tile][local block][kMr]
    std::vector<uint8_t> packed_b_qs;           // [N tile][K block][kNr * QK4_0/2]
    std::vector<uint16_t> packed_b_scales;      // [N tile][K block][kNr]
    std::vector<float> packed_c;                // [M tile][N tile][kMr * kNr]
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8CacheBlockingState>();
    if (!initialize_q4_0_ime<8>(request, input, *state))
        return {nullptr, "expected Q4_0/Q8_0 input"};
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    const size_t tiles_m = state->m() / kMr;
    const size_t tiles_n = state->n() / kNr;
    state->packed_c.resize(tiles_m * tiles_n * kTileFloats);

    auto m_major_a = std::move(state->packed_a_m8);
    const size_t panels = (blocks_k + kBlocksPerPanel - 1) / kBlocksPerPanel;
    state->packed_a_qs.resize(panels * tiles_m * kBlocksPerPanel * kAQuantBytesPerBlock);
    state->packed_a_scales.resize(panels * tiles_m * kBlocksPerPanel * kAScalesPerBlock);
    for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
        const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
        const size_t panel_base = (block_k / kBlocksPerPanel) * tiles_m * kBlocksPerPanel;
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            for (size_t local_block = 0; local_block < local_blocks; ++local_block) {
                const auto &source = m_major_a[tile_m * blocks_k + block_k + local_block];
                const size_t destination_block = panel_base + tile_m * kBlocksPerPanel + local_block;
                std::copy_n(source.d, kAScalesPerBlock, state->packed_a_scales.data() + destination_block * kAScalesPerBlock);
                for (size_t packed_chunk = 0; packed_chunk < kMr; ++packed_chunk) {
                    std::copy_n(source.qs + kPackedAChunkOrder[packed_chunk] * QK8_0, QK8_0, state->packed_a_qs.data() + destination_block * kAQuantBytesPerBlock + packed_chunk * QK8_0);
                }
            }
        }
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8CacheBlockingState *>(opaque);
    const size_t blocks_k = state.blocks_k();
    const size_t tiles_m = state.m() / kMr;
    const size_t tiles_n = state.n() / kNr;
    constexpr size_t kAChunkBytesPerTile = kBlocksPerPanel * (kAQuantBytesPerBlock + kAScalesBytesPerBlock);

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
            const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
            const size_t panel_base = (block_k / kBlocksPerPanel) * tiles_m * kBlocksPerPanel;
            const size_t chunk_tiles = std::max<size_t>(1, std::min(tiles_m, kAChunkBudgetBytes / kAChunkBytesPerTile));

            for (size_t chunk_begin = 0; chunk_begin < tiles_m; chunk_begin += chunk_tiles) {
                const size_t chunk_end = std::min(tiles_m, chunk_begin + chunk_tiles);
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
                for (size_t tile_n_index = 0; tile_n_index < tiles_n; ++tile_n_index) {
                    const uint8_t *b_qs_panel = state.packed_b_qs.data() + (tile_n_index * blocks_k + block_k) * kPackedBQBytesPerBlock;
                    const uint16_t *b_scales_panel = state.packed_b_scales.data() + (tile_n_index * blocks_k + block_k) * kNr;
                    for (size_t tile_m = chunk_begin; tile_m < chunk_end; ++tile_m) {
                        const int8_t *a_qs = state.packed_a_qs.data() + (panel_base + tile_m * kBlocksPerPanel) * kAQuantBytesPerBlock;
                        const float *a_scales = state.packed_a_scales.data() + (panel_base + tile_m * kBlocksPerPanel) * kAScalesPerBlock;
                        float *c_tile = state.packed_c.data() + (tile_m * tiles_n + tile_n_index) * kTileFloats;

                        float tile[kTileFloats];
                        SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed(a_qs, a_scales, b_qs_panel, b_scales_panel, tile, kNr, local_blocks, kNr);
                        if (block_k == 0) {
                            std::copy_n(tile, kTileFloats, c_tile);
                        } else {
                            for (size_t i = 0; i < kTileFloats; ++i)
                                c_tile[i] += tile[i];
                        }
                    }
                }
            }
        }

#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            for (size_t tile_n_index = 0; tile_n_index < tiles_n; ++tile_n_index) {
                const float *c_tile = state.packed_c.data() + (tile_m * tiles_n + tile_n_index) * kTileFloats;
                float *output_tile = state.output().data() + tile_m * kMr * state.n() + tile_n_index * kNr;
                for (size_t row = 0; row < kMr; ++row) {
                    std::copy_n(c_tile + row * kNr, kNr, output_tile + row * state.n());
                }
            }
        }
    }
}

} // namespace

void register_q4_0_ime_m8b_cb() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m8b-CB",
        .name = "Q4_0 x Q8_0 IME M8 batch reduction cache blocking",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<8, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy_common},
    });
}

} // namespace ime::bench::adapters
