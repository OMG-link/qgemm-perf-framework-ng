#include "adapter_common.h"
#include "q4_0_common.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMr = 4;
constexpr size_t kNr = 16;
// L1D budget: one K block contributes a packed B block (288 B) plus an
// activation block (4 floats + 4 * QK8_0 = 144 B). kBlocksPerPanel = 32
// therefore keeps the resident panel pair at 32 * 432 B = 13.5 KiB, leaving
// room for the microkernel scratch and the C tile in the 32 KiB L1D; 64 blocks
// would need 27 KiB and crowd out the streaming activation panel.
constexpr size_t kBlocksPerPanel = 32;
constexpr size_t kAChunkBudgetBytes = 128 * 1024;
constexpr size_t kABytesPerBlock = sizeof(block_q8_0_ime_m4);
constexpr size_t kBBytesPerBlock = sizeof(block_q4_0_ime_n16);
constexpr size_t kTileFloats = kMr * kNr;

static_assert(kABytesPerBlock == 4 * sizeof(float) + 4 * QK8_0);
static_assert(kBBytesPerBlock == kNr * (QK4_0 / 2) + kNr * sizeof(uint16_t));

// Cache-blocked M4 immediate reduction.
//
// The plain m4i kernel walks the whole packed B for every M tile, so the packed
// B stream is re-read once per M tile (m/4 passes) and dominates the cluster's
// shared L2/DRAM fill path. This variant splits K into panels and walks an
// activation chunk inside each panel, so one K panel of packed B is reused by
// every M tile of the chunk from L1D while the activation stream is served by
// the shared L2. The C tile is accumulated in a packed buffer across K panels.
struct M4CacheBlockingState : CommonState {
    size_t threads = 1;
    Q4_0M4N16ImePackedData q4;               // staged M-major activations and packed B
    std::vector<block_q8_0_ime_m4> packed_a; // [K panel][M tile][local block]
    std::vector<float> packed_c;             // [M tile][N tile][kMr * kNr]
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M4CacheBlockingState>();
    if (!initialize_q4_0_ime<4>(request, input, *state))
        return {nullptr, "expected Q4_0/Q8_0 input"};
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    const size_t tiles_m = state->m() / kMr;
    const size_t tiles_n = state->n() / kNr;
    state->packed_c.resize(tiles_m * tiles_n * kTileFloats);

    const auto &m_major_a = state->q4.packed_a_m4;
    const size_t panels = (blocks_k + kBlocksPerPanel - 1) / kBlocksPerPanel;
    state->packed_a.resize(panels * tiles_m * kBlocksPerPanel);
    for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
        const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
        const size_t panel_base = (block_k / kBlocksPerPanel) * tiles_m * kBlocksPerPanel;
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            const auto *source = m_major_a.data() + (tile_m * blocks_k + block_k) * kABytesPerBlock;
            std::memcpy(state->packed_a.data() + panel_base + tile_m * kBlocksPerPanel, source, local_blocks * kABytesPerBlock);
        }
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M4CacheBlockingState *>(opaque);
    const size_t blocks_k = state.blocks_k();
    const size_t tiles_m = state.m() / kMr;
    const size_t tiles_n = state.n() / kNr;
    constexpr size_t kAChunkBytesPerTile = kBlocksPerPanel * kABytesPerBlock;

    const auto *packed_a = reinterpret_cast<const uint8_t *>(state.packed_a.data());
    const auto *packed_b = reinterpret_cast<const uint8_t *>(state.q4.packed_b.data());

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
            const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
            const size_t panel_base = (block_k / kBlocksPerPanel) * tiles_m * kBlocksPerPanel;
            const size_t chunk_tiles = std::max<size_t>(1, std::min(tiles_m, kAChunkBudgetBytes / kAChunkBytesPerTile));

            for (size_t chunk_begin = 0; chunk_begin < tiles_m; chunk_begin += chunk_tiles) {
                const size_t chunk_end = std::min(tiles_m, chunk_begin + chunk_tiles);
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
                for (size_t tile_n_index = 0; tile_n_index < tiles_n; ++tile_n_index) {
                    const uint8_t *b_panel = packed_b + (tile_n_index * blocks_k + block_k) * kBBytesPerBlock;
                    for (size_t tile_m = chunk_begin; tile_m < chunk_end; ++tile_m) {
                        const uint8_t *a_panel = packed_a + (panel_base + tile_m * kBlocksPerPanel) * kABytesPerBlock;
                        float *c_tile = state.packed_c.data() + (tile_m * tiles_n + tile_n_index) * kTileFloats;

                        // The microkernel accumulates into the packed C tile, so the
                        // first K panel hands it a zeroed tile and every later panel
                        // adds its own K blocks on top: no staging tile, no second C
                        // pass.
                        if (block_k == 0)
                            std::fill_n(c_tile, kTileFloats, 0.0f);
                        SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl_Intrin(a_panel, b_panel, c_tile, local_blocks);
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

void register_q4_0_ime_m4i_cb() {
    register_kernel({
        .id = "q4_0-q8_0-IME-m4i-CB",
        .name = "Q4_0 x Q8_0 IME M4 immediate reduction cache blocking",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {validate_shape<4, 16, 32>, prepare, reset_common, run, export_common, checksum_common, destroy_common},
    });
}

} // namespace ime::bench::adapters
