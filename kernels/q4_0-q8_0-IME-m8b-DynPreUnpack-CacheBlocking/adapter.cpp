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
constexpr size_t kAQuantBytesPerBlock = kMr * QK8_0;
constexpr size_t kAScalesPerBlock = kMr;
constexpr size_t kPackedAChunkOrder[kMr] = {0, 1, 4, 5, 2, 3, 6, 7};
constexpr size_t kCacheLineBytes = 64;
static_assert(kPanelBytes == 8 * 1024);
static_assert(kAQuantBytesPerBlock == 256);
static_assert(sizeof(block_q8_0_ime_m8) == 288);

struct M8DynPreUnpackCacheBlockingState : CommonState {
    size_t threads = 1;
    std::vector<block_q8_0_ime_m8> packed_a_m8;
    std::vector<uint8_t> packed_a_qs_storage;
    std::vector<float> packed_a_scales_storage;
    int8_t *packed_a_qs = nullptr;
    float *packed_a_scales = nullptr;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
    std::vector<float> packed_c;
};

template <typename T> T *align_to_cache_line(std::vector<T> &storage, size_t payload_bytes) {
    void *pointer = storage.data();
    size_t space = storage.size() * sizeof(T);
    return static_cast<T *>(std::align(kCacheLineBytes, payload_bytes, pointer, space));
}

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8DynPreUnpackCacheBlockingState>();
    if (!initialize_q4_0_ime<8>(request, input, *state))
        return {nullptr, "expected Q4_0/Q8_0 input"};
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    const size_t tiles_m = state->m() / kMr;
    const size_t tiles_n = state->n() / kNr;
    state->packed_c.resize(tiles_m * tiles_n * kMr * kNr);
    auto m_major_a = std::move(state->packed_a_m8);

    const size_t packed_block_count = m_major_a.size();
    const size_t qs_bytes = packed_block_count * kAQuantBytesPerBlock;
    const size_t scale_count = packed_block_count * kAScalesPerBlock;
    const size_t scale_bytes = scale_count * sizeof(float);
    state->packed_a_qs_storage.resize(qs_bytes + kCacheLineBytes - 1);
    state->packed_a_scales_storage.resize(scale_count + kCacheLineBytes / sizeof(float) - 1);
    state->packed_a_qs = reinterpret_cast<int8_t *>(align_to_cache_line(state->packed_a_qs_storage, qs_bytes));
    state->packed_a_scales = align_to_cache_line(state->packed_a_scales_storage, scale_bytes);

    for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
        const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
        const size_t panel_base = block_k * tiles_m;
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            for (size_t local_block = 0; local_block < local_blocks; ++local_block) {
                const auto &source = m_major_a[tile_m * blocks_k + block_k + local_block];
                const size_t destination_block = panel_base + tile_m * local_blocks + local_block;
                std::copy_n(source.d, kAScalesPerBlock, state->packed_a_scales + destination_block * kAScalesPerBlock);
                for (size_t packed_chunk = 0; packed_chunk < kMr; ++packed_chunk) {
                    std::copy_n(source.qs + kPackedAChunkOrder[packed_chunk] * QK8_0, QK8_0, state->packed_a_qs + destination_block * kAQuantBytesPerBlock + packed_chunk * QK8_0);
                }
            }
        }
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8DynPreUnpackCacheBlockingState *>(opaque);
    const size_t blocks_k = state.blocks_k();
    const size_t tiles_m = state.m() / kMr;
    const size_t tiles_n = state.n() / kNr;

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::unique_ptr<int8_t[]> unpacked_b{new int8_t[state.n() * state.k()]};

#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
            const size_t n_block_base = (tile_n / kNr) * blocks_k;
            for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
                const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
                unpack_q4_0_ime_m8b_cache_blocking_rvv(state.packed_b_qs.data() + (n_block_base + block_k) * kNr * (QK4_0 / 2), unpacked_b.get() + (n_block_base + block_k) * kUnpackedBytesPerBlock,
                                                       local_blocks);
            }
        }

        for (size_t block_k = 0; block_k < blocks_k; block_k += kBlocksPerPanel) {
            const size_t local_blocks = std::min(kBlocksPerPanel, blocks_k - block_k);
            const size_t panel_base = block_k * tiles_m;
            const size_t a_bytes_per_tile = local_blocks * (kAQuantBytesPerBlock + kAScalesPerBlock * sizeof(float));
            const size_t chunk_tiles = std::max<size_t>(1, std::min(tiles_m, kAChunkBudgetBytes / a_bytes_per_tile));

            for (size_t chunk_begin = 0; chunk_begin < tiles_m; chunk_begin += chunk_tiles) {
                const size_t chunk_end = std::min(tiles_m, chunk_begin + chunk_tiles);
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
                for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
                    const size_t n_block_base = (tile_n / kNr) * blocks_k;
                    const int8_t *unpacked_panel = unpacked_b.get() + (n_block_base + block_k) * kUnpackedBytesPerBlock;
                    const uint16_t *panel_scales = state.packed_b_scales.data() + (n_block_base + block_k) * kNr;

                    for (size_t tile_m = chunk_begin; tile_m < chunk_end; ++tile_m) {
                        const size_t a_block = panel_base + tile_m * local_blocks;
                        const int8_t *a_qs = state.packed_a_qs + a_block * kAQuantBytesPerBlock;
                        const float *a_scales = state.packed_a_scales + a_block * kAScalesPerBlock;
                        const size_t tile_n_index = tile_n / kNr;
                        float *packed_c_tile = state.packed_c.data() + (tile_m * tiles_n + tile_n_index) * kMr * kNr;
                        SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack_CacheBlocking(a_qs, a_scales, unpacked_panel, panel_scales, packed_c_tile, local_blocks, kNr,
                                                                                                              block_k == 0);
                    }
                }
            }
        }

#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_m = 0; tile_m < tiles_m; ++tile_m) {
            for (size_t tile_n_index = 0; tile_n_index < tiles_n; ++tile_n_index) {
                const float *packed_c_tile = state.packed_c.data() + (tile_m * tiles_n + tile_n_index) * kMr * kNr;
                float *output_tile = state.output().data() + tile_m * kMr * state.n() + tile_n_index * kNr;
                for (size_t row = 0; row < kMr; ++row) {
                    std::copy_n(packed_c_tile + row * kNr, kNr, output_tile + row * state.n());
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
