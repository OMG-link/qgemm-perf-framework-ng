#include "adapter_common.h"
#include "llama_reference.h"
#include "types.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "kernel.h"
#include "tcm_arena.h"

namespace pfng::bench::adapters {
namespace {

constexpr size_t kAQuantBytesPerBlock = kImeRrMr * QK8_0;
constexpr size_t kPackedBQBytesPerBlock = kImeRrNr * (QK4_0 / 2);
constexpr size_t kPackedBScalesPerBlock = kImeRrColumnGroups * kImeRrScaleMatrixElements;
constexpr size_t kCacheLineBytes = 128;
constexpr size_t kCBytesPerMTile = kImeRrPackedCElements * sizeof(float);
constexpr size_t kMTilesPerChunk = 128;
constexpr size_t kCChunkBytes = kMTilesPerChunk * kCBytesPerMTile;

static_assert(kAQuantBytesPerBlock == 2 * 128);
static_assert(kPackedBQBytesPerBlock == kImeRrColumnGroups * 128);
static_assert(kCBytesPerMTile == 1024);
static_assert(kCChunkBytes == 128 * 1024);

struct RrState : CommonState {
    size_t threads = 1;
    std::vector<uint8_t> packed_a_qs_storage;
    std::vector<uint16_t> packed_a_scales_storage;
    int8_t *packed_a_qs = nullptr;
    uint16_t *packed_a_scales = nullptr;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

template <typename T> T *align_pointer(std::vector<T> &storage, size_t payload_bytes) {
    void *pointer = storage.data();
    size_t space = storage.size() * sizeof(T);
    return static_cast<T *>(std::align(kCacheLineBytes, payload_bytes, pointer, space));
}

__attribute__((noinline)) void unpack_c_chunk(const float *GGML_RESTRICT packed_c,
                                               float *GGML_RESTRICT output,
                                               size_t output_stride,
                                               size_t chunk_tiles) {
    for (size_t chunk_tile = 0; chunk_tile < chunk_tiles; ++chunk_tile) {
        const float *packed_tile = packed_c + chunk_tile * kImeRrPackedCElements;
        float *output_tile = output + chunk_tile * kImeRrMr * output_stride;
        for (size_t group = 0; group < kImeRrColumnGroups; ++group) {
            const float *matrix = packed_tile + group * 64;
            for (size_t row = 0; row < kImeRrMr; ++row) {
                std::copy_n(matrix + row * 8, 8,
                            output_tile + row * output_stride + group * 8);
            }
        }
    }
}

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input)
        return {nullptr, "expected Q4_0/Q8_0 input"};

    auto state = std::make_unique<RrState>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK8_0;
    state->output_value.resize(request.m * request.n);
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    const size_t a_tiles = request.m / kImeRrMr;
    const size_t a_scale_groups = (a_tiles + kImeRrActivationTilesPerScaleLoad - 1) / kImeRrActivationTilesPerScaleLoad;
    const size_t b_tiles = request.n / kImeRrNr;

    const size_t a_q_bytes = a_tiles * blocks_k * kAQuantBytesPerBlock;
    const size_t a_scale_count = blocks_k * a_scale_groups * kImeRrScaleMatrixElements;
    state->packed_a_qs_storage.resize(a_q_bytes + kCacheLineBytes - 1);
    state->packed_a_scales_storage.resize(a_scale_count + kCacheLineBytes / sizeof(uint16_t) - 1);
    state->packed_a_qs = reinterpret_cast<int8_t *>(align_pointer(state->packed_a_qs_storage, a_q_bytes));
    state->packed_a_scales = align_pointer(state->packed_a_scales_storage, a_scale_count * sizeof(uint16_t));
    std::fill_n(state->packed_a_scales, a_scale_count, uint16_t{0});

    std::vector<block_q8_0> row_blocks(kImeRrMr * blocks_k);
    for (size_t tile_m = 0; tile_m < request.m; tile_m += kImeRrMr) {
        for (size_t row = 0; row < kImeRrMr; ++row) {
            llama_reference::quantize_row_q8_0(input->activation.data() + (tile_m + row) * request.k,
                                               row_blocks.data() + row * blocks_k, request.k);
        }
        for (size_t block = 0; block < blocks_k; ++block) {
            const size_t tile_block = block * a_tiles + tile_m / kImeRrMr;
            int8_t *quant = state->packed_a_qs + tile_block * kAQuantBytesPerBlock;
            const size_t tile_m_index = tile_m / kImeRrMr;
            const size_t scale_group = tile_m_index / kImeRrActivationTilesPerScaleLoad;
            const size_t scale_column = tile_m_index % kImeRrActivationTilesPerScaleLoad;
            uint16_t *scales = state->packed_a_scales +
                               (block * a_scale_groups + scale_group) * kImeRrScaleMatrixElements;
            for (size_t row = 0; row < kImeRrMr; ++row) {
                const block_q8_0 &source = row_blocks[row * blocks_k + block];
                for (size_t window = 0; window < 2; ++window) {
                    std::copy_n(source.qs + window * 16, 16,
                                quant + window * 128 + row * 16);
                }
                std::memcpy(scales + row * 8 + scale_column, &source.d, sizeof(source.d));
            }
        }
    }

    state->packed_b_qs.resize(b_tiles * blocks_k * kPackedBQBytesPerBlock);
    state->packed_b_scales.resize(b_tiles * blocks_k * kPackedBScalesPerBlock, uint16_t{0});
    for (size_t tile_n = 0; tile_n < request.n; tile_n += kImeRrNr) {
        for (size_t block = 0; block < blocks_k; ++block) {
            const size_t tile_block = (tile_n / kImeRrNr) * blocks_k + block;
            uint8_t *quant = state->packed_b_qs.data() + tile_block * kPackedBQBytesPerBlock;
            uint16_t *scales = state->packed_b_scales.data() + tile_block * kPackedBScalesPerBlock;
            for (size_t column = 0; column < kImeRrNr; ++column) {
                const block_q4_0 &source = input->weight[(tile_n + column) * blocks_k + block];
                const size_t group = column / 8;
                const size_t group_column = column % 8;
                std::memcpy(quant + group * 128 + group_column * 16,
                            source.qs, QK4_0 / 2);
                std::memcpy(scales + group * kImeRrScaleMatrixElements + group_column * 8,
                            &source.d, sizeof(source.d));
            }
        }
    }

    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<RrState *>(opaque);
    const size_t blocks_k = state.blocks_k();
    const size_t a_tiles = state.m() / kImeRrMr;
    const size_t a_scale_groups = (a_tiles + kImeRrActivationTilesPerScaleLoad - 1) / kImeRrActivationTilesPerScaleLoad;
    const size_t b_tiles = state.n() / kImeRrNr;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_n_index = 0; tile_n_index < b_tiles; ++tile_n_index) {
            alignas(256) float fallback_c[kCChunkBytes / sizeof(float)];
            float *packed_c = static_cast<float *>(tcm::allocate(kCChunkBytes, 256));
            const bool c_in_tcm = packed_c != nullptr;
            if (packed_c == nullptr) {
                packed_c = fallback_c;
            }
            const size_t tile_n = tile_n_index * kImeRrNr;
            for (size_t chunk_begin = 0; chunk_begin < a_tiles; chunk_begin += kMTilesPerChunk) {
                const size_t chunk_tiles = std::min(kMTilesPerChunk, a_tiles - chunk_begin);
                std::fill_n(packed_c, chunk_tiles * kImeRrPackedCElements, 0.0f);

                for (size_t block = 0; block < blocks_k; ++block) {
                    const size_t a_block = block * a_tiles + chunk_begin;
                    const size_t a_scale_block = block * a_scale_groups + chunk_begin / kImeRrActivationTilesPerScaleLoad;
                    const size_t b_block = tile_n_index * blocks_k + block;
                    q4_0_q8_0_ime_rr_w_stationary(
                        state.packed_a_qs + a_block * kAQuantBytesPerBlock,
                        state.packed_a_scales + a_scale_block * kImeRrScaleMatrixElements,
                        state.packed_b_qs.data() + b_block * kPackedBQBytesPerBlock,
                        state.packed_b_scales.data() + b_block * kPackedBScalesPerBlock,
                        packed_c, chunk_tiles);
                }

                float *output_chunk = state.output().data() + chunk_begin * kImeRrMr * state.n() + tile_n;
                unpack_c_chunk(packed_c, output_chunk, state.n(), chunk_tiles);
            }
            if (c_in_tcm) {
                tcm::deallocate(packed_c);
            }
        }
    }
}

void destroy(KernelState state) noexcept { delete static_cast<RrState *>(state); }

} // namespace

void register_q4_0_ime_rr() {
    register_kernel({
        .id = "q4_0-q8_0-IME-RR",
        .name = "Q4_0 x Q8_0 IME scale outer-product reduction",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {peak_ime2_i8_i32_mac, validate_shape<kImeRrMr, kImeRrNr, QK8_0>,
                      prepare, reset_common, run, export_common, checksum_common, destroy},
    });
}

} // namespace pfng::bench::adapters