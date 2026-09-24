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

constexpr size_t kMr = 16;
constexpr size_t kNr = 32;
constexpr size_t kRowGroups = kMr / 8;     // one smt.vmadot consumes 8 rows
constexpr size_t kColumnGroups = kNr / 8;  // one smt.vmadot produces 8 columns
constexpr size_t kKWindows = QK8_0 / 16;   // one smt.vmadot consumes 16 K
constexpr size_t kAQuantBytesPerBlock = kMr * QK8_0;
constexpr size_t kAScalesPerBlock = kMr;
constexpr size_t kPackedBQBytesPerBlock = kNr * (QK4_0 / 2);
constexpr size_t kPackedBScalesPerBlock = kNr;
constexpr size_t kTileFloats = kMr * kNr;
constexpr size_t kCacheLineBytes = 64;
static_assert(kAQuantBytesPerBlock == 512);
static_assert(kPackedBQBytesPerBlock == 512);
static_assert(kAQuantBytesPerBlock == kRowGroups * kKWindows * 128);
static_assert(kPackedBQBytesPerBlock == kColumnGroups * 128);

struct M8State : CommonState {
    size_t threads = 1;
    std::vector<uint8_t> packed_a_qs_storage;
    std::vector<float> packed_a_scales_storage;
    int8_t *packed_a_qs = nullptr;
    float *packed_a_scales = nullptr;
    std::vector<uint8_t> packed_b_qs;
    std::vector<uint16_t> packed_b_scales;
};

template <typename T> T *align_to_cache_line(std::vector<T> &storage, size_t payload_bytes) {
    void *pointer = storage.data();
    size_t space = storage.size() * sizeof(T);
    return static_cast<T *>(std::align(kCacheLineBytes, payload_bytes, pointer, space));
}

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input)
        return {nullptr, "expected Q4_0/Q8_0 input"};

    auto state = std::make_unique<M8State>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK8_0;
    state->output_value.resize(request.m * request.n);
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    const size_t a_tiles = request.m / kMr;
    const size_t b_tiles = request.n / kNr;

    const size_t qs_bytes = a_tiles * blocks_k * kAQuantBytesPerBlock;
    const size_t scale_count = a_tiles * blocks_k * kAScalesPerBlock;
    state->packed_a_qs_storage.resize(qs_bytes + kCacheLineBytes - 1);
    state->packed_a_scales_storage.resize(scale_count + kCacheLineBytes / sizeof(float) - 1);
    state->packed_a_qs = reinterpret_cast<int8_t *>(align_to_cache_line(state->packed_a_qs_storage, qs_bytes));
    state->packed_a_scales = align_to_cache_line(state->packed_a_scales_storage, scale_count * sizeof(float));

    // Activations: quantize each row to Q8_0, then lay each K block out as four
    // 128-byte matrix-unit A registers, one per (row group, K window).
    std::vector<block_q8_0> row_blocks(kMr * blocks_k);
    for (size_t tile_m = 0; tile_m < request.m; tile_m += kMr) {
        for (size_t row = 0; row < kMr; ++row) {
            llama_reference::quantize_row_q8_0(input->activation.data() + (tile_m + row) * request.k, row_blocks.data() + row * blocks_k, request.k);
        }
        for (size_t block = 0; block < blocks_k; ++block) {
            const size_t tile_block = (tile_m / kMr) * blocks_k + block;
            int8_t *qs = state->packed_a_qs + tile_block * kAQuantBytesPerBlock;
            float *scales = state->packed_a_scales + tile_block * kAScalesPerBlock;
            for (size_t row = 0; row < kMr; ++row) {
                const block_q8_0 &source = row_blocks[row * blocks_k + block];
                const size_t group = row / 8;
                const size_t group_row = row % 8;
                for (size_t window = 0; window < kKWindows; ++window) {
                    std::copy_n(source.qs + window * 16, 16, qs + (group * kKWindows + window) * 128 + group_row * 16);
                }
                scales[row] = GGML_FP16_TO_FP32(source.d);
            }
        }
    }

    // Weights: one K block is four 128-byte B registers, byte index n * 16 + k
    // within a group of 8 columns. A Q4_0 byte already holds K k in its low
    // nibble and K k + 16 in its high nibble, which is exactly what the kernel
    // wants, so the bytes are copied through unchanged.
    state->packed_b_qs.resize(b_tiles * blocks_k * kPackedBQBytesPerBlock);
    state->packed_b_scales.resize(b_tiles * blocks_k * kPackedBScalesPerBlock);
    for (size_t tile_n = 0; tile_n < request.n; tile_n += kNr) {
        for (size_t block = 0; block < blocks_k; ++block) {
            const size_t tile_block = (tile_n / kNr) * blocks_k + block;
            uint8_t *qs = state->packed_b_qs.data() + tile_block * kPackedBQBytesPerBlock;
            uint16_t *scales = state->packed_b_scales.data() + tile_block * kPackedBScalesPerBlock;
            for (size_t column = 0; column < kNr; ++column) {
                const block_q4_0 &source = input->weight[(tile_n + column) * blocks_k + block];
                std::memcpy(qs + (column / 8) * 128 + (column % 8) * 16, source.qs, QK4_0 / 2);
                std::memcpy(scales + column, &source.d, sizeof(source.d));
            }
        }
    }

    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8State *>(opaque);
    const size_t blocks_k = state.blocks_k();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += kMr) {
            // The batched int32 component buffer. TCM takes whole-register
            // stores about 2.9x faster than L1D-resident ordinary memory, but a
            // TCM block is private to a hart pair, so a thread that is not on
            // one of the owning cpus falls back to the stack.
            alignas(256) int32_t fallback_components[kIme2ReductionComponentCount];
            int32_t *components = tcm::scratch(sizeof(fallback_components));
            if (components == nullptr) {
                components = fallback_components;
            }

            const size_t a_block = (tile_m / kMr) * blocks_k;
            const int8_t *a_qs = state.packed_a_qs + a_block * kAQuantBytesPerBlock;
            const float *a_scales = state.packed_a_scales + a_block * kAScalesPerBlock;
            float *output_rows = state.output().data() + tile_m * state.n();
            for (size_t tile_n = 0; tile_n < state.n(); tile_n += kNr) {
                const size_t b_block = (tile_n / kNr) * blocks_k;
                const uint8_t *b_qs = state.packed_b_qs.data() + b_block * kPackedBQBytesPerBlock;
                const uint16_t *b_scales = state.packed_b_scales.data() + b_block * kPackedBScalesPerBlock;

                float acc[kTileFloats] = {};
                SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_IME2_BatchRed(a_qs, a_scales, b_qs, b_scales, acc, blocks_k, components);

                for (size_t row = 0; row < kMr; ++row) {
                    std::copy_n(acc + row * kNr, kNr, output_rows + tile_n + row * state.n());
                }
            }
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
        .callbacks = {peak_ime2_i8_i32_mac, validate_shape<16, 32, 32>, prepare, reset_common, run, export_common, checksum_common, destroy},
    });
}

} // namespace pfng::bench::adapters
