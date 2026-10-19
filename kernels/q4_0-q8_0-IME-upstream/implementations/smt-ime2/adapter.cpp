#include "adapter_common.h"
#include "llama_reference.h"
#include "types.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ime2_dispatch.h"
#include "wrapper.h"

namespace pfng::bench::adapters {
namespace {

// The IME2 i8-i4 microkernels consume four activation rows and 32 weight
// columns per call.
constexpr size_t kActivationRows = 4;
constexpr size_t kWeightColumns = 32;

// Activation block of one K32 step and four rows:
// [f32 scale x 4][s16 block sum x 4][int8 quants x 4 x 32].
constexpr size_t kActivationRowBytes = spacemit_kernels::q8_blk_size(QK8_0, true);
constexpr size_t kActivationBlockBytes = kActivationRows * kActivationRowBytes;
// Weight block of one K32 step and 32 columns copied from the q4_0 layout:
// [fp16 scale x 32][int4 pairs x 32 x 16].
constexpr size_t kWeightColumnBytes = sizeof(uint16_t) + QK4_0 / 2;
constexpr size_t kWeightBlockBytes = kWeightColumns * kWeightColumnBytes;

static_assert(kActivationRowBytes == 38);
static_assert(kActivationBlockBytes == 152);
static_assert(kWeightColumnBytes == 18);
static_assert(kWeightBlockBytes == 576);

struct State : CommonState {
    size_t threads = 1;
    std::vector<uint8_t> packed_a; // [m / 4][blocks_k] activation blocks
    std::vector<uint8_t> packed_b; // [n / 32][blocks_k] weight blocks
    size_t row_stride_a = 0;       // blocks_k * kActivationRowBytes
    size_t col_stride_b = 0;       // blocks_k * kWeightColumnBytes
};

// q4_0 -> IME2 N32 x K32 block, as upstream's repack_q4_0_to_q4_0_32_bl does it:
// the 32 scales first, then each column's nibbles with the K pairs interleaved,
// so that byte b of a column carries K = 2b in its low nibble and K = 2b + 1 in
// its high nibble. The quantized values are moved, never re-derived, so the
// repack is lossless.
void pack_weight_block(const block_q4_0 *columns, size_t blocks_k, size_t block,
                       uint8_t *destination) {
    auto *scales = reinterpret_cast<uint16_t *>(destination);
    uint8_t *quants = destination + kWeightColumns * sizeof(uint16_t);
    for (size_t column = 0; column < kWeightColumns; ++column) {
        const block_q4_0 &source = columns[column * blocks_k + block];
        scales[column] = source.d;
        uint8_t *target = quants + column * (QK4_0 / 2);
        for (size_t pair = 0; pair < QK4_0 / 4; ++pair) {
            target[pair] = static_cast<uint8_t>((source.qs[2 * pair] & 0x0F) |
                                                ((source.qs[2 * pair + 1] & 0x0F) << 4));
            target[QK4_0 / 4 + pair] =
                static_cast<uint8_t>(((source.qs[2 * pair] & 0xF0) >> 4) |
                                     (source.qs[2 * pair + 1] & 0xF0));
        }
    }
}

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input) return {nullptr, "expected Q4_0/Q8_0 input"};

    auto state = std::make_unique<State>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK8_0;
    state->output_value.resize(request.m * request.n);
    state->threads = request.threads;

    const size_t blocks_k = state->blocks_k();
    state->row_stride_a = blocks_k * kActivationRowBytes;
    state->col_stride_b = blocks_k * kWeightColumnBytes;
    // The row stride is the distance between two activation rows, so a four-row
    // group occupies kActivationRows of them, and likewise for the 32 columns of
    // a weight group.
    state->packed_a.assign(request.m * state->row_stride_a, uint8_t{0});
    state->packed_b.assign((request.n / kWeightColumns) * kWeightColumns * state->col_stride_b,
                           uint8_t{0});

    // The upstream activation quantizer stores an fp32 amax/127 scale and rounds
    // with the vector unit's round-to-nearest-even, which is not the q8_0 the
    // reference defines: its blocks differ from the reference in the scale
    // precision and at rounding ties. Build the upstream block layout from the
    // reference quantization instead, so the measured GEMM is exactly the
    // q4_0 x q8_0 problem this kernel is benchmarked as. See ../../provenance.md.
    std::vector<block_q8_0> reference(blocks_k);
    for (size_t tile_m = 0; tile_m < request.m; tile_m += kActivationRows) {
        uint8_t *destination = state->packed_a.data() + tile_m * state->row_stride_a;
        for (size_t row = 0; row < kActivationRows; ++row) {
            llama_reference::quantize_row_q8_0(input->activation.data() + (tile_m + row) * request.k,
                                               reference.data(), request.k);
            for (size_t block = 0; block < blocks_k; ++block) {
                uint8_t *block_base = destination + block * kActivationBlockBytes;
                auto *scales = reinterpret_cast<float *>(block_base);
                auto *sums = reinterpret_cast<int16_t *>(block_base + kActivationRows * sizeof(float));
                auto *quants = reinterpret_cast<int8_t *>(
                    block_base + kActivationRows * sizeof(float) + kActivationRows * sizeof(int16_t));
                const block_q8_0 &source = reference[block];
                scales[row] = GGML_FP16_TO_FP32(source.d);
                int16_t sum = 0;
                for (size_t k = 0; k < QK8_0; ++k) {
                    quants[row * QK8_0 + k] = source.qs[k];
                    sum = static_cast<int16_t>(sum + source.qs[k]);
                }
                sums[row] = static_cast<int16_t>(-sum);
            }
        }
    }

    for (size_t tile_n = 0; tile_n < request.n; tile_n += kWeightColumns) {
        const block_q4_0 *columns = input->weight.data() + tile_n * blocks_k;
        for (size_t block = 0; block < blocks_k; ++block) {
            pack_weight_block(columns, blocks_k, block,
                              state->packed_b.data() +
                                  ((tile_n / kWeightColumns) * blocks_k + block) * kWeightBlockBytes);
        }
    }

    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    const size_t row_tiles = state.m() / kActivationRows;
    const size_t column_tiles = state.n() / kWeightColumns;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
#pragma omp parallel for schedule(static) num_threads(state.threads) if (state.threads > 1)
        for (size_t column_tile = 0; column_tile < column_tiles; ++column_tile) {
            const uint8_t *b_block =
                state.packed_b.data() + column_tile * kWeightColumns * state.col_stride_b;
            for (size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
                sqnbitgemm_spacemit_ime_ime2_i8i4(
                    QK8_0, state.blocks_k(),
                    state.packed_a.data() + row_tile * kActivationRows * state.row_stride_a,
                    state.row_stride_a, b_block, state.col_stride_b,
                    state.output().data() + row_tile * kActivationRows * state.n() +
                        column_tile * kWeightColumns,
                    state.n(), kActivationRows, kWeightColumns);
            }
        }
    }
}

} // namespace

void register_q4_0_ime_upstream() {
    register_kernel({
        .id = "q4_0-q8_0-IME-upstream",
        .name = "Q4_0 x Q8_0 IME upstream",
        .quantization = QuantizationType::WeightQ4_0ActivationQ8_0,
        .callbacks = {peak_ime2_i8_i32_mac, validate_shape<4, 32, 32>, prepare, reset_common, run,
                      export_common, checksum_common, destroy_common},
    });
}

} // namespace pfng::bench::adapters
