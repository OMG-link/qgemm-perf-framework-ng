#include "adapter_common.h"
#include "q4_0_common.h"

#include <algorithm>
#include <memory>

#include "kernel.h"

namespace ime::bench::adapters {
namespace {

constexpr size_t kMr = 8;
constexpr size_t kNr = 16;
constexpr size_t kAQuantBytesPerBlock = kMr * QK8_0;
constexpr size_t kAScalesPerBlock = kMr;
constexpr size_t kPackedAChunkOrder[kMr] = {0, 1, 4, 5, 2, 3, 6, 7};
constexpr size_t kCacheLineBytes = 64;
static_assert(kAQuantBytesPerBlock == 256);
static_assert(sizeof(block_q8_0_ime_m8) == 288);

struct M8DynPreUnpackState : CommonState {
    std::vector<block_q8_0_ime_m8> packed_a_m8;
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

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &input) {
    auto state = std::make_unique<M8DynPreUnpackState>();
    if (!initialize_q4_0_ime<8>(request, input, *state)) {
        return {nullptr, "expected Q4_0/Q8_0 input"};
    }

    auto m_major_a = std::move(state->packed_a_m8);
    const size_t packed_block_count = m_major_a.size();
    const size_t qs_bytes = packed_block_count * kAQuantBytesPerBlock;
    const size_t scale_count = packed_block_count * kAScalesPerBlock;
    const size_t scale_bytes = scale_count * sizeof(float);
    state->packed_a_qs_storage.resize(qs_bytes + kCacheLineBytes - 1);
    state->packed_a_scales_storage.resize(scale_count + kCacheLineBytes / sizeof(float) - 1);
    state->packed_a_qs = reinterpret_cast<int8_t *>(align_to_cache_line(state->packed_a_qs_storage, qs_bytes));
    state->packed_a_scales = align_to_cache_line(state->packed_a_scales_storage, scale_bytes);

    for (size_t block = 0; block < packed_block_count; ++block) {
        for (size_t packed_chunk = 0; packed_chunk < kMr; ++packed_chunk) {
            std::copy_n(m_major_a[block].qs + kPackedAChunkOrder[packed_chunk] * QK8_0, QK8_0, state->packed_a_qs + block * kAQuantBytesPerBlock + packed_chunk * QK8_0);
        }
        std::copy_n(m_major_a[block].d, kAScalesPerBlock, state->packed_a_scales + block * kAScalesPerBlock);
    }
    return {state.release(), {}};
}

void run(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<M8DynPreUnpackState *>(opaque);
    const size_t total_b_blocks = (state.n() / kNr) * state.blocks_k();

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        std::unique_ptr<int8_t[]> unpacked_b_qs{new int8_t[state.n() * state.k()]};
        unpack_q4_0_ime_m8b_rvv(state.packed_b_qs.data(), unpacked_b_qs.get(), total_b_blocks);
        for (size_t tile_m = 0; tile_m < state.m(); tile_m += kMr) {
            const size_t a_block = (tile_m / kMr) * state.blocks_k();
            const int8_t *a_qs = state.packed_a_qs + a_block * kAQuantBytesPerBlock;
            const float *a_scales = state.packed_a_scales + a_block * kAScalesPerBlock;
            SQ4BitGemmM8Kernel_CompInt8_ScaleFp16_Impl_Intrin_BatchRed_DynPreUnpack(a_qs, a_scales, unpacked_b_qs.get(), state.packed_b_scales.data(), state.output().data() + tile_m * state.n(),
                                                                                    state.n(), state.blocks_k(), state.n());
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
