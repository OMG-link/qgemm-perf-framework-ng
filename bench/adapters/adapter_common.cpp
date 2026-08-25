#include "adapter_common.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "ime1.h"
#include "llama_reference.h"

namespace ime::bench::adapters {
namespace {

block_q4_0x16 pack_q4_n16(const block_q4_0 *input) {
    block_q4_0x16 output{};
    for (size_t row = 0; row < 16; ++row) output.d[row] = input[row].d;
    for (size_t row = 0; row < 16; ++row) {
        for (size_t j = 0; j < QK4_0 / 4; ++j) {
            output.qs[row * QK4_0 / 4 + j] =
                (input[row].qs[j] & 0x0f) |
                static_cast<uint8_t>((input[row].qs[j + QK4_0 / 4] & 0x0f) << 4);
            output.qs[4 * QK4_0 + row * QK4_0 / 4 + j] =
                static_cast<uint8_t>((input[row].qs[j] & 0xf0) >> 4) |
                (input[row].qs[j + QK4_0 / 4] & 0xf0);
        }
    }
    return output;
}

} // namespace

ValidationResult validate_m4_shape(const BenchmarkRequest &request) {
    if (!request.m || !request.n || !request.k) return {false, "M, N and K must be positive"};
    if (request.m % 4) return {false, "M must be divisible by 4"};
    if (request.n % 16) return {false, "N must be divisible by 16"};
    if (request.k % 32) return {false, "K must be divisible by 32"};
    return {true, {}};
}

ValidationResult validate_m8_shape(const BenchmarkRequest &request) {
    auto result = validate_m4_shape(request);
    if (!result.supported) return result;
    if (request.m % 8) return {false, "M must be divisible by 8"};
    return {true, {}};
}

PrepareResult prepare_common(const BenchmarkRequest &request, const BenchmarkInput &generic_input) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic_input);
    if (!input) return {nullptr, "expected Q4_0/Q8_0 input"};
    auto state = std::make_unique<CommonState>();
    state->m = request.m;
    state->n = request.n;
    state->k = request.k;
    state->blocks_k = request.k / QK8_0;
    state->output.resize(request.m * request.n);

    const size_t m4_block_size = 4 * (sizeof(float) + QK8_0);
    state->packed_a_m4.resize((request.m / 4) * state->blocks_k * m4_block_size);
    std::vector<block_q8_0> reference_q8(4 * state->blocks_k);
    for (size_t tile_m = 0; tile_m < request.m; tile_m += 4) {
        auto *destination = state->packed_a_m4.data() + (tile_m / 4) * state->blocks_k * m4_block_size;
        sqnbitgemm_spacemit_ime::ime1::quantize_a_4row_i8(
            QK8_0, input->activation.data() + tile_m * request.k, request.k, destination);
        for (size_t row = 0; row < 4; ++row) {
            llama_reference::quantize_row_q8_0(
                input->activation.data() + (tile_m + row) * request.k,
                reference_q8.data() + row * state->blocks_k, request.k);
        }
        for (size_t block = 0; block < state->blocks_k; ++block) {
            auto *scales = reinterpret_cast<float *>(destination + block * m4_block_size);
            for (size_t row = 0; row < 4; ++row) {
                scales[row] = GGML_FP16_TO_FP32(reference_q8[row * state->blocks_k + block].d);
            }
        }
    }

    state->packed_b.resize((request.n / 16) * state->blocks_k);
    for (size_t tile_n = 0; tile_n < request.n; tile_n += 16) {
        for (size_t block = 0; block < state->blocks_k; ++block) {
            block_q4_0 rows[16];
            for (size_t row = 0; row < 16; ++row) {
                rows[row] = input->weight[(tile_n + row) * state->blocks_k + block];
            }
            state->packed_b[(tile_n / 16) * state->blocks_k + block] = pack_q4_n16(rows);
        }
    }
    return {state.release(), {}};
}

void reset_common(KernelState opaque) {
    auto &state = *static_cast<CommonState *>(opaque);
    std::fill(state.output.begin(), state.output.end(), 0.0f);
}

ExportResult export_common(KernelState opaque, std::span<float> output) {
    const auto &state = *static_cast<CommonState *>(opaque);
    if (output.size() != state.output.size()) return {false, "output size mismatch"};
    std::copy(state.output.begin(), state.output.end(), output.begin());
    return {true, {}};
}

double checksum_common(KernelState opaque) {
    const auto &state = *static_cast<CommonState *>(opaque);
    double sum = 0.0;
    for (size_t i = 0; i < state.output.size(); i += 257) sum += state.output[i];
    return sum;
}

void destroy_common(KernelState state) noexcept { delete static_cast<CommonState *>(state); }

} // namespace ime::bench::adapters