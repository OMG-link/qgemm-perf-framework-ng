// Derived from llama.cpp commit 314e72934. See provenance.md and LICENSE.
#include "llama_reference.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace ime::bench::llama_reference {

void quantize_row_q4_0(const float *input, block_q4_0 *output, size_t count) {
    assert(count % QK4_0 == 0);
    for (size_t block = 0; block < count / QK4_0; ++block) {
        float absolute_max = 0.0f;
        float signed_max = 0.0f;
        for (size_t i = 0; i < QK4_0; ++i) {
            const float value = input[block * QK4_0 + i];
            if (absolute_max < std::fabs(value)) {
                absolute_max = std::fabs(value);
                signed_max = value;
            }
        }
        const float scale = signed_max / -8.0f;
        const float inverse_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        output[block].d = GGML_FP32_TO_FP16(scale);
        for (size_t i = 0; i < QK4_0 / 2; ++i) {
            const float low = input[block * QK4_0 + i] * inverse_scale;
            const float high = input[block * QK4_0 + QK4_0 / 2 + i] * inverse_scale;
            const uint8_t qlow = std::min<int>(15, static_cast<int8_t>(low + 8.5f));
            const uint8_t qhigh = std::min<int>(15, static_cast<int8_t>(high + 8.5f));
            output[block].qs[i] = qlow | static_cast<uint8_t>(qhigh << 4);
        }
    }
}

void quantize_row_q8_0(const float *input, block_q8_0 *output, size_t count) {
    assert(count % QK8_0 == 0);
    for (size_t block = 0; block < count / QK8_0; ++block) {
        float absolute_max = 0.0f;
        for (size_t i = 0; i < QK8_0; ++i) {
            absolute_max = std::max(absolute_max, std::fabs(input[block * QK8_0 + i]));
        }
        const float scale = absolute_max / 127.0f;
        const float inverse_scale = scale != 0.0f ? 1.0f / scale : 0.0f;
        output[block].d = GGML_FP32_TO_FP16(scale);
        for (size_t i = 0; i < QK8_0; ++i) {
            output[block].qs[i] = static_cast<int8_t>(std::round(input[block * QK8_0 + i] * inverse_scale));
        }
    }
}

float dot_q4_0_q8_0(std::span<const block_q4_0> weight, std::span<const block_q8_0> activation) {
    assert(weight.size() == activation.size());
    float sum = 0.0f;
    for (size_t block = 0; block < weight.size(); ++block) {
        int32_t integer_sum = 0;
        for (size_t i = 0; i < QK4_0 / 2; ++i) {
            integer_sum += (static_cast<int>(weight[block].qs[i] & 0x0f) - 8) * activation[block].qs[i];
            integer_sum += (static_cast<int>(weight[block].qs[i] >> 4) - 8) * activation[block].qs[i + QK4_0 / 2];
        }
        sum += integer_sum * GGML_FP16_TO_FP32(weight[block].d) * GGML_FP16_TO_FP32(activation[block].d);
    }
    return sum;
}

std::vector<float> compute(QuantizationType type, const BenchmarkInput &generic_input) {
    if (type != QuantizationType::WeightQ4_0ActivationQ8_0) std::abort();
    const auto &input = std::get<Q4_0Q8_0Input>(generic_input);
    const size_t blocks_k = input.k / QK8_0;
    std::vector<block_q8_0> quantized_a(input.m * blocks_k);
    for (size_t m = 0; m < input.m; ++m) {
        quantize_row_q8_0(input.activation.data() + m * input.k,
                          quantized_a.data() + m * blocks_k, input.k);
    }
    std::vector<float> output(input.m * input.n);
    for (size_t m = 0; m < input.m; ++m) {
        for (size_t n = 0; n < input.n; ++n) {
            output[m * input.n + n] = dot_q4_0_q8_0(
                input.weight.subspan(n * blocks_k, blocks_k),
                std::span<const block_q8_0>(quantized_a).subspan(m * blocks_k, blocks_k));
        }
    }
    return output;
}

} // namespace ime::bench::llama_reference