// Derived from llama.cpp commit 314e72934. See provenance.md and LICENSE.
#include "llama_reference.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <array>

#include "iq2_xxs.h"

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

void quantize_row_q8_K(const float *input, block_q8_K *output, size_t count) {
    assert(count % QK_K == 0);
    for (size_t b = 0; b < count / QK_K; ++b) {
        float amax = 0.0f;
        for (size_t i = 0; i < QK_K; ++i)
            amax = std::max(amax, std::fabs(input[b * QK_K + i]));
        output[b].d = amax / 127.0f;
        const float inv = output[b].d ? 1.0f / output[b].d : 0.0f;
        for (size_t i = 0; i < QK_K; ++i)
            output[b].qs[i] = static_cast<int8_t>(std::round(input[b * QK_K + i] * inv));
        for (size_t i = 0; i < QK_K / 16; ++i) {
            output[b].bsums[i] = 0;
            for (size_t j = 0; j < 16; ++j)
                output[b].bsums[i] += output[b].qs[i * 16 + j];
        }
    }
}

void quantize_row_q4_K(const float *input, block_q4_K *output, size_t count) {
    assert(count % QK_K == 0);
    for (size_t b = 0; b < count / QK_K; ++b) {
        auto &o = output[b];
        float ls[8], lm[8], maxs = 0, maxm = 0;
        for (size_t sb = 0; sb < 8; ++sb) {
            float lo = input[b * QK_K + sb * 32], hi = lo;
            for (size_t j = 1; j < 32; ++j) {
                float v = input[b * QK_K + sb * 32 + j];
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            ls[sb] = (hi - lo) / 15.0f;
            lm[sb] = -lo / (ls[sb] ? ls[sb] : 1.0f);
            maxs = std::max(maxs, ls[sb]);
            maxm = std::max(maxm, lm[sb]);
        }
        const float ds = maxs / 63.0f, dm = maxm / 63.0f;
        o.d = GGML_FP32_TO_FP16(ds);
        o.dmin = GGML_FP32_TO_FP16(dm);
        uint8_t s[8], m[8];
        for (size_t i = 0; i < 8; ++i) {
            s[i] = std::min(63, (int)std::round(ls[i] / (ds ? ds : 1)));
            m[i] = std::min(63, (int)std::round(lm[i] / (dm ? dm : 1)));
        }
        for (size_t i = 0; i < 4; ++i) {
            o.scales[i] = s[i] | ((s[i + 4] & 3) << 6);
            o.scales[i + 4] = m[i] | ((m[i + 4] & 3) << 6);
            o.scales[i + 8] = (s[i + 4] >> 2) | ((m[i + 4] >> 2) << 4);
        }
        for (size_t sb = 0; sb < 4; ++sb)
            for (size_t j = 0; j < 32; ++j) {
                const int low = std::clamp(static_cast<int>(std::round(input[b * QK_K + sb * 32 + j] / (ls[sb] ? ls[sb] : 1.0f) + lm[sb])), 0, 15);
                const int high = std::clamp(static_cast<int>(std::round(input[b * QK_K + (sb + 4) * 32 + j] / (ls[sb + 4] ? ls[sb + 4] : 1.0f) + lm[sb + 4])), 0, 15);
                o.qs[sb * 32 + j] = static_cast<uint8_t>(low | (high << 4));
            }
    }
}

float dot_q4_K_q8_K(const block_q4_K &w, const block_q8_K &a) {
    uint8_t scales[8];
    uint8_t mins[8];
    for (size_t i = 0; i < 4; ++i) {
        scales[i] = w.scales[i] & 63;
        scales[i + 4] = (w.scales[i + 8] & 15) | ((w.scales[i] >> 6) << 4);
        mins[i] = w.scales[i + 4] & 63;
        mins[i + 4] = (w.scales[i + 8] >> 4) | ((w.scales[i + 4] >> 6) << 4);
    }
    const float d = GGML_FP16_TO_FP32(w.d);
    const float dmin = GGML_FP16_TO_FP32(w.dmin);
    float sum = 0.0f;
    for (size_t sb = 0; sb < 8; ++sb)
        for (size_t j = 0; j < 32; ++j) {
            const uint8_t packed = w.qs[(sb % 4) * 32 + j];
            const int q = sb < 4 ? packed & 15 : packed >> 4;
            sum += a.qs[sb * 32 + j] * (d * scales[sb] * q - dmin * mins[sb]);
        }
    return a.d * sum;
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

float dot_iq2_xxs_q8_K(const block_iq2_xxs &weight, const block_q8_K &activation) {
    const auto *packed = reinterpret_cast<const uint8_t *>(weight.qs);
    int32_t sum = 0;
    for (size_t subblock = 0; subblock < QK_K / 32; ++subblock) {
        int8_t values[32];
        int scale;
        iq2_xxs::decode_subblock(packed + subblock * 8, values, scale);
        int32_t subblock_sum = 0;
        for (size_t i = 0; i < 32; ++i)
            subblock_sum += values[i] * activation.qs[subblock * 32 + i];
        sum += scale * subblock_sum;
    }
    return 0.125f * GGML_FP16_TO_FP32(weight.d) * activation.d * sum;
}

namespace {

std::vector<float> compute_q4_0(const BenchmarkInput &generic_input) {
    const auto &input = std::get<Q4_0Q8_0Input>(generic_input);
    const size_t blocks = input.k / QK8_0;
    std::vector<block_q8_0> activation(input.m * blocks);
    for (size_t m = 0; m < input.m; ++m)
        quantize_row_q8_0(input.activation.data() + m * input.k, activation.data() + m * blocks, input.k);
    std::vector<float> output(input.m * input.n);
    for (size_t m = 0; m < input.m; ++m)
        for (size_t n = 0; n < input.n; ++n)
            output[m * input.n + n] = dot_q4_0_q8_0(input.weight.subspan(n * blocks, blocks),
                                                     std::span<const block_q8_0>(activation).subspan(m * blocks, blocks));
    return output;
}

std::vector<float> compute_q4_K(const BenchmarkInput &generic_input) {
    const auto &input = std::get<Q4_KQ8_KInput>(generic_input);
    const size_t blocks = input.k / QK_K;
    std::vector<block_q8_K> activation(input.m * blocks);
    for (size_t m = 0; m < input.m; ++m)
        quantize_row_q8_K(input.activation.data() + m * input.k, activation.data() + m * blocks, input.k);
    std::vector<float> output(input.m * input.n);
    for (size_t m = 0; m < input.m; ++m)
        for (size_t n = 0; n < input.n; ++n)
            for (size_t b = 0; b < blocks; ++b)
                output[m * input.n + n] += dot_q4_K_q8_K(input.weight[n * blocks + b], activation[m * blocks + b]);
    return output;
}

std::vector<float> compute_iq2_xxs(const BenchmarkInput &generic_input) {
    const auto &input = std::get<IQ2_XXSQ8_KInput>(generic_input);
    const size_t blocks = input.k / QK_K;
    std::vector<block_q8_K> activation(input.m * blocks);
    for (size_t m = 0; m < input.m; ++m)
        quantize_row_q8_K(input.activation.data() + m * input.k, activation.data() + m * blocks, input.k);
    std::vector<float> output(input.m * input.n);
    for (size_t m = 0; m < input.m; ++m)
        for (size_t n = 0; n < input.n; ++n)
            for (size_t b = 0; b < blocks; ++b)
                output[m * input.n + n] += dot_iq2_xxs_q8_K(input.weight[n * blocks + b], activation[m * blocks + b]);
    return output;
}

using ReferenceComputer = std::vector<float> (*)(const BenchmarkInput &);
constexpr std::array<ReferenceComputer, static_cast<size_t>(QuantizationType::Count)> computers = {
    compute_q4_0, compute_q4_K, compute_iq2_xxs};

} // namespace

std::vector<float> compute(QuantizationType type, const BenchmarkInput &generic_input) {
    return computers.at(static_cast<size_t>(type))(generic_input);
}

} // namespace ime::bench::llama_reference