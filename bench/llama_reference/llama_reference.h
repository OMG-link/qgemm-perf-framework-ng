#ifndef IME_BENCH_LLAMA_REFERENCE_H
#define IME_BENCH_LLAMA_REFERENCE_H

#include <span>
#include <vector>

#include "kernel_registry.h"

namespace ime::bench::llama_reference {

void quantize_row_q4_0(const float *input, block_q4_0 *output, size_t count);
void quantize_row_q8_0(const float *input, block_q8_0 *output, size_t count);
void quantize_row_q4_K(const float *input, block_q4_K *output, size_t count);
void quantize_row_q8_K(const float *input, block_q8_K *output, size_t count);
float dot_q4_0_q8_0(std::span<const block_q4_0> weight, std::span<const block_q8_0> activation);
float dot_q4_K_q8_K(const block_q4_K &weight, const block_q8_K &activation);
std::vector<float> compute(QuantizationType type, const BenchmarkInput &input);

} // namespace ime::bench::llama_reference

#endif