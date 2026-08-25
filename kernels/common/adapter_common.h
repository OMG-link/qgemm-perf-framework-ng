#ifndef IME_BENCH_ADAPTER_COMMON_H
#define IME_BENCH_ADAPTER_COMMON_H

#include <span>
#include <vector>

#include "kernel_registry.h"

namespace ime::bench::adapters {

struct CommonState {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    size_t blocks_k = 0;
    std::vector<std::byte> packed_a_m4;
    std::vector<block_q4_0x16> packed_b;
    std::vector<float> output;
};

ValidationResult validate_m4_shape(const BenchmarkRequest &request);
ValidationResult validate_m8_shape(const BenchmarkRequest &request);
PrepareResult prepare_common(const BenchmarkRequest &request, const BenchmarkInput &input);
void reset_common(KernelState state);
ExportResult export_common(KernelState state, std::span<float> output);
double checksum_common(KernelState state);
void destroy_common(KernelState state) noexcept;

void register_llama_dispatch();
void register_m4_batch_reduction();
void register_m4_immediate_reduction();
void register_m8_batch_reduction();

} // namespace ime::bench::adapters

#endif