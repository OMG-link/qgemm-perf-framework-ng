#ifndef IME_BENCH_BENCHMARK_H
#define IME_BENCH_BENCHMARK_H

#include "kernel_registry.h"

namespace ime::bench {

struct OwnedQ4_0Q8_0Input {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    std::vector<float> activation;
    std::vector<block_q4_0> weight;

    BenchmarkInput view() const;
};

struct OwnedQ4_KQ8_KInput {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    std::vector<float> activation;
    std::vector<block_q4_K> weight;
    BenchmarkInput view() const;
};

struct BenchmarkResult {
    std::string kernel_id;
    bool skipped = false;
    bool verified = false;
    bool verification_passed = false;
    std::string message;
    uint64_t min_cycles = 0;
    uint64_t median_cycles = 0;
    size_t iterations = 0;
    double checksum = 0.0;
    double fma_per_cycle = 0.0;
    double utilization_percent = 0.0;
    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;
};

OwnedQ4_0Q8_0Input create_input(QuantizationType type, const BenchmarkRequest &request);
OwnedQ4_KQ8_KInput create_q4_K_input(const BenchmarkRequest &request);
BenchmarkResult run_benchmark(const KernelRegistration &kernel, const BenchmarkRequest &request,
                              const BenchmarkInput &input);

} // namespace ime::bench

#endif