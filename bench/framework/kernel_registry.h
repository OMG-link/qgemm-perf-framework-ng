#ifndef IME_BENCH_KERNEL_REGISTRY_H
#define IME_BENCH_KERNEL_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "ggml.h"

namespace ime::bench {

struct BenchmarkRequest {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    size_t warmup_iterations = 3;
    size_t samples = 10;
    size_t iterations = 0;
    uint64_t target_cycles = 100'000'000;
    uint64_t seed = 1;
    bool verify = true;
};

enum class QuantizationType {
    WeightQ4_0ActivationQ8_0,
};

struct Q4_0Q8_0Input {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    std::span<const float> activation;
    std::span<const block_q4_0> weight;
};

using BenchmarkInput = std::variant<Q4_0Q8_0Input>;
using KernelState = void *;

struct ValidationResult {
    bool supported = false;
    std::string reason;
};

struct PrepareResult {
    KernelState state = nullptr;
    std::string error;
};

struct ExportResult {
    bool success = false;
    std::string error;
};

struct KernelCallbacks {
    ValidationResult (*validate)(const BenchmarkRequest &request);
    PrepareResult (*prepare)(const BenchmarkRequest &request, const BenchmarkInput &input);
    void (*reset)(KernelState state);
    void (*run)(KernelState state, size_t iterations) noexcept;
    ExportResult (*export_output)(KernelState state, std::span<float> row_major_output);
    double (*checksum)(KernelState state);
    void (*destroy)(KernelState state) noexcept;
};

struct KernelRegistration {
    std::string_view id;
    std::string_view name;
    QuantizationType quantization;
    KernelCallbacks callbacks;
};

void register_kernel(const KernelRegistration &registration);
std::span<const KernelRegistration> registered_kernels();
const KernelRegistration *find_kernel(std::string_view id);

} // namespace ime::bench

#endif