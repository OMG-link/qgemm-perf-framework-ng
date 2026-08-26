#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "llama_reference.h"
#include "perf.h"

namespace ime::bench {

BenchmarkInput OwnedQ4_0Q8_0Input::view() const {
    return Q4_0Q8_0Input{m, n, k, activation, weight};
}

OwnedQ4_0Q8_0Input create_input(QuantizationType type, const BenchmarkRequest &request) {
    if (type != QuantizationType::WeightQ4_0ActivationQ8_0) std::abort();
    OwnedQ4_0Q8_0Input input;
    input.m = request.m;
    input.n = request.n;
    input.k = request.k;
    input.activation.resize(request.m * request.k);
    std::vector<float> weight_f32(request.n * request.k);
    std::mt19937 generator(static_cast<uint32_t>(request.seed));
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : input.activation) value = distribution(generator);
    for (float &value : weight_f32) value = distribution(generator);
    input.weight.resize(request.n * (request.k / QK4_0));
    for (size_t n = 0; n < request.n; ++n) {
        llama_reference::quantize_row_q4_0(weight_f32.data() + n * request.k,
                                           input.weight.data() + n * (request.k / QK4_0), request.k);
    }
    return input;
}

BenchmarkResult run_benchmark(const KernelRegistration &kernel, const BenchmarkRequest &request,
                              const BenchmarkInput &input) {
    BenchmarkResult result;
    result.kernel_id = std::string(kernel.id);
    const auto validation = kernel.callbacks.validate(request);
    if (!validation.supported) {
        result.skipped = true;
        result.message = validation.reason;
        return result;
    }
    const auto prepared = kernel.callbacks.prepare(request, input);
    if (!prepared.state) {
        result.message = prepared.error;
        return result;
    }
    struct Cleanup {
        const KernelRegistration &kernel;
        KernelState state;
        ~Cleanup() { kernel.callbacks.destroy(state); }
    } cleanup{kernel, prepared.state};

    if (request.verify) {
        kernel.callbacks.reset(prepared.state);
        kernel.callbacks.run(prepared.state, 1);
        std::vector<float> actual(request.m * request.n);
        const auto exported = kernel.callbacks.export_output(prepared.state, actual);
        if (!exported.success) {
            result.message = exported.error;
            return result;
        }
        const auto expected = llama_reference::compute(kernel.quantization, input);
        result.verified = true;
        result.verification_passed = true;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float absolute = std::fabs(actual[i] - expected[i]);
            const float relative = absolute / std::max(std::fabs(expected[i]), 1.0e-6f);
            result.max_absolute_error = std::max(result.max_absolute_error, absolute);
            result.max_relative_error = std::max(result.max_relative_error, relative);
            if (absolute > 1.0e-3f && relative > 1.0e-3f) result.verification_passed = false;
        }
        if (!result.verification_passed) {
            result.message = "output differs from llama.cpp reference";
            return result;
        }
    }

    kernel.callbacks.reset(prepared.state);
    kernel.callbacks.run(prepared.state, request.warmup_iterations);
    const int cycles_fd = perf_event_cycles();
    size_t iterations = request.iterations;
    if (!iterations) {
        kernel.callbacks.reset(prepared.state);
        perf_reset(cycles_fd);
        kernel.callbacks.run(prepared.state, 1);
        perf_disable(cycles_fd);
        const uint64_t elapsed = std::max<uint64_t>(perf_read(cycles_fd), 1);
        iterations = std::max<size_t>(1, request.target_cycles / elapsed);
    }
    result.iterations = iterations;
    std::vector<uint64_t> samples;
    for (size_t sample = 0; sample < request.samples; ++sample) {
        kernel.callbacks.reset(prepared.state);
        asm volatile("" ::: "memory");
        perf_reset(cycles_fd);
        kernel.callbacks.run(prepared.state, iterations);
        perf_disable(cycles_fd);
        const uint64_t elapsed = perf_read(cycles_fd);
        asm volatile("" ::: "memory");
        samples.push_back(elapsed / iterations);
    }
    perf_close_event(cycles_fd);
    std::sort(samples.begin(), samples.end());
    result.min_cycles = samples.front();
    result.median_cycles = samples[samples.size() / 2];
    result.checksum = kernel.callbacks.checksum(prepared.state);
    const double fma = static_cast<double>(request.m) * request.n * request.k;
    result.fma_per_cycle = fma / result.median_cycles;
    result.utilization_percent = result.fma_per_cycle / 128.0 * 100.0;
    return result;
}

} // namespace ime::bench