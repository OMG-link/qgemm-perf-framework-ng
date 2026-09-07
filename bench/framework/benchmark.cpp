#include "benchmark.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <array>
#include <cstring>
#include <cstdio>

#include "llama_reference.h"
#include "perf.h"

namespace ime::bench {

BenchmarkInput OwnedQ4_0Q8_0Input::view() const {
    return Q4_0Q8_0Input{m, n, k, activation, weight};
}

BenchmarkInput OwnedQ4_KQ8_KInput::view() const {
    return Q4_KQ8_KInput{m, n, k, activation, weight};
}

BenchmarkInput OwnedIQ2_XXSQ8_KInput::view() const {
    return IQ2_XXSQ8_KInput{m, n, k, activation, weight};
}

namespace {

void fill_random(std::span<float> values, std::mt19937 &generator) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : values) value = distribution(generator);
}

OwnedBenchmarkInput create_q4_0_input(const BenchmarkRequest &request) {
    OwnedQ4_0Q8_0Input input;
    input.m = request.m;
    input.n = request.n;
    input.k = request.k;
    input.activation.resize(request.m * request.k);
    std::vector<float> weight_f32(request.n * request.k);
    std::mt19937 generator(static_cast<uint32_t>(request.seed));
    fill_random(input.activation, generator);
    fill_random(weight_f32, generator);
    input.weight.resize(request.n * (request.k / QK4_0));
    for (size_t n = 0; n < request.n; ++n) {
        llama_reference::quantize_row_q4_0(weight_f32.data() + n * request.k,
                                           input.weight.data() + n * (request.k / QK4_0), request.k);
    }
    return input;
}

OwnedBenchmarkInput create_q4_K_input(const BenchmarkRequest &request) {
    OwnedQ4_KQ8_KInput input;
    input.m = request.m;
    input.n = request.n;
    input.k = request.k;
    input.activation.resize(request.m * request.k);
    std::vector<float> weights(request.n * request.k);
    std::mt19937 generator(static_cast<uint32_t>(request.seed));
    fill_random(input.activation, generator);
    fill_random(weights, generator);
    input.weight.resize(request.n * (request.k / QK_K));
    for (size_t n = 0; n < request.n; ++n)
        llama_reference::quantize_row_q4_K(weights.data() + n * request.k, input.weight.data() + n * (request.k / QK_K), request.k);
    return input;
}

OwnedBenchmarkInput create_iq2_xxs_input(const BenchmarkRequest &request) {
    OwnedIQ2_XXSQ8_KInput input;
    input.m = request.m;
    input.n = request.n;
    input.k = request.k;
    input.activation.resize(request.m * request.k);
    input.weight.resize(request.n * (request.k / QK_K));
    std::mt19937 generator(static_cast<uint32_t>(request.seed));
    fill_random(input.activation, generator);
    std::uniform_int_distribution<unsigned> byte(0, 255);
    std::uniform_int_distribution<unsigned> grid_index(0, 31);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (auto &block : input.weight) {
        block.d = GGML_FP32_TO_FP16(scale(generator));
        auto *packed = reinterpret_cast<uint8_t *>(block.qs);
        for (size_t subblock = 0; subblock < QK_K / 32; ++subblock) {
            for (size_t group = 0; group < 4; ++group) packed[subblock * 8 + group] = grid_index(generator);
            uint32_t signs_and_scale = byte(generator) & 0x7f;
            signs_and_scale |= (byte(generator) & 0x7f) << 7;
            signs_and_scale |= (byte(generator) & 0x7f) << 14;
            signs_and_scale |= (byte(generator) & 0x7f) << 21;
            signs_and_scale |= (byte(generator) & 0x0f) << 28;
            std::memcpy(packed + subblock * 8 + 4, &signs_and_scale, sizeof(signs_and_scale));
        }
    }
    return input;
}

using InputFactory = OwnedBenchmarkInput (*)(const BenchmarkRequest &);
constexpr std::array<InputFactory, static_cast<size_t>(QuantizationType::Count)> input_factories = {
    create_q4_0_input, create_q4_K_input, create_iq2_xxs_input};

} // namespace

OwnedBenchmarkInput create_input(QuantizationType type, const BenchmarkRequest &request) {
    return input_factories.at(static_cast<size_t>(type))(request);
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
        result.compared_elements = actual.size();
        for (size_t i = 0; i < actual.size(); ++i) {
            const float absolute = std::fabs(actual[i] - expected[i]);
            const float relative = absolute / std::max(std::fabs(expected[i]), 1.0e-6f);
            result.max_absolute_error = std::max(result.max_absolute_error, absolute);
            result.max_relative_error = std::max(result.max_relative_error, relative);
            if (absolute > 1.0e-3f && relative > 1.0e-3f) {
                if (result.verification_passed) {
                    char detail[160];
                    std::snprintf(detail, sizeof(detail),
                                  "output differs from llama.cpp reference at index %zu: actual=%.9g expected=%.9g",
                                  i, actual[i], expected[i]);
                    result.message = detail;
                }
                result.verification_passed = false;
                ++result.error_elements;
            }
        }
        result.error_element_ratio = result.compared_elements
                                         ? static_cast<double>(result.error_elements) / result.compared_elements
                                         : 0.0;
        if (!result.verification_passed) {
            if (result.message.empty()) result.message = "output differs from llama.cpp reference";
            return result;
        }
    }

    kernel.callbacks.reset(prepared.state);
    kernel.callbacks.run(prepared.state, request.warmup_iterations);
    if (request.samples == 0 || request.perf_events.empty()) {
        result.message = "samples and perf event list must be non-empty";
        return result;
    }

    size_t iterations = request.iterations;
    if (!iterations) {
        PerfEventGroup calibration;
        const std::array calibration_events{cycles_event()};
        std::string error;
        if (!calibration.open(calibration_events, error)) {
            result.message = error;
            return result;
        }
        kernel.callbacks.reset(prepared.state);
        asm volatile("" ::: "memory");
        if (!calibration.start(error)) {
            result.message = error;
            return result;
        }
        kernel.callbacks.run(prepared.state, 1);
        asm volatile("" ::: "memory");
        std::vector<PerfEventValue> values;
        if (!calibration.stop_and_read(values, error)) {
            result.message = error;
            return result;
        }
        if (!values[0].time_enabled || values[0].time_running * 100 < values[0].time_enabled * 99) {
            result.message = "cycles calibration counter ran less than 99%";
            return result;
        }
        const uint64_t elapsed = std::max<uint64_t>(values[0].value, 1);
        iterations = std::max<size_t>(1, request.target_cycles / elapsed);
    }
    result.iterations = iterations;

    PerfEventGroup counters;
    std::string error;
    if (!counters.open(request.perf_events, error)) {
        result.message = error;
        return result;
    }
    std::vector<std::vector<double>> event_samples(request.perf_events.size());
    std::vector<double> minimum_running(request.perf_events.size(), 100.0);
    for (size_t sample = 0; sample < request.samples; ++sample) {
        kernel.callbacks.reset(prepared.state);
        asm volatile("" ::: "memory");
        if (!counters.start(error)) {
            result.message = error;
            return result;
        }
        kernel.callbacks.run(prepared.state, iterations);
        asm volatile("" ::: "memory");
        std::vector<PerfEventValue> values;
        if (!counters.stop_and_read(values, error)) {
            result.message = error;
            return result;
        }
        for (size_t event = 0; event < values.size(); ++event) {
            const auto &value = values[event];
            if (!value.time_enabled) {
                result.message = "perf event " + request.perf_events[event].name + " was never enabled";
                return result;
            }
            const double running = 100.0 * static_cast<double>(value.time_running) / value.time_enabled;
            minimum_running[event] = std::min(minimum_running[event], running);
            if (running < 99.0) {
                char detail[192];
                std::snprintf(detail, sizeof(detail), "perf event %s ran only %.3f%% (need >=99%%)", request.perf_events[event].name.c_str(), running);
                result.message = detail;
                return result;
            }
            event_samples[event].push_back(static_cast<double>(value.value) / iterations);
        }
    }

    bool has_cycles = false;
    for (size_t event = 0; event < request.perf_events.size(); ++event) {
        auto &samples = event_samples[event];
        std::sort(samples.begin(), samples.end());
        PerfMetricResult metric;
        metric.name = request.perf_events[event].name;
        metric.min_per_iteration = samples.front();
        metric.median_per_iteration = samples[samples.size() / 2];
        metric.min_running_percent = minimum_running[event];
        result.perf_metrics.push_back(metric);
        if (metric.name == "cycles") {
            has_cycles = true;
            result.min_cycles = static_cast<uint64_t>(metric.min_per_iteration);
            result.median_cycles = static_cast<uint64_t>(metric.median_per_iteration);
        }
    }
    result.checksum = kernel.callbacks.checksum(prepared.state);
    if (has_cycles && result.median_cycles) {
        const double fma = static_cast<double>(request.m) * request.n * request.k;
        result.fma_per_cycle = fma / result.median_cycles;
        result.utilization_percent = result.fma_per_cycle / 128.0 * 100.0;
    }
    return result;
}

} // namespace ime::bench