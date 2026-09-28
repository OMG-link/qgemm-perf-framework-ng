#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "adapter_common.h"
#include "benchmark.h"
#include "build_manifest.h"
#include "perf.h"
#include "platform.h"
#include "tcm_arena.h"

namespace {

bool parse_size(const char *text, size_t &value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end || !parsed || parsed > SIZE_MAX)
        return false;
    value = static_cast<size_t>(parsed);
    return true;
}

bool parse_perf_event(std::string_view text, pfng::bench::PerfEventSpec &event) {
    using namespace pfng::bench;
    const auto cache = [](uint64_t cache_id, uint64_t result) { return cache_id | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (result << 16); };
    if (text == "cycles")
        event = cycles_event();
    else if (text == "instructions")
        event = {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS};
    else if (text == "task-clock")
        event = {"task-clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK};
    else if (text == "page-faults")
        event = {"page-faults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS};
    else if (text == "dtlb-access")
        event = {"dtlb-access", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_RESULT_ACCESS)};
    else if (text == "dtlb-miss")
        event = {"dtlb-miss", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_RESULT_MISS)};
    else if (text == "l1d-access")
        event = {"l1d-access", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_RESULT_ACCESS)};
    else if (text == "l1d-miss")
        event = {"l1d-miss", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_RESULT_MISS)};
    else if (text == "llc-access")
        event = {"llc-access", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_RESULT_ACCESS)};
    else if (text == "llc-miss")
        event = {"llc-miss", PERF_TYPE_HW_CACHE, cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_RESULT_MISS)};
    else if (text.starts_with("raw:")) {
        const auto separator = text.find(':', 4);
        if (separator == std::string_view::npos || separator == 4 || separator + 1 == text.size())
            return false;
        const std::string config_text(text.substr(separator + 1));
        errno = 0;
        char *end = nullptr;
        const unsigned long long config = std::strtoull(config_text.c_str(), &end, 0);
        if (errno || end == config_text.c_str() || *end)
            return false;
        event = {std::string(text.substr(4, separator - 4)), PERF_TYPE_RAW, config};
    } else
        return false;
    return true;
}

bool append_perf_events(std::string_view list, std::vector<pfng::bench::PerfEventSpec> &events) {
    while (!list.empty()) {
        const auto comma = list.find(',');
        const auto token = list.substr(0, comma);
        pfng::bench::PerfEventSpec event;
        if (token.empty() || !parse_perf_event(token, event))
            return false;
        for (const auto &existing : events)
            if (existing.name == event.name)
                return false;
        events.push_back(std::move(event));
        if (comma == std::string_view::npos)
            break;
        list.remove_prefix(comma + 1);
    }
    return true;
}

void usage(const char *program) {
    std::printf("Usage: %s [--build-info|--platform-info|--check-platform|--list] "
                "[--kernel ID|all] --m M --n N --k K "
                "[--warmup N] [--samples N] [--iterations N] [--threads N] [--no-verify] "
                "[--perf-events LIST] [--perf-event EVENT]\n"
                "Events: cycles,instructions,task-clock,page-faults,dtlb-access,dtlb-miss,"
                "l1d-access,l1d-miss,llc-access,llc-miss,raw:NAME:CONFIG\n",
                program);
}

void print_build_info() {
    std::printf("platform: %.*s\n", static_cast<int>(pfng::bench::kBuildTargetPlatform.size()),
                pfng::bench::kBuildTargetPlatform.data());
    std::printf("platform display name: %.*s\n",
                static_cast<int>(pfng::bench::kBuildPlatformDisplayName.size()),
                pfng::bench::kBuildPlatformDisplayName.data());
    std::printf("arch: %.*s\n", static_cast<int>(pfng::bench::kBuildTargetArch.size()),
                pfng::bench::kBuildTargetArch.data());
    std::printf("tune: %.*s\n", static_cast<int>(pfng::bench::kBuildTargetTune.size()),
                pfng::bench::kBuildTargetTune.data());
    std::printf("compiler: %.*s\n", static_cast<int>(pfng::bench::kBuildCompiler.size()),
                pfng::bench::kBuildCompiler.data());
    std::printf("compiler version: %.*s\n",
                static_cast<int>(pfng::bench::kBuildCompilerVersion.size()),
                pfng::bench::kBuildCompilerVersion.data());
    std::printf("allowed CPUs: %zu-%zu\n", pfng::bench::kBuildFirstCpu,
                pfng::bench::kBuildLastCpu);
    std::printf("maximum threads: %zu\n", pfng::bench::kBuildMaxThreads);
}

void print_platform_info(const pfng::bench::PlatformDetectionResult &detected) {
    std::printf("detected platform: %.*s\n",
                static_cast<int>(pfng::bench::platform_name(detected.platform).size()),
                pfng::bench::platform_name(detected.platform).data());
    std::printf("device-tree model: %s\n", detected.model.empty() ? "unavailable" : detected.model.c_str());
    std::printf("device-tree compatible:\n");
    for (const auto &compatible : detected.compatibles)
        std::printf("  %s\n", compatible.c_str());
    if (!detected.error.empty()) std::printf("detection error: %s\n", detected.error.c_str());
}

} // namespace

int main(int argc, char **argv) {
    using namespace pfng::bench;
    adapters::register_builtin_kernels();

    BenchmarkRequest request;
    request.perf_events.push_back(cycles_event());
    std::string_view selected = "all";
    bool list = false;
    bool build_info = false;
    bool platform_info = false;
    bool check_platform = false;
    bool perf_events_explicit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--list")
            list = true;
        else if (argument == "--build-info")
            build_info = true;
        else if (argument == "--platform-info")
            platform_info = true;
        else if (argument == "--check-platform")
            check_platform = true;
        else if (argument == "--no-verify")
            request.verify = false;
        else if (argument == "--kernel" && i + 1 < argc)
            selected = argv[++i];
        else if (argument == "--m" && i + 1 < argc && parse_size(argv[++i], request.m)) {
        } else if (argument == "--n" && i + 1 < argc && parse_size(argv[++i], request.n)) {
        } else if (argument == "--k" && i + 1 < argc && parse_size(argv[++i], request.k)) {
        } else if (argument == "--warmup" && i + 1 < argc && parse_size(argv[++i], request.warmup_iterations)) {
        } else if (argument == "--samples" && i + 1 < argc && parse_size(argv[++i], request.samples)) {
        } else if (argument == "--iterations" && i + 1 < argc && parse_size(argv[++i], request.iterations)) {
        } else if (argument == "--threads" && i + 1 < argc && parse_size(argv[++i], request.threads) && request.threads <= kBuildMaxThreads) {
        } else if ((argument == "--perf-events" || argument == "--perf-event") && i + 1 < argc) {
            if (!perf_events_explicit) {
                request.perf_events.clear();
                perf_events_explicit = true;
            }
            if (!append_perf_events(argv[++i], request.perf_events)) {
                std::fprintf(stderr, "invalid or duplicate perf event: %s\n", argv[i]);
                usage(argv[0]);
                return 2;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (build_info) {
        print_build_info();
        return 0;
    }

    const auto detected_platform = detect_runtime_platform();
    if (platform_info) print_platform_info(detected_platform);
    std::string platform_error;
    if (!verify_runtime_platform(detected_platform, platform_error)) {
        if (!platform_info) print_platform_info(detected_platform);
        std::fprintf(stderr, "platform check failed: %s\n", platform_error.c_str());
        std::fprintf(stderr, "build platform: %.*s\n",
                     static_cast<int>(kBuildTargetPlatform.size()), kBuildTargetPlatform.data());
        return 3;
    }
    if (platform_info || check_platform) {
        std::printf("platform check: PASS\n");
        return 0;
    }
    if (list) {
        for (const auto &kernel : registered_kernels()) {
            std::printf("%-24.*s %.*s [platform=%.*s]\n", static_cast<int>(kernel.id.size()), kernel.id.data(), static_cast<int>(kernel.name.size()), kernel.name.data(),
                        static_cast<int>(platform_name(kernel.platform).size()), platform_name(kernel.platform).data());
        }
        return 0;
    }
    if (!request.m || !request.n || !request.k) {
        usage(argv[0]);
        return 2;
    }
    tcm::initialize();
    std::vector<const KernelRegistration *> kernels;
    if (selected == "all") {
        for (const auto &kernel : registered_kernels())
            kernels.push_back(&kernel);
    } else if (const auto *kernel = find_kernel(selected)) {
        kernels.push_back(kernel);
    } else {
        std::fprintf(stderr, "unknown kernel: %.*s\n", static_cast<int>(selected.size()), selected.data());
        return 2;
    }

    bool failed = false;
    for (size_t type_index = 0; type_index < static_cast<size_t>(QuantizationType::Count); ++type_index) {
        const auto type = static_cast<QuantizationType>(type_index);
        bool has_kernel = false;
        for (const auto *kernel : kernels)
            has_kernel |= kernel->quantization == type;
        if (!has_kernel)
            continue;
        const auto input_owner = create_input(type, request);
        const auto input = std::visit([](const auto &owner) { return owner.view(); }, input_owner);
        for (const auto *kernel : kernels) {
            if (kernel->quantization != type)
                continue;
            const auto result = run_benchmark(*kernel, request, input);
            std::printf("\n[%.*s] %.*s\n", static_cast<int>(kernel->id.size()), kernel->id.data(), static_cast<int>(kernel->name.size()), kernel->name.data());
            if (result.skipped) {
                std::printf("status: SKIP (%s)\n", result.message.c_str());
                continue;
            }
            if (result.verified) {
                std::printf("error elements: %zu/%zu (%.6f%%)\n", result.error_elements, result.compared_elements, result.error_element_ratio * 100.0);
            } else {
                std::printf("error elements: unavailable (verification disabled or not completed)\n");
            }
            if (!result.message.empty()) {
                std::printf("status: FAIL (%s), max_abs=%.6g, max_rel=%.6g\n", result.message.c_str(), result.max_absolute_error, result.max_relative_error);
                failed = true;
                continue;
            }
            std::printf("verify: %s, max_abs=%.6g, max_rel=%.6g\n", result.verified ? "PASS" : "disabled", result.max_absolute_error, result.max_relative_error);
            bool reported_cycles = false;
            for (const auto &metric : result.perf_metrics) {
                std::printf("perf: event=%s min=%.6f median=%.6f min_running_pct=%.6f\n", metric.name.c_str(), metric.min_per_iteration, metric.median_per_iteration, metric.min_running_percent);
                reported_cycles |= metric.name == "cycles";
            }
            if (reported_cycles) {
                std::printf("cycles: min=%llu median=%llu iterations=%zu samples=%zu\n", static_cast<unsigned long long>(result.min_cycles), static_cast<unsigned long long>(result.median_cycles),
                            result.iterations, request.samples);
                std::printf("performance: %.4f %s/cycle, %.2f%% of %.4f %s/cycle\n",
                            result.operations_per_cycle, result.operation_name.c_str(),
                            result.utilization_percent, result.peak_operations_per_cycle,
                            result.operation_name.c_str());
            } else {
                std::printf("cycles: unavailable iterations=%zu samples=%zu\n", result.iterations, request.samples);
                std::printf("performance: unavailable (cycles event not requested)\n");
            }
            std::printf("checksum: %.9g\n", result.checksum);
        }
    }
    return failed ? 1 : 0;
}