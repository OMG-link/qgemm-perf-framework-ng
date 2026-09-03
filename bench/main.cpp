#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <array>
#include <vector>

#include "adapter_common.h"
#include "benchmark.h"

namespace {

bool parse_size(const char *text, size_t &value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end || !parsed || parsed > SIZE_MAX) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

void usage(const char *program) {
    std::printf("Usage: %s [--list] [--kernel ID|all] --m M --n N --k K "
                "[--warmup N] [--samples N] [--iterations N] [--no-verify]\n", program);
}

} // namespace

int main(int argc, char **argv) {
    using namespace ime::bench;
    adapters::register_q4_0_ime_upstream();
    adapters::register_q4_0_ime_m4b();
    adapters::register_q4_0_ime_m4i();
    adapters::register_q4_0_ime_m8b();
    adapters::register_q4_0_ime_m4n32b();
    adapters::register_q4_0_ime_m32n4();
    adapters::register_q4_0_rvv_group();
    adapters::register_q4_0_rvv_my();
    adapters::register_q4_0_rvv_upstream();
    adapters::register_q4_K_rvv_group();
    adapters::register_q4_K_rvv_my();
    adapters::register_q4_K_rvv_upstream();
    adapters::register_iq2_xxs_rvv_my_br();
    adapters::register_iq2_xxs_rvv_my_ir();
    adapters::register_iq2_xxs_rvv_upstream();
    adapters::register_iq2_xxs_rvv_my_br_tcm();
    adapters::register_iq2_xxs_rvv_my_ir_tcm();

    BenchmarkRequest request;
    std::string_view selected = "all";
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--list") list = true;
        else if (argument == "--no-verify") request.verify = false;
        else if (argument == "--kernel" && i + 1 < argc) selected = argv[++i];
        else if (argument == "--m" && i + 1 < argc && parse_size(argv[++i], request.m)) {}
        else if (argument == "--n" && i + 1 < argc && parse_size(argv[++i], request.n)) {}
        else if (argument == "--k" && i + 1 < argc && parse_size(argv[++i], request.k)) {}
        else if (argument == "--warmup" && i + 1 < argc && parse_size(argv[++i], request.warmup_iterations)) {}
        else if (argument == "--samples" && i + 1 < argc && parse_size(argv[++i], request.samples)) {}
        else if (argument == "--iterations" && i + 1 < argc && parse_size(argv[++i], request.iterations)) {}
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (list) {
        for (const auto &kernel : registered_kernels()) {
            std::printf("%-24.*s %.*s\n", static_cast<int>(kernel.id.size()), kernel.id.data(),
                        static_cast<int>(kernel.name.size()), kernel.name.data());
        }
        return 0;
    }
    if (!request.m || !request.n || !request.k) {
        usage(argv[0]);
        return 2;
    }
    std::vector<const KernelRegistration *> kernels;
    if (selected == "all") {
        for (const auto &kernel : registered_kernels()) kernels.push_back(&kernel);
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
        for (const auto *kernel : kernels) has_kernel |= kernel->quantization == type;
        if (!has_kernel) continue;
        const auto input_owner = create_input(type, request);
        const auto input = std::visit([](const auto &owner) { return owner.view(); }, input_owner);
        for (const auto *kernel : kernels) {
            if (kernel->quantization != type) continue;
            const auto result = run_benchmark(*kernel, request, input);
            std::printf("\n[%.*s] %.*s\n", static_cast<int>(kernel->id.size()), kernel->id.data(),
                        static_cast<int>(kernel->name.size()), kernel->name.data());
            if (result.skipped) {
                std::printf("status: SKIP (%s)\n", result.message.c_str());
                continue;
            }
            if (result.verified) {
                std::printf("error elements: %zu/%zu (%.6f%%)\n",
                            result.error_elements, result.compared_elements,
                            result.error_element_ratio * 100.0);
            } else {
                std::printf("error elements: unavailable (verification disabled or not completed)\n");
            }
            if (!result.message.empty()) {
                std::printf("status: FAIL (%s), max_abs=%.6g, max_rel=%.6g\n", result.message.c_str(),
                            result.max_absolute_error, result.max_relative_error);
                failed = true;
                continue;
            }
            std::printf("verify: %s, max_abs=%.6g, max_rel=%.6g\n",
                        result.verified ? "PASS" : "disabled", result.max_absolute_error,
                        result.max_relative_error);
            std::printf("cycles: min=%llu median=%llu iterations=%zu samples=%zu\n", static_cast<unsigned long long>(result.min_cycles), static_cast<unsigned long long>(result.median_cycles),
                        result.iterations, request.samples);
            std::printf("performance: %.4f FMA/cycle, %.2f%% of 128 FMA/cycle\n",
                        result.fma_per_cycle, result.utilization_percent);
            std::printf("checksum: %.9g\n", result.checksum);
        }
    }
    return failed ? 1 : 0;
}