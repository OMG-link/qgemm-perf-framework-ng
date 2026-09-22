#ifndef PFNG_BENCH_PLATFORM_H
#define PFNG_BENCH_PLATFORM_H

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace pfng::bench {

enum class PlatformId {
    Unknown,
    SpacemitK1A60,
    SpacemitK1X60,
};

struct PlatformDetectionResult {
    PlatformId platform = PlatformId::Unknown;
    std::string model;
    std::vector<std::string> compatibles;
    std::string error;
};

std::string_view platform_name(PlatformId platform) noexcept;
PlatformId platform_from_name(std::string_view name) noexcept;
PlatformId build_target_platform() noexcept;
PlatformDetectionResult detect_runtime_platform(
    const std::filesystem::path &device_tree_root =
        "/sys/firmware/devicetree/base");
bool verify_runtime_platform(const PlatformDetectionResult &detected,
                             std::string &error);

} // namespace pfng::bench

#endif