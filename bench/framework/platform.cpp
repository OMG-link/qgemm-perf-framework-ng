#include "platform.h"

#include <algorithm>
#include <fstream>
#include <iterator>

#include "build_manifest.h"

namespace pfng::bench {
namespace {

std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

std::vector<std::string> split_compatible(std::string contents) {
    std::vector<std::string> values;
    size_t begin = 0;
    while (begin < contents.size()) {
        const size_t end = contents.find('\0', begin);
        const size_t length = (end == std::string::npos ? contents.size() : end) - begin;
        if (length) values.emplace_back(contents.substr(begin, length));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

void trim_trailing_nuls(std::string &value) {
    while (!value.empty() && (value.back() == '\0' || value.back() == '\n'))
        value.pop_back();
}

} // namespace

std::string_view platform_name(PlatformId platform) noexcept {
    switch (platform) {
    case PlatformId::SpacemitK1A60: return "spacemit-k1-a60";
    case PlatformId::SpacemitK1X60: return "spacemit-k1-x60";
    case PlatformId::Unknown: return "unknown";
    }
    return "unknown";
}

PlatformId platform_from_name(std::string_view name) noexcept {
    if (name == "spacemit-k1-a60") return PlatformId::SpacemitK1A60;
    if (name == "spacemit-k1-x60") return PlatformId::SpacemitK1X60;
    return PlatformId::Unknown;
}

PlatformId build_target_platform() noexcept {
    return platform_from_name(kBuildTargetPlatform);
}

PlatformDetectionResult detect_runtime_platform(
    const std::filesystem::path &device_tree_root) {
    PlatformDetectionResult result;
    const std::string compatible = read_file(device_tree_root / "compatible");
    if (compatible.empty()) {
        result.error = "unable to read Device Tree compatible";
        return result;
    }
    result.compatibles = split_compatible(compatible);
    result.model = read_file(device_tree_root / "model");
    trim_trailing_nuls(result.model);

    for (const auto expected : kBuildDeviceTreeCompatibles) {
        if (std::find(result.compatibles.begin(), result.compatibles.end(),
                      expected) != result.compatibles.end()) {
            result.platform = build_target_platform();
            return result;
        }
    }
    result.error = "unrecognized Device Tree compatible";
    return result;
}

bool verify_runtime_platform(const PlatformDetectionResult &detected,
                             std::string &error) {
    if (detected.platform == PlatformId::Unknown) {
        error = detected.error.empty() ? "unable to identify runtime platform"
                                       : detected.error;
        return false;
    }
    if (detected.platform != build_target_platform()) {
        error = "binary was built for " + std::string(kBuildTargetPlatform) +
                ", but runtime platform is " +
                std::string(platform_name(detected.platform));
        return false;
    }
    return true;
}

} // namespace pfng::bench