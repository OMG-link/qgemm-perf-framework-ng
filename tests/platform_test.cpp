#include "platform.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void write_binary(const std::filesystem::path &path, const std::string &value) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

bool expect(bool condition, const char *message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    using namespace pfng::bench;

    const fs::path root = fs::temp_directory_path() / "pfng-platform-test";
    fs::remove_all(root);
    fs::create_directories(root);

    bool passed = true;
    std::string compatible = "vendor,generic";
    compatible.push_back('\0');
    compatible += "spacemit,k1-x";
    compatible.push_back('\0');
    write_binary(root / "compatible", compatible);
    std::string model = "spacemit k1-x deb1 board";
    model.push_back('\0');
    write_binary(root / "model", model);
    auto detected = detect_runtime_platform(root);
    passed &= expect(detected.platform == build_target_platform(),
                     "recognizes the configured K1 target platform");
    passed &= expect(detected.compatibles.size() == 2,
                     "parses NUL-separated compatible values");
    passed &= expect(detected.model == "spacemit k1-x deb1 board",
                     "trims the model's trailing NUL");

    compatible = "vendor,unknown";
    compatible.push_back('\0');
    write_binary(root / "compatible", compatible);
    detected = detect_runtime_platform(root);
    passed &= expect(detected.platform == PlatformId::Unknown,
                     "rejects an unknown compatible");
    passed &= expect(!detected.error.empty(), "reports an unknown-compatible error");

    fs::remove(root / "compatible");
    detected = detect_runtime_platform(root);
    passed &= expect(detected.platform == PlatformId::Unknown,
                     "rejects a missing compatible file");

    fs::remove_all(root);
    return passed ? 0 : 1;
}