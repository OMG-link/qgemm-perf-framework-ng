#include "kernel_registry.h"

#include <cstdio>
#include <cstdlib>

namespace ime::bench {
namespace {
std::vector<KernelRegistration> &registry() {
    static std::vector<KernelRegistration> value;
    return value;
}

[[noreturn]] void registration_error(std::string_view message, std::string_view id) {
    std::fprintf(stderr, "kernel registration error for '%.*s': %.*s\n",
                 static_cast<int>(id.size()), id.data(), static_cast<int>(message.size()), message.data());
    std::abort();
}
} // namespace

void register_kernel(const KernelRegistration &registration) {
    if (registration.id.empty()) registration_error("empty id", registration.id);
    if (registration.name.empty()) registration_error("empty name", registration.id);
    const auto &cb = registration.callbacks;
    if (!cb.validate || !cb.prepare || !cb.reset || !cb.run || !cb.export_output || !cb.checksum || !cb.destroy) {
        registration_error("missing required callback", registration.id);
    }
    if (find_kernel(registration.id)) registration_error("duplicate id", registration.id);
    registry().push_back(registration);
}

std::span<const KernelRegistration> registered_kernels() { return registry(); }

const KernelRegistration *find_kernel(std::string_view id) {
    for (const auto &kernel : registry()) {
        if (kernel.id == id) return &kernel;
    }
    return nullptr;
}

} // namespace ime::bench