#include "adapter_common.h"

#include <algorithm>
namespace ime::bench::adapters {

void reset_common(KernelState opaque) {
    auto &state = *static_cast<CommonState *>(opaque);
    std::fill(state.output().begin(), state.output().end(), 0.0f);
}

ExportResult export_common(KernelState opaque, std::span<float> output) {
    const auto &state = *static_cast<CommonState *>(opaque);
    if (output.size() != state.output().size()) return {false, "output size mismatch"};
    std::copy(state.output().begin(), state.output().end(), output.begin());
    return {true, {}};
}

double checksum_common(KernelState opaque) {
    const auto &state = *static_cast<CommonState *>(opaque);
    double sum = 0.0;
    for (size_t i = 0; i < state.output().size(); i += 257) sum += state.output()[i];
    return sum;
}

void destroy_common(KernelState state) noexcept { delete static_cast<CommonState *>(state); }

} // namespace ime::bench::adapters