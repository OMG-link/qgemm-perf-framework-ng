#ifndef IME_BENCH_IQ2_XXS_H
#define IME_BENCH_IQ2_XXS_H

#include <array>
#include <cstdint>
#include <cstring>

#include "ggml.h"

namespace ime::bench::iq2_xxs {

// The input generator deliberately uses these 32 distinct entries. The kernels
// retain the complete llama.cpp table and therefore still support all 256.
inline constexpr std::array<uint64_t, 32> grid = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
};

inline uint8_t sign_mask(uint8_t index) {
    return index | static_cast<uint8_t>((__builtin_parity(index) != 0) << 7);
}

inline void decode_subblock(const uint8_t *packed, int8_t *values, int &scale) {
    uint32_t signs_and_scale;
    std::memcpy(&signs_and_scale, packed + 4, sizeof(signs_and_scale));
    for (size_t group = 0; group < 4; ++group) {
        uint64_t magnitudes = grid[packed[group]];
        const uint8_t signs = sign_mask((signs_and_scale >> (7 * group)) & 0x7f);
        for (size_t lane = 0; lane < 8; ++lane) {
            const int8_t magnitude = (magnitudes >> (lane * 8)) & 0xff;
            values[group * 8 + lane] = signs & (1u << lane) ? -magnitude : magnitude;
        }
    }
    scale = 2 * ((signs_and_scale >> 28) & 0x0f) + 1;
}

} // namespace ime::bench::iq2_xxs

#endif