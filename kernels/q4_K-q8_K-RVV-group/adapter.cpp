#include "adapter_common.h"
#include "llama_reference.h"
#include <array>
#include <cstring>
#include <memory>

void ggml_gemm_q4_K_8x32_q8_K_group(int, float *, size_t, const void *, const void *, int, int);
namespace ime::bench::adapters {
namespace {

uint8_t q4_scale(const block_q4_K &block, size_t index) {
    if (index < 4)
        return block.scales[index] & 0x3f;
    return (block.scales[index + 4] & 0x0f) | ((block.scales[index - 4] >> 6) << 4);
}

uint8_t q4_min(const block_q4_K &block, size_t index) {
    if (index < 4)
        return block.scales[index + 4] & 0x3f;
    return (block.scales[index + 4] >> 4) | ((block.scales[index] >> 6) << 4);
}

uint8_t q4_value(const block_q4_K &block, size_t index) {
    const uint8_t packed = block.qs[(index / 64) * 32 + index % 32];
    return index % 64 < 32 ? packed & 0x0f : packed >> 4;
}

void pack_group_subblock(uint8_t *destination, std::span<const block_q4_K> source, size_t subblock) {
    for (size_t column = 0; column < 16; ++column) {
        const uint8_t scale_low = q4_scale(source[column], subblock);
        const uint8_t scale_high = q4_scale(source[column + 16], subblock);
        const uint8_t min_low = q4_min(source[column], subblock);
        const uint8_t min_high = q4_min(source[column + 16], subblock);
        destination[column] = scale_low | ((scale_high >> 4) << 6);
        destination[16 + column] = min_low | ((min_high >> 4) << 6);
        destination[32 + column] = (scale_high & 0x0f) | ((min_high & 0x0f) << 4);
    }
}

struct PackedQ4K32 {
    _Float16 d[32], dmin[32];
    uint8_t scales[512];
    uint8_t qs[4096];
};
struct PackedQ8K4 {
    float d[4];
    int8_t qs[1024];
    uint16_t bsums[32];
};
struct State : CommonState {
    std::vector<PackedQ4K32> w;
    std::vector<PackedQ8K4> a;
};
PrepareResult prepare(const BenchmarkRequest &r, const BenchmarkInput &g) {
    auto in = std::get_if<Q4_KQ8_KInput>(&g);
    if (!in)
        return {nullptr, "expected Q4_K/Q8_K input"};
    auto s = std::make_unique<State>();
    s->m_value = r.m;
    s->n_value = r.n;
    s->k_value = r.k;
    s->blocks_k_value = r.k / QK_K;
    s->output_value.resize(r.m * r.n);
    s->w.resize(r.n / 32 * s->blocks_k());
    s->a.resize(r.m / 4 * s->blocks_k());
    for (size_t tn = 0; tn < r.n; tn += 32)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &o = s->w[tn / 32 * s->blocks_k() + b];
            for (size_t x = 0; x < 32; ++x) {
                auto &v = in->weight[(tn + x) * s->blocks_k() + b];
                std::memcpy(&o.d[x], &v.d, sizeof(v.d));
                std::memcpy(&o.dmin[x], &v.dmin, sizeof(v.dmin));
                if (x == 0) {
                    std::array<block_q4_K, 32> records;
                    for (size_t column = 0; column < 32; ++column)
                        records[column] = in->weight[(tn + column) * s->blocks_k() + b];
                    for (size_t sb = 0; sb < 4; ++sb) {
                        pack_group_subblock(o.scales + sb * 96, records, sb * 2);
                        pack_group_subblock(o.scales + sb * 96 + 48, records, sb * 2 + 1);
                    }
                }
                for (size_t group = 0; group < 4; ++group)
                    for (size_t z = 0; z < 32; ++z) {
                        const uint8_t low = q4_value(v, (group * 2) * 32 + z);
                        const uint8_t high = q4_value(v, (group * 2 + 1) * 32 + z);
                        o.qs[group * 1024 + z * 32 + x] = low | (high << 4);
                    }
            }
        }
    std::vector<block_q8_K> aq(r.m * s->blocks_k());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_K(in->activation.data() + m * r.k, aq.data() + m * s->blocks_k(), r.k);
    for (size_t tm = 0; tm < r.m; tm += 4)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &o = s->a[tm / 4 * s->blocks_k() + b];
            for (size_t x = 0; x < 4; ++x) {
                o.d[x] = aq[(tm + x) * s->blocks_k() + b].d;
                for (size_t i = 0; i < 256; ++i)
                    o.qs[i * 4 + x] = aq[(tm + x) * s->blocks_k() + b].qs[i];
                for (size_t i = 0; i < 8; ++i)
                    o.bsums[i * 4 + x] = 0;
                for (size_t i = 0; i < 256; ++i)
                    o.bsums[(i / 32) * 4 + x] += aq[(tm + x) * s->blocks_k() + b].qs[i];
            }
        }
    return {s.release(), {}};
}
void run(KernelState p, size_t it) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t q = 0; q < it; ++q)
        for (size_t m = 0; m < s.m(); m += 4)
            for (size_t n = 0; n < s.n(); n += 32)
                ggml_gemm_q4_K_8x32_q8_K_group(s.k(), s.output().data() + m * s.n() + n, s.n(), s.w.data() + n / 32 * s.blocks_k(), s.a.data() + m / 4 * s.blocks_k(), 4, 32);
}
} // namespace
void register_q4_K_rvv_group() {
    register_kernel({"q4_K-q8_K-RVV-group",
                     "Q4_K/Q8_K RVV grouped 4x32",
                     QuantizationType::WeightQ4_KActivationQ8_K,
                     {validate_shape<4, 32, 256>, prepare, reset_common, run, export_common, checksum_common, [](KernelState p) noexcept { delete static_cast<State *>(p); }}});
}
} // namespace ime::bench::adapters