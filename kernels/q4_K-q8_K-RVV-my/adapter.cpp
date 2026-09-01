#include "adapter_common.h"
#include "llama_reference.h"
#include <cstring>
#include <memory>
void ggml_gemm_q4_K_8x32_q8_K_my(int, float *, size_t, const void *, const void *, int, int);
namespace ime::bench::adapters {
namespace {

uint8_t q4_value(const block_q4_K &block, size_t index) {
    const uint8_t packed = block.qs[index % 128];
    return index < 128 ? packed & 0x0f : packed >> 4;
}

struct PackedQ4K32 {
    _Float16 d[32], dmin[32];
    int8_t scales[256], mins[256];
    uint8_t qs[4096];
};
struct PackedQ8K12 {
    float d[12];
    int8_t qs[3072];
    uint16_t bsums[96];
};
struct State : CommonState {
    std::vector<PackedQ4K32> w;
    std::vector<PackedQ8K12> a;
};
PrepareResult prepare(const BenchmarkRequest &r, const BenchmarkInput &g) {
    auto in = std::get_if<Q4_KQ8_KInput>(&g);
    if (!in)
        return {nullptr, "expected Q4_K/Q8_K input"};
    auto s = std::make_unique<State>();
    s->m_value = r.m;
    s->n_value = r.n;
    s->k_value = r.k;
    s->blocks_k_value = r.k / 256;
    s->output_value.resize(r.m * r.n);
    s->w.resize(r.n / 32 * s->blocks_k());
    s->a.resize(r.m / 12 * s->blocks_k());
    for (size_t tn = 0; tn < r.n; tn += 32)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &o = s->w[tn / 32 * s->blocks_k() + b];
            for (size_t x = 0; x < 32; ++x) {
                auto &v = in->weight[(tn + x) * s->blocks_k() + b];
                std::memcpy(&o.d[x], &v.d, sizeof(v.d));
                std::memcpy(&o.dmin[x], &v.dmin, sizeof(v.dmin));
                for (size_t i = 0; i < 4; ++i) {
                    o.scales[i * 32 + x] = v.scales[i] & 63;
                    o.mins[i * 32 + x] = v.scales[4 + i] & 63;
                    o.scales[(i + 4) * 32 + x] = (v.scales[8 + i] & 15) | ((v.scales[i] >> 6) << 4);
                    o.mins[(i + 4) * 32 + x] = (v.scales[8 + i] >> 4) | ((v.scales[4 + i] >> 6) << 4);
                }
                for (size_t subblock = 0; subblock < 8; ++subblock)
                    for (size_t pair = 0; pair < 16; ++pair) {
                        const size_t index = subblock * 32 + pair * 2;
                        const uint8_t low = q4_value(v, index);
                        const uint8_t high = q4_value(v, index + 1);
                        o.qs[(subblock * 16 + pair) * 32 + x] = low | (high << 4);
                    }
            }
        }

    std::vector<block_q8_K> aq(r.m * s->blocks_k());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_K(in->activation.data() + m * r.k, aq.data() + m * s->blocks_k(), r.k);
    for (size_t tm = 0; tm < r.m; tm += 12)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &o = s->a[tm / 12 * s->blocks_k() + b];
            for (size_t x = 0; x < 12; ++x) {
                o.d[x] = aq[(tm + x) * s->blocks_k() + b].d;
                for (size_t i = 0; i < 256; ++i)
                    o.qs[i * 12 + x] = aq[(tm + x) * s->blocks_k() + b].qs[i];
                for (size_t i = 0; i < 8; ++i)
                    o.bsums[i * 12 + x] = 0;
                for (size_t i = 0; i < 256; ++i)
                    o.bsums[(i / 32) * 12 + x] += aq[(tm + x) * s->blocks_k() + b].qs[i];
            }
        }
    return {s.release(), {}};
}
void run(KernelState p, size_t it) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t q = 0; q < it; ++q)
        for (size_t m = 0; m < s.m(); m += 12)
            for (size_t n = 0; n < s.n(); n += 32)
                ggml_gemm_q4_K_8x32_q8_K_my(s.k(), s.output().data() + m * s.n() + n, s.n(), s.w.data() + n / 32 * s.blocks_k(), s.a.data() + m / 12 * s.blocks_k(), 12, 32);
}
} // namespace
void register_q4_K_rvv_my() {
    register_kernel({"q4_K-q8_K-RVV-my",
                     "Q4_K/Q8_K RVV custom 12x32",
                     QuantizationType::WeightQ4_KActivationQ8_K,
                     {validate_shape<12, 32, 256>, prepare, reset_common, run, export_common, checksum_common, [](KernelState p) noexcept { delete static_cast<State *>(p); }}});
}
} // namespace ime::bench::adapters