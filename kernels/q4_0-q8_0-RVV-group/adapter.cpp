#include "adapter_common.h"
#include "ggml_def.h"
#include "llama_reference.h"

#include <cstring>
#include <memory>

void ggml_gemm_q4_0_12x32_q8_0(int k, float *s, size_t bs, const void *vx, const void *vy, int m, int n);

namespace ime::bench::adapters {
namespace {
struct State : CommonState {
    std::vector<block_q4_0x32> b32;
    std::vector<block_q8_0x12> a12;
};

PrepareResult prepare(const BenchmarkRequest &r, const BenchmarkInput &generic) {
    const auto *in = std::get_if<Q4_0Q8_0Input>(&generic);
    if (!in)
        return {nullptr, "expected Q4_0/Q8_0 input"};
    auto s = std::make_unique<State>();
    s->m_value = r.m;
    s->n_value = r.n;
    s->k_value = r.k;
    s->blocks_k_value = r.k / QK4_0;
    s->output_value.resize(r.m * r.n);
    s->b32.resize((r.n / 32) * s->blocks_k());
    s->a12.resize((r.m / 12) * s->blocks_k());
    std::vector<block_q8_0> aq(r.m * s->blocks_k());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_0(in->activation.data() + m * r.k, aq.data() + m * s->blocks_k(), r.k);
    for (size_t tm = 0; tm < r.m; tm += 12)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &a = s->a12[(tm / 12) * s->blocks_k() + b];
            for (size_t x = 0; x < 12; ++x)
                a.d[x] = aq[(tm + x) * s->blocks_k() + b].d;
            for (size_t z = 0; z < 16; ++z) {
                for (size_t x = 0; x < 8; ++x)
                    a.qs[z * 24 + x] = aq[(tm + x) * s->blocks_k() + b].qs[z];
                for (size_t x = 0; x < 4; ++x)
                    a.qs[z * 24 + 8 + x] = aq[(tm + 8 + x) * s->blocks_k() + b].qs[z];
                for (size_t x = 0; x < 4; ++x)
                    a.qs[z * 24 + 12 + x] = aq[(tm + x) * s->blocks_k() + b].qs[z + 16];
                for (size_t x = 0; x < 8; ++x)
                    a.qs[z * 24 + 16 + x] = aq[(tm + 4 + x) * s->blocks_k() + b].qs[z + 16];
            }
        }
    for (size_t tn = 0; tn < r.n; tn += 32)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &w = s->b32[(tn / 32) * s->blocks_k() + b];
            for (size_t x = 0; x < 32; ++x) {
                w.d[x] = in->weight[(tn + x) * s->blocks_k() + b].d;
                for (size_t z = 0; z < 16; ++z)
                    w.qs[z * 32 + x] = in->weight[(tn + x) * s->blocks_k() + b].qs[z];
            }
        }
    return {s.release(), {}};
}
void run(KernelState p, size_t it) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t q = 0; q < it; ++q)
        for (size_t m = 0; m < s.m(); m += 12)
            for (size_t n = 0; n < s.n(); n += 32)
                ggml_gemm_q4_0_12x32_q8_0(s.k(), s.output().data() + m * s.n() + n, s.n(), s.b32.data() + (n / 32) * s.blocks_k(), s.a12.data() + (m / 12) * s.blocks_k(), 12, 32);
}
} // namespace
void register_q4_0_rvv_group() {
    register_kernel({"q4_0-q8_0-RVV-group",
                     "Q4_0/Q8_0 RVV grouped 12x32",
                     QuantizationType::WeightQ4_0ActivationQ8_0,
                     {validate_shape<12, 32, 32>, prepare, reset_common, run, export_common, checksum_common, [](KernelState p) noexcept { delete static_cast<State *>(p); }}});
}
} // namespace ime::bench::adapters