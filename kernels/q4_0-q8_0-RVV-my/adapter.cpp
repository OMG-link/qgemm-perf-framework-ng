#include "adapter_common.h"
#include "types.h"
#include "llama_reference.h"
#include <cstring>
#include <memory>
void ggml_gemm_q4_0_8x32_q8_0(int k, float *s, size_t bs, const void *vx, const void *vy, int m, int n);
namespace ime::bench::adapters {
namespace {
struct State : CommonState {
    std::vector<block_q4_0_rvv_n32> b32;
    std::vector<block_q8_0_rvv_m8> a8;
};
PrepareResult prepare(const BenchmarkRequest &r, const BenchmarkInput &g) {
    auto *i = std::get_if<Q4_0Q8_0Input>(&g);
    if (!i)
        return {nullptr, "expected Q4_0/Q8_0 input"};
    auto s = std::make_unique<State>();
    s->m_value = r.m;
    s->n_value = r.n;
    s->k_value = r.k;
    s->blocks_k_value = r.k / 32;
    s->output_value.resize(r.m * r.n);
    s->b32.resize(r.n / 32 * s->blocks_k());
    s->a8.resize(r.m / 8 * s->blocks_k());
    std::vector<block_q8_0> a(r.m * s->blocks_k());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_0(i->activation.data() + m * r.k, a.data() + m * s->blocks_k(), r.k);
    for (size_t tm = 0; tm < r.m; tm += 8)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &x = s->a8[tm / 8 * s->blocks_k() + b];
            for (size_t q = 0; q < 8; ++q) {
                std::memcpy(&x.d[q], &a[(tm + q) * s->blocks_k() + b].d, sizeof(a[0].d));
                for (size_t z = 0; z < 16; ++z) {
                    x.qs[z * 16 + q] = a[(tm + q) * s->blocks_k() + b].qs[z];
                    x.qs[z * 16 + 8 + q] = a[(tm + q) * s->blocks_k() + b].qs[z + 16];
                }
            }
        }
    for (size_t tn = 0; tn < r.n; tn += 32)
        for (size_t b = 0; b < s->blocks_k(); ++b) {
            auto &w = s->b32[tn / 32 * s->blocks_k() + b];
            for (size_t q = 0; q < 32; ++q) {
                const auto &src = i->weight[(tn + q) * s->blocks_k() + b];
                std::memcpy(&w.d[q], &src.d, sizeof(src.d));
                for (size_t z = 0; z < 16; ++z)
                    w.qs[z * 32 + q] = src.qs[z];
            }
        }
    return {s.release(), {}};
}
void run(KernelState p, size_t it) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t q = 0; q < it; ++q)
        for (size_t m = 0; m < s.m(); m += 8)
            for (size_t n = 0; n < s.n(); n += 32)
                ggml_gemm_q4_0_8x32_q8_0(s.k(), s.output().data() + m * s.n() + n, s.n(), s.b32.data() + n / 32 * s.blocks_k(), s.a8.data() + m / 8 * s.blocks_k(), 8, 32);
}
} // namespace
void register_q4_0_rvv_my() {
    register_kernel({"q4_0-q8_0-RVV-my",
                     "Q4_0/Q8_0 RVV custom 8x32",
                     QuantizationType::WeightQ4_0ActivationQ8_0,
                     {validate_shape<8, 32, 32>, prepare, reset_common, run, export_common, checksum_common, [](KernelState p) noexcept { delete static_cast<State *>(p); }}});
}
} // namespace ime::bench::adapters