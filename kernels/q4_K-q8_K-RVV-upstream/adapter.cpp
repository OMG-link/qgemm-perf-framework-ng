#include "adapter_common.h"
#include "llama_reference.h"
#include <memory>

void ggml_vec_dot_q4_K_q8_K(int, float *, size_t, const void *, size_t, const void *, size_t, int);

namespace ime::bench::adapters {
namespace {
struct State : CommonState {
    std::vector<block_q4_K> w;
    std::vector<block_q8_K> a;
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
    s->w.assign(in->weight.begin(), in->weight.end());
    s->a.resize(r.m * s->blocks_k());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_K(in->activation.data() + m * r.k, s->a.data() + m * s->blocks_k(), r.k);
    return {s.release(), {}};
}
void run(KernelState p, size_t it) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t q = 0; q < it; ++q)
        for (size_t m = 0; m < s.m(); ++m)
            for (size_t n = 0; n < s.n(); ++n)
                ggml_vec_dot_q4_K_q8_K(s.k(), s.output().data() + m * s.n() + n, sizeof(float), s.w.data() + n * s.blocks_k(), sizeof(block_q4_K), s.a.data() + m * s.blocks_k(), sizeof(block_q8_K),
                                       1);
}
} // namespace
void register_q4_K_rvv_upstream() {
    register_kernel({"q4_K-q8_K-RVV-upstream",
                     "Q4_K/Q8_K RVV upstream dot",
                     QuantizationType::WeightQ4_KActivationQ8_K,
                     {validate_shape<1, 1, 256>, prepare, reset_common, run, export_common, checksum_common, [](KernelState p) noexcept { delete static_cast<State *>(p); }}});
}
} // namespace ime::bench::adapters