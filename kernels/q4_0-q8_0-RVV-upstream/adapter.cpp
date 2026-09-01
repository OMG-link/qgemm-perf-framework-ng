#include "adapter_common.h"
#include "llama_reference.h"

#include <memory>

void ggml_vec_dot_q4_0_q8_0(int n, float *, size_t, const void *, size_t, const void *, size_t, int);

namespace ime::bench::adapters {
namespace {
struct State : CommonState {
    std::vector<block_q4_0> b;
    std::vector<block_q8_0> a;
};
PrepareResult prepare(const BenchmarkRequest &r, const BenchmarkInput &generic) {
    const auto *input = std::get_if<Q4_0Q8_0Input>(&generic);
    if (!input)
        return {nullptr, "expected Q4_0/Q8_0 input"};
    auto s = std::make_unique<State>();
    s->m_value = r.m;
    s->n_value = r.n;
    s->k_value = r.k;
    s->blocks_k_value = r.k / QK8_0;
    s->output_value.resize(r.m * r.n);
    s->a.resize(r.m * s->blocks_k());
    s->b.assign(input->weight.begin(), input->weight.end());
    for (size_t m = 0; m < r.m; ++m)
        llama_reference::quantize_row_q8_0(input->activation.data() + m * r.k, s->a.data() + m * s->blocks_k(), r.k);
    return {s.release(), {}};
}
void run(KernelState p, size_t iterations) noexcept {
    auto &s = *static_cast<State *>(p);
    for (size_t it = 0; it < iterations; ++it)
        for (size_t m = 0; m < s.m(); ++m)
            for (size_t n = 0; n < s.n(); ++n)
                ggml_vec_dot_q4_0_q8_0(s.k(), s.output().data() + m * s.n() + n, sizeof(float), s.b.data() + n * s.blocks_k(), sizeof(block_q4_0), s.a.data() + m * s.blocks_k(), sizeof(block_q8_0),
                                       1);
}
} // namespace
void register_q4_0_rvv_upstream() {
    register_kernel({"q4_0-q8_0-RVV-upstream",
                     "Q4_0/Q8_0 RVV upstream dot",
                     QuantizationType::WeightQ4_0ActivationQ8_0,
                     {validate_shape<1, 1, 32>, prepare, reset_common, run, export_common, checksum_common, destroy_common}});
}
} // namespace ime::bench::adapters
