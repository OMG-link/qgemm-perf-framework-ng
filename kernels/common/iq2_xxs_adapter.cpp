#include "adapter_common.h"
#include "llama_reference.h"

#include <memory>

namespace gemm::link::outer {
void ggml_gemm_iq2_xxs_q8_K_br(int, float *, size_t, const void *, size_t,
                                const void *, size_t, int, int);
void ggml_gemm_iq2_xxs_q8_K_ir(int, float *, size_t, const void *, size_t,
                                const void *, size_t, int, int);
#ifdef IME_HAVE_TCM
void ggml_gemm_iq2_xxs_q8_K_br_tcm(int, float *, size_t, const void *, size_t,
                                    const void *, size_t, int, int);
void ggml_gemm_iq2_xxs_q8_K_ir_tcm(int, float *, size_t, const void *, size_t,
                                    const void *, size_t, int, int);
#endif
}
void ggml_vec_dot_iq2_xxs_q8_K(int, float *, size_t, const void *, size_t,
                                const void *, size_t, int);

namespace ime::bench::adapters {
namespace {

constexpr size_t mr = 12;
struct PackedQ8K12 {
    float d[mr];
    int8_t qs[QK_K * mr];
};

struct State : CommonState {
    std::vector<block_iq2_xxs> weight;
    std::vector<block_q8_K> activation;
    std::vector<PackedQ8K12> packed_activation;
};

PrepareResult prepare(const BenchmarkRequest &request, const BenchmarkInput &generic) {
    const auto *input = std::get_if<IQ2_XXSQ8_KInput>(&generic);
    if (!input) return {nullptr, "expected IQ2_XXS/Q8_K input"};
    auto state = std::make_unique<State>();
    state->m_value = request.m;
    state->n_value = request.n;
    state->k_value = request.k;
    state->blocks_k_value = request.k / QK_K;
    state->output_value.resize(request.m * request.n);
    state->weight.assign(input->weight.begin(), input->weight.end());
    state->activation.resize(request.m * state->blocks_k());
    for (size_t m = 0; m < request.m; ++m)
        llama_reference::quantize_row_q8_K(input->activation.data() + m * request.k,
                                            state->activation.data() + m * state->blocks_k(), request.k);
    if (request.m % mr == 0) {
        state->packed_activation.resize(request.m / mr * state->blocks_k());
        for (size_t tile_m = 0; tile_m < request.m; tile_m += mr)
            for (size_t block = 0; block < state->blocks_k(); ++block) {
                auto &output = state->packed_activation[tile_m / mr * state->blocks_k() + block];
                for (size_t row = 0; row < mr; ++row) {
                    const auto &source = state->activation[(tile_m + row) * state->blocks_k() + block];
                    output.d[row] = source.d;
                    for (size_t k = 0; k < QK_K; ++k) output.qs[k * mr + row] = source.qs[k];
                }
            }
    }
    return {state.release(), {}};
}

#ifdef IME_HAVE_TCM
extern "C" int fine_mm_init(size_t);
extern "C" void fine_mm_deinit();

struct TcmRuntime {
    TcmRuntime() : available(fine_mm_init(256 * 1024) == 0) {}
    ~TcmRuntime() { if (available) fine_mm_deinit(); }
    bool available;
};

PrepareResult prepare_tcm(const BenchmarkRequest &request, const BenchmarkInput &generic) {
    static TcmRuntime runtime;
    if (!runtime.available) return {nullptr, "failed to initialize the 256 KiB TCM allocator"};
    return prepare(request, generic);
}
#endif

template <auto Kernel> void run_gemm(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    for (size_t it = 0; it < iterations; ++it)
        Kernel(state.k(), state.output().data(), state.n(), state.weight.data(), state.blocks_k(),
               state.packed_activation.data(), state.blocks_k(), state.m(), state.n());
}

void run_upstream(KernelState opaque, size_t iterations) noexcept {
    auto &state = *static_cast<State *>(opaque);
    for (size_t it = 0; it < iterations; ++it)
        for (size_t m = 0; m < state.m(); ++m)
            for (size_t n = 0; n < state.n(); ++n)
                ggml_vec_dot_iq2_xxs_q8_K(
                    state.k(), state.output().data() + m * state.n() + n, sizeof(float),
                    state.weight.data() + n * state.blocks_k(), sizeof(block_iq2_xxs),
                    state.activation.data() + m * state.blocks_k(), sizeof(block_q8_K), 1);
}

void destroy(KernelState opaque) noexcept { delete static_cast<State *>(opaque); }

} // namespace

void register_iq2_xxs_rvv_my_br() {
    register_kernel({"iq2_xxs-q8_K-RVV-my-br", "IQ2_XXS/Q8_K RVV custom 12x32 batch-reduce",
                     QuantizationType::WeightIQ2_XXSActivationQ8_K,
                     {validate_shape<12, 32, 256>, prepare, reset_common,
                      run_gemm<gemm::link::outer::ggml_gemm_iq2_xxs_q8_K_br>, export_common,
                      checksum_common, destroy}});
}

void register_iq2_xxs_rvv_my_ir() {
    register_kernel({"iq2_xxs-q8_K-RVV-my-ir", "IQ2_XXS/Q8_K RVV custom 12x32 immediate-reduce",
                     QuantizationType::WeightIQ2_XXSActivationQ8_K,
                     {validate_shape<12, 32, 256>, prepare, reset_common,
                      run_gemm<gemm::link::outer::ggml_gemm_iq2_xxs_q8_K_ir>, export_common,
                      checksum_common, destroy}});
}

void register_iq2_xxs_rvv_upstream() {
    register_kernel({"iq2_xxs-q8_K-RVV-upstream", "IQ2_XXS/Q8_K upstream RVV dot",
                     QuantizationType::WeightIQ2_XXSActivationQ8_K,
                     {validate_shape<1, 1, 256>, prepare, reset_common, run_upstream,
                      export_common, checksum_common, destroy}});
}

void register_iq2_xxs_rvv_my_br_tcm() {
#ifdef IME_HAVE_TCM
    register_kernel({"iq2_xxs-q8_K-RVV-my-br-tcm", "IQ2_XXS/Q8_K RVV custom 12x32 batch-reduce (TCM)",
                     QuantizationType::WeightIQ2_XXSActivationQ8_K,
                     {validate_shape<12, 32, 256>, prepare_tcm, reset_common,
                      run_gemm<gemm::link::outer::ggml_gemm_iq2_xxs_q8_K_br_tcm>, export_common,
                      checksum_common, destroy}});
#endif
}

void register_iq2_xxs_rvv_my_ir_tcm() {
#ifdef IME_HAVE_TCM
    register_kernel({"iq2_xxs-q8_K-RVV-my-ir-tcm", "IQ2_XXS/Q8_K RVV custom 12x32 immediate-reduce (TCM)",
                     QuantizationType::WeightIQ2_XXSActivationQ8_K,
                     {validate_shape<12, 32, 256>, prepare_tcm, reset_common,
                      run_gemm<gemm::link::outer::ggml_gemm_iq2_xxs_q8_K_ir_tcm>, export_common,
                      checksum_common, destroy}});
#endif
}

} // namespace ime::bench::adapters