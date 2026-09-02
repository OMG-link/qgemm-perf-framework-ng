#ifndef IME_BENCH_ADAPTER_COMMON_H
#define IME_BENCH_ADAPTER_COMMON_H

#include <span>
#include <vector>

#include "kernel_registry.h"

namespace ime::bench::adapters {

class CommonState {
public:
    virtual ~CommonState() = default;
    size_t m() const noexcept { return m_value; }
    size_t n() const noexcept { return n_value; }
    size_t k() const noexcept { return k_value; }
    size_t blocks_k() const noexcept { return blocks_k_value; }
    std::vector<float> &output() noexcept { return output_value; }
    const std::vector<float> &output() const noexcept { return output_value; }

    size_t m_value = 0;
    size_t n_value = 0;
    size_t k_value = 0;
    size_t blocks_k_value = 0;
    std::vector<float> output_value;
};

template <size_t MMultiple, size_t NMultiple, size_t KMultiple>
ValidationResult validate_shape(const BenchmarkRequest &request) {
    static_assert(MMultiple > 0 && NMultiple > 0 && KMultiple > 0);
    if (!request.m || !request.n || !request.k) {
        return {false, "M, N and K must be positive"};
    }
    if (request.m % MMultiple) {
        return {false, "M must be divisible by " + std::to_string(MMultiple)};
    }
    if (request.n % NMultiple) {
        return {false, "N must be divisible by " + std::to_string(NMultiple)};
    }
    if (request.k % KMultiple) {
        return {false, "K must be divisible by " + std::to_string(KMultiple)};
    }
    return {true, {}};
}

void reset_common(KernelState state);
ExportResult export_common(KernelState state, std::span<float> output);
double checksum_common(KernelState state);
void destroy_common(KernelState state) noexcept;

void register_llama_dispatch();
void register_m4_batch_reduction();
void register_m4_immediate_reduction();
void register_m8_batch_reduction();
void register_q4_0_rvv_group();
void register_q4_0_rvv_my();
void register_q4_0_rvv_upstream();
void register_q4_K_rvv_group();
void register_q4_K_rvv_my();
void register_q4_K_rvv_upstream();
void register_iq2_xxs_rvv_my_br();
void register_iq2_xxs_rvv_my_ir();
void register_iq2_xxs_rvv_upstream();
void register_iq2_xxs_rvv_my_br_tcm();
void register_iq2_xxs_rvv_my_ir_tcm();

} // namespace ime::bench::adapters

#endif