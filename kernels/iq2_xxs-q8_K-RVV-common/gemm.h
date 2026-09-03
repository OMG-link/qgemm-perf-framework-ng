#include "ggml_min.h"

namespace gemm::link {

namespace outer {

constexpr int MR = 12;

void pack_a_iq2_xxs_br(int m, int k, const block_q8_K *in, block_q8_Kx<MR> *out);
void pack_a_iq2_xxs_ir(int m, int k, const block_q8_K *in, block_q8_Kx<MR> *out);
void ggml_gemm_iq2_xxs_q8_K_br(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx, const void *GGML_RESTRICT vy, size_t by, int m, int n);
void ggml_gemm_iq2_xxs_q8_K_ir(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx, const void *GGML_RESTRICT vy, size_t by, int m, int n);

} // namespace outer

} // namespace gemm::link