#include "ggml_min.h"

namespace gemm::link {

namespace outer {

constexpr int MR = 12;

void pack_a(int m, int k, const block_q8_K *in, block_q8_Kx<MR> *out);
void ggml_gemm_iq2_xxs_q8_K(int k, float *GGML_RESTRICT s, size_t bs, const void *GGML_RESTRICT vx, size_t bx, const void *GGML_RESTRICT vy, size_t by, int m, int n);

} // namespace outer

} // namespace gemm::link