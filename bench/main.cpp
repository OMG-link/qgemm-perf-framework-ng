#include <cstdio>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ime.h"
#include "perf.h"
#include "selected_cycles.h"

const int VL = 32;

#define FREQ 1.6
#define VLEN 256
#define ELEM_WID 16

using InBlock = block_q8_0x12;
using KerBlock = block_q4_0x32;

static bool parse_positive_int(const char *text, int &value) {
    errno = 0;
    char *end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char **argv) {

    //  A [480][1536]
    //  B [1536][1536]

    // parse command line args M/N/K
    if (argc != 4) {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return 1;
    }
    int m = 0;
    int n = 0;
    int k = 0;
    if (!parse_positive_int(argv[1], m) || !parse_positive_int(argv[2], n) || !parse_positive_int(argv[3], k)) {
        std::fprintf(stderr, "M, N and K must be positive integers.\n");
        return 2;
    }

    if (k % QK4_0 != 0 || m % 4 != 0 || n % VL != 0) {
        std::fprintf(stderr, "Unsupported dimensions: M must be divisible by 4, N by %d, and K by %d.\n", VL, QK4_0);
        return 2;
    }

    constexpr size_t blk_len = 32;
    constexpr size_t scale_stride = sizeof(uint16_t);
    constexpr size_t blk_bitwidth = 4;

    const size_t k_blks = ime_div_round_up(k, blk_len);

    // ---- A / per_gemm_ws ----
    const size_t lda = k_blks * ime_q8_block_size(blk_len);
    std::vector<std::byte> per_gemm_ws(m * lda);
    std::memset(per_gemm_ws.data(), 0x1, per_gemm_ws.size());

    // ---- B (packed i4) ----
    const size_t ldb = k_blks * (blk_len * blk_bitwidth / 8); // = k_blks * 16
    const size_t packed_b_stride = ldb + k_blks * scale_stride;

    std::vector<std::byte> packed_b(n * packed_b_stride);
    std::memset(packed_b.data(), 0x1, packed_b.size());

    // ---- C ----
    const size_t ldc = n;
    std::vector<float> c(m * ldc, 0.0f);

    // ---- args ----
    qnbitgemm_spacemit_ime_args args;
    args.a_ptr = nullptr; // 不使用
    args.lda = 0;         // 不使用
    args.packed_quant_b_data = packed_b.data();
    args.quant_b_scale = nullptr;
    args.quant_b_zp = nullptr; // 不使用 zero-point
    args.quant_b_blksum = nullptr;
    args.bias = nullptr;
    args.c_ptr = c.data();
    args.ldc = ldc;

    const double peak_fops = FREQ * VLEN / ELEM_WID;
    const double operations = static_cast<double>(m) * n * k;
    const int T = std::max(3, static_cast<int>((1e9 * peak_fops) / operations));

    // Warmup
    sqnbitgemm_spacemit_ime_i8i4(blk_len, k, &args, per_gemm_ws.data(),
                                 /* m_start */ 0,
                                 /* m_count */ m,
                                 /* n_start */ 0,
                                 /* n_count */ n);

    reset_selected_cycles();

    // Perf-setup
    int fd_cycles = perf_event_cycles();
    int fd_l1da = perf_event_l1d_access();
    int fd_l1dm = perf_event_l1d_miss();
    perf_reset(fd_cycles);
    perf_reset(fd_l1da);
    perf_reset(fd_l1dm);

    // Main test
    for (int t = 0; t < T; t++) {
        sqnbitgemm_spacemit_ime_i8i4(blk_len, k, &args, per_gemm_ws.data(),
                                     /* m_start */ 0,
                                     /* m_count */ m,
                                     /* n_start */ 0,
                                     /* n_count */ n);
    }

    // Perf-cleanup
    perf_disable(fd_cycles);
    perf_disable(fd_l1da);
    perf_disable(fd_l1dm);
    auto cycles = perf_read(fd_cycles);
    auto l1d_access = perf_read(fd_l1da);
    auto l1d_miss = perf_read(fd_l1dm);
    perf_close_event(fd_cycles);
    perf_close_event(fd_l1da);
    perf_close_event(fd_l1dm);
    const double miss_rate = l1d_access == 0 ? 0.0 : static_cast<double>(l1d_miss) / l1d_access * 100.0;
    printf("cycle = %llu \t l1d_access = %llu \t l1d_miss = %llu \t miss_rate = %.2f%%\n",
           static_cast<unsigned long long>(cycles), static_cast<unsigned long long>(l1d_access),
           static_cast<unsigned long long>(l1d_miss), miss_rate);

    int64_t cycle_total = cycles;
    int64_t cycle_per_test = cycle_total / T;

    int64_t selected_cycle_total = selected_cycles;
    int64_t selected_cycle_per_test = selected_cycle_total / T;
    printf("(selected) cycle_per_test: %ld, T: %d\n", selected_cycle_per_test, T);

    int64_t n_fma = static_cast<int64_t>(k) * m * n;
    int64_t fma_per_cycle;

#ifdef SPACEMIT_X60
    fma_per_cycle = 4 * 4 * 8;
#else
#ifdef XUANTIE_C910
    fma_per_cycle = 16;
#else
#error "Unknown CPU. Cannot determine peak FMA"
#endif
#endif

    int64_t theoretical_cycle = n_fma / fma_per_cycle;
    int64_t actual_cycle = cycle_per_test;
    double utilization = static_cast<double>(actual_cycle) / theoretical_cycle;

    printf("乘加数: %ld\n", n_fma);
    printf("理论需要周期数: %ld\n", theoretical_cycle);
    printf("实际执行周期数: %ld\n", actual_cycle);
    printf("实际-理论比值: %.2f (%.2f%%)\n", utilization, 100 / utilization);

    // free(s);
    // free(vx);
    // free(vy);
}
