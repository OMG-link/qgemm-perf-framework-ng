// K3 A100：把 TCM 当作微核中间缓冲用之前的最小确认。
//
// k3_tcm_bandwidth.cpp 只测了 TCM 的读带宽。要把 GEMM 微核的 int32 component
// 缓冲放进 TCM，需要先确认两件事：
//   1. 功能正确：写进去再读出来内容一致，含 vs2r.v 整寄存器写。
//   2. 写带宽：component 缓冲写多读少（每个 K block 写 2 KiB，每 8 个 block
//      读一遍），而 K3 普通内存的写峰值只有读的一半，TCM 的写没有数据。
//
// /dev/tcm 驱动缺陷带来的硬约束：
//   - 整个进程只允许存在一次 mmap，不能有并发或重复申请；
//   - 长度必须是 384 KiB 的整数倍。
// 违反会让驱动进入损坏状态，并在**下一次** TCM 调用时挂死不可恢复的 D 状态。
// 因此本程序只申请最小的一个块（384 KiB），不申请整颗 3 MiB：需要的只有
// 16 KiB，申请整颗只会增加和其它持有者碰撞的概率。
//
// 块按核对私有：设备树 soc/tcm@0 每簇 4 块，块 0/1 与 2/3 各共享一个 cpus
// phandle 对，即 (8,9)(10,11)(12,13)(14,15)，与矩阵单元配对一致。新鲜状态下
// 384 KiB 的申请拿到的是物理地址最小的块 0，归属 (8,9)，所以本程序只在
// cpu8 或 cpu9 上有意义，其它核会读到慢约 330 倍的外块。
#include "k3_a100.h"

#include <cerrno>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>

namespace {

constexpr size_t kBlockBytes = 384u << 10;  // TCM 的最小申请单位
constexpr size_t kBufferBytes = 16u << 10;  // smt-ime2 的 component 缓冲大小

void step(const char *what) {
    std::printf("# %s\n", what);
    std::fflush(stdout);
}

// 16 KiB = 64 条 e32m2 整寄存器写，和微核 component 落地的形态一致
// （每个 K block 8 条 vs2r.v，batch 8 共 64 条）。
__attribute__((noinline)) void store_pass(int32_t *base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e32, m2, ta, ma\n\t"
                 "vmv.v.i v8, 7\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\t"
                 ".rept 64\n\t"
                 "vs2r.v v8, (t1)\n\taddi t1, t1, 256\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "memory", "v8", "v9");
}

__attribute__((noinline)) void load_pass(int32_t *base, uint64_t rep) {
    asm volatile("vsetvli t0, zero, e32, m2, ta, ma\n\t"
                 "2:\n\t"
                 "mv t1, %0\n\t"
                 ".rept 64\n\t"
                 "vl2re32.v v8, (t1)\n\taddi t1, t1, 256\n\t"
                 ".endr\n\t"
                 "addi %1, %1, -1\n\tbnez %1, 2b\n\t"
                 : "+r"(base), "+r"(rep)
                 :
                 : "t0", "t1", "memory", "v8", "v9");
}

k3::CycleCounter g_counter;

double rate(void (*fn)(int32_t *, uint64_t), int32_t *base, int samples) {
    const uint64_t rep = (32ull << 20) / kBufferBytes;
    fn(base, 16);  // 预热
    std::vector<double> per_sample;
    for (int s = 0; s < samples; ++s) {
        g_counter.start();
        fn(base, rep);
        k3::CycleCounter::Reading r{};
        if (!g_counter.stop(r) || r.running_ratio < 0.99) {
            continue;
        }
        per_sample.push_back(double(kBufferBytes) * double(rep) / double(r.cycles));
    }
    if (per_sample.empty()) {
        return 0.0;
    }
    return *std::max_element(per_sample.begin(), per_sample.end());
}

} // namespace

int main(int argc, char **argv) {
    const int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    const int samples = (argc > 2) ? std::atoi(argv[2]) : 5;
    if (cpu != 8 && cpu != 9) {
        std::fprintf(stderr, "a 384 KiB request yields TCM block 0, which belongs to cpus 8 and 9\n");
        return 2;
    }

    // 主线程先只进入 A100 域，不钉核，与 k3_tcm_bandwidth.cpp 的顺序一致。
    step("entering A100 domain");
    if (!k3::enter_a100_domain()) {
        return 1;
    }

    step("open(/dev/tcm)");
    const int fd = open("/dev/tcm", O_RDWR | O_SYNC);
    if (fd < 0) {
        std::fprintf(stderr, "open(/dev/tcm) failed: %s\n", std::strerror(errno));
        return 1;
    }

    // 本进程唯一的一次 TCM mmap，最小合法尺寸。
    step("mmap(384 KiB, MAP_SHARED, /dev/tcm)");
    void *tcm = mmap(nullptr, kBlockBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (tcm == MAP_FAILED) {
        std::fprintf(stderr, "mmap failed: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }
    std::printf("# TCM va=%p len=%zu KiB\n", tcm, kBlockBytes >> 10);
    std::fflush(stdout);

    step("pinning to the target cpu");
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu) || !g_counter.open()) {
        munmap(tcm, kBlockBytes);
        close(fd);
        return 1;
    }
    std::printf("# cpu=%d vlenb=%llu samples=%d\n", cpu, (unsigned long long)k3::vlenb(), samples);
    std::fflush(stdout);

    auto *own = static_cast<int32_t *>(tcm);
    const size_t words = kBufferBytes / sizeof(int32_t);

    // ---- 1. 功能正确性 ----
    step("scalar write then read back");
    for (size_t i = 0; i < words; ++i) {
        own[i] = int32_t(i * 2654435761u);
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < words; ++i) {
        if (own[i] != int32_t(i * 2654435761u)) {
            ++mismatches;
        }
    }
    std::printf("scalar round trip over %zu KiB: %s (%zu mismatches)\n", kBufferBytes >> 10, mismatches == 0 ? "ok" : "FAILED", mismatches);
    std::fflush(stdout);

    step("vs2r.v write then read back");
    store_pass(own, 1);
    size_t vector_mismatches = 0;
    for (size_t i = 0; i < words; ++i) {
        if (own[i] != 7) {
            ++vector_mismatches;
        }
    }
    std::printf("vs2r.v round trip over %zu KiB: %s (%zu mismatches)\n", kBufferBytes >> 10, vector_mismatches == 0 ? "ok" : "FAILED", vector_mismatches);
    std::fflush(stdout);

    // ---- 2. 写/读带宽，TCM 自己的块 vs 匿名内存 ----
    step("measuring store and load rate");
    void *anon = mmap(nullptr, kBlockBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (anon == MAP_FAILED) {
        std::fprintf(stderr, "anon mmap failed\n");
        munmap(tcm, kBlockBytes);
        close(fd);
        return 1;
    }
    std::memset(anon, 0, kBlockBytes);

    const double tcm_store = rate(store_pass, own, samples);
    const double tcm_load = rate(load_pass, own, samples);
    const double anon_store = rate(store_pass, static_cast<int32_t *>(anon), samples);
    const double anon_load = rate(load_pass, static_cast<int32_t *>(anon), samples);

    std::printf("\n%-30s %14s %14s\n", "16 KiB buffer", "store B/cycle", "load B/cycle");
    std::printf("%-30s %14.2f %14.2f\n", "TCM block 0 (own pair)", tcm_store, tcm_load);
    std::printf("%-30s %14.2f %14.2f\n", "anon, L1D resident", anon_store, anon_load);
    std::fflush(stdout);

    munmap(anon, kBlockBytes);
    step("munmap TCM");
    if (munmap(tcm, kBlockBytes) != 0) {
        std::fprintf(stderr, "munmap failed: %s\n", std::strerror(errno));
    }
    close(fd);
    step("done, TCM released");
    return 0;
}
