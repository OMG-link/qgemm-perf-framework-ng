// A100 各级存储的访问延迟（指针追逐）。
//
// 带宽扫描无法区分 A100 的 64 KiB 私有 L1D 和 1 MiB 簇内 L2——两者对
// RVV 流式读取给出的带宽相同。延迟测量可以直接把各级存储分开。
//
// kernel 是纯依赖链：每次 load 的结果就是下一次 load 的地址，所以
// cycles/access 就是访问延迟。步长取 64 B（cache line），并在整个工作集
// 上做随机置换，避免硬件预取器把链路预测出来。
#include "k3_a100.h"

#include <random>
#include <vector>

#include <sys/mman.h>

namespace {

constexpr size_t kStride = 64;  // cache line

// 依赖链：uintptr_t 存的是下一跳的绝对地址。
uint64_t chase(const void * start, uint64_t steps) {
    const void * p = start;
    asm volatile("1:\n\t"
                 "ld %0, 0(%0)\n\t"
                 "addi %1, %1, -1\n\t"
                 "bnez %1, 1b\n\t"
                 : "+r"(p), "+r"(steps)
                 :
                 : "memory");
    return (uint64_t) (uintptr_t) p;
}

}  // namespace

int main(int argc, char ** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        return 1;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }

    const std::vector<size_t> sizes = {
        4u << 10,   8u << 10,   16u << 10,  32u << 10,  48u << 10,  64u << 10,
        96u << 10,  128u << 10, 192u << 10, 256u << 10, 512u << 10, 768u << 10,
        1u << 20,   2u << 20,   4u << 20,   16u << 20,  64u << 20,  256u << 20,
    };

    std::printf("# cpu=%d  pointer chase, 64 B stride, random permutation\n", cpu);
    std::printf("%12s %14s %12s\n", "working set", "cycles/access", "ns @1.8GHz");

    std::mt19937_64 rng(12345);
    for (size_t bytes : sizes) {
        const size_t nodes = bytes / kStride;
        if (nodes < 2) {
            continue;
        }
        void * raw = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            std::fprintf(stderr, "mmap %zu failed\n", bytes);
            continue;
        }
        auto * base = static_cast<uint8_t *>(raw);

        // 生成一个覆盖全部节点的单环随机置换。
        std::vector<size_t> order(nodes);
        for (size_t i = 0; i < nodes; ++i) {
            order[i] = i;
        }
        for (size_t i = nodes - 1; i > 0; --i) {
            std::swap(order[i], order[rng() % (i + 1)]);
        }
        for (size_t i = 0; i < nodes; ++i) {
            size_t from = order[i];
            size_t to   = order[(i + 1) % nodes];
            *reinterpret_cast<uintptr_t *>(base + from * kStride) =
                reinterpret_cast<uintptr_t>(base + to * kStride);
        }

        const uint64_t steps = std::max<uint64_t>(2'000'000ULL, nodes * 8);
        void * const   start = base + order[0] * kStride;

        chase(start, std::min<uint64_t>(steps, nodes * 4));  // 预热

        double best = 1e18;
        for (int s = 0; s < 5; ++s) {
            counter.start();
            uint64_t sink = chase(start, steps);
            k3::CycleCounter::Reading r{};
            counter.stop(r);
            asm volatile("" ::"r"(sink));
            if (r.running_ratio > 0.99) {
                best = std::min(best, double(r.cycles) / double(steps));
            }
        }
        std::printf("%11zuK %14.2f %12.2f\n", bytes >> 10, best, best / 1.8);
        std::fflush(stdout);
        munmap(raw, bytes);
    }
    return 0;
}
