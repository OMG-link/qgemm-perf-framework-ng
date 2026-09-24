// 打印 A100 核的基本运行时信息，并验证绑定方法与计数器可用性。
#include "k3_a100.h"

namespace {

std::string affinity_list() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return "<error>";
    }
    std::string out;
    for (int i = 0; i < 64; ++i) {
        if (CPU_ISSET(i, &set)) {
            if (!out.empty()) {
                out += ",";
            }
            out += std::to_string(i);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    int cpu = (argc > 1) ? std::atoi(argv[1]) : k3::kA100FirstCpu;

    std::printf("before: cpu=%d affinity=%s\n", sched_getcpu(), affinity_list().c_str());
    if (!k3::pin_to_a100(cpu)) {
        return 1;
    }
    std::printf("after:  cpu=%d affinity=%s\n", sched_getcpu(), affinity_list().c_str());
    if (!k3::verify_cpu(cpu)) {
        return 1;
    }

    std::printf("vlenb=%llu (VLEN=%llu bits)\n", (unsigned long long) k3::vlenb(),
                (unsigned long long) k3::vlenb() * 8);

    k3::CycleCounter counter;
    if (!counter.open()) {
        return 1;
    }

    // 已知指令数的整数加法链，用来同时校验 cycles 与 instructions 事件。
    constexpr uint64_t kIters = 200'000'000ULL;
    double             t0     = k3::wall_seconds();
    if (!counter.start()) {
        std::fprintf(stderr, "counter start failed\n");
        return 1;
    }
    uint64_t n = kIters;
    asm volatile(
        "1:\n\t"
        "addi %0, %0, -1\n\t"
        "bnez %0, 1b\n\t"
        : "+r"(n)::);
    k3::CycleCounter::Reading r{};
    if (!counter.stop(r)) {
        std::fprintf(stderr, "counter stop failed\n");
        return 1;
    }
    double t1 = k3::wall_seconds();

    std::printf("loop iters=%llu\n", (unsigned long long) kIters);
    std::printf("cycles=%llu instructions=%llu running_ratio=%.4f\n", (unsigned long long) r.cycles,
                (unsigned long long) r.instructions, r.running_ratio);
    std::printf("instructions / (2*iters) = %.4f\n", double(r.instructions) / (2.0 * kIters));
    std::printf("elapsed=%.6f s -> frequency = %.4f GHz\n", t1 - t0, r.cycles / (t1 - t0) / 1e9);
    std::printf("final cpu=%d\n", sched_getcpu());
    return 0;
}
