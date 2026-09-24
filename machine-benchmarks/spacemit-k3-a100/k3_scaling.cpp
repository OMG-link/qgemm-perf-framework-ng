// A100 多核扩展性测试。
//
// 带宽测试显示：即使工作集只有 16 KiB（完全在每核私有 L1D 内），同一个
// 4 核簇里 4 个线程的聚合读带宽也只有 ~56 B/cycle，单线程就能拿到 ~45。
// 这里用纯计算 kernel 交叉验证：如果标量 ALU 线性扩展而向量/矩阵单元不
// 扩展，说明簇内若干 hart 共享向量执行资源；如果标量也不扩展，说明是
// 测量方法或频率口径的问题。
#include "k3_a100.h"

#include <atomic>
#include <cmath>
#include <thread>

namespace {

using KernelFn = void (*)(uint64_t);

#define SCALE_KERNEL(fname, SETUP, BLOCK, REPT)                                                    \
    __attribute__((noinline)) void fname(uint64_t iters) {                                         \
        asm volatile("li t0, 0x01010101\n\t"                                                       \
                     "vsetvli t1, zero, e32, m8, ta, ma\n\t"                                       \
                     "vmv.v.x v0, t0\n\tvmv.v.x v8, t0\n\t"                                        \
                     "vmv.v.x v16, t0\n\tvmv.v.x v24, t0\n\t" SETUP                                \
                     "1:\n\t"                                                                      \
                     ".rept " #REPT "\n\t" BLOCK ".endr\n\t"                                       \
                     "addi %0, %0, -1\n\tbnez %0, 1b\n\t"                                          \
                     : "+r"(iters)                                                                 \
                     :                                                                             \
                     : "t0", "t1", "a4", "a5", "a6", "a7", "memory",                                \
                       "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",   \
                       "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",       \
                       "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");      \
    }

// 标量：4 条互相独立的 addi。
SCALE_KERNEL(k_scalar, "", "addi a4, a4, 1\n\taddi a5, a5, 1\n\taddi a6, a6, 1\n\taddi a7, a7, 1\n\t", 16)

// 向量 ALU：8 条独立的 vadd.vi e8,m1。
SCALE_KERNEL(k_vadd, "vsetvli t1, zero, e8, m1, tu, ma\n\t",
             "vadd.vi v8, v16, 1\n\tvadd.vi v9, v17, 1\n\tvadd.vi v10, v18, 1\n\t"
             "vadd.vi v11, v19, 1\n\tvadd.vi v12, v20, 1\n\tvadd.vi v13, v21, 1\n\t"
             "vadd.vi v14, v22, 1\n\tvadd.vi v15, v23, 1\n\t", 8)

// 矩阵单元：13 个独立累加器的 smt.vmadot i8。
#define MADOT13                                                                                    \
    "smt.vmadot v2, v28, v30\n\tsmt.vmadot v4, v28, v30\n\tsmt.vmadot v6, v28, v30\n\t"            \
    "smt.vmadot v8, v28, v30\n\tsmt.vmadot v10, v28, v30\n\tsmt.vmadot v12, v28, v30\n\t"          \
    "smt.vmadot v14, v28, v30\n\tsmt.vmadot v16, v28, v30\n\tsmt.vmadot v18, v28, v30\n\t"         \
    "smt.vmadot v20, v28, v30\n\tsmt.vmadot v22, v28, v30\n\tsmt.vmadot v24, v28, v30\n\t"         \
    "smt.vmadot v26, v28, v30\n\t"
SCALE_KERNEL(k_madot, "vsetvli t1, zero, e8, m1, tu, ma\n\t", MADOT13, 8)

// 浮点 FMA：8 条独立的 vfmacc.vv e32,m1。
SCALE_KERNEL(k_vfmacc, "vsetvli t1, zero, e32, m1, tu, ma\n\t",
             "vfmacc.vv v8, v16, v17\n\tvfmacc.vv v9, v18, v19\n\t"
             "vfmacc.vv v10, v20, v21\n\tvfmacc.vv v11, v22, v23\n\t"
             "vfmacc.vv v12, v24, v25\n\tvfmacc.vv v13, v26, v27\n\t"
             "vfmacc.vv v14, v28, v29\n\tvfmacc.vv v15, v30, v31\n\t", 8)

struct SpinBarrier {
    explicit SpinBarrier(int n) : target(n) {}
    void arrive_and_wait() {
        int gen = generation.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == target) {
            count.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == gen) {
                asm volatile("" ::: "memory");
            }
        }
    }
    std::atomic<int> count{0};
    std::atomic<int> generation{0};
    const int        target;
};

struct Sample {
    double   cpi     = 0;
    double   wall    = 0;
    uint64_t cycles  = 0;
    bool     ok      = false;
};

void worker(int cpu, KernelFn fn, int insts_per_iter, uint64_t iters, SpinBarrier & barrier,
            Sample & out) {
    if (!k3::pin_to_a100(cpu) || !k3::verify_cpu(cpu)) {
        barrier.arrive_and_wait();
        return;
    }
    k3::CycleCounter counter;
    if (!counter.open()) {
        barrier.arrive_and_wait();
        return;
    }
    fn(iters / 16 + 1);
    barrier.arrive_and_wait();
    double t0 = k3::wall_seconds();
    counter.start();
    fn(iters);
    k3::CycleCounter::Reading r{};
    bool                      ok = counter.stop(r);
    double                    t1 = k3::wall_seconds();
    out.cpi    = double(r.cycles) / (double(iters) * insts_per_iter);
    out.wall   = t1 - t0;
    out.cycles = r.cycles;
    out.ok     = ok && sched_getcpu() == cpu && r.running_ratio > 0.99;
}

struct Case {
    const char * name;
    KernelFn     fn;
    int          insts_per_iter;
};

const Case kCases[] = {
    {"addi (scalar)", k_scalar, 4 * 16},
    {"vadd.vi e8,m1", k_vadd, 8 * 8},
    {"vfmacc.vv e32,m1", k_vfmacc, 8 * 8},
    {"smt.vmadot i8", k_madot, 13 * 8},
};

const std::vector<std::vector<int>> kCpuSets = {
    {8},         {8, 9},       {8, 10},      {8, 11},      {9, 10},
    {8, 12},     {8, 9, 10},   {8, 9, 10, 11}, {8, 10, 12, 14},
    {8, 9, 10, 11, 12, 13, 14, 15},
};

std::string label(const std::vector<int> & cpus) {
    std::string s;
    for (size_t i = 0; i < cpus.size(); ++i) {
        s += (i ? "," : "") + std::to_string(cpus[i]);
    }
    return s;
}

}  // namespace

int main() {
    std::printf("%-18s %-24s %12s %14s %10s\n", "kernel", "cpus", "cyc/inst", "aggregate i/c",
                "scaling");
    for (const Case & c : kCases) {
        double single = 0;
        for (const std::vector<int> & cpus : kCpuSets) {
            const uint64_t iters = 200'000'000ULL / uint64_t(c.insts_per_iter);
            SpinBarrier    barrier(int(cpus.size()));
            std::vector<Sample>      samples(cpus.size());
            std::vector<std::thread> threads;
            for (size_t i = 0; i < cpus.size(); ++i) {
                threads.emplace_back(worker, cpus[i], c.fn, c.insts_per_iter, iters,
                                     std::ref(barrier), std::ref(samples[i]));
            }
            for (auto & t : threads) {
                t.join();
            }
            bool                ok = true;
            std::vector<double> cpis;
            for (const Sample & s : samples) {
                ok = ok && s.ok;
                cpis.push_back(s.cpi);
            }
            if (!ok) {
                std::printf("%-18s %-24s %12s %14s %10s\n", c.name, label(cpus).c_str(), "-", "-",
                            "invalid");
                continue;
            }
            double med = k3::median(cpis);
            double agg = double(cpus.size()) / med;
            if (cpus.size() == 1) {
                single = 1.0 / med;
            }
            std::printf("%-18s %-24s %12.4f %14.4f %9.2fx\n", c.name, label(cpus).c_str(), med, agg,
                        single > 0 ? agg / single : 0.0);
            std::fflush(stdout);
        }
    }
    return 0;
}
