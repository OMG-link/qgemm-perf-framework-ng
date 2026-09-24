// SpacemiT K3 A100 微基准公共支撑代码。
//
// K3 的 A100 核（cpu8-15）默认不在任何用户进程的 affinity mask 内：
// init 与所有子进程的 Cpus_allowed_list 都是 0-7（X100）。要把当前线程
// 迁移到 A100 上，必须先向 /proc/set_ai_thread 写入一个值，内核随后会把
// 该线程的 affinity mask 改成 8-15。这与 llama.cpp
// ggml/src/ggml-cpu/spacemit/ime.cpp 中的 bind_ai_thread() 一致。
#pragma once

#include <fcntl.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace k3 {

// A100 核在 /proc/cpuinfo 中的逻辑编号范围。
constexpr int kA100FirstCpu = 8;
constexpr int kA100LastCpu  = 15;

// K3 的 S 模式没有给用户态开放 cycle/time/instret CSR，`rdcycle` 会触发
// 非法指令异常。所有 cycle 计数必须走 perf_event_open。
class CycleCounter {
  public:
    CycleCounter() = default;
    CycleCounter(const CycleCounter &) = delete;
    CycleCounter & operator=(const CycleCounter &) = delete;

    ~CycleCounter() {
        if (inst_fd_ >= 0) {
            close(inst_fd_);
        }
        if (cycle_fd_ >= 0) {
            close(cycle_fd_);
        }
    }

    // 必须在线程已经绑定到目标核之后调用。
    bool open() {
        cycle_fd_ = open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
        if (cycle_fd_ < 0) {
            std::fprintf(stderr, "perf_event_open(cycles) failed: %s\n", std::strerror(errno));
            return false;
        }
        inst_fd_ = open_one(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, cycle_fd_);
        if (inst_fd_ < 0) {
            std::fprintf(stderr, "perf_event_open(instructions) failed: %s\n", std::strerror(errno));
            return false;
        }
        return true;
    }

    bool start() {
        return ioctl(cycle_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == 0 &&
               ioctl(cycle_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == 0;
    }

    struct Reading {
        uint64_t cycles       = 0;
        uint64_t instructions = 0;
        double   running_ratio = 0.0;  // time_running / time_enabled
    };

    bool stop(Reading & out) {
        if (ioctl(cycle_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }
        Raw c{}, i{};
        if (read(cycle_fd_, &c, sizeof(c)) != (ssize_t) sizeof(c)) {
            return false;
        }
        if (read(inst_fd_, &i, sizeof(i)) != (ssize_t) sizeof(i)) {
            return false;
        }
        out.cycles        = c.value;
        out.instructions  = i.value;
        out.running_ratio = c.time_enabled ? double(c.time_running) / double(c.time_enabled) : 0.0;
        return true;
    }

  private:
    struct Raw {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    };

    static int open_one(uint32_t type, uint64_t config, int group_fd) {
        perf_event_attr attr{};
        attr.type           = type;
        attr.size           = sizeof(attr);
        attr.config         = config;
        attr.disabled       = group_fd < 0;
        attr.pinned         = group_fd < 0;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        attr.read_format    = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        return (int) syscall(__NR_perf_event_open, &attr, 0, -1, group_fd, PERF_FLAG_FD_CLOEXEC);
    }

    int cycle_fd_ = -1;
    int inst_fd_  = -1;
};

inline double wall_seconds() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

inline uint64_t vlenb() {
    uint64_t v;
    asm volatile("csrr %0, 0xc22" : "=r"(v));  // vlenb
    return v;
}

// 向 /proc/set_ai_thread 写入，把调用线程迁移到 A100 核域。
inline bool enter_a100_domain() {
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) {
        std::fprintf(stderr, "open /proc/set_ai_thread failed: %s\n", std::strerror(errno));
        return false;
    }
    const char * payload = "0";
    ssize_t      n       = write(fd, payload, std::strlen(payload));
    close(fd);
    if (n < 0) {
        std::fprintf(stderr, "write /proc/set_ai_thread failed: %s\n", std::strerror(errno));
        return false;
    }
    return true;
}

// 迁移到 A100 核域，并把线程钉死在单个 A100 核上。
inline bool pin_to_a100(int cpu) {
    if (cpu < kA100FirstCpu || cpu > kA100LastCpu) {
        std::fprintf(stderr, "cpu %d is not an A100 core (expected %d-%d)\n", cpu, kA100FirstCpu, kA100LastCpu);
        return false;
    }
    if (!enter_a100_domain()) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        std::fprintf(stderr, "sched_setaffinity(cpu %d) failed: %s\n", cpu, std::strerror(errno));
        return false;
    }
    sched_yield();
    return true;
}

// 校验线程确实跑在预期的 A100 核上。
inline bool verify_cpu(int expected) {
    int actual = sched_getcpu();
    if (actual != expected) {
        std::fprintf(stderr, "running on cpu %d, expected %d\n", actual, expected);
        return false;
    }
    return true;
}

inline double median(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

inline double minimum(const std::vector<double> & v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

}  // namespace k3
