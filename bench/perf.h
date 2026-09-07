#ifndef PERF_H
#define PERF_H

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <linux/perf_event.h>
#include <span>
#include <string>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "framework/kernel_registry.h"

namespace ime::bench {

struct PerfEventValue {
    uint64_t value = 0;
    uint64_t time_enabled = 0;
    uint64_t time_running = 0;
};

class PerfEventGroup {
  public:
    PerfEventGroup() = default;
    PerfEventGroup(const PerfEventGroup &) = delete;
    PerfEventGroup &operator=(const PerfEventGroup &) = delete;
    PerfEventGroup(PerfEventGroup &&other) noexcept { *this = std::move(other); }
    PerfEventGroup &operator=(PerfEventGroup &&other) noexcept {
        if (this != &other) {
            close_all();
            fds_ = std::move(other.fds_);
            specs_ = std::move(other.specs_);
            other.fds_.clear();
        }
        return *this;
    }
    ~PerfEventGroup() { close_all(); }

    bool open(std::span<const PerfEventSpec> specs, std::string &error) {
        close_all();
        specs_.assign(specs.begin(), specs.end());
        for (size_t i = 0; i < specs_.size(); ++i) {
            perf_event_attr attr{};
            attr.type = specs_[i].type;
            attr.size = sizeof(attr);
            attr.config = specs_[i].config;
            attr.disabled = i == 0;
            attr.pinned = i == 0;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;
            attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            const int group_fd = fds_.empty() ? -1 : fds_.front();
            const int fd = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, group_fd,
                                                    PERF_FLAG_FD_CLOEXEC));
            if (fd < 0) {
                error = "perf_event_open(" + specs_[i].name + "): " + std::strerror(errno);
                close_all();
                return false;
            }
            fds_.push_back(fd);
        }
        return true;
    }

    bool start(std::string &error) {
        if (fds_.empty()) {
            error = "no perf events configured";
            return false;
        }
        if (ioctl(fds_.front(), PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 ||
            ioctl(fds_.front(), PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
            error = "perf group reset/enable: " + std::string(std::strerror(errno));
            return false;
        }
        return true;
    }

    bool stop_and_read(std::vector<PerfEventValue> &values, std::string &error) {
        if (ioctl(fds_.front(), PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0) {
            error = "perf group disable: " + std::string(std::strerror(errno));
            return false;
        }
        values.clear();
        values.reserve(fds_.size());
        for (size_t i = 0; i < fds_.size(); ++i) {
            PerfEventValue value;
            const ssize_t bytes = read(fds_[i], &value, sizeof(value));
            if (bytes != static_cast<ssize_t>(sizeof(value))) {
                error = "perf read(" + specs_[i].name + "): " +
                        (bytes < 0 ? std::string(std::strerror(errno))
                                   : "short read (event group may not fit available hardware counters)");
                return false;
            }
            values.push_back(value);
        }
        return true;
    }

  private:
    void close_all() noexcept {
        for (int fd : fds_) close(fd);
        fds_.clear();
        specs_.clear();
    }

    std::vector<int> fds_;
    std::vector<PerfEventSpec> specs_;
};

inline PerfEventSpec cycles_event() {
    return {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES};
}

} // namespace ime::bench

#endif
