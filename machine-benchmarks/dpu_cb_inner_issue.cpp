#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

using Kernel = void (*)(std::size_t, const void *, const void *, void *);
extern "C" {
void dpu_cb_issue_hot(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_fixed_w(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_fixed_a(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_reordered(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_prefetch_512(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_prefetch_2048(std::size_t, const void *, const void *, void *);
void dpu_cb_issue_production_prefetch_8192(std::size_t, const void *, const void *, void *);
}

namespace {
constexpr std::size_t kInnersPerSweep = 552960;
constexpr std::size_t kABytes = 60 * 48 * 256;
constexpr std::size_t kWBytes = 96 * 48 * 512;
constexpr std::size_t kOutBytes = 512;
struct Aligned {
    void *p{};
    explicit Aligned(std::size_t bytes) { if (posix_memalign(&p, 64, bytes)) p = nullptr; }
    ~Aligned() { std::free(p); }
};
struct Counter { std::uint32_t type; std::uint64_t config; std::string_view name; };
struct PerfRead { std::uint64_t value, enabled, running; };
struct Case { const char *name; Kernel fn; const char *definition; };

bool parse_size(std::string_view text, std::size_t &value) {
    if (text.empty() || text.front() == '+' || text.front() == '-') return false;
    errno = 0; char *end = nullptr;
    const auto parsed = std::strtoull(text.data(), &end, 0);
    if (errno || end == text.data() || end != text.data() + text.size()) return false;
    value = parsed; return true;
}
bool parse_raw(std::string_view text, std::uint64_t &value) {
    if (text.empty() || text.front() == '+' || text.front() == '-') return false;
    errno = 0; char *end = nullptr;
    const auto parsed = std::strtoull(text.data(), &end, 16);
    if (errno || end == text.data() || end != text.data() + text.size()) return false;
    value = parsed; return true;
}
int open_counter(const Counter &counter) {
    perf_event_attr attr{}; attr.type = counter.type; attr.size = sizeof(attr); attr.config = counter.config;
    attr.disabled = 1; attr.pinned = 1; attr.exclude_kernel = 1; attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
std::uint64_t checksum(const std::uint8_t *p) {
    std::uint64_t value = 0; for (std::size_t i = 0; i < kOutBytes; ++i) value = value * 131 + p[i]; return value;
}
void touch(const void *memory, std::size_t bytes) {
    const auto *p = static_cast<const volatile std::uint8_t *>(memory); unsigned value = 0;
    for (std::size_t i = 0; i < bytes; i += 64) value += p[i];
    asm volatile("" : : "r"(value) : "memory");
}
} // namespace

int main(int argc, char **argv) {
    std::size_t sweeps = 1, samples = 7;
    std::string_view selected;
    Counter counter{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--sweeps" && i + 1 < argc && parse_size(argv[++i], sweeps)) {
        } else if (arg == "--samples" && i + 1 < argc && parse_size(argv[++i], samples)) {
        } else if (arg == "--case" && i + 1 < argc) {
            selected = argv[++i];
        } else if (arg == "--counter" && i + 1 < argc) {
            const std::string_view value(argv[++i]);
            if (value == "cycles") counter = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"};
            else if (value == "instructions") counter = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "instructions"};
            else if (value.starts_with("raw:")) {
                std::uint64_t raw = 0;
                if (!parse_raw(value.substr(4), raw)) { std::fprintf(stderr, "invalid raw counter: %s\n", argv[i]); return 2; }
                counter = {PERF_TYPE_RAW, raw, value};
            } else { std::fprintf(stderr, "unknown counter: %s\n", argv[i]); return 2; }
        } else {
            std::fprintf(stderr, "usage: %s [--sweeps N] [--samples N>=7] [--case NAME] [--counter cycles|instructions|raw:HEX]\n", argv[0]);
            return 2;
        }
    }
    if (!sweeps || samples < 7 || sweeps > SIZE_MAX / kInnersPerSweep) {
        std::fprintf(stderr, "sweeps must be >0 and samples must be >=7\n"); return 2;
    }
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(0, &cpuset);
    if (const int error = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset)) {
        std::fprintf(stderr, "affinity: %s\n", std::strerror(error)); return 1;
    }
    Aligned a(kABytes), w(kWBytes), out(kOutBytes);
    if (!a.p || !w.p || !out.p) return 1;
    for (std::size_t i = 0; i < kABytes; ++i) static_cast<std::uint8_t *>(a.p)[i] = static_cast<std::uint8_t>(i * 17 + 3);
    for (std::size_t i = 0; i < kWBytes; ++i) static_cast<std::uint8_t *>(w.p)[i] = static_cast<std::uint8_t>(i * 29 + 7);
    std::memset(out.p, 0, kOutBytes);
    touch(a.p, kABytes); touch(w.p, kWBytes);

    const std::array<Case, 8> cases{{
        {"hot", dpu_cb_issue_hot, "production loop counts; fixed 256B A and 512B W hot set"},
        {"production", dpu_cb_issue_production, "panel->chunk(28/28/4)->N96->M->bk16->inner2 production addresses"},
        {"production_fixed_w", dpu_cb_issue_production_fixed_w, "production A addresses; fixed hot 512B W"},
        {"production_fixed_a", dpu_cb_issue_production_fixed_a, "production W addresses; fixed hot 256B A"},
        {"production_reordered", dpu_cb_issue_production_reordered, "production addresses; first update all 8 accumulators then second update all 8"},
        {"prefetch_w_512", dpu_cb_issue_production_prefetch_512, "production; prefetch next N W panel first 512B with four M tiles lead"},
        {"prefetch_w_2048", dpu_cb_issue_production_prefetch_2048, "production; prefetch next N W panel first 2KiB with four M tiles lead"},
        {"prefetch_w_8192", dpu_cb_issue_production_prefetch_8192, "production; prefetch complete next N 8KiB W panel with four M tiles lead"},
    }};
    if (!selected.empty() && std::none_of(cases.begin(), cases.end(), [&](const Case &c) { return selected == c.name; })) {
        std::fprintf(stderr, "unknown case: %.*s\n", static_cast<int>(selected.size()), selected.data()); return 2;
    }
    const int perf_fd = open_counter(counter);
    if (perf_fd < 0) { std::fprintf(stderr, "perf_event_open %.*s: %s\n", static_cast<int>(counter.name.size()), counter.name.data(), std::strerror(errno)); return 3; }
    const std::size_t total_inners = sweeps * kInnersPerSweep;
    std::printf("meta counter=%.*s cpu=0 actual_cpu=%d sweeps=%zu samples=%zu inners_per_sweep=%zu total_inners=%zu a_bytes=%zu w_bytes=%zu warmups=1 timed_region=kernel_call\n",
                static_cast<int>(counter.name.size()), counter.name.data(), sched_getcpu(), sweeps, samples, kInnersPerSweep, total_inners, kABytes, kWBytes);
    for (const auto &c : cases) {
        if (!selected.empty() && selected != c.name) continue;
        std::printf("case_definition case=%s definition=%s\n", c.name, c.definition);
        c.fn(1, a.p, w.p, out.p);
        std::vector<double> per_inner, running_pcts;
        for (std::size_t sample = 0; sample < samples; ++sample) {
            asm volatile("fence rw,rw" ::: "memory");
            if (ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0) || ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0)) { std::perror("perf start"); close(perf_fd); return 4; }
            c.fn(sweeps, a.p, w.p, out.p);
            if (ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0)) { std::perror("perf stop"); close(perf_fd); return 4; }
            PerfRead readout{};
            if (read(perf_fd, &readout, sizeof(readout)) != sizeof(readout)) { std::perror("perf read"); close(perf_fd); return 4; }
            const double running_pct = readout.enabled ? 100.0 * static_cast<double>(readout.running) / readout.enabled : 0.0;
            if (running_pct < 99.0) { std::fprintf(stderr, "case=%s sample=%zu running_pct=%.6f below 99%%\n", c.name, sample, running_pct); close(perf_fd); return 4; }
            const double value = static_cast<double>(readout.value) / total_inners;
            per_inner.push_back(value); running_pcts.push_back(running_pct);
            std::printf("raw case=%s sample=%zu value=%" PRIu64 " value_per_inner=%.6f running_pct=%.6f checksum=%" PRIu64 "\n",
                        c.name, sample, readout.value, value, running_pct, checksum(static_cast<const std::uint8_t *>(out.p)));
        }
        const auto [minimum, maximum] = std::minmax_element(per_inner.begin(), per_inner.end());
        std::printf("summary case=%s median_value_per_inner=%.6f min=%.6f max=%.6f median_running_pct=%.6f\n",
                    c.name, median(per_inner), *minimum, *maximum, median(running_pcts));
    }
    close(perf_fd); return 0;
}
