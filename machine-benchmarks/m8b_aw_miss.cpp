#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr std::size_t kBlocksK = 48;
constexpr std::size_t kNTiles = 96;
constexpr std::size_t kInners = kBlocksK * kNTiles * 2;
constexpr std::size_t kPanelBytes = kBlocksK * 256;
constexpr std::size_t kPayloadBytes = kNTiles * kPanelBytes;
constexpr std::size_t kMaxOffset = 4096;
constexpr std::size_t kOffsetPadding = kMaxOffset + 64;

struct TraceConfig {
    const std::uint8_t *first;
    const std::uint8_t *second;
    std::uint64_t first_n_stride;
    std::uint64_t second_n_stride;
};
extern "C" void m8b_aw_trace(const TraceConfig *, std::size_t, void *);
extern "C" void m8b_aw_trace_probe(const TraceConfig *, std::size_t, void *);
using TraceFn = void (*)(const TraceConfig *, std::size_t, void *);

struct Aligned {
    void *p{};
    std::size_t n{};
    explicit Aligned(std::size_t bytes) : n(bytes) { if (posix_memalign(&p, 4096, bytes)) p = nullptr; }
    ~Aligned() { std::free(p); }
};
struct EventRead { std::uint64_t value, enabled, running; };
struct Result { double misses_per_inner, cycles_per_inner, running_pct; std::uint64_t switches, migrations; };
struct Case { const char *name; bool first_w; bool w_long; bool a_long; };

int open_event(std::uint32_t type, std::uint64_t config, int group, bool leader) {
    perf_event_attr a{};
    a.type = type; a.size = sizeof(a); a.config = config;
    a.disabled = leader; a.pinned = leader; a.exclude_kernel = 1; a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, group, PERF_FLAG_FD_CLOEXEC));
}
void fill(void *p, std::size_t n, unsigned seed) {
    auto *q = static_cast<std::uint8_t *>(p);
    for (std::size_t i = 0; i < n; ++i) q[i] = static_cast<std::uint8_t>(i * 131 + seed);
}
double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
std::uint64_t checksum(const void *p) {
    const auto *q = static_cast<const std::uint8_t *>(p); std::uint64_t x = 0;
    for (unsigned i = 0; i < 256; ++i) x = x * 257 + q[i];
    return x;
}
bool parse_size(const char *s, std::size_t &out) {
    errno = 0; char *end = nullptr; const auto v = std::strtoull(s, &end, 0);
    if (errno || end == s || *end) return false; out = v; return true;
}
} // namespace

int main(int argc, char **argv) {
    std::size_t passes = 64, samples = 9, warmups = 3, a_offset = 0, w_offset = 0;
    bool probe = false, no_counter = false;
    std::string_view selected;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]); std::size_t value{};
        if ((arg == "--passes" || arg == "--samples" || arg == "--warmups" || arg == "--a-offset" || arg == "--w-offset") && ++i < argc && parse_size(argv[i], value)) {
            if (arg == "--passes") passes = value;
            else if (arg == "--samples") samples = value;
            else if (arg == "--warmups") warmups = value;
            else if (arg == "--a-offset") a_offset = value;
            else w_offset = value;
        } else if (arg == "--case" && ++i < argc) selected = argv[i];
        else if (arg == "--probe") probe = true;
        else if (arg == "--no-counter") no_counter = true;
        else { std::fprintf(stderr, "usage: %s [--case NAME] [--passes N] [--samples N] [--warmups N] [--a-offset N] [--w-offset N] [--probe] [--no-counter]\n", argv[0]); return 2; }
    }
    const auto valid_offset = [](std::size_t value) { return value <= kMaxOffset && value % 64 == 0; };
    if (!passes || samples < 1 || (!no_counter && samples < 5) || !valid_offset(a_offset) || !valid_offset(w_offset)) {
        std::fprintf(stderr, "passes>0, samples>=1 (--no-counter) or >=5, and 64-byte aligned A/W offsets in [0,4096] required\n"); return 2;
    }
    cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(0, &cpus);
    const int affinity = pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    if (affinity) { std::fprintf(stderr, "affinity: %s\n", std::strerror(affinity)); return 1; }

    Aligned a(kPayloadBytes + kOffsetPadding), w(kPayloadBytes + kOffsetPadding), out(256);
    if (!a.p || !w.p || !out.p) return 1;
    fill(a.p, a.n, 17); fill(w.p, w.n, 83); std::memset(out.p, 0, out.n);
    if (mlock(a.p, a.n) || mlock(w.p, w.n) || mlock(out.p, out.n)) { std::fprintf(stderr, "mlock: %s\n", std::strerror(errno)); return 1; }

    const std::array<Case, 5> cases{{
        {"A-only-short", false, false, false},
        {"W-only-long", true, true, true},
        {"mixed-W-first", true, true, false},
        {"mixed-A-first", false, true, false},
        {"role-swap-W-short-A-long", true, false, true},
    }};
    if (!selected.empty() && std::none_of(cases.begin(), cases.end(), [&](const Case &c) { return selected == c.name; })) { std::fprintf(stderr, "unknown case\n"); return 2; }

    const int cycles = no_counter ? -1 : open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, true);
    const std::uint64_t l1_miss = PERF_COUNT_HW_CACHE_L1D | (std::uint64_t(PERF_COUNT_HW_CACHE_OP_READ) << 8) | (std::uint64_t(PERF_COUNT_HW_CACHE_RESULT_MISS) << 16);
    const int misses = no_counter ? -1 : open_event(PERF_TYPE_HW_CACHE, l1_miss, cycles, false);
    const int switches = no_counter ? -1 : open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, cycles, false);
    const int migrations = no_counter ? -1 : open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, cycles, false);
    if (!no_counter && (cycles < 0 || misses < 0 || switches < 0 || migrations < 0)) { std::fprintf(stderr, "perf_event_open: %s\n", std::strerror(errno)); return 3; }
    const std::array<int, 4> fds{cycles, misses, switches, migrations};
    const TraceFn trace = probe ? m8b_aw_trace_probe : m8b_aw_trace;
    std::printf("meta blocks_k=%zu n_tiles=%zu inners_per_pass=%zu loads_per_inner=8 bytes_per_stream_per_inner=128 panel_bytes=%zu payload_bytes=%zu cpu=%d passes=%zu samples=%zu warmups=%zu a_offset=%zu w_offset=%zu trace=%s counters=%s pmu_boundary=trace_only\n", kBlocksK, kNTiles, kInners, kPanelBytes, kPayloadBytes, sched_getcpu(), passes, samples, warmups, a_offset, w_offset, probe ? "probe" : "baseline", no_counter ? "external" : "internal");

    for (const auto &c : cases) {
        if (!selected.empty() && selected != c.name) continue;
        const auto *ap = static_cast<const std::uint8_t *>(a.p) + a_offset;
        const auto *wp = static_cast<const std::uint8_t *>(w.p) + w_offset;
        TraceConfig cfg{};
        cfg.first = c.first_w ? wp : ap; cfg.second = c.first_w ? ap : wp;
        cfg.first_n_stride = (c.first_w ? c.w_long : c.a_long) ? kPanelBytes : 0;
        cfg.second_n_stride = (c.first_w ? c.a_long : c.w_long) ? kPanelBytes : 0;
        const char *a_extent = c.a_long ? "long" : "short";
        const char *w_extent = c.w_long ? "long" : "short";
        std::printf("case_config case=%s first=%s second=%s a_extent=%s w_extent=%s pair=%s/%s a_offset=%zu w_offset=%zu\n",
                    c.name, c.first_w ? "W" : "A", c.first_w ? "A" : "W", a_extent, w_extent, a_extent, w_extent, a_offset, w_offset);
        for (std::size_t i = 0; i < warmups; ++i) trace(&cfg, 1, out.p);
        if (no_counter) {
            for (std::size_t s = 0; s < samples; ++s) trace(&cfg, passes, out.p);
            std::printf("external case=%s calls=%zu checksum=%" PRIu64 "\n", c.name, samples, checksum(out.p));
            continue;
        }
        std::vector<double> miss_values, cycle_values;
        for (std::size_t s = 0; s < samples; ++s) {
            if (ioctl(cycles, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) || ioctl(cycles, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) { std::perror("perf start"); return 1; }
            asm volatile("" ::: "memory"); trace(&cfg, passes, out.p); asm volatile("" ::: "memory");
            if (ioctl(cycles, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) { std::perror("perf stop"); return 1; }
            std::array<EventRead, 4> r{};
            for (std::size_t i = 0; i < fds.size(); ++i) if (read(fds[i], &r[i], sizeof(r[i])) != sizeof(r[i])) { std::perror("perf read"); return 1; }
            const double denom = double(passes * kInners);
            const double run = r[0].enabled ? 100.0 * double(r[0].running) / double(r[0].enabled) : 0;
            const double mpi = double(r[1].value) / denom, cpi = double(r[0].value) / denom;
            if (run != 100.0 || r[2].value || r[3].value) { std::fprintf(stderr, "invalid sample case=%s sample=%zu running=%.9f switches=%" PRIu64 " migrations=%" PRIu64 "\n", c.name, s, run, r[2].value, r[3].value); return 4; }
            miss_values.push_back(mpi); cycle_values.push_back(cpi);
            std::printf("raw case=%s sample=%zu l1d_misses=%" PRIu64 " misses_per_inner=%.9f cycles=%" PRIu64 " cycles_per_inner=%.6f time_enabled=%" PRIu64 " time_running=%" PRIu64 " running_pct=%.9f context_switches=%" PRIu64 " migrations=%" PRIu64 " checksum=%" PRIu64 "\n", c.name, s, r[1].value, mpi, r[0].value, cpi, r[0].enabled, r[0].running, run, r[2].value, r[3].value, checksum(out.p));
        }
        const auto [mlo, mhi] = std::minmax_element(miss_values.begin(), miss_values.end());
        std::printf("summary case=%s median_l1d_misses_per_inner=%.9f min=%.9f max=%.9f median_cycles_per_inner=%.6f\n", c.name, median(miss_values), *mlo, *mhi, median(cycle_values));
    }
    for (int fd : fds) if (fd >= 0) close(fd);
    return 0;
}
