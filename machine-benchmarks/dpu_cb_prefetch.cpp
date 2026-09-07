#include <algorithm>
#include <cerrno>
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

extern "C" {
void dpu_cb_a_only(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_b_only(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_mixed(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_fixed_b(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_packed_a_only(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_packed_mixed(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_b_outer_b_only(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
void dpu_cb_b_outer_mixed(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
}
namespace {
constexpr std::size_t kTiles = 28, kBlocksPerTile = 16, kInnersPerBlock = 2;
constexpr std::size_t kABlock = 288, kAChunk = kTiles * kBlocksPerTile * kABlock, kABytesPerTile = kBlocksPerTile * kABlock;
constexpr std::size_t kLegacyBPanel = 8 * 1024, kBBlock = 512, kLine = 64, kMaxOffset = 4096, kMaxAllocation = 1024ULL * 1024 * 1024;
struct Aligned {void *p{};explicit Aligned(std::size_t n){if(posix_memalign(&p,4096,n))p=nullptr;}~Aligned(){std::free(p);}};
struct Read {std::uint64_t value,enabled,running;};
using Kernel = void (*)(const void *, const void *, std::size_t, void *, std::size_t, std::size_t, const std::uint64_t *, std::size_t);
enum class BPattern { None, Fixed, Alternate, Sequential, Shuffled };
struct Case {
    const char *name;
    Kernel run;
    BPattern pattern;
};
int open_event(std::uint32_t type,std::uint64_t config,int group,bool leader){perf_event_attr a{};a.type=type;a.size=sizeof(a);a.config=config;a.disabled=leader;a.pinned=leader;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;return int(syscall(SYS_perf_event_open,&a,0,-1,group,PERF_FLAG_FD_CLOEXEC));}
void touch(void *memory,std::size_t bytes){auto *p=static_cast<volatile unsigned char*>(memory);unsigned v=0;for(std::size_t i=0;i<bytes;i+=kLine)v+=p[i];asm volatile(""::"r"(v):"memory");}
bool parse(const char *text,std::size_t &v){errno=0;char *end=nullptr;auto n=std::strtoull(text,&end,0);if(errno||end==text||*end)return false;v=n;return true;}
}
int main(int argc,char **argv){
    std::size_t passes = 4096, samples = 7, warmup_passes = 0, a_offset = 0, b_offset = 0, scrub_kib = 64, cpu = 0;
    std::size_t b_panels = 96, a_tiles = 60, b_panel_kib = 8;
    bool lock_memory = false, continuous = false;
    std::string_view selected = "all";
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--passes" && i + 1 < argc && parse(argv[++i], passes)) {
        } else if (a == "--samples" && i + 1 < argc && parse(argv[++i], samples)) {
        } else if (a == "--warmup-passes" && i + 1 < argc && parse(argv[++i], warmup_passes)) {
        } else if (a == "--a-offset" && i + 1 < argc && parse(argv[++i], a_offset)) {
        } else if (a == "--b-offset" && i + 1 < argc && parse(argv[++i], b_offset)) {
        } else if (a == "--scrub-kib" && i + 1 < argc && parse(argv[++i], scrub_kib)) {
        } else if (a == "--cpu" && i + 1 < argc && parse(argv[++i], cpu)) {
        } else if (a == "--b-panels" && i + 1 < argc && parse(argv[++i], b_panels)) {
        } else if (a == "--b-panel-kib" && i + 1 < argc && parse(argv[++i], b_panel_kib)) {
        } else if (a == "--a-tiles" && i + 1 < argc && parse(argv[++i], a_tiles)) {
        } else if (a == "--case" && i + 1 < argc) {
            selected = argv[++i];
        } else if (a == "--lock-memory") {
            lock_memory = true;
        } else if (a == "--continuous") {
            continuous = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--passes N] [--samples N] [--warmup-passes N] [--a-offset 0..4096] [--b-offset 0..4096] [--scrub-kib N] [--cpu N] [--b-panels N] [--b-panel-kib 4|8] [--a-tiles "
                         "N] [--case NAME|all] "
                         "[--lock-memory] [--continuous]\n",
                         argv[0]);
            return 2;
        }
    }
    const std::size_t b_panel_bytes = b_panel_kib * 1024;
    const std::size_t b_blocks_per_panel = b_panel_bytes / kBBlock;
    if (!passes || !samples || !scrub_kib || scrub_kib > 1024 || cpu >= CPU_SETSIZE || !b_panels || !a_tiles || (b_panel_kib != 4 && b_panel_kib != 8) ||
        b_panels > (kMaxAllocation - kMaxOffset) / b_panel_bytes || a_tiles > (kMaxAllocation - kMaxOffset) / kABytesPerTile || a_offset > kMaxOffset || b_offset > kMaxOffset || a_offset % kLine ||
        b_offset % kLine)
        return 2;
    const std::size_t a_payload_bytes = a_tiles * kABytesPerTile;
    const std::size_t b_payload_bytes = b_panels * b_panel_bytes;
    const std::size_t a_bytes = std::max(kAChunk, a_payload_bytes) + kMaxOffset;
    const std::size_t b_bytes = std::max(kLegacyBPanel, b_payload_bytes) + kMaxOffset;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (int e = pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
        std::fprintf(stderr, "affinity: %s\n", std::strerror(e));
        return 1;
    }
    const std::size_t scrub_bytes = scrub_kib * 1024;
    Aligned A(a_bytes), B(b_bytes), scrub(scrub_bytes), out(64);
    if (!A.p || !B.p || !scrub.p || !out.p)
        return 1;
    auto *a_data = static_cast<unsigned char *>(A.p) + a_offset;
    auto *b_data = static_cast<unsigned char *>(B.p) + b_offset;
    std::memset(A.p, 3, a_bytes);
    std::memset(B.p, 5, b_bytes);
    std::memset(scrub.p, 1, scrub_bytes);
    std::memset(out.p, 0, 64);
    if (lock_memory && (mlock(A.p, a_bytes) || mlock(B.p, b_bytes) || mlock(scrub.p, scrub_bytes) || mlock(out.p, 64))) {
        std::fprintf(stderr, "mlock: %s\n", std::strerror(errno));
        return 1;
    }
    std::printf("config cpu=%zu actual_cpu=%d a=%p b=%p scrub=%p a_offset=%zu b_offset=%zu scrub_kib=%zu b_panels=%zu b_panel_kib=%zu blocks_per_panel=%zu b_payload_kib=%zu a_tiles=%zu "
                "warmup_passes=%zu lock_memory=%s continuous=%s\n",
                cpu, sched_getcpu(), a_data, b_data, scrub.p, a_offset, b_offset, scrub_kib, b_panels, b_panel_kib, b_blocks_per_panel, b_payload_bytes / 1024, a_tiles, warmup_passes,
                lock_memory ? "true" : "false", continuous ? "true" : "false");
    const std::uint64_t miss_config = PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    const int cycles_fd = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1, true);
    const int misses_fd = cycles_fd < 0 ? -1 : open_event(PERF_TYPE_HW_CACHE, miss_config, cycles_fd, false);
    const int context_switches_fd = cycles_fd < 0 ? -1 : open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, cycles_fd, false);
    const int migrations_fd = cycles_fd < 0 ? -1 : open_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, cycles_fd, false);
    if (cycles_fd < 0 || misses_fd < 0 || context_switches_fd < 0 || migrations_fd < 0) {
        std::fprintf(stderr, "perf_event_open: %s\n", std::strerror(errno));
        if (cycles_fd >= 0)
            close(cycles_fd);
        return 3;
    }
    std::vector<std::uint64_t> fixed_order(b_panels, 0), alternate_order(b_panels), sequential_order(b_panels), shuffled_order(b_panels);
    for (std::size_t i = 0; i < b_panels; ++i) {
        alternate_order[i] = (i % std::min<std::size_t>(2, b_panels)) * b_panel_bytes;
        sequential_order[i] = i * b_panel_bytes;
        shuffled_order[i] = i * b_panel_bytes;
    }
    std::uint64_t random_state = 0x9e3779b97f4a7c15ULL;
    for (std::size_t i = b_panels; i > 1; --i) {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        std::swap(shuffled_order[i - 1], shuffled_order[random_state % i]);
    }
    if (b_panels > 81) {
        const std::size_t requested_prefix[] = {37, 4, 81};
        for (std::size_t position = 0; position < 3; ++position) {
            const auto wanted = requested_prefix[position] * b_panel_bytes;
            const auto found = std::find(shuffled_order.begin() + position, shuffled_order.end(), wanted);
            std::iter_swap(shuffled_order.begin() + position, found);
        }
    }
    const Case cases[] = {{"a_only", dpu_cb_a_only, BPattern::None},
                          {"b_only", dpu_cb_b_only, BPattern::None},
                          {"mixed", dpu_cb_mixed, BPattern::None},
                          {"fixed_b", dpu_cb_fixed_b, BPattern::None},
                          {"packed_a_only", dpu_cb_packed_a_only, BPattern::None},
                          {"packed_mixed", dpu_cb_packed_mixed, BPattern::None},
                          {"b_pattern_fixed", dpu_cb_b_outer_mixed, BPattern::Fixed},
                          {"b_pattern_alternate", dpu_cb_b_outer_mixed, BPattern::Alternate},
                          {"b_pattern_sequential", dpu_cb_b_outer_mixed, BPattern::Sequential},
                          {"b_pattern_shuffled", dpu_cb_b_outer_mixed, BPattern::Shuffled}};
    bool ran_case = false;
    for (const auto &c : cases) {
        const bool selected_pattern_group = selected == "b_patterns" && c.pattern != BPattern::None;
        if (selected != "all" && selected != c.name && !selected_pattern_group)
            continue;
        ran_case = true;
        const bool b_pattern = c.pattern != BPattern::None;
        const std::size_t measured_a_tiles = b_pattern ? a_tiles : kTiles;
        const std::size_t measured_b_panels = b_pattern ? b_panels : 1;
        const std::size_t measured_blocks_per_panel = b_pattern ? b_blocks_per_panel : kBlocksPerTile;
        const std::uint64_t *b_order = nullptr;
        switch (c.pattern) {
        case BPattern::Fixed:
            b_order = fixed_order.data();
            break;
        case BPattern::Alternate:
            b_order = alternate_order.data();
            break;
        case BPattern::Sequential:
            b_order = sequential_order.data();
            break;
        case BPattern::Shuffled:
            b_order = shuffled_order.data();
            break;
        case BPattern::None:
            break;
        }
        const std::size_t inners = passes * measured_b_panels * measured_a_tiles * measured_blocks_per_panel * kInnersPerBlock;
        std::vector<double> misses;
        misses.reserve(samples);
        if (b_pattern) {
            std::printf("order case=%s first_panels=", c.name);
            for (std::size_t i = 0; i < std::min<std::size_t>(8, b_panels); ++i)
                std::printf("%s%llu", i ? "," : "", static_cast<unsigned long long>(b_order[i] / b_panel_bytes));
            std::printf("\n");
        }
        for (std::size_t sample = 0; sample < samples; ++sample) {
            if (!continuous || sample == 0) {
                touch(a_data, b_pattern ? a_payload_bytes : kAChunk);
                touch(scrub.p, scrub_bytes);
                touch(b_data, b_pattern ? b_panel_bytes : kLegacyBPanel);
                if (warmup_passes)
                    c.run(a_data, b_data, warmup_passes, out.p, b_panels, a_tiles, b_order, b_blocks_per_panel);
            }
            asm volatile("fence rw,rw" ::: "memory");
            if (ioctl(cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 || ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
                return 4;
            c.run(a_data, b_data, passes, out.p, b_panels, a_tiles, b_order, b_blocks_per_panel);
            if (ioctl(cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
                return 4;
            Read cycles{}, miss{}, context_switches{}, migrations{};
            if (read(cycles_fd, &cycles, sizeof(cycles)) != sizeof(cycles) || read(misses_fd, &miss, sizeof(miss)) != sizeof(miss) ||
                read(context_switches_fd, &context_switches, sizeof(context_switches)) != sizeof(context_switches) || read(migrations_fd, &migrations, sizeof(migrations)) != sizeof(migrations))
                return 4;
            double cr = cycles.enabled ? double(cycles.running) / cycles.enabled : 0, mr = miss.enabled ? double(miss.running) / miss.enabled : 0;
            double csr = context_switches.enabled ? double(context_switches.running) / context_switches.enabled : 0;
            double migr = migrations.enabled ? double(migrations.running) / migrations.enabled : 0;
            if (cr < .99 || mr < .99 || csr < .99 || migr < .99) {
                std::fprintf(stderr, "counter multiplexed\n");
                return 4;
            }
            const double per_inner = double(miss.value) / inners;
            const double per_panel_visit = double(miss.value) / (passes * measured_b_panels);
            const double per_pass = double(miss.value) / passes;
            misses.push_back(per_inner);
            std::printf("raw case=%s sample=%zu passes=%zu warmup_passes=%zu inners=%zu b_panels=%zu b_panel_kib=%zu blocks_per_panel=%zu a_tiles=%zu a_offset=%zu b_offset=%zu scrub_kib=%zu "
                        "cycles_per_inner=%.6f l1d_miss_per_inner=%.6f "
                        "l1d_miss_per_panel_visit=%.6f l1d_miss_per_pass=%.6f context_switches=%llu migrations=%llu running_pct=%.3f\n",
                        c.name, sample, passes, warmup_passes, inners, measured_b_panels, b_pattern ? b_panel_kib : 8, measured_blocks_per_panel, measured_a_tiles, a_offset, b_offset, scrub_kib,
                        double(cycles.value) / inners, per_inner, per_panel_visit, per_pass, static_cast<unsigned long long>(context_switches.value), static_cast<unsigned long long>(migrations.value),
                        100 * std::min({cr, mr, csr, migr}));
        }
        std::sort(misses.begin(), misses.end());
        const double median_per_inner = misses[misses.size() / 2];
        std::printf("summary case=%s passes=%zu inners=%zu b_panels=%zu b_panel_kib=%zu blocks_per_panel=%zu a_tiles=%zu a_offset=%zu b_offset=%zu scrub_kib=%zu median_l1d_miss_per_inner=%.6f "
                    "median_l1d_miss_per_panel_visit=%.6f median_l1d_miss_per_pass=%.6f\n",
                    c.name, passes, inners, measured_b_panels, b_pattern ? b_panel_kib : 8, measured_blocks_per_panel, measured_a_tiles, a_offset, b_offset, scrub_kib, median_per_inner,
                    median_per_inner * measured_a_tiles * measured_blocks_per_panel * kInnersPerBlock,
                    median_per_inner * measured_b_panels * measured_a_tiles * measured_blocks_per_panel * kInnersPerBlock);
    }
    if (!ran_case) {
        std::fprintf(stderr, "unknown case: %.*s\n", static_cast<int>(selected.size()), selected.data());
        close(migrations_fd);
        close(context_switches_fd);
        close(misses_fd);
        close(cycles_fd);
        return 2;
    }
    close(migrations_fd);
    close(context_switches_fd);
    close(misses_fd);
    close(cycles_fd);
    return 0;
}
