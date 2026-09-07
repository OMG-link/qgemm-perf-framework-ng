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
void dpu_cb_a_only(const void *,const void *,std::size_t,void *);
void dpu_cb_b_only(const void *,const void *,std::size_t,void *);
void dpu_cb_mixed(const void *,const void *,std::size_t,void *);
void dpu_cb_fixed_b(const void *,const void *,std::size_t,void *);
void dpu_cb_packed_a_only(const void *, const void *, std::size_t, void *);
void dpu_cb_packed_mixed(const void *, const void *, std::size_t, void *);
}
namespace {
constexpr std::size_t kTiles = 28, kBlocksPerTile = 16, kInnersPerBlock = 2;
constexpr std::size_t kABlock = 288, kAChunk = kTiles * kBlocksPerTile * kABlock;
constexpr std::size_t kBPanel = 8 * 1024, kLine = 64, kMaxOffset = 4096;
struct Aligned {void *p{};explicit Aligned(std::size_t n){if(posix_memalign(&p,4096,n))p=nullptr;}~Aligned(){std::free(p);}};
struct Read {std::uint64_t value,enabled,running;};
using Kernel=void(*)(const void*,const void*,std::size_t,void*);
struct Case {const char *name;Kernel run;};
int open_event(std::uint32_t type,std::uint64_t config,int group,bool leader){perf_event_attr a{};a.type=type;a.size=sizeof(a);a.config=config;a.disabled=leader;a.pinned=leader;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;return int(syscall(SYS_perf_event_open,&a,0,-1,group,PERF_FLAG_FD_CLOEXEC));}
void touch(void *memory,std::size_t bytes){auto *p=static_cast<volatile unsigned char*>(memory);unsigned v=0;for(std::size_t i=0;i<bytes;i+=kLine)v+=p[i];asm volatile(""::"r"(v):"memory");}
bool parse(const char *text,std::size_t &v){errno=0;char *end=nullptr;auto n=std::strtoull(text,&end,0);if(errno||end==text||*end)return false;v=n;return true;}
}
int main(int argc,char **argv){
    std::size_t passes = 4096, samples = 7, warmup_passes = 0, a_offset = 0, b_offset = 0, scrub_kib = 64, cpu = 0;
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
        } else if (a == "--case" && i + 1 < argc) {
            selected = argv[++i];
        } else if (a == "--lock-memory") {
            lock_memory = true;
        } else if (a == "--continuous") {
            continuous = true;
        } else {
            std::fprintf(
                stderr,
                "usage: %s [--passes N] [--samples N] [--warmup-passes N] [--a-offset 0..4096] [--b-offset 0..4096] [--scrub-kib N] [--cpu N] [--case NAME|all] [--lock-memory] [--continuous]\n",
                argv[0]);
            return 2;
        }
    }
    if (!passes || !samples || !scrub_kib || scrub_kib > 1024 || cpu >= CPU_SETSIZE || a_offset > kMaxOffset || b_offset > kMaxOffset || a_offset % kLine || b_offset % kLine)
        return 2;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (int e = pthread_setaffinity_np(pthread_self(), sizeof(set), &set)) {
        std::fprintf(stderr, "affinity: %s\n", std::strerror(e));
        return 1;
    }
    const std::size_t scrub_bytes = scrub_kib * 1024;
    Aligned A(kAChunk + kMaxOffset), B(kBPanel + kMaxOffset), scrub(scrub_bytes), out(64);
    if (!A.p || !B.p || !scrub.p || !out.p)
        return 1;
    auto *a_data = static_cast<unsigned char *>(A.p) + a_offset;
    auto *b_data = static_cast<unsigned char *>(B.p) + b_offset;
    std::memset(A.p, 3, kAChunk + kMaxOffset);
    std::memset(B.p, 5, kBPanel + kMaxOffset);
    std::memset(scrub.p, 1, scrub_bytes);
    std::memset(out.p, 0, 64);
    if (lock_memory && (mlock(A.p, kAChunk + kMaxOffset) || mlock(B.p, kBPanel + kMaxOffset) || mlock(scrub.p, scrub_bytes) || mlock(out.p, 64))) {
        std::fprintf(stderr, "mlock: %s\n", std::strerror(errno));
        return 1;
    }
    std::printf("config cpu=%zu actual_cpu=%d a=%p b=%p scrub=%p a_offset=%zu b_offset=%zu scrub_kib=%zu warmup_passes=%zu lock_memory=%s continuous=%s\n", cpu, sched_getcpu(), a_data, b_data,
                scrub.p, a_offset, b_offset, scrub_kib, warmup_passes, lock_memory ? "true" : "false", continuous ? "true" : "false");
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
    const Case cases[] = {
        {"a_only", dpu_cb_a_only}, {"b_only", dpu_cb_b_only}, {"mixed", dpu_cb_mixed}, {"fixed_b", dpu_cb_fixed_b}, {"packed_a_only", dpu_cb_packed_a_only}, {"packed_mixed", dpu_cb_packed_mixed}};
    const std::size_t inners = passes * kTiles * kBlocksPerTile * kInnersPerBlock;
    bool ran_case = false;
    for (const auto &c : cases) {
        if (selected != "all" && selected != c.name)
            continue;
        ran_case = true;
        std::vector<double> misses;
        misses.reserve(samples);
        for (std::size_t sample = 0; sample < samples; ++sample) {
            if (!continuous || sample == 0) {
                touch(a_data, kAChunk);
                touch(scrub.p, scrub_bytes);
                touch(b_data, kBPanel);
                if (warmup_passes)
                    c.run(a_data, b_data, warmup_passes, out.p);
            }
            asm volatile("fence rw,rw" ::: "memory");
            if (ioctl(cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 || ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
                return 4;
            c.run(a_data, b_data, passes, out.p);
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
            double per_inner = double(miss.value) / inners;
            misses.push_back(per_inner);
            std::printf("raw case=%s sample=%zu passes=%zu warmup_passes=%zu inners=%zu a_offset=%zu b_offset=%zu scrub_kib=%zu cycles_per_inner=%.6f l1d_miss_per_inner=%.6f context_switches=%llu "
                        "migrations=%llu running_pct=%.3f\n",
                        c.name, sample, passes, warmup_passes, inners, a_offset, b_offset, scrub_kib, double(cycles.value) / inners, per_inner, static_cast<unsigned long long>(context_switches.value),
                        static_cast<unsigned long long>(migrations.value), 100 * std::min({cr, mr, csr, migr}));
        }
        std::sort(misses.begin(), misses.end());
        std::printf("summary case=%s passes=%zu inners=%zu a_offset=%zu b_offset=%zu scrub_kib=%zu median_l1d_miss_per_inner=%.6f\n", c.name, passes, inners, a_offset, b_offset, scrub_kib,
                    misses[misses.size() / 2]);
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
