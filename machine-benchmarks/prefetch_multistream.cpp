#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <random>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

extern "C" {
std::uintptr_t prefetch_multistream_gap16(const void *, std::size_t, std::size_t);
std::uintptr_t prefetch_multistream_gap64(const void *, std::size_t, std::size_t);
}

namespace {
constexpr std::size_t kPage = 4096;
constexpr std::size_t kStreamStride = 4160;
constexpr std::size_t kLine = 64;
constexpr std::size_t kTrashBytes = 96 * 1024 * 1024;
struct Aligned { void *p{}; explicit Aligned(std::size_t n){if(posix_memalign(&p,kPage,n))p=nullptr;} ~Aligned(){std::free(p);} };
struct Read { std::uint64_t value, enabled, running; };
using Sequence = std::uintptr_t (*)(const void *, std::size_t, std::size_t);

int open_event(std::uint32_t type, std::uint64_t config, int group, bool leader) {
    perf_event_attr attr{}; attr.type=type; attr.size=sizeof(attr); attr.config=config;
    attr.disabled=leader; attr.pinned=leader; attr.exclude_kernel=1; attr.exclude_hv=1;
    attr.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open,&attr,0,-1,group,PERF_FLAG_FD_CLOEXEC));
}
void thrash(void *memory) { auto *p=static_cast<volatile unsigned char *>(memory); unsigned v=0; for(std::size_t i=0;i<kTrashBytes;i+=kLine)v+=p[i]; asm volatile(""::"r"(v):"memory"); }
bool parse(const char *text,std::size_t &value) { errno=0; char *end=nullptr; auto n=std::strtoull(text,&end,0); if(errno||end==text||*end)return false; value=n; return true; }
}

int main(int argc,char **argv) {
    std::size_t trials=1024,samples=7,max_streams=32,gap=64;
    for(int i=1;i<argc;++i) { std::string_view arg=argv[i];
        if(arg=="--trials"&&i+1<argc&&parse(argv[++i],trials)){}
        else if(arg=="--samples"&&i+1<argc&&parse(argv[++i],samples)){}
        else if(arg=="--max-streams"&&i+1<argc&&parse(argv[++i],max_streams)){}
        else if(arg=="--gap"&&i+1<argc&&parse(argv[++i],gap)&&(gap==16||gap==64)){}
        else { std::fprintf(stderr,"usage: %s [--trials N] [--samples N] [--max-streams N] [--gap 16|64]\n",argv[0]); return 2; }
    }
    if(!trials||!samples||!max_streams||max_streams>64)return 2;
    cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(0,&cpus);
    if(int e=pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus)){std::fprintf(stderr,"affinity: %s\n",std::strerror(e));return 1;}

    const std::size_t trial_bytes=((max_streams*kStreamStride+4*kLine+kPage-1)/kPage)*kPage;
    Aligned regions(trials*trial_bytes),trash_memory(kTrashBytes);
    if(!regions.p||!trash_memory.p)return 1;
    auto *bytes=static_cast<unsigned char *>(regions.p);
    for(std::size_t i=0;i<trials*trial_bytes;++i)bytes[i]=static_cast<unsigned char>((i*131+17)|1);
    std::memset(trash_memory.p,1,kTrashBytes);
    std::vector<void *> order(trials);
    for(std::size_t i=0;i<trials;++i)order[i]=bytes+i*trial_bytes;
    std::mt19937_64 random(0x6d756c7469737472ULL); std::shuffle(order.begin(),order.end(),random);

    const std::uint64_t miss_config=PERF_COUNT_HW_CACHE_L1D|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16);
    const int cycles_fd=open_event(PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,-1,true);
    const int misses_fd=cycles_fd<0?-1:open_event(PERF_TYPE_HW_CACHE,miss_config,cycles_fd,false);
    if(cycles_fd<0||misses_fd<0){std::fprintf(stderr,"perf_event_open: %s\n",std::strerror(errno));if(cycles_fd>=0)close(cycles_fd);return 3;}
    const Sequence run=gap==16?prefetch_multistream_gap16:prefetch_multistream_gap64;

    for(std::size_t streams=1;streams<=max_streams;++streams) {
        std::vector<double> medians; medians.reserve(samples);
        for(std::size_t sample=0;sample<samples;++sample) {
            thrash(trash_memory.p);
            std::uintptr_t pointer_sink=0; for(void *p:order)pointer_sink^=reinterpret_cast<std::uintptr_t>(p);
            asm volatile("fence rw,rw"::"r"(pointer_sink):"memory");
            if(ioctl(cycles_fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)<0||ioctl(cycles_fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)<0)return 4;
            const auto sink=run(order.data(),trials,streams);
            if(ioctl(cycles_fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)<0)return 4;
            Read cycles{},miss{}; if(read(cycles_fd,&cycles,sizeof(cycles))!=sizeof(cycles)||read(misses_fd,&miss,sizeof(miss))!=sizeof(miss))return 4;
            const double cr=cycles.enabled?double(cycles.running)/cycles.enabled:0,mr=miss.enabled?double(miss.running)/miss.enabled:0;
            if(cr<.99||mr<.99){std::fprintf(stderr,"counter multiplexed\n");return 4;}
            const double misses_per_stream=double(miss.value)/(trials*streams); medians.push_back(misses_per_stream);
            std::printf("raw gap=%zu streams=%zu sample=%zu trials=%zu cycles_per_stream=%.6f l1d_miss_per_stream=%.6f running_pct=%.3f sink=%" PRIuPTR "\n",gap,streams,sample,trials,double(cycles.value)/(trials*streams),misses_per_stream,100*std::min(cr,mr),sink);
        }
        std::sort(medians.begin(),medians.end());
        std::printf("summary gap=%zu streams=%zu loads_per_stream=4 expected_no_prefetch=4 expected_tracked=3 median_l1d_miss_per_stream=%.6f\n",gap,streams,medians[medians.size()/2]);
    }
    close(misses_fd); close(cycles_fd); return 0;
}
