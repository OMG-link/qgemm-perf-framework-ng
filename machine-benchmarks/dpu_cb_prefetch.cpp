#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

extern "C" {
void dpu_cb_a_only(const void *,const void *,std::size_t,void *);
void dpu_cb_b_only(const void *,const void *,std::size_t,void *);
void dpu_cb_mixed(const void *,const void *,std::size_t,void *);
void dpu_cb_fixed_b(const void *,const void *,std::size_t,void *);
}
namespace {
constexpr std::size_t kABlock=288,kBPanel=8*1024,kLine=64,kTrash=96*1024*1024;
struct Aligned {void *p{};explicit Aligned(std::size_t n){if(posix_memalign(&p,4096,n))p=nullptr;}~Aligned(){std::free(p);}};
struct Read {std::uint64_t value,enabled,running;};
using Kernel=void(*)(const void*,const void*,std::size_t,void*);
struct Case {const char *name;Kernel run;};
int open_event(std::uint32_t type,std::uint64_t config,int group,bool leader){perf_event_attr a{};a.type=type;a.size=sizeof(a);a.config=config;a.disabled=leader;a.pinned=leader;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;return int(syscall(SYS_perf_event_open,&a,0,-1,group,PERF_FLAG_FD_CLOEXEC));}
void touch(void *memory,std::size_t bytes){auto *p=static_cast<volatile unsigned char*>(memory);unsigned v=0;for(std::size_t i=0;i<bytes;i+=kLine)v+=p[i];asm volatile(""::"r"(v):"memory");}
bool parse(const char *text,std::size_t &v){errno=0;char *end=nullptr;auto n=std::strtoull(text,&end,0);if(errno||end==text||*end)return false;v=n;return true;}
}
int main(int argc,char **argv){
 std::size_t blocks=65536,samples=7;
 for(int i=1;i<argc;++i){std::string_view a=argv[i];if(a=="--blocks"&&i+1<argc&&parse(argv[++i],blocks)){}else if(a=="--samples"&&i+1<argc&&parse(argv[++i],samples)){}else{std::fprintf(stderr,"usage: %s [--blocks N] [--samples N]\n",argv[0]);return 2;}}
 if(!blocks||!samples)return 2;cpu_set_t set;CPU_ZERO(&set);CPU_SET(0,&set);if(int e=pthread_setaffinity_np(pthread_self(),sizeof(set),&set)){std::fprintf(stderr,"affinity: %s\n",std::strerror(e));return 1;}
 Aligned A(blocks*kABlock),B(kBPanel),trash(kTrash),out(64);if(!A.p||!B.p||!trash.p||!out.p)return 1;std::memset(A.p,3,blocks*kABlock);std::memset(B.p,5,kBPanel);std::memset(trash.p,1,kTrash);std::memset(out.p,0,64);
 const std::uint64_t miss_config=PERF_COUNT_HW_CACHE_L1D|(PERF_COUNT_HW_CACHE_OP_READ<<8)|(PERF_COUNT_HW_CACHE_RESULT_MISS<<16);const int cycles_fd=open_event(PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,-1,true);const int misses_fd=cycles_fd<0?-1:open_event(PERF_TYPE_HW_CACHE,miss_config,cycles_fd,false);if(cycles_fd<0||misses_fd<0){std::fprintf(stderr,"perf_event_open: %s\n",std::strerror(errno));if(cycles_fd>=0)close(cycles_fd);return 3;}
 const Case cases[]={{"a_only",dpu_cb_a_only},{"b_only",dpu_cb_b_only},{"mixed",dpu_cb_mixed},{"fixed_b",dpu_cb_fixed_b}};
 for(const auto &c:cases){std::vector<double> misses;misses.reserve(samples);for(std::size_t sample=0;sample<samples;++sample){touch(trash.p,kTrash);touch(B.p,kBPanel);asm volatile("fence rw,rw":::"memory");if(ioctl(cycles_fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)<0||ioctl(cycles_fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)<0)return 4;c.run(A.p,B.p,blocks,out.p);if(ioctl(cycles_fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)<0)return 4;Read cycles{},miss{};if(read(cycles_fd,&cycles,sizeof(cycles))!=sizeof(cycles)||read(misses_fd,&miss,sizeof(miss))!=sizeof(miss))return 4;double cr=cycles.enabled?double(cycles.running)/cycles.enabled:0,mr=miss.enabled?double(miss.running)/miss.enabled:0;if(cr<.99||mr<.99){std::fprintf(stderr,"counter multiplexed\n");return 4;}double per_inner=double(miss.value)/(blocks*2);misses.push_back(per_inner);std::printf("raw case=%s sample=%zu blocks=%zu cycles_per_inner=%.6f l1d_miss_per_inner=%.6f running_pct=%.3f\n",c.name,sample,blocks,double(cycles.value)/(blocks*2),per_inner,100*std::min(cr,mr));}std::sort(misses.begin(),misses.end());std::printf("summary case=%s blocks=%zu inners=%zu median_l1d_miss_per_inner=%.6f\n",c.name,blocks,blocks*2,misses[misses.size()/2]);}
 close(misses_fd);close(cycles_fd);return 0;
}
