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
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
using Fn=void(*)(size_t,const void*,const void*,void*);
extern "C" { void dyn_real(size_t,const void*,const void*,void*); void dyn_hot(size_t,const void*,const void*,void*); void dyn_fixed_b(size_t,const void*,const void*,void*); void dyn_fixed_a(size_t,const void*,const void*,void*); }
constexpr size_t A_BYTES=280ull*192*288, B_BYTES=96ull*280*512, INNERS=5160960, POLLUTE=32ull<<20;
struct Buf{void*p{};size_t n;explicit Buf(size_t n):n(n){if(posix_memalign(&p,4096,n))p=nullptr;}~Buf(){free(p);}};
struct Ev{const char*n;uint32_t type;uint64_t config;};
constexpr Ev EXEC[]={{"cycles",PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES},{"instructions",PERF_TYPE_HARDWARE,PERF_COUNT_HW_INSTRUCTIONS},{"load_inst",PERF_TYPE_RAW,0x2b},{"vector_load_inst",PERF_TYPE_RAW,0x39}};
constexpr Ev CACHE[]={{"l1d_load_access",PERF_TYPE_RAW,0x6},{"l1d_load_miss",PERF_TYPE_RAW,0x5},{"l2_load_access",PERF_TYPE_RAW,0xb8},{"l2_load_miss",PERF_TYPE_RAW,0xb9}};
constexpr Ev PATH[]={{"l1d_prefetch_hit",PERF_TYPE_RAW,0xad},{"l1d_prefetch_refill",PERF_TYPE_RAW,0xae},{"l2_ar_request",PERF_TYPE_RAW,0xba},{"l2_ar_stall_cycle",PERF_TYPE_RAW,0xbb}};
constexpr Ev STALL[]={{"cycles",PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES},{"eu_stall",PERF_TYPE_RAW,0x1f}};
struct Counter{Ev e;int fd=-1;};
static uint64_t sink;
static void touch(const void*p,size_t n){auto*q=(volatile const uint8_t*)p;uint64_t x=0;for(size_t i=0;i<n;i+=64)x+=q[i];sink=x;}
static double median(std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];}
static bool open_group(std::vector<Counter>&c,const Ev*e,size_t n){int leader=-1;for(size_t i=0;i<n;i++){perf_event_attr a{};a.type=e[i].type;a.size=sizeof(a);a.config=e[i].config;a.disabled=i==0;a.exclude_kernel=1;a.exclude_hv=1;a.pinned=i==0;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;int fd=(int)syscall(SYS_perf_event_open,&a,0,-1,leader,PERF_FLAG_FD_CLOEXEC);if(fd<0)return false;if(i==0)leader=fd;c.push_back({e[i],fd});}return true;}
int main(int argc,char**argv){int samples=7;size_t sweeps=1;std::string_view cs="real_cold",group="execution";for(int i=1;i<argc;i++){std::string_view a=argv[i];if(a=="--case"&&++i<argc)cs=argv[i];else if(a=="--group"&&++i<argc)group=argv[i];else if(a=="--samples"&&++i<argc)samples=atoi(argv[i]);else if(a=="--sweeps"&&++i<argc)sweeps=strtoull(argv[i],0,0);else{fprintf(stderr,"usage: %s --case real_cold|real_preheated|hot|fixed_b|fixed_a --group execution|cache|path|stall [--samples >=7] [--sweeps N]\n",argv[0]);return 2;}}if(samples<7)return 2;
 cpu_set_t set;CPU_ZERO(&set);CPU_SET(0,&set);int er=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);if(er){fprintf(stderr,"affinity: %s\n",strerror(er));return 1;}Buf A(A_BYTES),B(B_BYTES),O(512),P(POLLUTE);if(!A.p||!B.p||!O.p||!P.p)return 1;for(size_t i=0;i<A.n;i++)((uint8_t*)A.p)[i]=(i*17+3);for(size_t i=0;i<B.n;i++)((uint8_t*)B.p)[i]=(i*29+7);memset(P.p,1,P.n);memset(O.p,0,O.n);
 Fn fn=nullptr;bool cold=false,preheat=false;if(cs=="real_cold"){fn=dyn_real;cold=true;}else if(cs=="real_preheated"){fn=dyn_real;preheat=true;}else if(cs=="hot")fn=dyn_hot;else if(cs=="fixed_b")fn=dyn_fixed_b;else if(cs=="fixed_a")fn=dyn_fixed_a;else return 2;
 const Ev*ev=nullptr;size_t nev=0;if(group=="execution"){ev=EXEC;nev=std::size(EXEC);}else if(group=="cache"){ev=CACHE;nev=std::size(CACHE);}else if(group=="path"){ev=PATH;nev=std::size(PATH);}else if(group=="stall"){ev=STALL;nev=std::size(STALL);}else return 2;std::vector<Counter> ctr;if(!open_group(ctr,ev,nev)){fprintf(stderr,"perf_event_open group=%.*s failed: %s\n",(int)group.size(),group.data(),strerror(errno));return 3;}
 printf("meta case=%.*s group=%.*s cpu=%d samples=%d sweeps=%zu inners_per_sweep=%zu A_bytes=%zu B_bytes=%zu timed=assembly_compute_only\n",(int)cs.size(),cs.data(),(int)group.size(),group.data(),sched_getcpu(),samples,sweeps,INNERS,A_BYTES,B_BYTES);
 for(int w=0;w<2;w++)fn(1,A.p,B.p,O.p);std::vector<double> cyc;
 for(int s=0;s<samples;s++){if(cold){touch(B.p,B.n);touch(P.p,P.n);}else if(preheat){touch(A.p,A.n);touch(B.p,B.n);}else{touch(A.p,8192);touch(B.p,16384);}asm volatile("":::"memory");ioctl(ctr[0].fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP);ioctl(ctr[0].fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP);fn(sweeps,A.p,B.p,O.p);ioctl(ctr[0].fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP);printf("raw case=%.*s group=%.*s sample=%d",(int)cs.size(),cs.data(),(int)group.size(),group.data(),s);for(auto&c:ctr){struct{uint64_t v,en,run;}r{};if(read(c.fd,&r,sizeof(r))!=(ssize_t)sizeof(r))return 1;double pct=r.en?100.0*r.run/r.en:0,per=double(r.v)/(INNERS*sweeps);if(pct<99.99){fprintf(stderr,"\nINVALID running %.6f\n",pct);return 4;}printf(" %s=%" PRIu64 " %s_per_inner=%.6f %s_running_pct=%.6f",c.e.n,r.v,c.e.n,per,c.e.n,pct);if(std::string_view(c.e.n)=="cycles")cyc.push_back(per);}printf(" sink=%" PRIu64 "\n",sink);}
 if(!cyc.empty())printf("summary case=%.*s median_cycles_per_inner=%.6f min=%.6f max=%.6f\n",(int)cs.size(),cs.data(),median(cyc),*std::min_element(cyc.begin(),cyc.end()),*std::max_element(cyc.begin(),cyc.end()));for(auto&c:ctr)close(c.fd);return 0;}
