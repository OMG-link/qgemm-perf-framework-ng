#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <linux/perf_event.h>
#include <pthread.h>
#include <random>
#include <sched.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

extern "C" {
void pc_corridor(std::size_t);
void pc_vmadot_raw(std::size_t,const void*,const void*,void*);
void pc_vmadot_nop(std::size_t);
std::uintptr_t pc_pointer_chase(std::size_t,void*,std::size_t);
void pc_dyn_inner(std::size_t,const void*,const void*,void*,std::size_t);
void pc_vl8r_vset_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_nop_vset_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_vset_nop_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_vset_vle_nop_hot(const void*,const void*,std::size_t);
void pc_vl8r_vset_vle_nop_control(const void*,const void*,std::size_t);
void pc_vl8r_vset_vadd_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_vset_vadd15_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_vsetm8_vadd_vsetm1_vle_hot(const void*,const void*,std::size_t);
void pc_vl8r_vset_vle_vadd_hot(const void*,const void*,std::size_t);
}
struct Aligned { void *p{}; explicit Aligned(std::size_t n){if(posix_memalign(&p,256,n))p=nullptr;} ~Aligned(){free(p);} };
struct Read {std::uint64_t cycles,enabled,running;};
static int cycles_fd(){perf_event_attr a{};a.type=PERF_TYPE_HARDWARE;a.size=sizeof(a);a.config=PERF_COUNT_HW_CPU_CYCLES;a.disabled=1;a.exclude_kernel=1;a.exclude_hv=1;a.pinned=1;a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;return int(syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC));}
static void make_cycle(void *mem,std::size_t bytes,std::uint64_t seed){const std::size_t n=bytes/64;std::vector<std::size_t> order(n);for(std::size_t i=0;i<n;i++)order[i]=i;std::mt19937_64 g(seed);std::shuffle(order.begin(),order.end(),g);auto *b=static_cast<unsigned char*>(mem);for(std::size_t i=0;i<n;i++)*reinterpret_cast<void**>(b+64*order[i])=b+64*order[(i+1)%n];}
static void thrash(void *p,std::size_t n){volatile unsigned char *q=static_cast<volatile unsigned char*>(p);unsigned x=0;for(std::size_t i=0;i<n;i+=64)x+=q[i];asm volatile(""::"r"(x):"memory");}
int main(int argc,char **argv){
 std::string_view name;std::size_t iters=20000000;int repeats=1;bool list=false,no_counter=false;
 for(int i=1;i<argc;i++){std::string_view a=argv[i];if(a=="--case"&&++i<argc)name=argv[i];else if(a=="--iterations"&&++i<argc)iters=strtoull(argv[i],nullptr,0);else if(a=="--repeats"&&++i<argc)repeats=atoi(argv[i]);else if(a=="--no-counter")no_counter=true;else if(a=="--list")list=true;else{std::fprintf(stderr,"usage: %s --case NAME [--iterations N] [--repeats N] [--no-counter]\n",argv[0]);return 2;}}
 const char *names[]={"corridor","vmadot_raw","vmadot_nop","ptr_hot_immediate","ptr_hot_gap","ptr_cold_immediate","ptr_cold_gap","dyn_hot","dyn_cold","vl8r_vset_vle_hot","vl8r_nop_vset_vle_hot","vl8r_vset_nop_vle_hot","vl8r_vset_vle_nop_hot","vl8r_vset_vle_nop_control","vl8r_vset_vadd_vle_hot","vl8r_vset_vadd15_vle_hot","vl8r_vsetm8_vadd_vsetm1_vle_hot","vl8r_vset_vle_vadd_hot"};
 if(list){for(auto n:names)puts(n);return 0;}if(name.empty()||std::none_of(std::begin(names),std::end(names),[&](auto n){return name==n;})){std::fprintf(stderr,"invalid case\n");return 2;}
 cpu_set_t set;CPU_ZERO(&set);CPU_SET(0,&set);if(int e=pthread_setaffinity_np(pthread_self(),sizeof(set),&set)){std::fprintf(stderr,"affinity: %s\n",strerror(e));return 1;}
 constexpr std::size_t hot_bytes=16*1024,cold_bytes=64*1024*1024,thrash_bytes=96*1024*1024;Aligned hot(hot_bytes),cold(cold_bytes),trash(thrash_bytes),Ahot(128*64),Bhot(256*64),Acold(128*32768),Bcold(256*32768),out(512);if(!hot.p||!cold.p||!trash.p||!Ahot.p||!Bhot.p||!Acold.p||!Bcold.p||!out.p)return 1;
 std::memset(Ahot.p,3,128*64);std::memset(Bhot.p,5,256*64);std::memset(Acold.p,7,128*32768);std::memset(Bcold.p,11,256*32768);std::memset(trash.p,1,thrash_bytes);std::memset(out.p,0,512);make_cycle(hot.p,hot_bytes,0x1234);make_cycle(cold.p,cold_bytes,0x5678);
 const int fd=no_counter?-1:cycles_fd();if(!no_counter&&fd<0){std::fprintf(stderr,"cycles open failed: %s\n",strerror(errno));return 3;}
 for(int r=0;r<repeats;r++){
  if(name.find("cold")!=std::string_view::npos)thrash(trash.p,thrash_bytes);
  if(name=="vl8r_vset_vle_hot")pc_vl8r_vset_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_nop_vset_vle_hot")pc_vl8r_nop_vset_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_nop_vle_hot")pc_vl8r_vset_nop_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_vle_nop_hot")pc_vl8r_vset_vle_nop_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_vle_nop_control")pc_vl8r_vset_vle_nop_control(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_vadd_vle_hot")pc_vl8r_vset_vadd_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_vadd15_vle_hot")pc_vl8r_vset_vadd15_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vsetm8_vadd_vsetm1_vle_hot")pc_vl8r_vsetm8_vadd_vsetm1_vle_hot(Bhot.p,Ahot.p,4096);
  else if(name=="vl8r_vset_vle_vadd_hot")pc_vl8r_vset_vle_vadd_hot(Bhot.p,Ahot.p,4096);
  void *head=(name.find("ptr_cold")!=std::string_view::npos)?cold.p:hot.p;std::uintptr_t sink=0;
  asm volatile("":::"memory");auto begin=std::chrono::steady_clock::now();if(fd>=0){ioctl(fd,PERF_EVENT_IOC_RESET,0);ioctl(fd,PERF_EVENT_IOC_ENABLE,0);}asm volatile("":::"memory");
  if(name=="corridor")pc_corridor(iters);
  else if(name=="vmadot_raw")pc_vmadot_raw(iters,Ahot.p,Bhot.p,out.p);
  else if(name=="vmadot_nop")pc_vmadot_nop(iters);
  else if(name.find("ptr_")==0)sink=pc_pointer_chase(iters,head,name.find("_gap")!=std::string_view::npos);
  else if(name=="dyn_hot")pc_dyn_inner(iters,Ahot.p,Bhot.p,out.p,63);
  else if(name=="vl8r_vset_vle_hot")pc_vl8r_vset_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_nop_vset_vle_hot")pc_vl8r_nop_vset_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_nop_vle_hot")pc_vl8r_vset_nop_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_vle_nop_hot")pc_vl8r_vset_vle_nop_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_vle_nop_control")pc_vl8r_vset_vle_nop_control(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_vadd_vle_hot")pc_vl8r_vset_vadd_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_vadd15_vle_hot")pc_vl8r_vset_vadd15_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vsetm8_vadd_vsetm1_vle_hot")pc_vl8r_vsetm8_vadd_vsetm1_vle_hot(Bhot.p,Ahot.p,iters);
  else if(name=="vl8r_vset_vle_vadd_hot")pc_vl8r_vset_vle_vadd_hot(Bhot.p,Ahot.p,iters);
  else pc_dyn_inner(iters,Acold.p,Bcold.p,out.p,32767);
  asm volatile("":::"memory");if(fd>=0)ioctl(fd,PERF_EVENT_IOC_DISABLE,0);auto end=std::chrono::steady_clock::now();Read x{};if(fd>=0&&read(fd,&x,sizeof(x))!=sizeof(x))return 1;auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();double ratio=fd>=0&&x.enabled?double(x.running)/x.enabled:1;
  std::printf("raw case=%.*s repeat=%d iterations=%zu cycles=%" PRIu64 " duration_ns=%" PRIu64 " cycles_per_iter=%.6f running_pct=%.6f counter=%s sink=%" PRIuPTR "\n",int(name.size()),name.data(),r,iters,x.cycles,std::uint64_t(ns),fd>=0?double(x.cycles)/iters:0.0,100*ratio,fd>=0?"internal":"external",sink);if(fd>=0&&ratio<.99){std::fprintf(stderr,"counter multiplexed\n");return 4;}
 }
 if(fd>=0)close(fd);return 0;
}
