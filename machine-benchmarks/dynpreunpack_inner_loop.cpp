#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <chrono>
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
void dyn_init_operands();
void dyn_body_full(std::size_t,const void*,const void*,void*); void dyn_body_no_bload(std::size_t,const void*,const void*,void*);
void dyn_body_no_aloads(std::size_t,const void*,const void*,void*); void dyn_body_no_dots(std::size_t,const void*,const void*,void*);
void dyn_body_b_fanout1(std::size_t,const void*,const void*,void*); void dyn_dep_load_dot(std::size_t,const void*,const void*,void*);
void dyn_dep_load_base(std::size_t,const void*,const void*,void*); void dyn_dep_dot_chain(std::size_t,const void*,const void*,void*);
void dyn_dep_dot_base(std::size_t,const void*,const void*,void*);
}
struct Aligned { void *p{}; explicit Aligned(std::size_t n){if(posix_memalign(&p,64,n))p=nullptr;} ~Aligned(){free(p);} };
struct Case {const char *name; Kernel fn;};
struct PerfRead {std::uint64_t cycles, time_enabled, time_running;};
static std::uint64_t checksum(const std::uint8_t *p){std::uint64_t x=0; for(int i=0;i<512;i++) x=x*131+p[i]; return x;}
static double median(std::vector<double> x){std::sort(x.begin(),x.end()); return x[x.size()/2];}
static int open_cycles(){
 perf_event_attr a{}; a.type=PERF_TYPE_HARDWARE; a.size=sizeof(a); a.config=PERF_COUNT_HW_CPU_CYCLES;
 a.disabled=1; a.exclude_kernel=1; a.exclude_hv=1; a.pinned=1;
 a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
 return static_cast<int>(syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC));
}
int main(int argc,char **argv){
 std::size_t iters=200000; int samples=7; std::string_view selected;
 for(int i=1;i<argc;i++){
  const std::string_view arg(argv[i]);
  if(arg=="--iterations"&&++i<argc) iters=strtoull(argv[i],nullptr,0);
  else if(arg=="--samples"&&++i<argc) samples=atoi(argv[i]);
  else if(arg=="--case"&&++i<argc) selected=argv[i];
  else {std::fprintf(stderr,"usage: %s [--iterations N] [--samples N] [--case NAME]\n",argv[0]); return 2;}
 }
 if(samples<7||iters<200000){std::fprintf(stderr,"samples must be >=7 and iterations >=200000\n");return 2;}
 cpu_set_t set; CPU_ZERO(&set); CPU_SET(0,&set); const int affinity=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
 if(affinity){std::fprintf(stderr,"affinity: %s\n",strerror(affinity));return 1;}
 Aligned A(8192),B(16384),O(512); if(!A.p||!B.p||!O.p)return 1;
 for(int i=0;i<8192;i++)((std::uint8_t*)A.p)[i]=(i*17+3)&255;
 for(int i=0;i<16384;i++)((std::uint8_t*)B.p)[i]=(i*29+7)&255; std::memset(O.p,0,512);
 const std::array<Case,9> cases={{{"A_load_use",dyn_dep_load_dot},{"A_load_base",dyn_dep_load_base},{"A_dot_chain",dyn_dep_dot_chain},{"A_dot_base",dyn_dep_dot_base},{"B_full",dyn_body_full},{"B_no_bload",dyn_body_no_bload},{"B_no_aloads",dyn_body_no_aloads},{"B_b_fanout1",dyn_body_b_fanout1},{"B_no_dots",dyn_body_no_dots}}};
 if(!selected.empty()&&std::none_of(cases.begin(),cases.end(),[&](const Case& c){return selected==c.name;})){std::fprintf(stderr,"unknown case: %.*s\n",int(selected.size()),selected.data());return 2;}
 const int perf_fd=open_cycles();
 if(perf_fd<0){std::fprintf(stderr,"perf_event_open cycles:u failed: %s; use --case NAME under external perf stat\n",strerror(errno));return 3;}
 std::printf("meta counter=perf_event_open event=cycles:u pid=current_thread cpu_affinity=0 cpu_actual=%d iterations=%zu samples=%d warmups=3 timed_region=kernel_call checksum_and_output=excluded\n",sched_getcpu(),iters,samples);
 for(const auto &c:cases){
  if(!selected.empty()&&selected!=c.name)continue;
  for(int w=0;w<3;w++){dyn_init_operands();c.fn(4096,A.p,B.p,O.p);}
  std::vector<double> cpis,rates,freqs;
  for(int s=0;s<samples;s++){
   dyn_init_operands();
   asm volatile("":::"memory"); const auto wall_b=std::chrono::steady_clock::now();
   if(ioctl(perf_fd,PERF_EVENT_IOC_RESET,0)||ioctl(perf_fd,PERF_EVENT_IOC_ENABLE,0)){std::perror("perf ioctl start");close(perf_fd);return 1;}
   asm volatile("":::"memory"); c.fn(iters,A.p,B.p,O.p); asm volatile("":::"memory");
   if(ioctl(perf_fd,PERF_EVENT_IOC_DISABLE,0)){std::perror("perf ioctl stop");close(perf_fd);return 1;}
   const auto wall_e=std::chrono::steady_clock::now(); asm volatile("":::"memory");
   PerfRead r{}; if(read(perf_fd,&r,sizeof(r))!=sizeof(r)){std::perror("perf read");close(perf_fd);return 1;}
   const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(wall_e-wall_b).count();
   const double rate=r.time_enabled?100.0*double(r.time_running)/double(r.time_enabled):0.0;
   if(rate<99.0){std::fprintf(stderr,"case=%s sample=%d running_pct=%.6f below 99%%; rerun required\n",c.name,s,rate);close(perf_fd);return 4;}
   const double cpi=double(r.cycles)/double(iters); const double ghz=ns?double(r.cycles)/double(ns):0.0;
   cpis.push_back(cpi); rates.push_back(rate); freqs.push_back(ghz);
   std::printf("raw case=%s sample=%d hw_cycles=%" PRIu64 " cycles_per_iter=%.6f time_enabled=%" PRIu64 " time_running=%" PRIu64 " running_pct=%.6f duration_ns=%" PRIu64 " implied_ghz=%.6f checksum=%" PRIu64 "\n",c.name,s,r.cycles,cpi,r.time_enabled,r.time_running,rate,(std::uint64_t)ns,ghz,checksum((std::uint8_t*)O.p));
  }
  const auto [lo,hi]=std::minmax_element(cpis.begin(),cpis.end());
  std::printf("summary case=%s median_hw_cycles_per_iter=%.6f min=%.6f max=%.6f spread_pct=%.3f median_running_pct=%.6f median_implied_ghz=%.6f\n",c.name,median(cpis),*lo,*hi,100.0*(*hi-*lo)/median(cpis),median(rates),median(freqs));
 }
 close(perf_fd); return 0;
}
