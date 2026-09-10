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
void m8b_exact_hot(std::size_t,const void*,const void*,void*);
void m8b_exact_reordered(std::size_t,const void*,const void*,void*);
void m8b_no_load(std::size_t,const void*,const void*,void*);
void m8b_no_unpack(std::size_t,const void*,const void*,void*);
void m8b_no_dots(std::size_t,const void*,const void*,void*);
void m8b_stream_exact(std::size_t,const void*,const void*,void*);
void m8b_stream_a_fixed_w(std::size_t,const void*,const void*,void*);
void m8b_fixed_a_stream_w(std::size_t,const void*,const void*,void*);
}
struct Aligned { void *p{}; explicit Aligned(std::size_t n){if(posix_memalign(&p,64,n))p=nullptr;} ~Aligned(){free(p);} };
struct Case {const char *name; Kernel fn; const char *definition;};
struct Counter {std::uint32_t type; std::uint64_t config; std::string_view name;};
struct PerfRead {std::uint64_t value, time_enabled, time_running;};
static std::uint64_t checksum(const std::uint8_t *p){std::uint64_t x=0; for(int i=0;i<512;i++) x=x*131+p[i]; return x;}
static double median(std::vector<double> x){std::sort(x.begin(),x.end()); return x[x.size()/2];}
static int open_counter(const Counter &counter){
 perf_event_attr a{}; a.type=counter.type; a.size=sizeof(a); a.config=counter.config;
 a.disabled=1; a.exclude_kernel=1; a.exclude_hv=1; a.pinned=1;
 a.read_format=PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
 return static_cast<int>(syscall(SYS_perf_event_open,&a,0,-1,-1,PERF_FLAG_FD_CLOEXEC));
}
static bool parse_raw_config(std::string_view text,std::uint64_t &config){
 if(text.empty()||text.front()=='+'||text.front()=='-')return false;
 errno=0; char *end=nullptr; const auto parsed=strtoull(text.data(),&end,16);
 if(errno==ERANGE||end==text.data()||end!=text.data()+text.size())return false;
 config=parsed; return true;
}
int main(int argc,char **argv){
 std::size_t iters=200000; int samples=7; std::string_view selected;
 Counter counter{PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,"cycles"};
 for(int i=1;i<argc;i++){
  const std::string_view arg(argv[i]);
  if(arg=="--iterations"&&++i<argc) iters=strtoull(argv[i],nullptr,0);
  else if(arg=="--samples"&&++i<argc) samples=atoi(argv[i]);
  else if(arg=="--case"&&++i<argc) selected=argv[i];
  else if(arg=="--counter"&&++i<argc){
   const std::string_view value(argv[i]);
   if(value=="cycles")counter={PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,"cycles"};
   else if(value=="instructions")counter={PERF_TYPE_HARDWARE,PERF_COUNT_HW_INSTRUCTIONS,"instructions"};
   else if(value.starts_with("raw:")){
    std::uint64_t config=0;
    if(!parse_raw_config(value.substr(4),config)){std::fprintf(stderr,"invalid raw counter: %s\n",argv[i]);return 2;}
    counter={PERF_TYPE_RAW,config,value};
   }else{std::fprintf(stderr,"unknown counter: %s\n",argv[i]);return 2;}
  }
  else {std::fprintf(stderr,"usage: %s [--iterations N] [--samples N] [--case NAME] [--counter cycles|instructions|raw:HEX]\n",argv[0]); return 2;}
 }
 if(samples<7||iters<200000){std::fprintf(stderr,"samples must be >=7 and iterations >=200000\n");return 2;}
 if(iters>SIZE_MAX/128){std::fprintf(stderr,"iterations too large\n");return 2;}
 cpu_set_t set; CPU_ZERO(&set); CPU_SET(0,&set); const int affinity=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
 if(affinity){std::fprintf(stderr,"affinity: %s\n",strerror(affinity));return 1;}
 const std::size_t stream_bytes=iters*128;
 Aligned A(stream_bytes),W(stream_bytes),O(512); if(!A.p||!W.p||!O.p)return 1;
 for(std::size_t i=0;i<stream_bytes;i++)((std::uint8_t*)A.p)[i]=(i*17+3)&255;
 for(std::size_t i=0;i<stream_bytes;i++)((std::uint8_t*)W.p)[i]=(i*29+7)&255; std::memset(O.p,0,512);
 const std::array<Case,8> cases={{{"exact_hot",m8b_exact_hot,"exact .LBB0_8 body; A/W wrap every 64 inners"},{"exact_reordered",m8b_exact_reordered,"exact_hot except dots are first-update-all then second-update-all"},{"no_load",m8b_no_load,"eight loads and their eight address instructions replaced one-for-one by scalar nops"},{"no_unpack",m8b_no_unpack,"16 unpack ops replaced by nops; low-nibble sources preinitialized, loaded packed bytes feed high-source dots"},{"no_dots",m8b_no_dots,"exact_hot with 16 dots replaced one-for-one by scalar nops"},{"stream_exact",m8b_stream_exact,"exact .LBB0_8 body; A/W advance 128 bytes per inner without wrap"},{"stream_a_fixed_w",m8b_stream_a_fixed_w,"exact .LBB0_8 arithmetic body; A advances 128 bytes per inner, W wraps every 64 inners"},{"fixed_a_stream_w",m8b_fixed_a_stream_w,"exact .LBB0_8 arithmetic body; A wraps every 64 inners, W advances 128 bytes per inner"}}};
 if(!selected.empty()&&std::none_of(cases.begin(),cases.end(),[&](const Case& c){return selected==c.name;})){std::fprintf(stderr,"unknown case: %.*s\n",int(selected.size()),selected.data());return 2;}
 const int perf_fd=open_counter(counter);
 if(perf_fd<0){std::fprintf(stderr,"perf_event_open %.*s failed: %s; use --case NAME under external perf stat\n",int(counter.name.size()),counter.name.data(),strerror(errno));return 3;}
 std::printf("meta counter=%.*s counter_type=%" PRIu32 " counter_config=0x%" PRIx64 " pid=current_thread cpu_affinity=0 cpu_actual=%d iterations=%zu samples=%d warmups=3 timed_region=kernel_call checksum_and_output=excluded hot_wrap_inners=64 stream_stride_bytes=128\n",int(counter.name.size()),counter.name.data(),counter.type,counter.config,sched_getcpu(),iters,samples);
 const bool is_cycles=counter.type==PERF_TYPE_HARDWARE&&counter.config==PERF_COUNT_HW_CPU_CYCLES;
 for(const auto &c:cases){
  if(!selected.empty()&&selected!=c.name)continue;
  std::printf("case_definition case=%s definition=%s\n",c.name,c.definition);
  for(int w=0;w<3;w++)c.fn(std::min<std::size_t>(iters,4096),A.p,W.p,O.p);
  std::vector<double> values_per_inner,rates,freqs;
  for(int s=0;s<samples;s++){
   asm volatile("":::"memory"); const auto wall_b=std::chrono::steady_clock::now();
   if(ioctl(perf_fd,PERF_EVENT_IOC_RESET,0)||ioctl(perf_fd,PERF_EVENT_IOC_ENABLE,0)){std::perror("perf ioctl start");close(perf_fd);return 1;}
   asm volatile("":::"memory"); c.fn(iters,A.p,W.p,O.p); asm volatile("":::"memory");
   if(ioctl(perf_fd,PERF_EVENT_IOC_DISABLE,0)){std::perror("perf ioctl stop");close(perf_fd);return 1;}
   const auto wall_e=std::chrono::steady_clock::now(); asm volatile("":::"memory");
   PerfRead r{}; if(read(perf_fd,&r,sizeof(r))!=sizeof(r)){std::perror("perf read");close(perf_fd);return 1;}
   const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(wall_e-wall_b).count();
   const double rate=r.time_enabled?100.0*double(r.time_running)/double(r.time_enabled):0.0;
   if(rate<99.0){std::fprintf(stderr,"case=%s sample=%d running_pct=%.6f below 99%%; rerun required\n",c.name,s,rate);close(perf_fd);return 4;}
   const double value_per_inner=double(r.value)/double(iters);
   values_per_inner.push_back(value_per_inner); rates.push_back(rate);
   if(is_cycles){
    const double ghz=ns?double(r.value)/double(ns):0.0; freqs.push_back(ghz);
    std::printf("raw case=%s sample=%d value=%" PRIu64 " value_per_inner=%.6f cycles_per_inner=%.6f time_enabled=%" PRIu64 " time_running=%" PRIu64 " running_pct=%.6f duration_ns=%" PRIu64 " implied_ghz=%.6f checksum=%" PRIu64 "\n",c.name,s,r.value,value_per_inner,value_per_inner,r.time_enabled,r.time_running,rate,(std::uint64_t)ns,ghz,checksum((std::uint8_t*)O.p));
   }else{
    std::printf("raw case=%s sample=%d value=%" PRIu64 " value_per_inner=%.6f time_enabled=%" PRIu64 " time_running=%" PRIu64 " running_pct=%.6f duration_ns=%" PRIu64 " checksum=%" PRIu64 "\n",c.name,s,r.value,value_per_inner,r.time_enabled,r.time_running,rate,(std::uint64_t)ns,checksum((std::uint8_t*)O.p));
   }
  }
  const auto [lo,hi]=std::minmax_element(values_per_inner.begin(),values_per_inner.end()); const double med=median(values_per_inner);
  if(is_cycles)std::printf("summary case=%s median_cycles_per_inner=%.6f min=%.6f max=%.6f spread_pct=%.3f median_running_pct=%.6f median_implied_ghz=%.6f\n",c.name,med,*lo,*hi,100.0*(*hi-*lo)/med,median(rates),median(freqs));
  else std::printf("summary case=%s median_value_per_inner=%.6f min=%.6f max=%.6f spread_pct=%.3f median_running_pct=%.6f\n",c.name,med,*lo,*hi,100.0*(*hi-*lo)/med,median(rates));
 }
 close(perf_fd); return 0;
}
