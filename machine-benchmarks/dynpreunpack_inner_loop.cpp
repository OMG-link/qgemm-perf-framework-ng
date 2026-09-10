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

#include <riscv_vector.h>
#include <smt_vector.h>

using Kernel = void (*)(std::size_t, const void *, const void *, void *);
extern "C" {
void dyn_init_operands();
void dyn_body_full(std::size_t,const void*,const void*,void*); void dyn_body_no_bload(std::size_t,const void*,const void*,void*);
void dyn_body_no_aloads(std::size_t,const void*,const void*,void*); void dyn_body_no_dots(std::size_t,const void*,const void*,void*);
void dyn_body_b_fanout1(std::size_t,const void*,const void*,void*); void dyn_dep_load_dot(std::size_t,const void*,const void*,void*);
void dyn_dep_load_base(std::size_t,const void*,const void*,void*); void dyn_dep_dot_chain(std::size_t,const void*,const void*,void*);
void dyn_dep_dot_base(std::size_t,const void*,const void*,void*);
void dyn_body_intrinsic_reordered_fixed_addr(std::size_t, const void *, const void *, void *);
void dyn_body_prod_two_inner_hot_l1(std::size_t, const void *, const void *, void *);
}

extern "C" [[gnu::noinline]] void dyn_body_full_intrinsic(std::size_t iterations, const void *a_data, const void *b_data, void *out) {
    const auto *A = static_cast<const std::int8_t *>(a_data);
    const auto *B = static_cast<const std::int8_t *>(b_data);
    auto *O = static_cast<std::int32_t *>(out);
    vint32m2_t acc0 = __riscv_vmv_v_x_i32m2(0, 16), acc1 = __riscv_vmv_v_x_i32m2(0, 16);
    vint32m2_t acc2 = __riscv_vmv_v_x_i32m2(0, 16), acc3 = __riscv_vmv_v_x_i32m2(0, 16);
    vint32m2_t acc4 = __riscv_vmv_v_x_i32m2(0, 16), acc5 = __riscv_vmv_v_x_i32m2(0, 16);
    vint32m2_t acc6 = __riscv_vmv_v_x_i32m2(0, 16), acc7 = __riscv_vmv_v_x_i32m2(0, 16);
#pragma clang loop unroll(disable)
    for (std::size_t i = 0; i < iterations; ++i) {
        const std::size_t index = i & 63;
        const auto *ap = A + index * 128;
        const auto *bp = B + index * 256;
        const vint8m8_t b = __riscv_vle8_v_i8m8(bp, __riscv_vsetvlmax_e8m8());
        const vint8m1_t b0 = __riscv_vget_v_i8m8_i8m1(b, 0), b1 = __riscv_vget_v_i8m8_i8m1(b, 1);
        const vint8m1_t b2 = __riscv_vget_v_i8m8_i8m1(b, 2), b3 = __riscv_vget_v_i8m8_i8m1(b, 3);
        const vint8m1_t b4 = __riscv_vget_v_i8m8_i8m1(b, 4), b5 = __riscv_vget_v_i8m8_i8m1(b, 5);
        const vint8m1_t b6 = __riscv_vget_v_i8m8_i8m1(b, 6), b7 = __riscv_vget_v_i8m8_i8m1(b, 7);
        const std::size_t vl = __riscv_vsetvlmax_e8m1();
        const vint8m1_t a0 = __riscv_vle8_v_i8m1(ap, vl), a1 = __riscv_vle8_v_i8m1(ap + 32, vl);
        const vint8m1_t a2 = __riscv_vle8_v_i8m1(ap + 64, vl), a3 = __riscv_vle8_v_i8m1(ap + 96, vl);
        asm volatile("" ::"vr"(b0), "vr"(b1), "vr"(b2), "vr"(b3), "vr"(b4), "vr"(b5), "vr"(b6), "vr"(b7), "vr"(a0), "vr"(a1), "vr"(a2), "vr"(a3));
        acc0 = __riscv_smt_vmadot_i32m2(acc0, a0, b0, 3, 0);
        acc1 = __riscv_smt_vmadot_i32m2(acc1, a0, b1, 3, 0);
        acc2 = __riscv_smt_vmadot_i32m2(acc2, a0, b2, 3, 0);
        acc3 = __riscv_smt_vmadot_i32m2(acc3, a0, b3, 3, 0);
        acc4 = __riscv_smt_vmadot_i32m2(acc4, a2, b0, 3, 0);
        acc5 = __riscv_smt_vmadot_i32m2(acc5, a2, b1, 3, 0);
        acc6 = __riscv_smt_vmadot_i32m2(acc6, a2, b2, 3, 0);
        acc7 = __riscv_smt_vmadot_i32m2(acc7, a2, b3, 3, 0);
        acc0 = __riscv_smt_vmadot_i32m2(acc0, a1, b4, 3, 0);
        acc1 = __riscv_smt_vmadot_i32m2(acc1, a1, b5, 3, 0);
        acc2 = __riscv_smt_vmadot_i32m2(acc2, a1, b6, 3, 0);
        acc3 = __riscv_smt_vmadot_i32m2(acc3, a1, b7, 3, 0);
        acc4 = __riscv_smt_vmadot_i32m2(acc4, a3, b4, 3, 0);
        acc5 = __riscv_smt_vmadot_i32m2(acc5, a3, b5, 3, 0);
        acc6 = __riscv_smt_vmadot_i32m2(acc6, a3, b6, 3, 0);
        acc7 = __riscv_smt_vmadot_i32m2(acc7, a3, b7, 3, 0);
    }
    __riscv_vse32_v_i32m2(O + 0, acc0, 16);
    __riscv_vse32_v_i32m2(O + 16, acc1, 16);
    __riscv_vse32_v_i32m2(O + 32, acc2, 16);
    __riscv_vse32_v_i32m2(O + 48, acc3, 16);
    __riscv_vse32_v_i32m2(O + 64, acc4, 16);
    __riscv_vse32_v_i32m2(O + 80, acc5, 16);
    __riscv_vse32_v_i32m2(O + 96, acc6, 16);
    __riscv_vse32_v_i32m2(O + 112, acc7, 16);
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
 const std::array<Case, 12> cases = {{{"A_load_use", dyn_dep_load_dot},
                                      {"A_load_base", dyn_dep_load_base},
                                      {"A_dot_chain", dyn_dep_dot_chain},
                                      {"A_dot_base", dyn_dep_dot_base},
                                      {"B_full", dyn_body_full},
                                      {"B_full_intrinsic", dyn_body_full_intrinsic},
                                      {"B_intrinsic_reordered_fixed_addr", dyn_body_intrinsic_reordered_fixed_addr},
                                      {"B_prod_two_inner_hot_l1", dyn_body_prod_two_inner_hot_l1},
                                      {"B_no_bload", dyn_body_no_bload},
                                      {"B_no_aloads", dyn_body_no_aloads},
                                      {"B_b_fanout1", dyn_body_b_fanout1},
                                      {"B_no_dots", dyn_body_no_dots}}};
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
