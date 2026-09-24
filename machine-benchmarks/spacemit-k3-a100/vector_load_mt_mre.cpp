#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

constexpr long kIters = 20000000;   // 4 loads of 1024 B each per iteration

alignas(4096) static char buf[8][(2 << 20) + 4096];        // one 2 MiB slot per thread

double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char ** argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    memset(buf, 1, sizeof buf);     // back every slot with its own pages

    double t0 = now();
#pragma omp parallel num_threads(threads)
    {
        const char * p = buf[omp_get_thread_num()];
        long n = kIters;
        asm volatile("vsetvli t0, zero, e8, m8, ta, ma\n\t"
                     "addi t1, %0, 1024\n\taddi t2, t1, 1024\n\taddi t3, t2, 1024\n"
                     "1:\n\t"
                     "vle8.v v0,  (%0)\n\t"
                     "vle8.v v8,  (t1)\n\t"
                     "vle8.v v16, (t2)\n\t"
                     "vle8.v v24, (t3)\n\t"
                     "addi %1, %1, -1\n\tbnez %1, 1b\n\t"
                     : "+r"(p), "+r"(n)
                     :
                     : "t0", "t1", "t2", "t3", "memory",
                       "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                       "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                       "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                       "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
    double sec = now() - t0;

    double bits = 4.0 * 1024 * 8 * kIters * threads;
    printf("%d thread(s), %.0f b in %.1f ms -> %.1f b/cycle @1.8GHz\n",
           threads, bits, sec * 1e3, bits / (sec * 1.8e9));
    return 0;
}
