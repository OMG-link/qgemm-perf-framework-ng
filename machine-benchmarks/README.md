# SpacemiT K1 L2 bandwidth benchmark

This is an independent machine benchmark. Its CMake target is
`k1-l2-bandwidth`; it does not link the GEMM framework.

The tested K1 topology is two shared-L2 clusters: CPUs 0--3 share 512 KiB and
CPUs 4--7 share 512 KiB. Each CPU has a 32 KiB L1D and the cache line is 64 B.
The default test therefore uses CPUs 0,1,2,3, 256 KiB per single-core worker,
and 96 KiB per concurrent worker. The latter totals 384 KiB and is larger than
each L1D while remaining below the shared L2 capacity.

## Build

From the repository root:

```bash
cmake --build /home/linkai/projects/triton-riscv/gemm-kernel-opt/ggml-kernel-perf/perf-framework-ng/build --target k1-l2-bandwidth
```

## Direct run

```bash
./build/k1-l2-bandwidth --mode all --cpus 0,1,2,3
```

The program pins each worker, warms its buffer, and reports the fastest of the
requested samples. The concurrent aggregate uses the slowest worker duration
as its common interval, so its byte count is not the sum of unrelated time
windows.

## perf measurement

Use `run_k1_l2_perf.py` on the build host. It copies the exact cross-built
binary to the target and runs each perf event group separately. This avoids
silently accepting PMU multiplexing. A group is valid only if every event has
non-zero `time_running` and `time_running/time_enabled` is at least 99%.

```bash
python3 machine-benchmarks/run_k1_l2_perf.py \
  --remote spacemit-k1 \
  --binary /home/linkai/projects/triton-riscv/gemm-kernel-opt/ggml-kernel-perf/perf-framework-ng/build/k1-l2-bandwidth
```

The runner records raw `perf stat -x ';' --no-scale` output and benchmark
output for every run. It measures these groups independently:

```text
cycles,instructions,load_inst,vector_load_inst
L1-dcache-loads,L1-dcache-load-misses,l2_load_access,l2_load_miss
l2_ar_channel_request,l2_ar_channel_stall_cycle,cycles
```

The K1 native `l2_*` events are preferred over generic `LLC-*` events. The
runner also performs an optional `--validate` pass with 16 KiB, 256 KiB and
16 MiB buffers. Native event counts are treated as evidence only after they
show the expected L1/L2/DRAM trend. `l2_load_access * 64` is not automatically
claimed to be physical L2 traffic.

## Experimental path-count benchmark

`k1-memory-path-bandwidth` is the fixed-workload experiment. It uses eight
independent 256-bit RVV loads per inner iteration, records each worker's TID,
CPU, byte count and time interval, and reports both common overlap and the
correct aggregate makespan. Worker count alone is not interpreted as channel
count; the report combines scaling and interference experiments. It supports:

```text
--mode l1       16 KiB ordinary-memory working set
--mode l2       256 KiB for one worker; 96 KiB/worker for concurrent tests
--mode dram     64 MiB ordinary-memory working set
--mode tcm      four 128 KiB TCM mappings, when the runtime is available
--cpus 0,1,2,3  explicit worker CPU list (any non-empty size)
--pmu execution per-thread cycles/load/L1 counters
--pmu path      per-thread prefetch/L2 counters
--load-lmul 1|8 eight 32-byte m1 loads or one 256-byte m8 load per 256 B
```

Worker count and interval overlap prove only simultaneous requesters, not
parallel channel count. The TCM mode prints the
allocator mapping VA and `tcm_va_to_pa()` result before measuring the four
workers. Its four mappings are allocated with the existing `/dev/tcm` runtime;
the program does not infer their identity from device-tree names. Invalid
physical addresses make bank identity unavailable, but do not prevent a
black-box bandwidth test of mappings returned by `/dev/tcm`. Such a test can
establish allocator capacity and a concurrent-access lower bound; it cannot
associate a mapping with a named physical bank or CPU-local path.

A sample is valid for bandwidth scaling only when every worker's `task-clock`
is at least 98% of its wall time and every hardware event has at least 99%
PMU running time. `time_enabled` is not CPU residency and is never used to
detect preemption. All worker timed regions must also have a common overlap of
at least 90% of the longest worker duration. Duration spread is reported but
is not rejected by itself: unequal durations can be a real consequence of
shared-resource arbitration.

Collect the standard scaling matrix and retain machine-readable results with:

```bash
python3 machine-benchmarks/run_memory_path_scaling.py \
  --remote spacemit-k1 \
  --binary build/machine-benchmarks/k1-memory-path-bandwidth \
  --output memory-path-scaling.json
```

The current measured results and the limitations on experimentally inferring
path count are documented in
`docs/2026-09-02-k1-memory-path-bandwidth.md`.