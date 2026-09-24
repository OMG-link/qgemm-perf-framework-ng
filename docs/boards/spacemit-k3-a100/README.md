# SpacemiT K3 / A100 基础特性

本目录记录 SpacemiT K3 上 **A100 核** 的实测基础数据。同一颗 K3 上还有
8 个 X100 核，本轮测试不涉及。

- [instruction-throughput.md](instruction-throughput.md) — 指令吞吐
- [memory-bandwidth.md](memory-bandwidth.md) — 各级存储带宽与延迟

## 1. 板级拓扑

```text
Device Tree compatible : spacemit,k3-pico-itx
Linux                  : 6.18.3-generic
逻辑 CPU               : 0-15
内存                   : 31 GiB
```

| | X100 | A100 |
|---|---|---|
| 逻辑 CPU | 0-7 | 8-15 |
| `marchid` | `0x8000000058000002` | `0x8000000041000002` |
| `mimpid` | `0x33d8a600` | `0x10000000d5686200` |
| VLEN | 256 bit | **1024 bit**（`vlenb` = 128） |
| 矩阵扩展 | `XSMTVDot`（Matrix 1.0） | `XSMTVDotII`（Matrix 2.0） |
| `cpuinfo_max_freq` | 2.4 GHz | 2.0 GHz |
| 实测运行频率 | — | 1.79 GHz |
| L1I / L1D | 64 KiB / 64 KiB 每核 | 64 KiB / 64 KiB 每核 |
| L2 | 4 MiB / 4 核 | **1 MiB / 4 核** |

A100 的 8 个核分成两个 4 核簇，各带一个 1 MiB L2：

```text
                         ┌ cpu8  ┐
主存 ─┬─ A100 L2 #0 ─────┤ cpu9  │  1 MiB，4 核共享
      │   (1 MiB)        │ cpu10 │
      │                  └ cpu11 ┘
      │                  ┌ cpu12 ┐
      └─ A100 L2 #1 ─────┤ cpu13 │  1 MiB，4 核共享
          (1 MiB)        │ cpu14 │
                         └ cpu15 ┘
```

`sysfs` 的 `shared_cpu_list` 给出 `8-11` 与 `12-15`，与实测带宽的簇内竞争
一致。

## 2. 把线程绑到 A100 核上

K3 默认不允许普通进程调度到 A100 核：`init` 及其所有子进程的
`Cpus_allowed_list` 都是 `0-7`，`taskset -c 8` 直接返回 `EINVAL`。

迁移方式是向 `/proc/set_ai_thread` 写入一个值，内核随后把 **调用线程** 的
affinity mask 改成 `8-15`：

```c
int fd = open("/proc/set_ai_thread", O_WRONLY);
write(fd, "0", 1);
close(fd);
// 此后 Cpus_allowed_list 变成 8-15，可以再用 sched_setaffinity 钉到单核
```

这与 llama.cpp 中 `bind_ai_thread()` 的做法一致，见
`/home/linkai/projects/llama.cpp/ggml/src/ggml-cpu/spacemit/ime.cpp`。要点：

- 写入只影响调用线程，**每个 worker 线程都要自己写一次**；
- 写完之后才能用 `sched_setaffinity` 固定到某个具体 A100 核；
- `/proc/set_ai_thread` 的权限是 `--w--w--w-`，不需要 root。

本目录的所有基准都通过 `machine-benchmarks/spacemit-k3-a100/k3_a100.h` 里的
`k3::pin_to_a100(cpu)` 完成这一步，并在测量前后校验 `sched_getcpu()`。

## 3. 计数口径

**用户态读不到 cycle CSR。** `rdcycle` / `rdtime` / `csrr instret` 在 K3 上
都会触发非法指令异常（`scounteren` 未开放）。所有 cycle 计数都走
`perf_event_open`（`PERF_COUNT_HW_CPU_CYCLES`，`exclude_kernel=1`），并检查
`time_running / time_enabled ≥ 99%` 排除事件复用。`vlenb`（CSR `0xc22`）可以
正常读取。

A100 的 cpufreq governor 是 `userspace`，频率固定在 1.8 GHz，不会自动调频。
实测运行频率 1.786–1.795 GHz。所有带宽和吞吐都以 **cycle** 为分母，因此对
频率不敏感；只有换算成 GB/s、TOPS 时才用到 1.79 GHz。

## 4. 构建与运行

编译器必须是带 `XSMTVDotII` 支持的 K3 分支：

```bash
CXX=/home/linkai/projects/triton-riscv/llvm-project-spacemit-x60/builders/x86-k3/llvm/bin/clang++
```

这些基准不接入 pfng 的 CMake 工程（pfng 的平台 profile 目前只覆盖
`spacemit-k1-*`），而是用一个独立脚本静态编译：

```bash
cd machine-benchmarks/spacemit-k3-a100
./build.sh k3_probe.cpp k3_insn_throughput.cpp k3_madot_rate.cpp \
           k3_madot_shape.cpp k3_scaling.cpp \
           k3_mem_bandwidth.cpp k3_mem_latency.cpp \
           k3_path_bandwidth.cpp k3_l1d_peak.cpp \
           k3_load_sharing.cpp
scp out/* spacemit-k3:/tmp/
ssh spacemit-k3 /tmp/k3_probe 8
ssh spacemit-k3 /tmp/k3_insn_throughput 8 5
ssh spacemit-k3 /tmp/k3_path_bandwidth 8
ssh spacemit-k3 /tmp/k3_l1d_peak 8
ssh spacemit-k3 /tmp/k3_load_sharing
ssh spacemit-k3 /tmp/k3_mem_bandwidth rd-m8
ssh spacemit-k3 /tmp/k3_mem_latency 8
ssh spacemit-k3 /tmp/k3_scaling
```

关键编译选项：

```text
-mcpu=spacemit-a100                      # VLEN=1024 + XSMTVDotII
-mllvm -riscv-v-vector-bits-min=1024
-mllvm -riscv-v-vector-bits-max=1024
```

注意 `main` 启动时线程还在 X100 上（VLEN=256）。任何依赖 1024 bit VLEN 或
`XSMTVDotII` 的代码都必须在 `pin_to_a100()` 之后执行。

## 5. 结论摘要

| 项目 | 实测值 |
|---|---|
| 运行频率 | 1.79 GHz |
| VLEN | 1024 bit |
| 向量 ALU 数据通路 | 512 bit/cycle（`e8,m1` 的 `vadd.vi` 2 cycle） |
| `smt.vmadot` 吞吐 | 1 条/cycle，**每 2 个核共享一个矩阵单元** |
| `smt.vmadot` 单条工作量 | int8 1024 MAC；int4 2048 MAC |
| 单核 int8 矩阵峰值 | 3.67 TOPS（独占矩阵单元时） |
| 全 8 核 int8 矩阵峰值 | 14.7 TOPS（4 个矩阵单元） |
| L1D 命中延迟 / L2 / 主存 | 2 / 35 / 277–375 cycle |
| L1D → 寄存器带宽（单核峰值） | 60.2 B/cycle（108 GB/s），官方 64 B/cycle 的 94% |
| L2 → L1D 带宽（单核） | 57.3 B/cycle（103 GB/s） |
| 主存 → L2 带宽（单核） | 9.68 B/cycle（17 GB/s） |
| TCM → 寄存器（单核，读自己归属的块） | 57.63 B/cycle（103 GB/s）；零地址运算 64.00 B/cycle（115 GB/s） |
| TCM → 寄存器（2 线程异簇） | 聚合 115.25 B/cycle（206 GB/s） |
| TCM → 寄存器（4 线程同簇） | 聚合 230.48 B/cycle（413 GB/s）；零地址运算 255.95 |
| TCM → 寄存器（8 线程） | 聚合 460.93 B/cycle（825 GB/s）；零地址运算 511.86（916 GB/s） |
| TCM 块归属 | 每块只对设备树 `cpus` 指定的 2 个核满速（其他核 0.17 B/cycle，慢 330 倍） |
| 读带宽全芯片上限 | 157 B/cycle（282 GB/s），每簇 2 核活跃时取到 |
| A100 侧主存读上限 | 17.4 B/cycle（约 31 GB/s） |

三点需要注意：L1D → 寄存器与 L2 → L1D 的带宽测不出差别，L1D 只降低延迟
（2 cycle 对 35 cycle）不提供额外带宽；向量 load 的带宽在 4 核簇内共享（两簇之间严格可加），
单簇峰值出现在 2 核活跃时；要接近 64 B/cycle 必须用 LMUL = m8 或 `vl8re8.v`，LMUL ≤ m4 的
天花板只有 51.2 B/cycle。细节见 [memory-bandwidth.md](memory-bandwidth.md)。
