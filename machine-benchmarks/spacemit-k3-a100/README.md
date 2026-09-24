# SpacemiT K3 / A100 machine benchmarks

这些是独立的板级微基准，**不接入 pfng 的 CMake 工程**（`platforms/` 目前只
定义了 `spacemit-k1-a60` 和 `spacemit-k1-x60`）。用 `build.sh` 直接交叉编译
成静态可执行文件，输出到 `out/`。

实测结果记录在 [docs/boards/spacemit-k3-a100/](../../docs/boards/spacemit-k3-a100/)。

| 程序 | 作用 |
|---|---|
| `k3_probe` | 打印 A100 的绑定状态、VLEN、实测频率，校验 PMU 可用 |
| `k3_insn_throughput` | 标准 RVV 与 XSMTVDotII 指令的稳态吞吐 |
| `k3_madot_rate` | `smt.vmadot` 的依赖延迟与所需独立累加器数量 |
| `k3_madot_shape` | 单条 dot 指令的输出形状与 MAC 数量 |
| `k3_scaling` | 标量/向量/矩阵单元的多核扩展性 |
| `k3_path_bandwidth` | L1D→寄存器 / L2→L1D / 主存→L2 的分层带宽（斜率法 + 大页） |
| `k3_l1d_peak` | L1D→寄存器峰值的定向验证（零地址运算 + LMUL 扫描） |
| `k3_load_sharing` | 向量/标量 load 带宽的簇内共享结构（1/2/3/4 核 × 簇内外） |
| `k3_mem_bandwidth` | 读/写/copy 带宽的工作集与核数扫描 |
| `k3_mem_latency` | 指针追逐测各级存储延迟 |
| `k3_tcm_bandwidth` | TCM → 寄存器带宽（单次 mmap 整颗 TCM，按 384 KiB 切片给各核） |
| `k3_bandwidth_diagnose` | 标量/m1/m8 × L1/L2 × 1/4/8 核的 load 带宽对照 |
| `k3_pmu_diagnose` | PMU 事件校准（哪些 cache/stall 事件在 A100 上可用） |
| `vector_load_mt_mre` | 向量 load 簇内不扩展的最小复现（OpenMP，板上 g++ 编译） |

## 构建与运行

```bash
export CXX=/home/linkai/projects/triton-riscv/llvm-project-spacemit-x60/builders/x86-k3/llvm/bin/clang++
./build.sh k3_probe.cpp k3_insn_throughput.cpp k3_madot_rate.cpp \
           k3_madot_shape.cpp k3_scaling.cpp \
           k3_mem_bandwidth.cpp k3_mem_latency.cpp \
           k3_path_bandwidth.cpp k3_l1d_peak.cpp \
           k3_load_sharing.cpp
scp out/* spacemit-k3:/tmp/
ssh spacemit-k3 /tmp/k3_probe 8
```

大部分程序的第一个参数是要绑定的 A100 核编号（8–15，默认 8）。
`k3_mem_bandwidth` 的第一个参数是模式：`rd-m8` / `rd-m1` / `write` / `copy`。

`vector_load_mt_mre.cpp` 用 OpenMP 且不自绑定，在板上直接编译运行：

```bash
g++ -O2 -fopenmp -march=rv64gcv -o vlmre vector_load_mt_mre.cpp
echo 0 > /proc/set_ai_thread
taskset -c 8    ./vlmre 1     # 单核
taskset -c 8,9  ./vlmre 2     # 同簇 2 核
taskset -c 8,12 ./vlmre 2     # 跨簇 2 核
```

## 两个平台特性

- **A100 核必须先写 `/proc/set_ai_thread` 才能绑定。** 默认 affinity mask 是
  `0-7`（X100），`taskset -c 8` 返回 `EINVAL`。`k3_a100.h` 的
  `k3::pin_to_a100()` 封装了这一步。
- **用户态读不到 cycle CSR。** `rdcycle` 会触发非法指令异常，所有 cycle
  计数走 `perf_event_open`（`k3::CycleCounter`）。
