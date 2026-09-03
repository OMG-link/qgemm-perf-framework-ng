# SpacemiT K1 数据通路带宽模型

## 1. 结论摘要

本报告统一以 **B/cycle** 记录带宽。测试期间 CPU 工作在固定的
1.6 GHz，统计口径为：

```text
B/cycle = timed region 内传输的字节数 / 同一时间段的 CPU cycles
```

多核聚合 B/cycle 使用全芯片公共时间轴上的 CPU cycle，不累加各核
cycle。最终采用的数据通路模型如下：

| 数据通路 | 数量与共享关系 | 单通路持续读取带宽 | 共享单元/全芯片带宽 |
|---|---|---:|---:|
| L1D → 寄存器 | 8 核独立，每核一条 | 约 15.88 B/cycle | 8 核模型约 127.00 B/cycle |
| L2 → L1D | 8 核独立，每 4 核共享一个 L2，共 2 个 cluster | 约 5.17 B/cycle | 每 cluster 约 19.54 B/cycle；双 cluster 模型约 39.08 B/cycle |
| 主存 → L2 | 2 个 L2 cluster 可独立发起主存流量，但共享主存总带宽 | 单 cluster 约 4.73 B/cycle | 两个 cluster 合计上限约 5.15 B/cycle |
| TCM → 寄存器 | 4 个可并行黑盒 TCM 分配/读取路径 | 约 15.82 B/cycle | 4 路约 63.28 B/cycle |

主存通路的核心特征是：两个 L2 cluster 能够同时发起请求，但带宽不能
相加，最终共同受约 **5.15 B/cycle** 的全局上限约束。

## 2. 硬件共享模型

采用下列层级关系解释测试结果：

```text
                                CPU0: L1D0 → registers0
                                CPU1: L1D1 → registers1
主存 ─┬─ L2 cluster 0 ───────── CPU2: L1D2 → registers2
      │   4 核共享               CPU3: L1D3 → registers3
      │
      │                         CPU4: L1D4 → registers4
      │                         CPU5: L1D5 → registers5
      └─ L2 cluster 1 ───────── CPU6: L1D6 → registers6
          4 核共享               CPU7: L1D7 → registers7

TCM0 → CPU0 registers
TCM1 → CPU1 registers
TCM2 → CPU2 registers
TCM3 → CPU3 registers
```

共享情况：

- L1D→寄存器通路按核心独立，共 8 条；
- L2→L1D 接口按核心独立，共 8 条；
- CPU0～CPU3 共享 L2 cluster 0；
- CPU4～CPU7 共享 L2 cluster 1；
- 两个 L2 cluster 可并发产生主存请求；
- 两个 cluster 共享约 5.15 B/cycle 的主存总上限；
- TCM 黑盒分配和并发读取实验显示 4 路独立扩展。

## 3. 测试方法

### 3.1 读取 kernel

峰值测试使用一条 LMUL=m8 的 RVV load，每条读取 256 B：

```text
1 × vle8.v e8,m8，每条读取 256 B
```

`vsetvli e8,m8` 只在手写汇编函数入口执行一次，inner loop 仅包含
`vle8.v`、地址递增和分支。另保留 LMUL=m1 对照模式：每 256 B 使用
8 条 32 B load。m1 对照约为 12.94 B/cycle，说明它受 load 指令数量和
前端开销限制，不能代表 L1 数据通路峰值。

`../vle` 的原始 m8 手写循环已重新构建并复现：24 KiB L1 工作集得到
约 15.49～15.56 B/cycle。相同循环接入严格 harness 后，单核中位数为
15.88 B/cycle，与 16 B/cycle 的预期一致。

### 3.2 聚合带宽

聚合带宽使用：

```text
所有 worker 读取的总字节数
--------------------------------------
最早 worker 开始到最晚 worker 结束的时间
```

worker 时间区间的交集只用于验证是否真正并发，不作为带宽分母。

### 3.3 有效样本条件

每个 worker 在自己的线程中通过 `perf_event_open` 采集 PMU。样本必须
同时满足：

```text
requested CPU == actual CPU
task-clock / wall-time >= 98%
PMU time_running / time_enabled >= 99%
共同 timed-region overlap / 最长 worker duration >= 90%
LMUL=m1 时 vector_load_inst 与软件读取字节数匹配
checksum 正确
```

每个数据点取 5 个有效样本的中位数。

K1 的 `vector_load_inst` 事件能统计 LMUL=m1 的 `vle8.v`，但对 LMUL=m8
返回 0。因此 m8 峰值测试不把该事件的数值作为有效性条件；m8 的执行
由最终 ELF 中的手写汇编、紧贴 kernel 的 `rdcycle`、checksum 以及缓存
路径 PMU 共同验证。

### 3.4 环境和换算口径

```text
处理器：SpacemiT K1 / X60
系统：Linux 6.6.63
CPU：0～7 在线
CPU 频率：1.6 GHz
Governor：performance
PMU：每线程 perf_event_open，无事件 multiplex
```

目标机为共享机器。八个长时间 worker 会占满全部 CPU，使后台任务抢占
某个 worker。因此 8 核聚合模型由两个四核 cluster 的独立结果组成，
不采用被抢占污染的八 worker wall-clock 结果。

## 4. L1D → 寄存器

每核使用独立的 16 KiB 工作集，使数据驻留在该核 L1D 中。

| 活跃核心数 | 聚合带宽（B/cycle） | 相对单核 |
|---:|---:|---:|
| 1 | 15.88 | 1.00x |
| 4 | 63.50 | 4.00x |

四核结果为单核的 4.00 倍，说明各核 L1D→寄存器通路不共享数据带宽。
每个四核组能够达到约 63.50 B/cycle。按 8 核独立模型：

```text
2 × 63.50 ≈ 127.00 B/cycle
```

记录为：

```text
数量：8
共享：无，每核独立
单核带宽：约 15.88 B/cycle
全芯片聚合：约 127.00 B/cycle
```

## 5. L2 → L1D

### 5.1 工作集和 PMU 证据

并发测试每核使用 96 KiB：

```text
4 × 96 KiB = 384 KiB
```

该工作集大于每核 L1D，但低于一个四核 cluster 的 L2 容量，避免旧版
测试因 `4 × 256 KiB` 超出 L2 而混入主存流量。

一个代表性单核样本：

```text
软件读取字节：       805,306,368 B
l2_load_access：       12,694,714
l1d_prefetch_refill：  12,294,065
l2_load_miss：             15,362
```

换算得到：

```text
805,306,368 / 12,694,714 ≈ 63.44 B / L2 access
```

这与 64 B cache line 相符。`l1d_prefetch_refill` 与 `l2_load_access`
数量接近，说明顺序流量主要由预取器沿 L2→L1D 通路传输。

### 5.2 扩展曲线

| 活跃核心数 | 聚合带宽（B/cycle） | 相对单核 |
|---:|---:|---:|
| 1 | 5.17 | 1.00x |
| 4 | 19.54 | 3.78x |

同一 cluster 内四核达到单核的 3.78 倍，表现为接近线性扩展。

### 5.3 数量和共享关系

每个四核 cluster 的实测聚合带宽约为 19.54 B/cycle。按两个 cluster
和八个核心接口独立的模型：

```text
2 × 19.54 ≈ 39.08 B/cycle
```

记录为：

```text
数量：8 条核心侧 L2→L1D 通路
共享：每 4 核共享一个 L2，共 2 个 L2 cluster
单核带宽：约 5.17 B/cycle
单 cluster 四核聚合：约 19.54 B/cycle
双 cluster 八核模型：约 39.08 B/cycle
```

这里的“8 条”表示八个核心具有独立的 L2→L1D 接收路径，不表示一个
L2 内部一定存在四个完全独立的 tag/data-array bank 或 read port。

## 6. 主存 → L2

每 worker 使用 64 MiB 工作集。另对 16～256 MiB 工作集进行了尺寸
扫描，带宽基本稳定，排除了数据驻留 L2 的可能。

| 请求配置 | 共享关系 | 聚合带宽（B/cycle） |
|---|---|---:|
| CPU0 | cluster 0 单请求者 | 4.73 |
| CPU0、CPU4 | 两个 L2 cluster | 5.15 |

结果具有两个特征：

1. 单个 cluster 已能获得约 4.73 B/cycle；
2. 两个 cluster 同时请求时提升到约 5.15 B/cycle，但继续增加请求者不再
   提升。

采用如下模型：

```text
cluster-facing 数量：2
共享关系：两个 cluster 独立发起主存请求
全局共享瓶颈：约 5.15 B/cycle
```

两个 cluster 的主存带宽不能简单相加。该模型描述软件可见的数据流和
共享上限，不进一步区分内存控制器、NoC 端口和 DDR PHY 的内部数量。

顺序主存流量中，预取器会在 demand load 到达前填充 cache，因此
`l2_load_miss` 不能代表全部主存传输量，不得用 `l2_load_miss × 64`
估算主存带宽。

## 7. TCM → 寄存器

TCM runtime 通过 `/dev/tcm` 分配 128 KiB mapping，每个 worker 重复读取
其中 64 KiB。

| 并发 TCM mapping/worker | 聚合带宽（B/cycle） | 相对单路 |
|---:|---:|---:|
| 1 | 15.82 | 1.00x |
| 2 | 31.75 | 2.01x |
| 3 | 47.55 | 3.01x |
| 4 | 63.28 | 4.00x |

四路扩展接近线性。前四个 128 KiB mapping 分配成功，第五个分配失败。
代表性的四 worker PMU 结果满足：

```text
共同 overlap ratio：95.7%
PMU running：100%
l1d_prefetch_refill：0
L2 事件相对 4 GiB 软件读取量可忽略
```

记录为：

```text
数量：4 个可并行黑盒 TCM 分配/读取路径
单路带宽：约 15.82 B/cycle
四路聚合：约 63.28 B/cycle
共享：实测未观察到明显共享带宽瓶颈
```

当前 `tcm_va_to_pa()` 返回值不在预期 TCM 物理窗口内，所以无法通过
该接口证明 mapping N 与物理 TCM bank N 或 CPU N 的一一对应关系。

## 8. 最终通路表

| 通路 | 数量 | 单路带宽（B/cycle） | 共享情况 | 聚合/上限（B/cycle） |
|---|---:|---:|---|---:|
| L1D → 寄存器 | 8 | 15.88 | 每核独立 | 约 127.00 |
| L2 → L1D | 8 | 5.17 | 每 4 核共享一个 L2，共 2 个 cluster | 每 cluster 19.54；全芯片模型 39.08 |
| 主存 → L2 | 2 个 cluster-facing 路径 | 单 cluster 4.73 | 两 cluster 独立请求、共享下游总带宽 | 全局约 5.15 |
| TCM → 寄存器 | 4 | 15.82 | 黑盒实验近似独立 | 约 63.28 |

## 9. 复现

构建：

```bash
cmake --build \
  /home/linkai/projects/triton-riscv/gemm-kernel-opt/ggml-kernel-perf/perf-framework-ng/build \
  --target k1-memory-path-bandwidth
```

运行单项测试：

```bash
./build/machine-benchmarks/k1-memory-path-bandwidth \
  --mode l2 --cpus 0,1,2,3 --bytes 98304 --rounds 8192 --pmu execution
```

自动采集扩展曲线：

```bash
python3 machine-benchmarks/run_memory_path_scaling.py \
  --remote spacemit-k1 \
  --binary build/machine-benchmarks/k1-memory-path-bandwidth \
  --output memory-path-scaling.json
```