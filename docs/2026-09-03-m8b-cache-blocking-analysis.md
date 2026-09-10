2026.9.7

# mr=8的批量归约方案优化尝试

本文梳理基于"mr=8的批量归约方案"的 `q4_0-IME` 动态预解压和 Cache Blocking 的理论预期与实测结果。

## 实验参数

除特别说明外，完整 GEMM 数据来自同一当前二进制、同一目标机 CPU 0 和同一 shape：

```text
M = 480, N = 1536, K = 1536
warmup = 3
samples = 7
iterations/sample = 3
PMU running = 100%
```

机器相关实测上限：

| 数据通路 | 单核持续读取带宽 |
|---|---:|
| L1D → registers | 15.88 B/cycle |
| L2 → L1D | 5.17 B/cycle |
| DRAM → L2，单 cluster | 4.73 B/cycle |
| DRAM → L2，全芯片共享上限 | 5.15 B/cycle |

下文的“显式 payload”按源码中的 load/store 字节数计算，不包括 cache-line overfetch、write allocate、scale、reduction scratch 和输出流量。

## 1. 动态解压：为什么不需要为持续带宽分块

> **本节结论：**基础 m8b 的连续输入已经能被硬件预取有效覆盖，不存在需要用分块解决的持续带宽瓶颈，因此动态解压不分块也能取得较好的性能。

### 1.1 一个 K16 inner 做什么

一个 `8×16×16` inner 完成 2,048 个 FMA：

| 操作 | 数据或指令数 |
|---|---:|
| packed W load | `4 × 32 B = 128 B` |
| A load | `4 × 32 B = 128 B` |
| Q4 低 nibble | 4 × `vand.vi` |
| Q4 高 nibble | 4 × `vsrl.vi` |
| zero-point 修正 | 8 × `vadd.vi` |
| 点积 | 16 × `smt.vmadot` |

X60 实测稳定吞吐对应的第一阶串行预算为：

| 操作 | 指令数 | cycles/inst | 周期预算 |
|---|---:|---:|---:|
| W loads | 4 | 2 | 8 |
| A loads | 4 | 2 | 8 |
| `vand.vi` | 4 | 2 | 8 |
| `vsrl.vi` | 4 | 2 | 8 |
| `vadd.vi` | 8 | 1 | 8 |
| `smt.vmadot` | 16 | 1 | 16 |
| **总计** |  |  | **56 cycles/inner** |

56 cycles 是按各组稳定吞吐直接相加的串行资源模型。不同指令可能部分重叠，因此它不是严格硬件下限，但适合估算该指令流的数据供给节奏。

### 1.2 理论带宽需求

每个 inner 的显式输入 payload 为：

```text
128 B packed W + 128 B A = 256 B
```

对应需求：

```text
256 B / 56 cycles = 4.57 B/cycle
4.57 / 5.17 = 88.4% of measured single-core L2 bandwidth
```

因此，动态解压版本接近但没有超过 L2→L1D 的持续流带宽。Q4 解压的 24-cycle 静态预算给硬件预取留下了搬运下一批数据的时间。

### 1.3 实测 L1D miss

当前同一二进制在代表 shape 上测得：

| 情况 | L1D misses/K16 inner |
|---|---:|
| 无硬件预取的理论基线 | 4 |
| 实测动态解压 m8b | **0.514325** |

每个 inner 读取 128 B packed W 和 128 B A，共覆盖 4 条新的 64 B cache line。`0.514325 miss/inner` 是整个 kernel 的总 missed-load event，除 inner A/W 主数据外还包含 scale、component 和 accumulator 等访问，因此不能直接用它计算四条输入 line 的预取覆盖率。

对各 load 的 PC 归因得到：

| 来源 | L1D misses/K16 inner | 占全部 miss |
|---|---:|---:|
| W main loads | 0.04516 | 8.79% |
| A main loads | 0.14995 | 29.20% |
| W scale loads | 0.03345 | 6.51% |
| A scale loads | 0.05368 | 10.52% |
| Component loads | 0.16412 | 31.96% |
| Accumulator loads | 0.04563 | 8.88% |
| Output copy / kernel 未 probe | 约 0.0083 | 1.62% |
| Kernel 外 setup/reset/memset | 约 0.0082 | 1.59% |

归因结果合计约 `0.50849 miss/inner`，与未插桩测得的 `0.514325 miss/inner` 接近。A/W main loads 合计为 `0.19511 miss/inner`，约占全部 miss 的 38%；原“其它”部分主要来自 component loads（31.96%）和 accumulator loads（8.88%），output copy、未 probe 区域及 kernel 外 setup/reset/memset 仅占较小部分。

A/W main 在其它尺寸下的长稳态测试结果如下：

| M | N | K | W main miss/inner | A main miss/inner |
|---:|---:|---:|---:|---:|
| 480 | 1536 | 256 | 0.247472 | 0.168362 |
| 480 | 1536 | 768 | 0.105055 | 0.316308 |
| 480 | 1536 | 1024 | 0.075244 | 0.296834 |
| 480 | 1536 | 1536 | 0.044517 | 0.150767 |
| 480 | 1536 | 4096 | 0.020084 | 0.038843 |
| 480 | 1536 | 8192 | 0.011071 | 0.021376 |
| 480 | 1536 | 8960 | 0.009442 | 0.021369 |
| 480 | 1536 | 11008 | 0.009119 | 0.015119 |

`K=1536` 下 A/W main miss 的 `3.387` 比例是该特定规模 cache 状态下的表现，不代表核心循环存在普遍、固定的 A/W miss 不对称：`K=256` 时 W miss 反而高于 A，而在较大的真实矩阵尺寸下两者都进入低 miss 区，差距也明显缩小。跨尺寸看，A/W 的绝对 miss 会随 working set 非单调波动，但硬件预取总体上同时降低了两条顺序数据流的 miss 率；矩阵尺寸主要改变未被覆盖的少量残余 refill，而不是改变硬件预取对 A/W main load 的基本覆盖能力。代表 shape 中 A/W main 合计仅 `0.19511 miss/inner`，即约 `95.1%` 的潜在 main-load demand miss 未形成实际 load refill。

## 2. 动态预解压：为什么变慢

> **本节结论：**动态预解压虽然省去了 Q4 解压计算，却使 W 的读取量翻倍，所需的 L2→L1D 带宽超过硬件上限；增加的访存等待超过了省下的计算时间，因此整体性能反而下降。

### 2.1 算术换流量

未分块 DynPreUnpack 在每次 GEMM 开始时将完整 Q4_0 W 展开成 int8。compute inner 不再执行 mask、shift 和 zero-point add，但需要读取两倍大小的 W：

| 每个 compute inner | 动态解压 m8b | 未分块 DynPreUnpack | 变化 |
|---|---:|---:|---:|
| A load | 128 B | 128 B | 0 |
| W load | 128 B packed | 256 B unpacked | +100% |
| compute payload | 256 B | 384 B | +50% |
| inner 解压 ALU | 24-cycle | 0 | -24 cycles |
| `smt.vmadot` | 16 | 16 | 0 |

### 2.2 总流量和 L2 压力

| 显式 payload/GEMM | 动态解压 m8b | 未分块 DynPreUnpack |
|---|---:|---:|
| A compute reads | 67.5 MiB | 67.5 MiB |
| W compute reads | 67.5 MiB | 135 MiB |
| unpack packed-W reads | 0 | 1.125 MiB |
| unpack int8-W writes | 0 | 2.25 MiB |
| **合计** | **135 MiB** | **205.875 MiB** |
| **相对动态解压** | 1.000× | **1.525×** |

动态预解压删除 Q4 解压指令，但 W loads 从 4 条增加到 8 条。每个 compute inner 的理论周期从 56 降为 40，而输入 payload 增加到 384 B。对应的 L2→L1D 带宽压力为：

| 版本 | compute payload/inner | 理论 cycles/inner | L2→L1D 带宽压力 | 相对 5.17 B/cycle 上限 |
|---|---:|---:|---:|---:|
| 动态解压 | 256 B | 56 | 4.57 B/cycle | 88.4% |
| 动态预解压 | 384 B | 40 | **9.60 B/cycle** | **185.7%** |

动态解压的理论数据需求仍低于 L2→L1D 持续带宽；动态预解压则需要 9.60 B/cycle，是实测上限的 1.86 倍。删除解压指令后缩短的理论周期无法容纳翻倍后的 W 数据流，compute 阶段转为受 L2→L1D 数据供给限制。

### 2.3 端到端结果

| kernel | median cycles | FMA/cycle | 相对基础 m8b |
|---|---:|---:|---:|
| m8b，动态解压 | 125,360,000 | 9.0337 | baseline |
| m8b，未分块 DynPreUnpack | 155,440,947 | 7.2855 | **+24.00% cycles** |

## 3. Cache Blocking：理论预期与实测结果

> **本节结论：**Cache Blocking 理论上可以通过复用 W 解除 L2→L1D 带宽瓶颈。实测中，它比未分块 DynPreUnpack 少 13.71% cycles，但仍比基础 m8b 多 6.99% cycles，因此理论收益只兑现了一部分。

### 3.1 分块参数的选择

微内核尺寸为 `MR = 8`、`NR = 16`。动态预解压后，每个 inner 读取 256 B W 和 128 B A，因此优先在 L1D 中复用 W。

给 `KC × NR` 的 W micro-panel 分配 8 KiB L1D 预算。W 为 `int8_t`，所以：

```text
KC = 8 KiB / (NR × sizeof(int8_t))
   = 8,192 B / 16 B
   = 512
```

每个量化 block 覆盖 32 个 K 元素，`KC = 512` 对应：

```text
KC / 32 = 16 blocks
```

一个 `MR × KC` 的 A micro-panel 包含 16 个 `block_q8_0_ime_m8`：

```text
16 × sizeof(block_q8_0_ime_m8)
= 16 × 288 B
= 4,608 B
```

给 `MC × KC` 的 A block 分配 128 KiB L2 预算。一个 A block 可容纳 28 个完整的 A micro-panels，因此：

```text
MC / MR = floor(128 KiB / 4,608 B) = 28
MC = 28 × MR = 224
```

实际分块参数为：

```text
MR = 8
NR = 16
KC = 512
MC = 224
W micro-panel (KC × NR) = 8 KiB
A block (MC × KC) = 28 × 4,608 B = 129,024 B
```

实际循环顺序为：

```text
for k0 in 0..K step KC:                 # K block
    for m0 in 0..M step MC:             # A block: MC × KC
        for n0 in 0..N step NR:         # W micro-panel: KC × NR
            for m1 in m0..min(m0+MC,M) step MR:
                compute A[m1:m1+MR, k0:k0+KC]
                      × W[k0:k0+KC, n0:n0+NR]
```

因此，在一个 `MC × KC` 的 A block 内，同一个 `KC × NR` W micro-panel 会被连续用于最多 `MC / MR = 28` 个 A micro-panels；切换到下一个 A block 后，再从头使用同一个 W micro-panel。

### 3.2 分块方案的预期

动态预解压的 compute 每个 inner 要求 L2→L1D 提供 128 B A 和 256 B W，理论带宽压力为：

```text
(128 B + 256 B) / 40 cycles = 9.60 B/cycle
```

这超过单核 L2→L1D 持续带宽 `5.17 B/cycle`。

整个 M 维包含 `M / MR` 个 A micro-panels，每个 A block 包含 `MC / MR` 个 A micro-panels。同一个 W micro-panel 从 L2 载入 L1D 的次数因此从 `M / MR` 降为：

```text
ceil(M / MC)
```

以 `M = 480`、`MR = 8`、`MC = 224` 为例，共有 60 个 A micro-panels，分成 `28 + 28 + 4` 三个 A blocks。每个 inner 摊销的 W 流量为：

```text
256 B × 3 / 60 = 12.8 B
```

分块后的理论 L2→L1D 带宽压力为：

```text
(128 B + 12.8 B) / 40 cycles = 3.52 B/cycle
```

`3.52 B/cycle` 低于 `5.17 B/cycle`。因此分块的预期是通过在 L1D 中复用 W micro-panel，解除动态预解压造成的 L2→L1D 带宽瓶颈。

### 3.3 端到端结果

| kernel | median cycles | FMA/cycle | 相对基础 m8b |
|---|---:|---:|---:|
| 动态解压 m8b | 125,360,000 | 9.0337 | baseline |
| 未分块 DynPreUnpack | 155,440,947 | 7.2855 | **+24.00% cycles** |
| DynPreUnpack + Cache Blocking | 134,128,047 | 8.4431 | **+6.99% cycles** |

Cache Blocking 比未分块 DynPreUnpack 少 13.71% cycles，说明分块复用产生了端到端收益；但它仍比基础 m8b 多 6.99% cycles，尚未完全兑现理论模型预测的收益。

### 3.4 开销测试结果

| 组分 | m8b | m8b-DPU | m8b-DPU-CB |
|---|---:|---:|---:|
| 动态预解压 | 0.000M cycles | 8.290M cycles | 7.667M cycles |
| Inner loop | 45.596M cycles | 51.649M cycles | 44.320M cycles |
| Inner loop output writeback | 37.746M cycles | 42.144M cycles | 37.359M cycles |
| Batch reduction | 33.369M cycles | 37.616M cycles | 37.423M cycles |
| 最终 C writeback | 6.387M cycles | 6.745M cycles | 计入 batch reduction |
| Unpack C | 0.000M cycles | 0.000M cycles | 2.284M cycles |
| 其他执行与控制开销 | 2.255M cycles | 8.808M cycles | 4.957M cycles |
| **端到端实测** | **125.360M cycles** | **155.441M cycles** | **134.128M cycles** |

### 3.5 m8b-DPU-CB 相对 m8b 的变化分析

m8b-DPU-CB 的端到端 cycles 比 m8b 多：

```text
134.128M - 125.360M = 8.768M cycles（+6.99%）
```

差异首先来自 Inner loop 没有兑现预期中的明显收益。代表 shape 每次 kernel pass 执行 `552,960` 个 K16 inner，因此：

```text
m8b Inner loop：
45.596M / 552,960 = 82.458 cycles/inner

m8b-DPU-CB Inner loop：
44.320M / 552,960 = 80.152 cycles/inner

实际改善：
82.458 - 80.152 = 2.306 cycles/inner（2.80%）
```

预解包移除了 compute inner 中的 Q4 解包指令，理论执行预算由约 `56 cycles/inner` 降至约 `40 cycles/inner`，理论收益为 `16 cycles/inner`。但在数据命中 L1D 的受控测试中，m8b 和 m8b-DPU-CB 的 exact body 分别为 `56.667` 和 `47.002 cycles/inner`，实际兑现的纯执行收益只有：

```text
56.667 - 47.002 = 9.665 cycles/inner
```

生产 Inner loop 的 miss 测试结果为：

| kernel | L1D miss/inner | L2 miss/inner |
|---|---:|---:|
| m8b | 0.2214 | 0.0097 |
| m8b-DPU-CB | 0.2317 | 0.0376 |
| 变化 | **+0.0103** | **+0.0279** |

m8b-DPU-CB 的 L1D miss 总量只略有上升，但 L2 miss 增至 m8b 的约 `3.9` 倍。这说明 Cache Blocking 降低 L2→L1D 总体带宽压力后，复杂的 A/W 访问与复用模式仍造成了状态敏感的 cache 命中层级恶化，使一部分访问从 L2 hit 落入高延迟的 L2 miss/DRAM 路径。

独立 pointer-chase 测得 L1 miss且L2 hit的增量 penalty 约为 `35.36 cycles`，L2 miss相对L2 hit还需增加约 `390.77 cycles`。据此估算，m8b-DPU-CB 相对 m8b 的 miss变化增加：

```text
L1D miss增量：
0.0103 × 35.36 ≈ 0.36 cycles/inner

L2 miss增量：
0.0279 × 390.77 ≈ 10.90 cycles/inner

合计：
0.36 + 10.90 ≈ 11.26 cycles/inner
```

该估算假设 miss完全串行，生产执行中实际存在一定重叠，因此不作为精确的逐周期分解；但它说明了额外开销的量级。约 `11.26 cycles/inner` 的 miss增量代价超过了热L1测试中实际兑现的 `9.67 cycles/inner` 预解包收益，其中约 `97%` 来自 L2 miss。最终，生产 Inner loop 只改善 `2.306 cycles/inner`，没有形成足以主导端到端性能的收益。

在 Inner loop 收益有限的同时，m8b-DPU-CB 还承担了 m8b 不存在或更低的额外阶段成本：

| 额外或增加的阶段 | m8b-DPU-CB 相对 m8b |
|---|---:|
| 动态预解压 | +7.667M cycles |
| Unpack C | +2.284M cycles |
| Batch reduction | +4.054M cycles |
| 其他执行与控制开销 | +2.702M cycles |

部分组分的统计边界不同，例如 m8b-DPU-CB 的最终 C writeback 计入 Batch reduction，因此这些差值不直接相加作为端到端差值。端到端结果仍以实测的 `+8.768M cycles` 为准。

综上，m8b-DPU-CB 未能超过 m8b 的原因是：**Inner loop 的实际收益只有 `2.306 cycles/inner`，远低于预解包的理论收益；这点有限收益不足以覆盖动态预解压、Unpack C、较高的 Batch reduction 及其他额外开销。**

## 4. 结论

| 阶段 | 做了什么 | 理论预期 | 实测结果 |
|---|---|---|---:|
| 动态解压 m8b | baseline | baseline | 125.360M cycles |
| 未分块 DynPreUnpack | 提前展开完整 W | 删除解压，但 L2→L1D 压力升至 9.60 B/cycle | 155.441M，+24.00% |
| Cache Blocking | 在多个 A micro-panels 间复用 8 KiB W micro-panel | 理论带宽压力降至 3.52 B/cycle | 134.128M，+6.99% |

最终结论是：

1. **基础 m8b 的顺序数据流大部分已被硬件预取覆盖。**总计 `0.514325 miss/inner` 中，inner A/W 主数据约占 `0.179 miss/inner`；相对 4 misses/inner 的无预取基线，约 96% 的潜在 main-load demand miss 已被避免。
2. **动态预解压是算术换流量。**删除解压后 W loads 从 4 条增加到 8 条，compute payload 增至 384 B，所需 L2→L1D 带宽达到 9.60 B/cycle；动态预解压本身约占 8.290M cycles，未分块 DPU 端到端比基础 m8b 多 24.00% cycles。
3. **Cache Blocking 降低了带宽压力，但 Inner loop 没有形成明显收益。**在多个 A micro-panels 间复用 W micro-panel，将理论带宽压力从 9.60 降至 3.52 B/cycle；但状态敏感的 cache 命中层级恶化基本抵消了预解包实际兑现的执行收益，Inner loop 仅改善 2.306 cycles/inner。有限的 Inner loop 收益不足以覆盖动态预解压、Unpack C、较高的 Batch reduction 及其他额外开销，因此 m8b-DPU-CB 虽比未分块 DPU 少 13.71% cycles，端到端仍比基础 m8b 多 6.99% cycles。
