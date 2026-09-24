# SpacemiT K3 / A100 指令吞吐

口径与 [docs/x60-vector-instruction-throughput.md](../../x60-vector-instruction-throughput.md)
一致：只记录收敛到明确硬件速率的稳态吞吐。测量条件为 VLEN = **1024 bit**、
`spacemit-k3` 的 cpu8、寄存器互不依赖、访存用例的数据常驻 L1D、展开长度
足够长使循环和前端开销可忽略。`Cycles/inst` 是吞吐，不是依赖延迟。

每个用例都用两个展开长度（每个块重复 4 次和 16 次）各测 5 遍取最小值，
两者相对偏差记为 spread。表格按助记符排序，同一助记符再按向量配置排序。

基准程序：`machine-benchmarks/spacemit-k3-a100/k3_insn_throughput.cpp`。

## 1. 标准 RVV

A100 的向量数据通路是 **512 bit/cycle**：`e8,m1`（1024 bit）的 ALU 操作是
2 cycle，`e8,m8`（8192 bit）是 16 cycle，比例严格线性。

| Mnemonic | Vector configuration | Cycles/inst | Inst/cycle |
|---|---|---:|---:|
| `vadd.vi` | `e8,m1` | 2 | 0.5 |
| `vadd.vv` | `e8,m8` | 16 | 0.0625 |
| `vand.vi` | `e8,m1` | 2 | 0.5 |
| `vfcvt.f.x.v` | `e32,m2` | 4 | 0.25 |
| `vfmacc.vv` | `e32,m1` | 2 | 0.5 |
| `vfmacc.vv` | `e32,m2` | 4 | 0.25 |
| `vfmul.vf` | `e32,m2` | 4 | 0.25 |
| `vfwcvt.f.f.v` | `e16,m1` to `e32,m2` | 4 | 0.25 |
| `vl1re16.v` | 一个 1024-bit 寄存器 | 2.48 | 0.404 |
| `vl2re32.v` | 两个寄存器 / 2048 bit | 4.95 | 0.202 |
| `vl8re8.v` | 八个寄存器 / 8192 bit | 17.09 | 0.0585 |
| `vle8.v` | `e8,m1` | 2.48 | 0.404 |
| `vle8.v` | `e8,m2` | 4.95 | 0.202 |
| `vle8.v` | `e8,m4` | 10.15 | 0.0985 |
| `vle8.v` | `e8,m8` | 17.01 | 0.0588 |
| `vmv2r.v` | 两个寄存器 / 2048 bit | 4 | 0.25 |
| `vse32.v`, masked | `e32,m1`；固定非平凡掩码 | 5 | 0.2 |

`vsrl.vi e8,m1` 与 `vand.vi e8,m1` 完全相同（2 cycle），不单独列行。

**load 指令的吞吐对地址分布极其敏感，上表的数值是在 24 个互不相同的
基址寄存器、循环体内零地址运算的条件下取到的最小值。** 换成常见写法会显著
变慢，不是硬件吞吐：

| 条件 | `vle8.v e8,m8` cycles/inst |
|---|---:|
| 24 个不同地址，循环内零地址运算 | 17.01 |
| 16 个不同地址 | 17.56 |
| 4 个不同地址（反复读同一块 4 KiB） | 19.77 |
| 每条 load 跟一条 `addi` 推进指针 | 21.44 |
| 8 条指令只用 2 个地址 | 22.10 |
| 8 条指令全部访问同一地址 | 39.4 |

换算成字节：`e8,m1` 是 128 B / 2.48 cycle = **51.7 B/cycle**，
`e8,m8` 是 1024 B / 17.01 cycle = **60.2 B/cycle**。m1/m2/m4 一律是每
4 个 cache line 5 个 cycle（51.2 B/cycle），只有 m8 跳出这个规律、能接近
官方宣称的 64 B/cycle 数据通路上限。完整分析见
[memory-bandwidth.md](memory-bandwidth.md) 第 3 节。

## 2. XSMTVDotII 矩阵指令

**所有 dot 形式的吞吐都是 1 条/cycle，与数据类型无关。** 编译器单元测试
`llvm/test/MC/RISCV/xsmtvdotii-valid.s` 与 `xsmtvdotii-invalid.s` 给出
`XSMTVDotII` 支持的全部类型：整数形式只接受 `i4` / `i8`（`i8` 是默认），
浮点形式只接受 `fp16` / `bfp16`。

| Mnemonic | Type | Vector configuration | Cycles/inst | Inst/cycle |
|---|---|---|---:|---:|
| `smt.vmadot` | `i4` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadot` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotu` | `i4` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotu` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotsu` | `i4` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotsu` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotus` | `i4` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadotus` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadot1` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadot2` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadot3` | `i8` | `e8,m1`；M2 累加器 | 1 | 1 |
| `smt.vmadot.hp` | `i4` | `e8,m1`；掩码 v0；imm 2 | 1 | 1 |
| `smt.vmadot.hp` | `i8` | `e8,m1`；掩码 v0；imm 2 | 1 | 1 |
| `smt.vmadot.sp` | `i4` | `e8,m1`；掩码 v0；imm 1 | 1 | 1 |
| `smt.vmadot.sp` | `i8` | `e8,m1`；掩码 v0；imm 1 | 1 | 1 |
| `smt.vfwmadot` | `fp16` | `e16,m1` | 1 | 1 |
| `smt.vfwmadot1` | `fp16` | `e16,m1` | 1 | 1 |
| `smt.vfwmadot2` | `fp16` | `e16,m1` | 1 | 1 |
| `smt.vfwmadot3` | `fp16` | `e16,m1` | 1 | 1 |

`smt.vmadot1/2/3` 是 slide 形式，汇编器只接受 `i8`（`i4` 报
`smt.vmadot with slide only supports i8 type`），因此没有 `i4` 行。

**`bfp16` 汇编器接受但硬件不实现。** `smt.vfwmadot v2, v4, v6, bfp16` 在
A100 上触发 SIGILL，`smt.vfwmadot1/2/3` 的 `bfp16` 形式同样。这是本轮测试
中唯一一个编译器与硬件不一致的编码。

### 2.1 依赖延迟与独立累加器数量

`smt.vmadot` 的依赖延迟是 **8 cycle**，远大于 1 cycle 的吞吐，所以循环体里
必须放足够多的独立累加器才能测到吞吐。定向扫描（`k3_madot_rate.cpp`）：

| 独立累加器个数 | Cycles/vmadot |
|---:|---:|
| 1 | 8.0 |
| 2 | 4.0 |
| 4 | 2.0 |
| 8 | 1.111 |
| 12 | 1.0 |

1/2/4 个累加器时结果就是 `8 / 累加器数`，纯粹由延迟决定。主表统一用 13 个
独立累加器（`v2`…`v26`，源固定为 `v28`/`v30`，掩码留给 `v0`）。

只改累加器、源寄存器全部相同的变体给出同样的 1.111（8 累加器），说明限制
在累加器侧，不在源读端口。

### 2.2 矩阵单元由核对共享

矩阵单元 **不是每核一个**：偶奇成对的两个 hart 共享一个。多线程扫描
（`k3_scaling.cpp`，每线程 13 个独立累加器）：

| 线程绑定 | 每线程 cycles/vmadot | 聚合 vmadot/cycle |
|---|---:|---:|
| 8 | 1.0 | 1.0 |
| 8, 9 | 2.0 | 1.0 |
| 8, 10 | 1.0 | 2.0 |
| 8, 11 | 1.0 | 2.0 |
| 9, 10 | 1.0 | 2.0 |
| 8, 12 | 1.0 | 2.0 |
| 8, 9, 10, 11 | 2.0 | 2.0 |
| 8, 10, 12, 14 | 1.0 | 4.0 |
| 8–15（全部） | 2.0 | 4.0 |

只有 `{8,9}` 这种同一对内的组合会互相减半，`{8,10}`、`{8,11}`、`{9,10}`
都能线性扩展。因此 8 个 A100 核共 **4 个矩阵单元**，分组为
`(8,9) (10,11) (12,13) (14,15)`。

标量与向量单元没有这个限制，同样的扫描下 `addi`、`vadd.vi e8,m1`、
`vfmacc.vv e32,m1` 从 1 核到 8 核都是严格 8.00 倍线性扩展。

### 2.3 单条指令的实际计算量

用全 1 输入、累加器清零、执行一条指令后回写累加器测得
（`k3_madot_shape.cpp`）。VLEN = 1024 时 M2 累加器是 256 B。

| Mnemonic | Type | 输出元素 | 归约深度 K | MAC/inst |
|---|---|---:|---:|---:|
| `smt.vmadot` | `i8` | 64 × int32 | 16 | 1024 |
| `smt.vmadot` | `i4` | 64 × int32 | 32 | 2048 |
| `smt.vmadot1` | `i8` | 64 × int32 | 16 | 1024 |
| `smt.vmadot2` | `i8` | 64 × int32 | 16 | 1024 |
| `smt.vmadot3` | `i8` | 64 × int32 | 16 | 1024 |
| `smt.vfwmadot` | `fp16` | 64 × fp32 | 8 | 512 |

即 `smt.vmadot i8` 每条指令算一个 8×8 输出块、K = 16 的整数 GEMM 微块：
`e8,m1` 的 1024 bit 正好是 8×16 的 int8 A 块和 16×8 的 int8 B 块。`i4` 把 K
翻倍到 32。`smt.vmadot1/2/3` 的 MAC 数与 `smt.vmadot` 相同，只是在 slide
维度上选取不同的 A 子块。

由此得到峰值（按实测 1.79 GHz、1 条/cycle/矩阵单元）：

| | MAC/cycle | op/cycle | 峰值 |
|---|---:|---:|---:|
| 单个矩阵单元 int8 | 1024 | 2048 | 3.67 TOPS |
| 单个矩阵单元 int4 | 2048 | 4096 | 7.33 TOPS |
| 单个矩阵单元 fp16→fp32 | 512 | 1024 | 1.83 TFLOPS |
| 全芯片 A100（4 个单元）int8 | 4096 | 8192 | 14.7 TOPS |
| 全芯片 A100（4 个单元）int4 | 8192 | 16384 | 29.3 TOPS |
| 全芯片 A100（4 个单元）fp16→fp32 | 2048 | 4096 | 7.33 TFLOPS |

作为对照，`vfmacc.vv e32,m1` 的 fp32 峰值是 32 lane × 2 op / 2 cycle =
32 op/cycle，全 8 核 256 op/cycle，即 0.46 TFLOPS。矩阵单元的 fp16 峰值是
它的 16 倍。

## 3. XSMTVDotII pack / unpack 指令

| Mnemonic | Vector configuration | Cycles/inst | Inst/cycle |
|---|---|---:|---:|
| `smt.vnpack.vv` | `e8,m1`；imm 1 | 2 | 0.5 |
| `smt.vnpack4.vv` | `e8,m1`；imm 1 | 2 | 0.5 |
| `smt.vnspack.vv` | `e8,m1`；imm 1 | 2 | 0.5 |
| `smt.vnspack4.vv` | `e8,m1`；imm 1 | 2 | 0.5 |
| `smt.vpack.vv` | `e8,m1`；imm 1 | 4 | 0.25 |
| `smt.vupack.vv` | `e8,m1`；imm 1 | 13 | 0.0769 |

`smt.vupack.vv` 的 13 cycle 在两个展开长度下完全一致（spread 0.00%），但
明显不是 512 bit/cycle 数据通路的整数倍，推测是微码展开。

## 4. 排除项

沿用 X60 文档的排除规则：结果在多次运行间数值一致还不够，必须在不同展开
长度和相关的操作数/地址变体下都收敛到明确的硬件速率。本轮排除：

- **所有 store 指令的绝对吞吐**：表中的 `vse32.v`（masked）等 store 行是在
  少量地址、每条 store 跟一条地址运算的条件下测的。load 侧的经验表明这两个
  条件都会造成 20%–30% 的低估，store 侧没有重做同等严格的测量，所以 store
  的数值只能用于相互比较，不能当作硬件上限。
- **`vse8.v e8,m1`（unmasked）**：两个展开长度给出 5.66 和 5.79
  cycles/inst，spread 2.2%，且随地址分布变化明显（8 条指令共用同一地址时
  是 39 cycle）。当前测量没有建立与上下文无关的速率。
- **`smt.vfwmadot* bfp16`**：硬件不实现，SIGILL。
- **`smt.vmadot.hp` / `smt.vmadot.sp` 的功能语义**：吞吐可测（1 条/cycle），
  但功能行为没有测通。`.hp` 写入 32 个 int32 lane（相当于 64 个 fp16），
  全 1 整数输入和 fp16 1.0 输入都只得到 `0x7e00`（fp16 NaN），8 个 imm 取值
  结果相同；`.sp` 在掩码全 1、imm 0–3 下累加器保持为 0。两者应当还需要
  本轮没有设置的额外状态或特定的数据布局，因此没有给出 MAC/inst。
