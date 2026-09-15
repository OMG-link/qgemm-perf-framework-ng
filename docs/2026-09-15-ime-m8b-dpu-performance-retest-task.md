# IME-m8b-DPU 性能重测任务书

## 1. 任务目标

基于当前工作区中的 128 B dynamic-unpack repack 实现，重新测量以下两个内核在真实模型矩阵尺寸和 1–4 线程下的硬件周期性能：

- `q4_0-q8_0-IME-m8b-dyn-pre-unpack`
- `q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking`

用本次测得的数据完整替换两个内核已有的 `performance.md` 数据，并在 `Hardware cycle results` 表格中增加 `Speedup` 列。

## 2. 交付文件

必须更新：

- `kernels/q4_0-q8_0-IME-m8b-dyn-pre-unpack/performance.md`
- `kernels/q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking/performance.md`

必须另外保存一份包含全部原始 benchmark stdout 的文本文件。可以保存在 `/tmp`，但完成任务时必须在汇报中给出其绝对路径。不要把临时 binary、构建产物或大体积原始日志提交到仓库。

## 3. 重要上下文与边界

1. 本次重测针对当前工作区的 128 B repack 修改。两个 unpack 函数在源码层面应对每个 K16 inner 执行一次 128 B `e8,m4` load 和两次 128 B `e8,m4` store：
   - `kernels/q4_0-q8_0-IME-m8b-dyn-pre-unpack/kernel.cpp`
   - `kernels/q4_0-q8_0-IME-m8b-DynPreUnpack-CacheBlocking/kernel.cpp`
2. 当前工作区可能包含尚未提交的生产代码和 machine microbenchmark。不得执行 `git reset`、`git checkout .`、`git clean` 或其他会丢弃现有改动的命令。
3. 不要修改内核算法、adapter、线程策略或 benchmark framework。本任务只允许为完成重测而更新上述两个 `performance.md`；如果发现 correctness 或构建问题，应停止并汇报，不要自行改变 kernel 行为。
4. 所有正式测试必须使用同一个 Release binary。构建完成后记录该 binary 的 SHA-256；正式测试过程中不得重新链接 binary。
5. 两个内核及不同线程配置必须顺序执行，禁止并行启动多个固定到相同 CPU 的 benchmark 进程。

## 4. 测试平台

目标机：

```text
SSH host: spacemit-k1
Board:    bpi-f3
CPU:      Spacemit X60
```

正式采集前记录并核实：

- OS 和 Linux kernel 版本；
- CPU frequency governor；
- 当前和最大 CPU 频率；
- 编译器版本及 commit；
- `git rev-parse HEAD`；
- `git status --short`；
- benchmark binary SHA-256；
- ASLR 是否按正式命令禁用；
- 每项 PMU 的 `time_running/time_enabled`。

目标机应使用 performance governor，CPU 频率应稳定在 1.6 GHz。若环境不满足，应先修正环境或停止并汇报，不得把环境异常时的结果写入文档。

## 5. 构建与 binary 审计

在仓库根目录执行：

```bash
./compile-and-test.sh build
sha256sum build/ime-llama-bench
```

必须确认是 Release build，并检查最终 ELF 中两个 unpack 函数使用 `e8,m4` 和 128 B 数据宽度，没有退化回原来的 32 B operand loop。可使用：

```bash
OBJDUMP=/home/linkai/projects/triton-riscv/llvm-project-spacemit-x60/builders/x86-riscv/llvm/bin/llvm-objdump

"$OBJDUMP" -d --demangle --no-show-raw-insn build/ime-llama-bench \
  | sed -n '/<unpack_q4_0_ime_m8b_rvv/,/^$/p'

"$OBJDUMP" -d --demangle --no-show-raw-insn build/ime-llama-bench \
  | sed -n '/<unpack_q4_0_ime_m8b_cache_blocking_rvv/,/^$/p'
```

预期每个 QK32 block 包含两个 128 B packed-W loads 和四个 128 B unpacked-W stores。若最终 ELF 不符合该特征，不得继续正式采集。

## 6. Correctness 前置验证

在正式性能测试前，对两个内核分别验证 1、2、3、4 线程：

```text
M=24, N=32, K=256
warmup=1
samples=3
iterations=3
verification enabled
```

命令模板：

```bash
./compile-and-test.sh run \
  --kernel KERNEL_ID \
  --m 24 --n 32 --k 256 \
  --threads THREADS \
  --warmup 1 --samples 3 --iterations 3
```

八项 correctness 必须全部满足：

- `verify: PASS`；
- `error elements: 0/768`；
- checksum 在相同输入下与 reference 一致；
- PMU running 不低于 99%。

任一项失败时停止任务，不得写入正式性能结果。

## 7. 正式测试矩阵

每个内核都要测以下 6 个 shape，每个 shape 都测 1、2、3、4 线程，共计：

```text
2 kernels × 6 shapes × 4 thread counts = 48 runs
```

| Model configuration | Matrix dimensions | M | N | K |
|---|---|---:|---:|---:|
| Qwen2 1.5B | hidden → hidden | 480 | 1,536 | 1,536 |
| Qwen2 1.5B | hidden → intermediate | 480 | 8,960 | 1,536 |
| Qwen2 1.5B | intermediate → hidden | 480 | 1,536 | 8,960 |
| Llama 2 7B | hidden → hidden | 480 | 4,096 | 4,096 |
| Llama 2 7B | hidden → intermediate | 480 | 11,008 | 4,096 |
| Llama 2 7B | intermediate → hidden | 480 | 4,096 | 11,008 |

## 8. 正式测量参数

每项正式测试统一使用：

```text
warmup:       3
samples:      7
iterations:   3
verification: disabled with --no-verify
perf event:   hardware cycles
CPU affinity: 0-(threads-1)
OMP_DYNAMIC:  FALSE
ASLR:         disabled with setarch -R
```

使用同一个已经计算 SHA-256 的 binary，并为本次任务建立唯一的远端临时目录。正式命令必须等价于：

```bash
env OMP_NUM_THREADS=THREADS OMP_DYNAMIC=FALSE \
  setarch -R taskset -c 0-$((THREADS - 1)) \
  /path/to/ime-llama-bench \
  --kernel KERNEL_ID \
  --m M --n N --k K \
  --threads THREADS \
  --warmup 3 --samples 7 --iterations 3 --no-verify
```

可以使用 `compile-and-test.sh run` 协助部署，但必须确认实际远端命令满足上述 affinity、OpenMP 和 ASLR 条件。如果现有 wrapper 没有禁用 ASLR，应手工上传一次 binary 并使用上述命令，不要在文档中错误声称 ASLR 已禁用。

每项结果必须满足：

- 使用硬件 `cycles`，不能用 wall-clock 或 nominal-frequency 估算；
- `min_running_pct=100%`，至少不得低于 99%；
- 文档中的 cycles 使用 benchmark 输出的 normalized median cycles；
- 48 项运行均来自相同 SHA-256 的 binary；
- 不得挑选单次最好结果代替 7-sample median；
- 如果发生频率变化、调度干扰或 PMU multiplexing，应丢弃受影响的整项运行并重测。

## 9. 文档更新要求

两个 `performance.md` 都必须：

1. 更新平台、环境、Git revision、worktree 描述和 binary SHA-256；
2. 增加 `Measurement method`，说明 warmup、samples、iterations、CPU pinning、OpenMP、ASLR、PMU running 和 timed region；
3. 记录 correctness 验证范围和结果；
4. 用本次 48 项中的对应 24 项完全替换旧表格数据；
5. 在 `Hardware cycle results` 表中增加 `Speedup` 列，不创建独立的 scaling 表。

表头格式应与 m4i 文档保持一致：

```markdown
| Model configuration | Matrix dimensions | M | N | K | Threads | Median cycles | Average per-core utilization | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
```

### 9.1 Average per-core utilization

使用：

```text
(M * N * K / median_cycles) / (threads * 128 FMA/cycle) * 100%
```

保留两位小数并写 `%`。

### 9.2 多线程加速比

`Speedup` 定义为同一内核、同一 shape 的单线程 median cycles 除以当前线程的 median cycles：

```text
Speedup = one_thread_median_cycles / current_thread_median_cycles
```

要求：

- 1 线程行写 `1.00x`；
- 2、3、4 线程行分别相对该 shape 的 1 线程结果计算；
- 保留两位小数并使用小写 `x`，例如 `2.73x`；
- 即使 4 线程比 3 线程慢，也必须写真实计算结果，不得做单调化或人工修正。

## 10. 机械核对

更新文档后必须编写一次性脚本或使用等价机械方法核对：

1. 每个文档正好有 24 条结果行；
2. 两个文档合计正好有 48 条结果行；
3. 每个 kernel/shape 都包含 threads 1、2、3、4，且无重复或缺失；
4. 文档 cycles 与原始 stdout 逐项一致；
5. utilization 按公式计算并四舍五入到两位后逐项一致；
6. 48 个 speedup 按公式计算并四舍五入到两位后逐项一致；
7. 每条 Markdown 结果行的列数与表头一致；
8. `git diff --check` 通过；
9. 最终 diff 没有修改本任务允许范围之外的生产代码；
10. 远端本任务临时目录已删除。

不要只目视检查表格。

## 11. Acceptance Criteria

- 两个目标 kernel 的 `performance.md` 均记录了当前 128 B repack binary 的最新真实性能数据；
- 两个 `performance.md` 的 `Hardware cycle results` 均包含 `Speedup` 列；
- 每个文档包含 6 个真实 shape × 1–4 线程的 24 项结果；
- cycles、utilization 和 speedup 已与原始数据机械核对；
- correctness 前置验证全部通过；
- 所有正式结果使用同一个 binary SHA-256，PMU running 不低于 99%；
- Release build 和 `git diff --check` 通过；
- 仅更新两个目标 `performance.md`，不改动 kernel 行为；
- 最终汇报提供 binary SHA-256、原始结果绝对路径、correctness 摘要和机械核对结果。