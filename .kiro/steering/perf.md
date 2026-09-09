本文档说明 perf 使用时的注意事项

- 使用 perf 采样 PC 分布确定热点时，采样带来的额外开销一般控制在 +5% 以内。
  - 允许的例外：当采集的目标发生频率极低时，允许提高采样频率以采集到有统计学意义的数据
- 使用 perf 采样 load miss 事件源时，参考 `docs/skills/l1d-miss-pc-sampling-with-nop-sleds.md` 的技术方案。
