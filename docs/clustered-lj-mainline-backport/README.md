# Clustered-LJ 主线 backport 计划

- 日期：2026-09-02
- 目标分支：backport/clustered-lj-mainline
- 目标基点：b78aeaba848fc4703489e0d5b1609644ce0417f9（sponge-upstream/master）
- 性能实现检查点：19856deb
- 计划与审计检查点：71046ec
- 共同祖先：4c694ebab7032b0ef28d8312115d0f3253800125
- 状态：B0–B7 已完成；backport 已通过最终集成门槛与残留审计

## 目标

把性能分支中已经完成 correctness、SASS、NCU、36 replay 和 36 production 门槛的 clustered neighbor/LJ 实现移植到最新 SPONGEMM 主线，同时保留主线在 H5 bundled I/O、restart、input materialization、force-field parsing 和构建系统上的后续改动。

本 backport 明确不移植 SPONGE Manager：

- 不加入 SPONGE/manager；
- 不加入 cmake/targets/SPONGE_MANAGER.cmake 或 Manager target；
- 不加入 Manager REMD benchmark、fixture、CLI 与文档；
- 不加入只为 Manager worker 控制服务的 SPONGE/worker_protocol 和主程序 worker-mode plumbing。

若 clustered runtime 的历史提交与 Manager 混合，按最终运行时职责重新形成主题提交，不按原提交路径整段 cherry-pick。

## 文档

| 文档 | 职责 |
|---|---|
| [00-scope-and-baseline.md](00-scope-and-baseline.md) | 精确基点、范围、排除项与三类对照基线 |
| [01-topic-batches.md](01-topic-batches.md) | 主题批次、依赖顺序、提交边界和停止条件 |
| [02-conflict-map.md](02-conflict-map.md) | 与最新主线的重叠文件、三方合并规则与 Manager 泄漏检查 |
| [03-validation-and-acceptance.md](03-validation-and-acceptance.md) | CPU/CUDA correctness、SASS、NCU、replay 与 production 验收 |
| [04-consumer-follow-ups.md](04-consumer-follow-ups.md) | consumer 多体性能问题、primitive 抽取设计、迁移批次与验收门槛 |

## 实施原则

1. 以最新 upstream 为第一父历史，不把 231 个源分支提交整体 merge 进主线。
2. 以最终职责形成少量可审查主题提交；实验、被撤回实现和 Manager 历史不进入 backport。
3. 主线独有逻辑默认保留；clustered 最终实现按 contract/provider/builder/LJ/consumer 的依赖顺序接入。
4. 不恢复 native/legacy LJ、solvent fast path、virial-only specialization 或旧 neighbor-list façade。
5. 不新增 production probe、feature gate、双路径 fallback、通用 evaluator 或 Manager 兼容层。
6. 每个代码批次都有明确父版本、构建/correctness 和 SASS 检查；device 或 launch 有风险时执行 NCU；完整 36/36 replay 与 production 在集成检查点执行。

## 完成标准

- upstream H5/input/restart 测试保持；
- CPU 与 CUDA clustered contract、manybody、LJ/soft-LJ、SITS/REST2 oracle 通过；
- Manager target 和 Manager 源码未引入；
- source reference 与 backport candidate 的目标 kernel SASS/资源差异可解释；
- replay 36/36、full 18/18 matched；
- production 36/36，六个 cell 单独通过 3% gate；
- mainline tree 不保留 legacy/native neighbor/LJ 双路径；插件 ABI 所需 half-list 仅作为兼容边界存在。
