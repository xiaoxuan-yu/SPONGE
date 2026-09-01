# 主题批次与依赖顺序

## 1. 为什么不用整段 cherry-pick

源历史包含五个 merge、Manager/worker 协议、microbench 实验、clustered 算法迁移、provider 架构和后续清理。部分早期提交同时修改 runtime 与 Manager，直接重放会把排除项带入；后续 upstream 又修改了同一批输入和 force 文件。

正式历史使用“最终状态主题 backport”：每批从 4c694eba..19856de 的受控 pathspec 生成候选 diff，在 upstream worktree 中三方应用并人工解决重叠文件。只有边界独立、冲突很小的晚期提交才直接 cherry-pick。

## 2. 批次

### B0：计划与基线

- 建立本 worktree 和文档组；
- 固定 upstream/source/reference SHA；
- 不改生产源码。

### B1：独立 Python benchmark harness

- replay matrix runner；
- production matrix runner；
- migration gate；
- 三者的纯 Python 单元测试；
- wat160k、wat600k 与 DNA_COU staging/dry-run 所需的 canonical tracked inputs。

依赖审计证明 native contract/manybody oracle、NBNXM microbench 与 snapshot producer 都依赖尚未迁移的 provider/builder/LJ 闭包，因此延后到对应主题批次。本批只迁移 benchmarks/performance/clustered_lj，不修改 CMake，不影响默认 configure，也不声称 native target 已可构建。

实施结果：从 19856de 恢复 6 个 runner/gate/test 文件及三组最小输入闭包；首次测试以 52/56 暴露缺失 tracked inputs，补齐后 56/56 通过。没有放宽 staging、idle、route、matched 或 3% gate。

### B2：contract、provider 与 builder foundation

- neighbor_list/contract；
- Provider config/domain/lifecycle/state；
- 三个 owning domain 与 build request；
- builder internal API、CPU/GPU payload、candidate、active refresh、record stream；
- Cornerstone 与中立 buffer/traversal primitive；
- runtime CMake source/object ownership。
- 在本闭包可构建后加入 clustered spatial-view test；manybody oracle 随 consumer 批，microbench/snapshot 随 LJ 批。

约束：

- 不引入 LJ compatibility façade；
- candidate/payload object 每个最终 target 只注入一次；
- Manager/worker source 不进入 runtime source list；
- 本批结束时 contract/oracle target 和 CPU/CUDA SPONGE 至少可编译。

实施拆分：

- B2.1 先迁移 contract、Provider 的只读 view 接口与参数/state 定义、Cornerstone gitlink，以及独立 `CLUSTERED_SPATIAL_VIEW_TEST`；不接入默认 SPONGE source list。
- 为使独立 target 可达，`cmake/utils/targets.cmake` 改为仅在调用方未设置时默认 `SPONGE`，不再以 `FORCE` 覆盖 `-DTARGETS=`；默认构建集合未改变，也未加入 Manager。
- B2.1 在 `dev-cpu` 与 `dev-cuda13`/SM89 下均完成 24 并行构建，`ClusteredSpatialViewContract` 各 1/1 通过。
- B2.1 没有生产 source、kernel、launch 或默认 binary 变化，因此不触发 SASS/NCU/A-B；这些门槛从 B2.2 builder runtime 接入默认 SPONGE 起执行。
- B2.2 再迁移 lifecycle、builder 实现与 runtime source/object ownership，并以 CPU/CUDA SPONGE 构建闭包结束。

### B3：regular LJ、soft-LJ 与主生命周期

- clustered workspace/gather；
- regular LJ force-only/full；
- standalone soft-LJ force-only/full；
- main.cpp 中唯一 Provider/workspace ownership、build/gather/clear；
- pair shift、PME direct 与 output semantics。

冲突处理必须保留 upstream H5 input/restart/materialization。不得恢复 native/legacy LJ、solvent fast path或 virial-only variant。

### B4：Selective Interaction、SITS 与 REST2

- 新 Selective_Interaction 目录；
- sparse product ownership；
- regular SITS、REST2 correction、soft-core SITS 三个物理模式；
- main force dispatch 与 selective policy façade。

旧 SPONGE/SITS 中 upstream H5 状态逻辑需要迁入新 owner，而不是保留两套 SITS。

### B5：custom 与 manybody consumers

- custom pair；
- EAM；
- SW/EDIP/Tersoff；
- ReaxFF bond-order、EEQ、VDW、hydrogen bond；
- CPU clustered paths、center lists 与 lookup primitive。

约束：保留 upstream native/H5 initialization；只替换邻居消费算法，不合并不同算子的数学循环。

### B6：清理、源码树与最终 source owner

- 应用 K1–K7 已验证的 lookup/buffer/offset/SITS/soft-LJ 清理；
- 删除 migration 后不可达的旧 source、header、CMake entry；
- 收口 include 与 target source list；
- 加入最终架构/迁移文档。

本批禁止新算法优化。若清理导致目标 kernel SASS 变化，先回退并拆批。

### B7：集成门槛

- 全部 CPU/CUDA oracle；
- upstream H5/input/restart smoke；
- source reference 对 candidate 的 SASS/NCU；
- 36 replay；
- 36 production；
- Manager 泄漏和 legacy/native 残留审计。

## 3. 每批提交模板

    Batch:
    Target parent:
    Source reference:
    Path whitelist:
    Upstream behavior preserved:
    Manager exclusions:
    Expected device/launch change:
    Correctness:
    SASS/NCU:
    Decision:

## 4. 停止条件

- 需要 Manager 类型、worker protocol 或 Manager target 才能编译：停止并重新划定 runtime 接口；
- upstream H5/input 行为只能通过恢复旧 owner 才能保留：停止并设计显式适配；
- 出现第二套 Provider、neighbor list 或 LJ 生产路径：拒绝；
- device batch 无法在 source reference 与 candidate 间解释 SASS/资源差异：拒绝；
- 任一 production cell 回退超过 3%：拒绝。
