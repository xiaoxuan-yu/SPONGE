# 范围、基点与对照

## 1. 已确认的历史事实

目标与源分支的共同祖先是 4c694eba。从该点起：

- 性能/重构源分支到 71046ec 有 231 个提交；
- 最新 upstream 到 b78aeaba 有 39 个提交；
- 源端改动 345 个文件，upstream 改动 366 个文件；
- 两侧有 25 个重叠路径，集中在 main.cpp、control/MD core、soft-LJ、custom force、manybody、GROMACS loader、CMake 与 pixi。

因此本任务不是无冲突的 commit backport。整段 merge 或连续 cherry-pick 会同时带入 Manager 历史、已撤回实验和与主线 H5/input 改动的冲突。

## 2. 迁移范围

必须迁移：

- SPONGE/neighbor_list/contract；
- SPONGE/neighbor_list/provider；
- SPONGE/neighbor_list/builder；
- clustered workspace、gather、regular LJ 与 soft-LJ 最终 kernel/host boundary；
- SITS/REST2/selective interaction 的 clustered consumer；
- custom pair、EAM、SW、EDIP、Tersoff、ReaxFF 等已经直接消费 clustered view 的路径；
- DeviceBuffer、traversal primitive、center-list host primitive 等被最终实现真实依赖的中立头；
- microbench、snapshot producer、contract/manybody oracle、clustered performance runner；
- Cornerstone 依赖及其准确构建属性。

可在对应主题末尾迁移最终架构文档、测试 CMake target 和 fixture/runner。

## 3. 明确排除

- SPONGE/manager；
- cmake/targets/SPONGE_MANAGER.cmake；
- Manager REMD/scheduler/worker lifecycle；
- Manager benchmark、fixture、skill 与用户文档；
- SPONGE/worker_protocol 及只为 Manager worker-mode 增加的 CLI/runtime plumbing；
- 源分支已删除或撤回的 probe、gate、legacy/native、solvent 与 virial-only 路径；
- 与 clustered runtime 无关的性能实验和原始 .tmp artifact。

若最终 runtime CMake 采用共享 source-list 文件，该文件只列 SPONGE runtime 必需源，不因历史实现复制 worker protocol 或 Manager source。

## 4. 三类基线

| 基线 | 提交 | 用途 |
|---|---|---|
| Upstream functional baseline | b78aeaba | H5/input/restart、构建与非 clustered 行为对照 |
| Clustered performance reference | 19856de | 最终运行时代码、SASS、NCU、replay、production 对照 |
| Backport candidate | 当前主题批次 HEAD | 主线融合后的 correctness 与性能候选 |

71046ec 只比 19856de 多 K8 审计文档，不改变 runtime，可用于查阅计划但不能替代性能 binary。

## 5. 路径选择规则

- source-only clustered 目录以 19856de 最终状态为语义来源；
- upstream-only H5/tests/schemas 默认原样保留；
- 重叠文件必须三方融合，不允许整文件选边；
- 不追求源端 231 个提交的 SHA/拓扑复刻；主题提交必须能从最终 diff、caller/owner 与验证证据解释；
- 每批只暂存白名单路径，并运行 Manager 泄漏检查。
