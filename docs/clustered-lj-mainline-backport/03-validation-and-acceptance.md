# 验证与验收

## 1. 环境

- 编译固定 24 并行；
- GPU/NCU 在沙箱外运行；
- CUDA 使用 pixi dev-cuda13、固定 SM89；
- 临时 benchmark runner 只允许移除已确认图形负载下的 pre/post idle hard return，telemetry 与所有 correctness/performance gate 保持；
- 不提交 .tmp、NCU report、SASS dump 或 benchmark case。

## 2. 三方验证

### Upstream baseline

用于证明未破坏最新主线：

- upstream CPU/CUDA configure/build；
- H5 bundled I/O、restart/input、force-field loader 测试；
- upstream 自有 unit/integration tests。

### Clustered reference

19856de 提供：

- 真实 clustered kernel SASS/资源；
- NCU reference；
- canonical snapshots；
- 36 replay 和 36 production 性能锚点。

### Backport candidate

每批与 upstream 做功能对照；device/runtime 集成检查点与 19856de 做 clustered 对照。

## 3. 每批快速门槛

1. git diff --check；
2. Manager/worker leakage path check；
3. CPU SPONGE build；
4. clustered contract与manybody oracle；
5. 按范围补充 regular/soft-LJ、SITS/REST2、custom/manybody fixture；
6. CUDA SPONGE 与受影响测试 target build；
7. 目标 kernel SASS occurrence/body/resource检查。

纯文档批不运行性能门槛；纯 host 批仍需最终集成 A/B，不以“理论无影响”替代。
未接入默认 SPONGE、且不包含生产 kernel/launch 的独立 contract target 批次，只执行其 CPU/CUDA configure、build 与 contract test；它不伪造无意义的 NCU 或 A/B 结果。首次接入默认 runtime source list 的后续批次补齐生产构建、SASS 与适用的性能门槛。

对于已进入默认 binary、但尚无生产调用点的 device foundation，验收为精确父版本全函数 normalized SASS 集合比较：所有 common function 必须 exact，允许 only-added 的新模块函数，不允许 changed 或 removed。NCU 与 A/B 从首次出现可达 launch 和 clustered route 的批次开始。

## 4. Device 批次

任何 kernel body、模板实例、launch或 TU owner 变化必须：

1. 先采集 19856de 对应 kernel 的 ncu --set full；
2. 记录 grid/block、register、shared/local、spill、occupancy、warp stall、L1/L2/DRAM；
3. 修改后重采样同一输入；
4. 对比 normalized SASS；
5. 资源或 executed instruction 增长必须有明确性能收益，否则回退。

重点覆盖：

- regular LJ force-only/full；
- standalone soft-LJ force-only/full；
- candidate fixed-light count；
- record aggregate/compact；
- fused gather；
- SITS/REST2/soft-core SITS；
- 受迁移影响的 manybody/ReaxFF hot kernel。

## 5. Replay

固定：

- systems：wat160k、wat600k、dna_cou；
- modes：force-only、full；
- implementations：reference、candidate；
- 3 runs，2000 iterations；
- 36/36 valid；
- full 18/18 matched；
- route、layout、finite/sanity 全部符合 runner contract；
- 每个 system/mode paired median regression 不超过 3%。

## 6. Production

固定：

- wat160k、wat600k、dna_cou；
- NVT、NPT；
- reference/candidate 交替；
- 每格 3 runs、10000 steps；
- 36/36 valid；
- 每个 system/ensemble 单独通过 3%；
- 报告 ns/day、wall time、Calculate_Force；
- 历史 peak 只作诊断，不替代 matched A/B。

## 7. Upstream 专项

最新 upstream 的 39 个提交引入 H5/input/force-field变化，最终至少补充：

- bundled H5 topology/protocol/restart smoke；
- warning 后继续执行；
- Amber/GROMACS NONBONDED_PARM_INDEX、CHAMBER 1-4 LJ、nonbond override fixture；
- ReaxFF native initialization；
- NVT/NPT restart continuity。

## 8. 最终验收

以下条件是合取关系：

- upstream tests通过；
- clustered CPU/CUDA correctness通过；
- source reference与candidate的目标 SASS/NCU无不可解释回退；
- replay 36/36；
- production 36/36；
- Manager/worker泄漏检查为空；
- legacy/native双路径检查为空。

## 9. B3 集成检查点

B3 是 clustered provider 首次进入 LJ 生产调用点的批次，因此执行完整 device 与运行时门槛。

### Correctness 与 binary

- CPU/CUDA SPONGE 与 `CLUSTERED_SPATIAL_VIEW_TEST` 均以 24 并行构建；contract 1/1 通过，SM89 ELF 检查通过。
- wat160k、wat600k、dna_cou 的 force-only/full snapshot pair oracle 6/6 exact；full output 3/3 matched。
- standalone soft-LJ CUDA oracle 通过，最大绝对误差 `0.03`，容差 `2`。
- benchmark runner 单元测试 20/20 通过。

### SASS 与 NCU

- source/candidate 的 14 个 regular/soft-LJ、gather 与 microbench target 实例 normalized SASS body hash、register、stack、shared 和 local resource exact。
- regular force-only/full 代表 NCU 中 grid/block/resource 相同，duration 与主要 compute/memory/stall 指标在测量波动内。
- 整步 NCU 发现 production 回退来自非 LJ 的 `PME_Final` host launch：block 为 `(8,128)`，原子索引使用 `threadIdx.y`，主线却按 `blockSize.x` 计算 grid。修复前 candidate 为 82,272 blocks、约 `471 us`；修复后与 reference 同为 5,142 blocks，定向 full profile 为 `331.744 us` 对 `331.968 us`。

### Replay 与 production

- replay 使用同一批新鲜 canonical snapshots，3 systems × 2 output modes × 2 implementations × 3 runs，共 36/36 valid，full 18/18 matched；六格 median throughput delta 均在 `-0.51%` 到 `+0.61%`。
- production 输入显式固定 `skin=2.0`，避免 source/mainline 默认值差异污染 A/B；reference 仅在临时 benchmark worktree 中对齐 mainline barostat 与 thermostat RNG，未进入候选提交。
- 图形负载偶尔超过 5% 时，临时 runner 只移除 pre/post idle hard return；telemetry、route、finite、matched 与 performance gate 未改变，临时 runner 和产物不提交。
- 最终 production 36/36 valid。paired median：wat160k NVT `+0.25%`、NPT `+0.11%`；wat600k NVT `+0.55%`、NPT `-0.06%`；dna_cou NVT `-0.69%`、NPT `+0.61%`。
- 最终 water median ns/day：wat160k NVT `100.009 → 100.474`、NPT `91.590 → 91.688`；wat600k NVT `28.222 → 28.429`、NPT `25.821 → 25.808`。DNA NVT `398.485 → 395.547`、NPT `376.389 → 378.687`。
