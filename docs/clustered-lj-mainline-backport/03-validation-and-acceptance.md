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
