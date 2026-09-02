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

## 10. B4 Selective Interaction 检查点

B4 将 selective interaction 接入唯一 clustered provider，并移除旧 SITS owner；本批同时验证空 pair-list 语义和 upstream H5 restart 闭环。

### Correctness 与生命周期

- CPU/SM89 SPONGE、`CLUSTERED_SPATIAL_VIEW_TEST` 与 `NBNXM_MICROBENCH` 目标构建通过；CPU/CUDA contract 通过。
- standalone SITS façade 与 reproducibility 在 CPU/CUDA 各 2/2 通过；H5 runtime closure 1/1 通过。
- regular SITS、REST2 correction、soft-core SITS 由同一 façade 管理 sparse product、stamp 与 clear；没有第二套 selective owner，也没有 Manager 依赖。
- upstream H5 `nk` restart precedence、atom selection、runtime restart Export/Apply 与 trajectory pending append 均保留。
- `sci_count == cj_count == 0` 被定义为合法空 gmxpacked product；partial payload 继续报错，regular/soft-LJ 的空表路径为 no-op。

### SASS 与 NCU

- source-first NCU 覆盖 sparse builder、regular direct 与 soft direct 三个代表 kernel；每个 profile 均完成 44 passes。
- normalized SASS body source/candidate exact：builder 2424 条、regular 544 条、soft 752 条；register、static/dynamic shared 与 spill 状态均一致。
- builder launch 为 `(100,1,1) × (128,1,1)`，37 registers、12 B static shared、0 spill，duration `101.54 → 101.79 us`。
- regular launch 为 `(29,8,4) × (8,8,1)`，64 registers、0 spill，duration `10.24 → 10.82 us`；soft launch 为 `(38,8,4) × (8,8,1)`，72 registers、0 spill，duration `17.95 → 15.97 us`。branch efficiency、L1 hit rate 与 achieved occupancy 无不可解释差异。

### Replay 与 production

- replay 为 3 systems × 2 output modes × 2 implementations × 3 runs，共 36/36 valid；paired delta：wat160k force-only `+0.251%`、full `-0.054%`，wat600k force-only `+0.087%`、full `-0.067%`，DNA force-only `-2.115%`、full `+0.243%`。
- production 为 3 systems × NVT/NPT × 2 implementations × 3 runs，共 36/36 valid；所有进程返回 0，GPU idle 为 2%，锁频为 2520 MHz。
- production paired speed delta：wat160k NVT `-0.312%`、NPT `-0.010%`；wat600k NVT `-0.299%`、NPT `+0.248%`；DNA NVT `+0.294%`、NPT `-0.072%`。
- 官方 replay + production migration gate 通过；本批没有使用宽松 gate，也没有提交临时 runner 或性能产物。

## 11. B5.1 SW/EDIP/Tersoff 检查点

B5.1 将首组三体 manybody consumers 接到唯一 clustered provider；精确批次父版本为 `d7cdb87`。EAM、custom 与 ReaxFF 未进入本批，因此 A/B 只判断本批新增 consumer 是否扰动既有 LJ 生产路径。

### Correctness 与边界

- CPU 与 CUDA13/SM89 的 SPONGE、`CLUSTERED_SPATIAL_VIEW_TEST`、`MANYBODY_CLUSTERED_ORACLE_TEST` 构建通过；contract 与 manybody oracle 各 2/2 通过。
- `MANYBODY_CLUSTERED_ORACLE_TEST` 以单 target 配置、构建和执行 1/1 通过，未借用 SPONGE target 的 `sponge_toml` 定义。
- EDIP/Tersoff oracle 覆盖非中心 periodic shift、pair active mask、exclusion、非对称/严格 cutoff、directed relation、三元组 image identity、EDIP z 与 dE/dz；CPU 与 CUDA 均通过。
- SW 使用现有 LAMMPS diamond 对照 fixture，10648 atoms，1/1 通过；没有新增 production probe 或临时 gate。
- 三个 consumer 的 view contract 在 Provider `AcquireView` 集中验证；CUDA 要求 pair-shift metadata/rcell，SW 额外要求 endpoint incidence。consumer 只检查自己的算法缓冲区，避免重复 contract 逻辑。
- SW/EDIP/Tersoff 保留 upstream H5/native initialization，并显式拒绝 non-PBC 与 multi-rank。EAM、custom 与 ReaxFF 的 legacy full-list owner 未在本批改动。

### SASS 与 NCU

- source-first EDIP/Tersoff oracle NCU 覆盖 8 个 kernel 实例，共 44 passes；source/candidate normalized SASS 均为 4768 条指令，SHA-256 均为 `b170562d2b657c5ca4f7a4e6c5db6408598a3d0a23ea892ba0d17ab3fea0cf4d`。
- EDIP/Tersoff 的 grid/block、register、shared、occupancy limit 与 branch 指标一致。代表 duration（source → candidate，微秒）：EDIP builder false `8.384 → 4.864`、true `6.272 → 4.960`，EDIP force `3.968 → 3.968`，Tersoff builder false `4.832 → 4.800`、true `5.824 → 4.928`，Tersoff force `6.272 → 6.240`。
- SW diamond NCU 覆盖 builder false/true 与 cached-center full，共 44 passes；normalized SASS 均为 3216 条指令，SHA-256 均为 `3471d7bcc80a1cb615cb021a7e87ecbcdbfdcec543060cb3abea9cfd1b549a72`。
- SW 三个实例资源与 launch exact；duration 为 builder false `260.58 → 265.31 us`、true `270.11 → 268.83 us`、cached full `101.44 → 102.30 us`。最大单项变化 `+1.82%`，无 spill、occupancy 或 launch 形态回退。

### Replay 与 production

- replay 使用精确父提交 `d7cdb87` 与当前 candidate 各自构建的 microbench，3 systems × 2 output modes × 2 implementations × 3 runs，warmup 200、iterations 2000，共 36/36 valid，full 全部 matched。
- replay paired median kernel-time delta：wat160k force-only `-0.211%`、full `-0.151%`；wat600k force-only `-0.199%`、full `+0.016%`；DNA force-only `+0.647%`、full `+0.020%`。
- production 使用同一父/candidate 的 SPONGE binary，3 systems × NVT/NPT × 2 implementations × 3 runs，每次 10000 steps，共 36/36 valid。四次图形负载越过 5% 的尝试由正式 runner 标无效并重试，未计入矩阵。
- production paired speed delta：wat160k NVT `+0.325%`、NPT `-0.441%`；wat600k NVT `-0.129%`、NPT `-0.015%`；DNA NVT `-0.208%`、NPT `+0.069%`。
- 官方 replay + production migration gate 在每个 cell 的 3% 合取门限下通过；未修改 runner、未放宽 idle/performance gate，NCU report、SASS dump 与矩阵产物均只保留在 `.tmp`。
