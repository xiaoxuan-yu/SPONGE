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

## 12. B5.2 EAM 检查点

B5.2 将 EAM 接到唯一 clustered provider；精确批次父版本为 `722dddc`，性能实现参考为 `19856de`。custom pair 与 ReaxFF 未进入本批。

### Correctness、初始化与空表边界

- CPU 与 CUDA13/SM89 的 SPONGE 构建通过；CUDA `NBNXM_MICROBENCH` 同步重建。
- Cu funcfl 与 Cu-Ni setfl/alloy 两套现有 LAMMPS fixture 在 CPU、CUDA 各 6/6 通过。GPU 最大 force 差分别约为 `1.75e-2` 与 `1.17e-1`，force cosine 均不低于 `0.999999`；CPU 结果同量级并通过原有容差。
- EAM 保留主线 `NativeEAMDefinition`、funcfl、setfl/alloy 与 H5 typed reader；只替换邻居消费算法。non-PBC 与 multi-rank 继续显式拒绝，不引入 legacy fallback。
- 两条 EAM 路径都按 rho accumulation、embedding derivative、pair force 三阶段执行；CUDA 使用 gmxpacked tile，CPU 使用对应 clustered traversal。只有 force-only/full 两种模板实例。
- H5 restart-load 小体系暴露 builder 曾发布 `sci>0、cj=0` 的 partial payload。CPU payload、host compact 与 device compact 的失败闭包现统一将四个计数归零；修复后 SPONGE 完整执行该 case，原 `clustered spatial view has no gmxpacked pair payload` 异常消失。
- 空 payload 只跳过 rho 与 pair-force traversal，不跳过逐原子的 embedding 阶段。使用现有 synthetic funcfl（`F(0)=1 eV`）派生的无邻居临时输入与精确父版本对照，10976 原子的 legacy 与 clustered EAM 均报告 `253131.22 kcal/mol`；`dF/dρ` 同样在空表上刷新。
- opt-in H5 CTest 仍被既有运行时编译 warning 插入 `Read Nk from ...` 日志行所触发的精确字符串断言阻断；SPONGE 返回 0 且完成两帧，故本批不修改 H5 harness，也不把该断言记为 EAM correctness 失败。

### SASS 与 NCU

- source-first NCU 分别覆盖 full 与 force-only case；每个 report 包含 gather、rho、embedding derivative 与 force 四个 kernel，各完成 44 passes。
- normalized SASS source/candidate exact。full 的指令数与 SHA-256：gather 88 / `8914328d314682c03fe1c28550a779514d0b0e77a7a522019cd904dabb3d55df`，rho 1120 / `e3816cf8527b4f8fb1caf6caf9730360e5dea984eb04e4d7e5d2e4f1e27d1671`，derivative 88 / `1eca499f895c9d4287ec99b8507ac72d5fb16f837ac476077bbe20e615c5882e`，force 1840 / `c53b9db7fda417b80b1438cf4615f1cf3186a19ac33f48fa1e08e211a6fc29bf`。
- force-only 的指令数与 SHA-256：gather 88 / `d805730e330e32b021eed876e8b0e86824c984c45fef7d698500d3b196a2ce25`，rho 1120 / `f2f486d1885e8008bd0d28650daaf1fe039e2e43ab15a0761d4b1eef28234698`，derivative 80 / `6a40740b0a71816a82a8349bade33e84504e44b3035ed8861f2736e0aba3d1f8`，force 1568 / `9165023ce0ad2c8a2a78c1f14e412c344d075cc33a2e2b952c5230194a8e4ea8`。
- full 的 register 数为 gather/rho/derivative/force `20/55/19/72`；force-only 对应资源、grid/block 与 source exact，全部 0 spill。
- 代表 duration（source → candidate，微秒）：full rho `35.232 → 34.464`、derivative `17.600 → 17.632`、force `363.392 → 364.544`；force-only rho `37.22 → 34.34`、derivative `4.45 → 2.05`、force `42.21 → 43.14`。短 gather launch 的计时受 warm/cache 波动影响，但 SASS、资源与 executed instruction exact，无结构性回退。
- 后续空 payload 修复同时改变 builder 失败收口与 EAM host dispatch，但不改变 kernel body。修复前后 report 再次 source-first NCU：六个模板实例的 normalized SASS、launch 与资源 exact；full force 为 `364.54 → 367.39 us`，force-only force 为 `43.14 → 44.54 us`，其余差异均为短 launch/执行采样波动，无结构性变化。

### Replay 与 production

- replay 使用精确父提交 `722dddc` 与当前 candidate 各自构建的 microbench，3 systems × 2 output modes × 2 implementations × 3 runs，warmup 200、iterations 2000，共 36/36 valid，full 全部 matched；pre/post idle SM 为 `2–3%`。
- replay paired median kernel-time delta：wat160k force-only `+0.241%`、full `-0.388%`；wat600k force-only `+0.039%`、full `-0.002%`；DNA force-only `-1.214%`、full `+0.760%`。
- production 使用同一父/candidate 的 SPONGE binary，3 systems × NVT/NPT × 2 implementations × 3 runs，每次 10000 steps，共 36/36 valid；wat 系统走 comb，DNA 走 packed-AB，pre/post idle SM 均不超过 5%。
- production paired speed delta：wat160k NVT `+0.475%`、NPT `-0.450%`；wat600k NVT `-0.108%`、NPT `+0.076%`；DNA NVT `+0.010%`、NPT `+0.552%`。
- 空表 host-dispatch 修复发生在正式 A/B 后；六个 A/B case 均不初始化 EAM，microbench target 未改变，且 post-fix EAM SASS/launch/resource exact，因此该矩阵仍直接覆盖受影响的 LJ 生产路径。
- 官方 replay + production migration gate 在每个 cell 的 3% 合取门限下通过；runner、idle/performance gate、NCU report 与矩阵产物均未进入提交。

## 13. B5.3 custom pair 检查点

B5.3 将运行时 JIT custom-pair consumer 接到唯一 clustered provider；精确批次父版本为 `314184a`，性能实现参考为 `19856de`。ReaxFF 未进入本批。

### Correctness、JIT 与空表边界

- CPU 与 CUDA13/SM89 的 SPONGE 构建通过；CUDA `NBNXM_MICROBENCH` 与 `SPONGE_CLUSTERED_SNAPSHOT_PRODUCER` 同步构建。沙箱外 clustered contract 与 manybody oracle 为 2/2 通过。
- Morse/LAMMPS 对照在 CPU、CUDA 各连续运行三次并通过能量、压力、应力与逐原子 force 检查。GPU 最大绝对 force 误差为 `2.11e-4` 到 `2.84e-4`，CPU 为 `1.65e-4` 到 `2.56e-4`。
- 19³ periodic fixture 的 canonical pair oracle 为 `500707/500707` exact，duplicate、missing、extra 均为 0；单原子 CPU/GPU 空表 case 的 potential 与 Morse force 均为 0。
- custom pair 保留主线 native/H5 输入和 JIT 势函数，仅替换邻居遍历与 launch contract。JIT 源显式携带 clustered lane-valid/local primitive，避免依赖宿主 translation unit 的预处理状态；生产实现只有 force-only/full 两种 specialization。
- `SPONGE_CLUSTERED_SNAPSHOT_PRODUCER` 复用主程序初始化与 clustered provider，并通过 `SPONGE_EMBEDDED_RUNTIME` 排除可执行入口。公共 runtime source 清单只有一个 CMake owner，未引入第二套生产实现。

### SASS 与 NCU

- source reference 的 custom-pair JIT 源缺少 `Clustered_Lane_Is_Valid`/`Clustered_Lane_Is_Local` 定义，无法重新生成 live reference report；该缺陷未带入候选。对照使用此前同实现的 final NCU report，同时对候选重新执行完整 NCU。
- force-only 候选 kernel 为约 `38.30 us`，历史 final 为 `42.30 us`；69 registers、0 spill，理论 occupancy `58.33%`，achieved occupancy 约 `34.5–35.3%`，launch 为 2480 blocks × 64 threads。
- full 候选 kernel 为约 `60.93 us`，历史 final 为 `68.54 us`；72 registers、`16.38 KiB` dynamic shared、`1.28 KiB` static shared、0 spill，理论/实际 occupancy 为 `20.83%/18.19%`。资源和 launch 形态无回退。
- full 候选 executed instruction 比历史 report 多约 0.90M，但实测 duration 更低，且无 spill、occupancy 或 memory-write 放大；因此接受该差异。空表 host-dispatch 修复不改变 JIT kernel source。

### Replay 与 production

- replay 使用精确父提交 `314184a` 与当前 candidate 的 microbench，3 systems × 2 output modes × 2 implementations × 3 runs，warmup 200、iterations 2000，共 36/36 valid。首轮一次 post-run idle SM 为 7% 被正式 runner 判无效并中止；保留原记录后按同一 5% 阈值完整重跑通过，未放宽 gate。
- replay paired median kernel-time delta：wat160k force-only `-1.055%`、full `-0.396%`；wat600k force-only `-0.247%`、full `+0.204%`；DNA force-only `+0.640%`、full `+0.059%`。
- production 使用同一父/candidate 的 SPONGE binary，3 systems × NVT/NPT × 2 implementations × 3 runs，每次 10000 steps，共 36/36 valid。median ns/day（父版本 → 候选）：wat160k NVT `91.202 → 92.153`、NPT `84.132 → 84.737`；wat600k NVT `26.408 → 26.483`、NPT `24.496 → 24.380`；DNA NVT `355.003 → 358.464`、NPT `341.352 → 342.527`。
- 对应 median throughput delta 为 wat160k NVT `+1.043%`、NPT `+0.719%`；wat600k NVT `+0.283%`、NPT `-0.475%`；DNA NVT `+0.975%`、NPT `+0.344%`。wat160k NVT 的首个父版本样本出现一次慢启动，但另两组配对样本正常，且它只使候选侧看起来更快，不会隐藏性能回退。
- 官方 replay + production migration gate 在每个 cell 的 3% 合取门限下通过；runner、idle/performance gate、NCU report、snapshots 与矩阵产物只保留在 `.tmp`。

## 14. B5.4 ReaxFF VDW 检查点

B5.4 只迁移 ReaxFF VDW；精确批次父版本为 `49298b8`，性能实现参考为 `19856de`。EEQ、bond-order、bond、angle/torsion 与 hydrogen-bond 不在本批改动范围，仍保留 legacy neighbor-list consumer。

### Correctness 与边界

- CPU 与 CUDA13/SM89 的 SPONGE、`CLUSTERED_SPATIAL_VIEW_TEST`、`MANYBODY_CLUSTERED_ORACLE_TEST` 构建通过；clustered contract 与 manybody oracle 在 CPU/CUDA 各 2/2 通过。
- PETN 16240 的现有 LAMMPS 单帧对照在 CPU、CUDA 均通过。相对精确父版本的 step 0/1 A/B 中，CPU total potential 差为 `+0.12 kcal/mol`、pressure 差不超过 `0.03 bar`、force max/RMS 差为 `0.003365/0.000135`；GPU total potential 差为 `-0.24/-0.12 kcal/mol`、pressure 差为 `-0.07 bar`、force max/RMS 差为 `0.004598/0.000191`。
- VDW 只替换邻居遍历：native/H5 参数与 type 初始化保持，其他 ReaxFF 数学阶段和 legacy list 不变。GPU 使用 per-pair image shift 与 active mask；CPU 继续遵守其 SCI-shift backend contract。PBC 与 single-PP-rank 在初始化时 fail-fast，空 SCI payload 在 coordinate gather 前作为合法 no-op 收口。
- GPU 以 packed traversal 的局部 j endpoint 聚合 atom energy/virial；这些数组在 ReaxFF 路径只进入总能量/总 virial 归约，端点归属不是可观察物理语义。审计期间尝试恢复 legacy 的较小 atom-id 归属会增加热循环随机 atomic，NCU 显示 full kernel 回退 `10.48%`，因此未纳入提交；恢复 subgroup 聚合后最终 SASS 与性能参考 exact。
- opt-in H5 ReaxFF/EDIP runtime smoke 在 candidate 与精确父版本上都于既有 Tersoff clustered force 中触发同一 SIGSEGV，发生在 VDW 调用之前；本批不将该父版本已有问题归因于 VDW，也不越界修复 Tersoff。

### SASS 与 NCU

- legacy VDW kernel 的 source-first NCU 为约 `4.48 ms`，127 blocks × 128 threads、40 registers、0 spill；achieved occupancy 约 `8.13%`，no-eligible 约 `98.81%`，L1TEX scoreboard 约 `80.6 cycles`，过量 sectors 约占 `81%`。
- clustered VDW launch 为 `(708,8,1)` blocks × `(8,8,1)` threads。full 为 64 registers、理论/实际 occupancy 约 `66.67%/51%`、0 spill；force-only 为 72 registers、理论/实际 occupancy 约 `58.33%/43%`、0 spill。
- 最终 source/candidate 的两个 specialization normalized SASS exact：full 为 1472 个 encoding，force-only 为 2944 个 encoding；launch 与资源一致。final full 两次 duration 为 `667.97/695.81 us`，均值 `681.89 us`，相对 source 均值 `680.88 us` 为 `+0.15%`；force-only 候选均值约 `271.50 us`，相对 source `270.38 us` 为 `+0.41%`。
- 对 legacy 单 kernel，clustered full 约加速 `6.6×`，force-only 约加速 `16.5×`。最终 full executed instructions 为 `99,462,397`，force-only 为约 `77.91M`；两者没有 spill、launch 形态或 occupancy 上限退化。

### PETN throughput、replay 与 production

- PETN 16240 NVE 使用现有 throughput fixture，2000 steps、parent/current 各三次并交错执行。parent elapsed 为 `28.262/28.424/28.380 s`，current 为 `19.324/19.220/19.233 s`；median steps/s 为 `70.472 → 103.987`，median ns/day 为 `0.608878 → 0.898445`，提升 `47.557%`。
- replay 为 3 systems × 2 output modes × 2 implementations × 3 runs，共 36/36 valid。paired median kernel-time delta：wat160k force-only `+0.158%`、full `-0.209%`；wat600k force-only `-0.280%`、full `+0.061%`；DNA force-only `-1.251%`、full `+0.467%`。
- production 为 3 systems × NVT/NPT × 2 implementations × 3 runs，每次 10000 steps，共 36/36 valid，pre/post idle SM 均为 `3%`。median ns/day（父版本 → 候选）：wat160k NVT `91.097 → 91.326`、NPT `84.020 → 84.435`；wat600k NVT `26.416 → 26.309`、NPT `24.017 → 24.056`；DNA NVT `356.530 → 359.714`、NPT `341.395 → 343.904`。
- 对应 throughput delta 为 wat160k NVT `+0.251%`、NPT `+0.494%`；wat600k NVT `-0.405%`、NPT `+0.162%`；DNA NVT `+0.893%`、NPT `+0.735%`。官方 replay + production migration gate 在 3% 合取门限下通过；runner、gate、NCU/SASS 与性能产物均未进入提交，只保留在 `.tmp`。
