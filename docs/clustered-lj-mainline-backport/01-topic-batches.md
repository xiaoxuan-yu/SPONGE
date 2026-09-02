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

B2.2 实施结果：

- 从 `19856de` 恢复 provider lifecycle、endpoint incidence、pair shift 与完整 builder 源码树；regular/soft LJ、gather、Manager 和 worker 均未进入本批。
- 新增 `SPONGE_clustered_neighbor_sources.cmake`，只管理中立 neighbor foundation 及六个职责明确的 object target；主线原有 `SPONGE.cmake` 的 `main.cpp`、H5/toml/jit/HighFive/HDF5 source 与链接结构保持。
- 主线 H5 使 builder 的 `MD_core.h` 间接依赖 `hdf5.h`；因此 object target 显式继承 HighFive/HDF5 usage requirements，不通过全局 include 绕过依赖。
- CPU builder 需要 CUDA 风格公共 primitive 的 host shim；只恢复 `cpu_api.h` 的 builtin dimension、mask/shuffle/ballot、bit-cast、atomic declaration 与 `warpSize=1`。旧 CPU 生产调用均在 GPU guard 内或在 CPU 分支忽略 warp 参数，审计未发现行为变化。
- `dev-cpu` 与 `dev-cuda13`/SM89 均以 24 并行完整构建 SPONGE 和 contract target；contract 各 1/1 通过。
- CPU tests 为 28 pass、1 fail、4 skip；唯一失败是 tracked H5 manifest 写死 `/home/youmans/sidereus/SPONGE`，精确父版本 `634a797` 同项同因失败，分类为既有 fixture 污染。
- SM89 SASS：父 434、候选 797；434 个 common function 全部 exact，0 changed、0 removed，只新增 363 个未接入主循环的 builder/Cornerstone function。原有 LJ、PME、neighbor 代表 kernel hash 全相同。
- 本批尚无可达 builder launch，故 NCU 没有采样对象，migration A/B 也尚无 clustered route；B3 接入 Provider/LJ 生命周期时首次执行 NCU、replay 与 production A/B。

### B3：regular LJ、soft-LJ 与主生命周期

- clustered workspace/gather；
- regular LJ force-only/full；
- standalone soft-LJ force-only/full；
- main.cpp 中唯一 Provider/workspace ownership、build/gather/clear；
- pair shift、PME direct 与 output semantics。

冲突处理必须保留 upstream H5 input/restart/materialization。不得恢复 native/legacy LJ、solvent fast path或 virial-only variant。

B3 实施结果：

- 从 `19856de` 恢复 clustered workspace/gather、regular LJ、standalone soft-LJ 与 warp-record kernel；默认 binary 只保留 force-only/full 两种 output 语义，删除 solvent fast path，不恢复 virial-only specialization。
- `main.cpp` 保留 upstream H5 input/restart/finalize 与 force-field materialization，建立唯一 Provider/workspace owner；纯 clustered-LJ 体系不再初始化第二套 legacy neighbor list。legacy list 暂只服务尚未进入 B4/B5 的消费者，不作为 LJ fallback。
- PBC 大位移检查读取实际活动邻居表的 effective rebuild skin。pure clustered 路径使用 Provider skin；仍有 legacy consumer 时继续使用 legacy skin，避免两套安全半径分叉。
- 每步 force reset 显式清除 `need_pressure`，保证 force-only/full dispatch 不受上一步 barostat/output 请求污染。
- 生产 A/B 首轮定位出主线既有 `PME_Final` launch-grid 错误：kernel 以 `threadIdx.y` 映射原子，却按 `blockSize.x` 计算 grid，形成 16 倍过度 launch。该独立 prerequisite 修复为按 `blockSize.y` 计算；NCU 从 `(82272,1,1)`/约 `471 us` 恢复到 reference 的 `(5142,1,1)`/约 `332 us`。
- CPU 与 SM89 SPONGE/contract 构建通过；pair oracle 6/6 exact，soft-LJ oracle 通过；source/candidate 的 14 个 B3 target 实例 normalized SASS body/resource exact。
- NCU 对 regular force-only/full 的资源与主要指标无不可解释差异；修复后的 `PME_Final` 同 launch 为 candidate `331.744 us`、reference `331.968 us`。
- replay 36/36、full 18/18 matched；六格 median throughput delta 为 `-0.51%` 到 `+0.61%`。
- production 36/36；paired median 为 wat160k NVT `+0.25%`、NPT `+0.11%`，wat600k NVT `+0.55%`、NPT `-0.06%`，DNA NVT `-0.69%`、NPT `+0.61%`，全部通过 3% gate。

### B4：Selective Interaction、SITS 与 REST2

- 新 Selective_Interaction 目录；
- sparse product ownership；
- regular SITS、REST2 correction、soft-core SITS 三个物理模式；
- main force dispatch 与 selective policy façade。

旧 SPONGE/SITS 中 upstream H5 状态逻辑需要迁入新 owner，而不是保留两套 SITS。

B4 实施结果：

- 新建唯一的 `Selective_Interaction` owner/façade，合并 regular SITS、REST2 correction 与 soft-core SITS；删除旧 `SPONGE/SITS` owner，不引入 Manager 类型、worker protocol 或 Manager 构建依赖。
- sparse clustered product 的构建、stamp、失效与释放由 façade 统一管理；GPU 与两条 CPU 路径均保留，主循环只执行一次 selective dispatch。
- upstream H5 `nk` restart 优先级、atom selection、runtime restart 导入导出与 trajectory pending append 迁入新 owner，H5 runtime closure 通过。
- clustered contract 明确允许 `sci_count == cj_count == 0` 的合法空 pair list；regular/soft-LJ GPU 对空表执行 no-op，同时继续拒绝 partial payload。selective dispatch 额外要求实际 LJ operator 可用，避免仅配置 SITS 时误入 direct LJ 路径。
- CPU/SM89 SPONGE 与 contract target 构建通过；CPU/CUDA contract、standalone SITS façade/reproducibility 及 H5 runtime closure 均通过。
- source/candidate 的 sparse builder、regular direct、soft direct 三个代表 kernel normalized SASS 完全一致：分别为 2424、544、752 条指令，register/shared/spill 资源相同。NCU 中 builder 约 `101.54 → 101.79 us`，regular `10.24 → 10.82 us`，soft `17.95 → 15.97 us`，无 spill 或 launch 形态变化。
- replay 36/36 valid，六格 paired delta 为 `-2.115%` 到 `+0.251%`；production 36/36 valid，六格 paired speed delta 为 `-0.312%` 到 `+0.294%`，组合 migration gate 通过。

### B5：custom 与 manybody consumers

- custom pair；
- EAM；
- SW/EDIP/Tersoff；
- ReaxFF bond-order、EEQ、VDW、hydrogen bond；
- CPU clustered paths、center lists 与 lookup primitive。

约束：保留 upstream native/H5 initialization；只替换邻居消费算法，不合并不同算子的数学循环。

B5.1 实施结果（SW/EDIP/Tersoff）：

- SW、EDIP 与 Tersoff 不再申请 legacy full neighbor list；三者从主循环中的唯一 `ClusteredNeighborProvider` 获取 all-local gmxpacked view，EAM、custom 与 ReaxFF 仍停留在 legacy 边界，留给后续 B5 子批。
- upstream H5/native 初始化保持；三种势均显式限制为 PBC、单 PP rank，避免在尚无 typed halo 与 directed-edge ownership 时产生隐式错误。
- GPU 构建按 per-pair shift/active mask 生成 directed center relation，CPU 使用对应 gmxpacked relation；最终 SW/EDIP/Tersoff 数学循环保持独立，不引入通用 evaluator 或第二套 provider。
- 新增 CPU gmxpacked traversal helper 与 EDIP/Tersoff manybody oracle；oracle target 可独立配置，不依赖 `SPONGE` target 偶然定义的库。
- CPU/CUDA SPONGE、contract 与 manybody oracle 构建/测试通过；SW LAMMPS diamond fixture 通过。source/candidate 的 SW 与 EDIP/Tersoff 目标 SASS exact，NCU launch/resource 无结构性变化。
- 相对精确父提交 `d7cdb87` 的 replay 36/36、production 36/36 通过；官方组合 migration gate 通过。完整数值见验证文档的 B5.1 检查点。

B5.2 实施结果（EAM）：

- EAM 不再申请或消费 legacy full neighbor list；主循环从唯一 `ClusteredNeighborProvider` 获取 all-local gmxpacked view，并调用独立的 CPU/GPU clustered EAM 两阶段路径。
- funcfl、setfl/alloy 与 H5 typed initialization 保持主线实现；迁移只替换 rho、embedding derivative 与 pair-force 的邻居遍历，保留 force-only/full 两种输出语义，不增加 virial-only 变体、fallback 或 evaluator。
- CPU/GPU 的 Cu funcfl 与 Cu-Ni alloy LAMMPS 对照各 6/6 通过；EAM 目标 kernel 的 source/candidate normalized SASS、launch 与资源一致。完整 NCU 数值见验证文档的 B5.2 检查点。
- 修复 builder 在合法空 pair list 上遗留 `sci>0、cj=0` 未发布状态的问题；CPU payload、host compact 与 device compact 现在都以 all-valid 或 all-zero 原子计数收口，consumer contract 仍拒绝 partial payload。
- 相对精确父提交 `722dddc` 的 replay 36/36、production 36/36 通过；官方组合 migration gate 通过。

B5.3 实施结果（custom pair）：

- custom pair 不再申请或消费 legacy full neighbor list；主循环从唯一 `ClusteredNeighborProvider` 获取 all-local gmxpacked view，CPU/GPU 都遍历同一 clustered pair product。
- 保留主线 native/H5 初始化与运行时 JIT 势函数，只把 JIT launch contract 收口为 clustered force-only/full 两种变体；不增加 virial-only、legacy fallback、probe、gate 或通用 evaluator。
- sorted id/coordinate/charge/type 由 custom-pair owner 的复用缓冲区提供；exclusion mask、local/ghost 权重、能量、virial 与 force 的语义保持。合法空 pair list 为 no-op，partial payload 继续拒绝。
- 为验证运行时 JIT 路径，加入独立 custom-pair oracle 和 clustered snapshot producer；SPONGE 的 runtime source 清单抽为单一 CMake owner，producer 与主程序共享该清单而不复制生产源码枚举。
- CPU/CUDA SPONGE、microbench、snapshot producer 与 clustered contract 构建/测试通过；三组 Morse/LAMMPS 对照、500707-pair canonical oracle 和单原子空表均通过。完整 NCU 与 A/B 数值见验证文档的 B5.3 检查点。
- 相对精确父提交 `314184a` 的 replay 36/36、production 36/36 通过；官方组合 migration gate 通过。

B5.4 实施结果（ReaxFF VDW）：

- ReaxFF VDW 不再消费 legacy half list；主循环为 VDW 从唯一 `ClusteredNeighborProvider` 获取 all-local gmxpacked view。EEQ、bond-order、bond、angle/torsion 与 hydrogen-bond 仍保留原邻居表，避免在一个提交中混合迁移。
- GPU 直接遍历 8-way packed partitions，并使用 per-pair shift/active mask、exclusion 与 subgroup endpoint reduction；CPU 使用对应 clustered SCI traversal。实现只有 force-only/full 两种 specialization，没有 virial-only、fallback、probe 或 evaluator。
- VDW 初始化不再单独请求 legacy full list；bond 等未迁移 ReaxFF consumers 仍维持 legacy list owner。PBC/single-PP-rank 约束在初始化时 fail-fast，合法空 payload 在 coordinate gather 前直接收口。
- CPU/CUDA PETN/LAMMPS 对照、clustered contract 与 manybody oracle 通过；source/candidate 最终 VDW kernel normalized SASS exact，NCU 无 launch、register、occupancy 或 spill 回退。完整数值见验证文档的 B5.4 检查点。
- PETN 16240 NVE 2000-step、三次交错 A/B 的 median throughput 为 `0.609 → 0.898 ns/day`，提升 `47.56%`；相对精确父提交 `49298b8` 的 replay 36/36、production 36/36 与官方组合 migration gate 均通过。

B5.5 实施结果（ReaxFF EEQ）：

- EEQ 不再读取 legacy full-neighbor `ATOM_GROUP`。CPU/GPU 都从同一 all-local gmxpacked view 构建对称 H-matrix CSR，后续 CG matrix-vector 与 EEQ force 复用该 CSR；其他 ReaxFF consumers 仍保留 legacy list，避免混批。
- GPU count/fill 使用 8-way packed partitions、per-pair image shift、active mask 与 exclusion；CPU 使用 SCI shift。主线 charge capture 与 native/H5 参数初始化保持，只补齐 clustered coordinate/fill scratch；合法空 SCI payload跳过零尺寸 launch。
- CPU/CUDA 的三组 EEQ water/LAMMPS charge 对照和 PETN 全 ReaxFF 单帧对照均通过；clustered contract 与 manybody oracle 在 CPU/CUDA 各通过。候选 binary 已无旧 EEQ Count/Fill kernel，clustered H 两个模板实例均为 48 registers、0 local spill。
- PETN source-first NCU 中，EEQ gather + count + fill + force 总时延由 `1.537–1.542 ms` 降至 `1.263–1.270 ms`，约减少 `17.4–18.1%`；PETN 2000-step 三次交错 A/B 为 `0.9005 → 0.9136 ns/day`，提升 `1.45%`。
- 相对精确父提交 `580dcb11` 的 replay 36/36、production 36/36 通过；官方组合 migration gate 通过。

B5.6 实施结果（ReaxFF bond-order/bond）：

- bond-order 不再扫描 legacy `ATOM_GROUP`；CPU/GPU 都直接遍历同一 all-local gmxpacked payload，生成 canonical bond pair、原始 bond order 与距离，后续 correction、CSR、angle/torsion/over-under 和 force projection 继续复用这份稀疏结果。
- bond energy 不再重新扫描 full neighbor list 并对 CSR 做逐邻居 lookup，而是每个 canonical bond 直接计算一次能量和三组 bond-order 导数；删除 bond owner 的 full-list 请求及未使用的坐标、force、virial、CSR 参数。
- GPU CSR prefix 从单线程串行 kernel 改为 device exclusive scan；CPU 保留原顺序实现。angle/torsion 同步删除已经无效的 `ATOM_GROUP` 裸参数，但不合并各自数学 kernel。
- legacy/raw BO、legacy bond-force 与 GPU 串行 prefix kernel 均已从候选 binary 消失；full neighbor list 当前只由尚未迁移的 hydrogen-bond 请求和消费。
- CPU/CUDA PETN/LAMMPS 单帧对照、clustered contract 与 manybody oracle 通过；source-first NCU 显示 raw BO `969.86 → 179.30 us`、bond energy/derivative `567.36 → 71.10 us`，新 kernel 均无 spill。完整数值见验证文档的 B5.6 检查点。
- PETN 16240 NVE 2000-step、三次交错 A/B 的 median throughput 为 `0.9134 → 1.1329 ns/day`，提升 `24.01%`；相对精确父提交 `2d772b9a` 的 replay 36/36、production 36/36 均通过 3% gate。

B5 后续保持小批提交：下一批迁移 ReaxFF hydrogen-bond，消除最后一个 legacy full-list consumer；angle/torsion 已直接消费 bond-order CSR，不需要重新扫描 spatial neighbor list。每个 device 子批单独执行 source-first NCU、SASS 与完整 A/B gate。

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
