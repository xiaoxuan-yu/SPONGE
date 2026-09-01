# 主线冲突图与解决规则

## 1. 总体分类

### Source-only，可按最终职责迁入

- SPONGE/neighbor_list；
- SPONGE/Selective_Interaction；
- clustered LJ workspace/gather；
- tools/nbnxm_microbench；
- clustered benchmark、contract 和 manybody oracle。

这些目录仍需检查 include/CMake 与 upstream 文件的连接点，但其内部语义以 19856de 为准。

### Upstream-only，默认保留

- SPONGE/utils/h5md；
- tests/h5_bundle；
- benchmarks/bundled_io；
- schema/install/test scaffolding；
- upstream 后续新增的 ReaxFF native initialization 和 H5 lifecycle。

### 必须三方融合

- SPONGE/main.cpp；
- cmake/targets/SPONGE.cmake；
- SPONGE/MD_core/MD_core.cpp；
- SPONGE/control.cpp、control.h；
- SPONGE/Lennard_Jones_force/LJ_soft_core.cpp；
- SPONGE/custom_force/pairwise_force.cpp、pairwise_force.h；
- SPONGE/barostat/MC_barostat.cpp、MC_barostat.h；
- SPONGE/manybody/eam、edip、sw、tersoff、reaxff；
- SPONGE/xponge/load/gromacs.hpp；
- pixi.toml、pixi.lock。

## 2. 关键规则

### main.cpp

保留 upstream：

- H5 input plan validation/materialization；
- topology/protocol sidecar；
- dynamic restart/state read-write；
- upstream warning与input生命周期。

加入 clustered：

- 唯一 ClusteredNeighborProvider；
- 唯一 LJClusteredWorkspace；
- config/domain binding；
- build/gather/force/clear 顺序；
- SITS/REST2/custom/manybody clustered dispatch。

不加入 Manager worker-mode、schedule或transport入口。

### CMake

保留 upstream：

- HighFive/HDF5；
- JIT header target；
- ReaxFF native init；
- 当前 install/test选项。

加入 clustered：

- provider/builder/contract/gather source；
- candidate/payload/record-stream object target；
- microbench/snapshot/oracle target；
- C++/CUDA/HIP 20 与 Cornerstone include。

不创建 SPONGE_MANAGER target，不列 worker protocol/Manager source。

### SITS

最终只能有新 Selective_Interaction/SITS owner。Upstream 旧 SPONGE/SITS 中仍有效的 H5 input/restart字段与初始化逻辑迁入新 owner，然后删除旧生产实现和 CMake entry。

### manybody/ReaxFF

clustered traversal 与 upstream native/H5 initialization 正交保留。任何以“选 source 整文件”解决冲突的做法都不合格。

## 3. Manager 泄漏检查

每批暂存前检查相对 b78aeaba 的新增路径是否命中：

    ^SPONGE/manager/
    ^SPONGE/worker_protocol/
    ^cmake/targets/SPONGE_MANAGER.cmake$
    ^benchmarks/.*/remd/

并在 CMakeLists、cmake 与 SPONGE 中检索 SPONGE_MANAGER、manager_main、worker_protocol 和 sponge::manager。任何新命中都必须删除或逐项解释，不能用 compatibility façade 掩盖。

## 4. Legacy 泄漏检查

最终不得重新出现：

- native/legacy LJ production dispatch；
- solvent LJ fast path；
- virial-only clustered specialization；
- 旧 ATOM_GROUP neighbor consumer；
- 旧 shared clustered cache façade；
- 新旧 SITS 双 owner；
- Manager-only runtime gate。
