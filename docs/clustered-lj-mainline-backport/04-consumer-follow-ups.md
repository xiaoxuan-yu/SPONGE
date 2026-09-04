# Clustered consumer 后续问题与 primitive 审查

- 日期：2026-09-03
- 范围：`backport/clustered-lj-mainline` 最终 consumer 状态
- 性质：后续问题记录、设计边界与实施日志；不改变 B0–B7 backport 验收结论
- 状态：C1、C2、C3、C4 已完成并通过独立验收

## 0. 结论

当前 regular/soft LJ、SITS、custom pair、EAM、SW、EDIP、Tersoff 与 ReaxFF 已不再消费 legacy spatial neighbor list。后续工作不是继续删除 legacy fallback，而是收敛 clustered consumer 内重复的 pair-layout 解码、CPU traversal 和 derived-CSR 生命周期。

建议保持以下架构边界：

```text
neighbor_list/contract
    public layout、mask、shift、ownership、cursor primitive
                         |
                         v
consumer-side traversal adapter
    只枚举/解码 pair，不包含势函数
                         |
             +-----------+-----------+
             |                       |
             v                       v
direct pair consumer          derived-product consumer
LJ/EAM/VDW/SITS/...           SW/EDIP/Tersoff/EEQ/BO/HB
                                     |
                                     v
                         model-specific math kernel
```

优先抽取小而稳定的位布局 primitive；EDIP/Tersoff 这类高度同构的路径也先只共享
storage/scan/stamp，再以 profile 决定是否共享 builder enumerator。不要为了减少源码
行数建立跨模型 evaluator。

### 0.1 当前推进状态

截至 2026-09-03，C1 已完成以下源码工作：

- 在 `neighbor_list/contract/traversal.cuh` 增加
  `Clustered_Gmxpacked_Effective_Imask()` 与
  `Clustered_Gmxpacked_I_Entry_Is_Active()`；
- 将 EAM、ReaxFF VDW、bond-order、EEQ、EDIP、Tersoff、soft-LJ、SITS
  的主要 GPU pair 解码迁移到公共 helper；
- regular LJ 只复用 effective-imask helper，保留其 `SCI_SHIFT_ONLY`、
  nullable metadata、动态 cluster stride 和展开方式；
- soft-LJ 与 SITS 的 CPU 路径复用 effective-imask helper；
- 为 split、exclusion、active-I 解码和 CPU pair-mask orientation 增加 host
  contract cases。

本批次已通过 CPU 与 CUDA 编译、`ClusteredSpatialViewContract`、
`ManybodyClusteredOracle` 和独立 CUDA/host primitive parity。精确父提交的 `sm_89`
对照显示 937 个 device function 的资源记录完全一致；41 个目标 kernel 的 normalized
SASS body 并非全部等价，因此又对 regular LJ force-only/full-output 与 EAM rho/force
执行 NCU。代表实例没有新增 register、spill、global-memory transaction 或 occupancy
限制，regular LJ 的单次 NCU duration 漂移为 `+0.13%`/`+0.14%`，五组交错
force-only timing 的均值漂移为 `+0.47%`，低于 3% 门槛。C1 据此验收完成；这些数据
只证明此次抽取没有可测的性能退化，不证明 helper 本身带来性能收益。

C2 已完成 CPU traversal contract 与目标 consumer 的源码迁移：

- CPU pair-mask 与 iterator 的权威实现已迁入
  `neighbor_list/contract/cpu_traversal.h`；旧的
  `manybody/clustered_gmxpacked_cpu.h` 只保留 forwarding include；
- EAM、ReaxFF VDW/bond-order/EEQ/HB 使用 J-major pair iterator；
- SW、EDIP、Tersoff 使用 I-tile builder iterator；
- regular/soft LJ 与 SITS 使用 local-I/J-tile iterator；
- visitor 仍拥有 cutoff、自相互作用、pair ownership、模型参数、累加器和
  OpenMP scheduling，因此迁移没有建立跨模型 evaluator；
- contract test 已覆盖空 SCI、partial exclusion、local/ghost、self/reverse
  ownership、pair shift、动态 cluster stride 与三种遍历顺序；CPU/CUDA
  build 和两个 clustered oracle 已通过，EAM 父/candidate CPU 输出逐字节一致。

C2 的 soft-LJ、SITS、ReaxFF 扩大 CPU 场景与最终 CUDA 复测均已通过，批次已独立
验收。

C3 已新增 consumer-side `ClusteredPayloadStamp` 与 `ClusteredCSRStorage`，并把
SW、EDIP、Tersoff 的重复 raw pointer/capacity 生命周期迁入该公共层；GPU 路径的
全量 counts D2H、host prefix-sum、offsets H2D 已替换为 CUB/hipCUB device exclusive
scan，只保留 item-count 标量 D2H。CPU 路径与各模型的 count/fill 枚举和 cutoff
predicate 保持不变。CPU/CUDA 构建、clustered contract/manybody oracle，以及
SW/EDIP/Tersoff 的 9 组 LAMMPS 对照均已通过。ReaxFF 的复用评估也已完成，结论是
保留其现有 device scan 和 typed parallel arrays。device scan 现以 64 位总和检测
`INT_MAX` overflow，并拒绝任一负 count；zero/sparse/tail/negative/overflow 边界、
normalized SASS/resource、NCU 与 20 次强制 rebuild 时间线均已闭合。C3 据此独立
验收完成。C4 作为独立批次执行，没有与已闭合的 traversal/CSR 迁移混合。

C4 已完成 profile-driven 评估，没有保留新的 force-kernel 源码改动。160k-water
NVT 的 2,000-step 时间线中，clustered payload 只在初始化及第
`533/1032/1434/1839` 次 rebuild check 后构建，稳态间隔中位数为 452 steps；因此
derived CSR 的构建成本在代表轨迹中高度摊销。SW 同输入下把 cache capacity 从 64
临时降到 8、强制走 center-cursor fallback 后，force kernel 从 `101.824 us` 退化到
`6.689 ms`，不支持删除 CSR/cache。Tersoff 双 `k` 扫描当前不是 DRAM-bound，ReaxFF
HB 的 16-lane 方案也已优于已测的 warp、8-lane 与单线程方案；两者均不再改写。
最终 replay 36/36、production 36/36 与官方 3% migration gate 全部通过。

## 0.2 当前 builder、provider 与 consumer 架构

生产数据流为：

```text
neighbor_list facade
    |
    v
ClusteredNeighborProvider
    |-- lifecycle/config/domain binding
    |-- generation、lease、rebuild cache
    |
    v
builder
    geometry -> candidate -> record stream -> compact gmxpacked payload
       CPU: cpu_builder.cpp
       GPU: spatial/candidate/payload/active-refresh + detail kernels
    |
    +--> provider-derived metadata
    |       pair-shift metadata
    |       endpoint incidence（按 consumer requirement 延迟生成）
    v
CLUSTERED_SPATIAL_VIEW + requirements validator
    |
    +--> direct consumers
    |       regular/soft LJ、SITS、EAM、ReaxFF VDW、custom pair
    |
    +--> derived-product consumers
            SW/EDIP/Tersoff center CSR
            ReaxFF bond CSR、EEQ H-matrix CSR、HB endpoint incidence
```

各层职责如下：

- `neighbor_list/neighbor_list.*` 是主程序 facade，不暴露 builder 内部 storage；
- `neighbor_list/provider/` 持有唯一空间布局与 pair payload owner，负责 build/reuse、
  generation/lease、view 发布，以及按能力需求准备 pair shift 和 endpoint incidence；
- `neighbor_list/builder/` 只生成空间布局、candidate、record stream 与 compact
  gmxpacked payload；CPU/GPU backend 在内部收口，不向 consumer 暴露两套 API；
- `neighbor_list/contract/` 定义 payload 类型、view validation、GPU 位布局 primitive
  和 CPU traversal adapter；它不拥有模型参数或势函数逻辑；
- consumer 只通过 `AcquireView(requirements)` 消费不可变 view。需要模型私有邻接关系时，
  以 `provider_incarnation + payload_generation` 为 stamp 派生并缓存，而不把 CSR
  放回 provider。

这使 builder 与 consumer 的依赖方向保持单向：consumer 可以派生数据，但不能反向改变
provider payload，也不能重新引入第二个空间邻居表 owner。

## 0.3 Legacy 路径残留结论

目标 consumer 中已经没有待拆的 legacy spatial-neighbor fallback。审查覆盖
regular/soft LJ、SITS、custom pair、EAM、SW、EDIP、Tersoff 与 ReaxFF
VDW/bond-order/EEQ/HB；它们的运行时 pair source 均为 `CLUSTERED_SPATIAL_VIEW`。

仍可见但不属于待删除 legacy neighbor 路径的内容有：

- regular LJ 的“falling back to regular gmxpacked”是从 clustered fast
  specialization 回退到 clustered 通用 kernel，不是 native/legacy neighbor list；
- `EDIP_Force_With_Full_Neighbor_*` 等 `Full_Neighbor` 名称指 clustered payload
  派生的模型私有 center CSR，名称陈旧但数据源不是 legacy；
- `xponge/load/native/*`、`NativeEAMDefinition`、ReaxFF native H5 与
  `write_legacy_file` 属于输入/输出兼容，不是运行时 neighbor owner；
- custom pair/JIT 保留手写 packed traversal 与插件 ABI 所需 half-list 兼容边界；
  它是 ABI 特例，不能直接套用静态 C++ visitor；
- GPU center-cursor 和 endpoint-incidence 是 clustered contract 的派生访问方式，
  不是另一套邻居表 provider。

因此下一步应处理重复解码、派生 CSR 生命周期与性能，而不是继续按字符串搜索删除
`legacy`、`native` 或 `Full_Neighbor`。

## 1. 多体 kernel 与 center CSR 的性能问题

### 问题

SW、EDIP 和 Tersoff 目前把 gmxpacked pair payload 派生为按中心原子组织的 directed CSR，再由模型 kernel 遍历 CSR。需要确认以下两类改造是否能带来端到端收益：

1. 保留 CSR，但消除 count/fill 之间的 host prefix-sum 和 D2H/H2D 同步；
2. 不再物化 CSR，改为直接遍历 gmxpacked 或 endpoint-incidence，并在模型 kernel 内聚合中心邻居。

### 当前事实

- SW、EDIP、Tersoff 的 CSR 只在 provider incarnation 或 payload generation 改变时重建；其余 step 复用已有关系。
- C3 之前，三者的 GPU builder 均采用 count kernel、counts D2H、host exclusive
  scan、offsets H2D、fill kernel；当前工作树已改为 count kernel、device exclusive
  scan、单个 item-count 标量 D2H、fill kernel。count/fill 仍会重复解码 gmxpacked，
  这是本批刻意保留的遍历边界。
- SW/EDIP/Tersoff 的三体主循环约为 `O(sum_i degree(i)^2)`；CSR 提供连续的 center row，避免把 exclusion、active mask、pair shift 和 endpoint 聚合带入二次循环。
- SW 还存在超出 cached-neighbor capacity 时的 center-cursor fallback，可作为 direct-vs-CSR 原型的现成对照入口。
- 最终 backport 的小型 oracle NCU 数据为：SW builder `265.31 + 268.83 us`、cached full `102.30 us`；EDIP builder `4.864 + 4.960 us`、force `3.968 us`；Tersoff builder `4.800 + 4.928 us`、force `6.240 us`。这些数字只说明重建阶段的成本结构，不代表生产 workload 的摊销收益。

### 判断

- 不应把“删除 center CSR”视为机械清理。直接 gmxpacked traversal 只有在 payload 高频变化、邻居度较低，或能够以 shared/register tile 保持 endpoint 聚合和邻居复用时才可能更快。
- 第一优先级是 device-side scan 和 CSR builder 同步消除。这比重写三体数学 kernel 风险低，也不改变 directed cutoff、周期 image 和三元组 identity。
- Tersoff 每个 `j` 对同一 row 分别为 zeta 和导数扫描两次 `k`，存在几何和参数复用机会；但必须监测 register、spill、occupancy 和原子写入。
- EDIP 的 `Get_Z -> force/dE_dz -> Redistribute_Z` 存在跨 kernel 数据依赖，不宜在没有 NCU 证据时强行融合。
- ReaxFF hydrogen-bond 最终 clustered 主核约 `440.54 us`，加 map 后约 `442.49 us`，相对旧 `444.38 us` 仅改善约 `0.43%`，并且 achieved occupancy 约 `29%`；这是比 EAM、VDW 或 bond-order 更值得继续 profile 的多体路径。

### 实施前置证据

任何重写必须先记录代表 workload 的：

- payload generation 变化次数和 CSR rebuild interval；
- count、scan/sync、fill、force 各自耗时；
- neighbor degree 分布、CSR 容量和 center-cursor fallback 比例；
- registers、spill、achieved occupancy、warp stall、atomic throughput 和 memory sectors；
- 每步总耗时与生产 throughput，而不仅是单 kernel duration。

若没有上述证据，不启动 direct-gmxpacked 三体 kernel 重写。

## 2. Consumer 重复 primitive 审查

### 重复矩阵

| Consumer | GPU pair 解码 | CPU pair 解码 | derived product | 当前特殊边界 |
|---|---|---|---|---|
| regular LJ | effective mask 已共享；active 判定仍特化 | local-I/J-tile iterator | 无 | dense layout、work partition、SCI shift |
| soft-LJ | effective mask 已共享；active 判定仍特化 | local-I/J-tile iterator | 无 | soft 坐标、soft 参数、SCI shift + min-image |
| SITS | effective/active mask 已共享 | local-I/J-tile iterator | sparse gmxpacked view | selection、REST2、per-pair shift、enhancing 输出 |
| custom pair | JIT 参数化 | JIT 参数化 | sorted gather | 用户参数与 JIT ABI |
| EAM | rho/force 均已共享 | J-major pair iterator | rho、`df/drho` | 两阶段 endpoint accumulation |
| ReaxFF VDW | effective/active mask 已共享 | J-major pair iterator | 无 | Reax ownership 与输出归约 |
| ReaxFF bond-order | effective/active mask 已共享 | J-major pair iterator | canonical pair + bond CSR | bond index 与 BO 导数 |
| ReaxFF EEQ | GPU count/fill 已共享 | J-major pair iterator | H-matrix CSR | 对称矩阵与 CG 复用 |
| ReaxFF HB | 使用 center cursor | J-major pair iterator | endpoint incidence | 16-lane/H 与 bond lookup |
| SW | 使用 center cursor/CSR | I-tile iterator | sorted-center CSR | cached-center capacity |
| EDIP | effective/active mask 已共享 | I-tile iterator | atom-id directed CSR | 非对称 cutoff、`z/dE_dz` |
| Tersoff | effective/active mask 已共享 | I-tile iterator | atom-id directed CSR | 非对称 cutoff、双 `k` 扫描 |

表中的“重复”只表示 clustered layout 解码重复，不表示物理公式可以合并。

### P1：split、exclusion 和 active-mask 解码

这是最明确、最适合首先抽取的重复，也是当前已经实施的 C1 主体。

EAM、ReaxFF VDW、bond-order、EEQ、EDIP、Tersoff、soft-LJ 和 SITS 的 GPU kernel 都重复以下步骤：

1. 从 `j_lane` 计算 split 和 split-local lane；
2. 读取 `packed.split[split]`；
3. 按 `exclusion_index` 读取 pair word；
4. 计算 `effective_mask = imask & pair_bits`；
5. 计算 `(jm, i_local)` 对应的 packed bit；
6. 合并 `Clustered_Get_Pair_Active_I_Mask()`；
7. 从 pair-shift word 解码 image shift。

`neighbor_list/contract/traversal.cuh` 原先已经提供 lane、shift、active-mask 和 ownership primitive；本批在同一层补充了 split/exclusion/effective-mask 解码。新增接口保持为纯 `__host__ __device__ __forceinline__` 标量 primitive，不包含 cutoff、势函数或输出逻辑。

优先避免创建完整通用 evaluator；目标是消除位布局和 exclusion indexing 的重复，而不是统一所有 pair kernel。

已增加两个标量 primitive，没有引入大型 context 对象：

```cpp
__host__ __device__ __forceinline__ unsigned int
Clustered_Gmxpacked_Effective_Imask(
    const CLUSTERED_GMXPACKED_CJ& packed,
    const CLUSTERED_GMXPACKED_EXCLUSION* exclusions,
    int split, int split_j_lane, int i_lane,
    int i_cluster_size = kClusteredClusterSize);

__host__ __device__ __forceinline__ bool
Clustered_Gmxpacked_I_Entry_Is_Active(
    unsigned int effective_imask, uint64_t pair_shift_bits,
    int split, int jm, int i_local);
```

第一个 primitive 只负责 `imask & exclusion-word`；可选的
`i_cluster_size` 参数保留 regular LJ 动态 stride 的布局语义。第二个负责 packed bit 与
active-I mask。pair shift ID 继续使用现有 `Clustered_Get_Pair_Shift_Id()`，避免一个
helper 同时承担过多职责。

实际归属：`SPONGE/neighbor_list/contract/traversal.cuh`。它们描述公开 payload 的位布局，不属于任何物理模型，也不应放回 LJ 或 manybody 目录。

本批次没有强制所有 consumer 同时使用第二个 helper：regular LJ 还要组合
`SCI_SHIFT_ONLY`、nullable pair-shift metadata、动态 stride 和原有展开；soft-LJ 还要
组合独立的 `active_i_mask`。这些路径只抽取完全等价的 effective-mask 部分，避免用
“统一”掩盖语义差异。

GPU helper 不应：

- 持有 view 或复制整个 SCI/CJ 对象；
- 隐式读取 global memory；
- 内置 cutoff、自相互作用或 local/ghost ownership；
- 改变现有 loop nesting 或 work partition。

这些约束用于降低 register live range 和 SASS 漂移风险。

### P1：复用现有 CPU pair-mask helper

`Clustered_Gmxpacked_CPU_Pair_Mask()` 已迁至
`neighbor_list/contract/cpu_traversal.h`。它支持可选的动态 I-cluster stride；exclusion
索引使用实际 stride，但输出继续保持固定 8×8 lane-mask orientation。旧的
`manybody/clustered_gmxpacked_cpu.h` 仅作为兼容 include，避免一次修改所有 include
拓扑。

contract case 已覆盖两个 split、相邻 lane、partial/full exclusion、动态
`i_cluster_size=4`、local/ghost ownership、self/reverse ownership 和周期 image
metadata。目标 CPU consumer 已不再各自实现 SCI/CJ 位布局解码。

### P2：CPU SCI/CJ 遍历骨架

多个 CPU consumer 都重复：

```text
SCI -> CJ-packed -> jm -> split/lane -> mask -> shift -> atom id
```

已抽取只负责枚举和解码的模板 iterator，由 visitor 决定：

- pair ownership；
- cutoff 和参数读取；
- canonical 或 directed endpoint；
- force、energy、virial或 CSR 写入。

不要把 OpenMP、模型局部累加器或物理公式放入 iterator。

iterator 向 visitor 暴露最小 candidate record：

```text
sci_id、packed_id、jm、i_local
cluster_i/j、i/j lane、sorted_i/j、atom_i/j
pair shift ID 或 displacement 所需 shift
valid/local ownership facts
```

是否计算 displacement 仍由调用方决定。CSR builder 需要距离筛选，而部分 consumer
只需要 pair identity；iterator 不读取坐标、不计算 cutoff，也不决定 ownership。

为了保持既有浮点累加和 builder 插入顺序，最终没有强行建立一种遍历形式，而是提供三种
显式 adapter：

- `For_Each_Pair_In_SCI`：保持
  `packed -> jm -> j_lane -> i_lane -> i_local` 的 J-major 顺序，用于 EAM 与
  ReaxFF 直接 pair consumer；
- `For_Each_I_Tile_In_SCI`：保持
  `i_local -> packed -> jm` 的 builder 顺序，返回 pair mask，用于
  SW/EDIP/Tersoff；
- `For_Each_Local_I_In_SCI` + `For_Each_J_Tile_For_Local_I`：保持 I-major
  累加与每 tile 的 J flush，用于 regular/soft LJ 和 SITS。

SCI/OpenMP scheduling、begin/end-J 生命周期、模型局部 accumulator、per-pair 或
SCI-level shift 的选择均留在 consumer。归属最终确定为
`SPONGE/neighbor_list/contract/cpu_traversal.h`，因为 LJ、SITS 与 manybody 都已使用。

### P2：directed CSR storage、stamp 和 scan

EDIP 与 Tersoff 的 center-CSR builder 和 ensure 生命周期几乎同构，只有 cutoff predicate 和参数布局不同。SW 的 row key 是 sorted atom，EDIP/Tersoff 的 row key 是 atom id，因此 SW 只能共享 storage/scan/stamp，不能直接套用相同 enumerator。

当前实现拆成：

- `ClusteredCSRStorage`：counts、offsets、items、capacity 和总数；
- `ClusteredPayloadStamp`：provider incarnation 和 payload generation；
- GPU backend-specific exclusive scan；
- 模型各自保留的 packed directed-pair enumerator、cutoff predicate 和 row-key
  mapping。

ReaxFF bond-order 和 EEQ 也有 canonical-pair/双向 CSR 的 count、scan、fill，但 GPU
路径已经分别使用 `thrust::exclusive_scan`，EEQ 还用 `thrust::reduce` 取得总项数，
不存在 SW/EDIP/Tersoff 原有的全量 host prefix-sum。bond-order 维护并行的
`bond_nbr`/`bond_idx`，EEQ 维护 `jlist`/matrix value，因此当前三字段 `int items`
storage 不能无损容纳二者。C3 只记录可复用的 stamp/scan 生命周期，不为追求统一而
强行迁移这两条路径。

建议接口边界：

```cpp
struct ClusteredPayloadStamp
{
    long long provider_incarnation;
    long long payload_generation;
};

struct ClusteredCSRStorage
{
    int* counts;
    int* offsets;
    int* items;
    int counts_capacity;
    int offsets_capacity;
    int items_capacity;
    int item_count;
};
```

公共层只提供 clear/reserve、stamp match/capture、exclusive scan 和 count→offset 收口。EDIP/Tersoff 的共用 enumerator使用模板 predicate；SW 继续保留 center-cursor enumerator，只复用 storage/scan/stamp。

实际归属：`SPONGE/manybody/clustered_csr.h`。实现保持 header-only，以便 CPU/CUDA/HIP
编译单元共享 reserve/clear/stamp；GPU scan 在 `USE_GPU` 下选择 CUB 或 hipCUB。这是
consumer 派生数据，不进入 provider state；否则 provider 会重新承担模型私有邻接表
owner 的职责。

### P2：view validation 和 launch layout

`Clustered_Validate_Spatial_View()` 已覆盖 generation、payload shape、pair-shift 和 endpoint-incidence capability。ReaxFF consumer 已广泛使用；EAM 等模块仍有手写 count/pointer/layout 检查。应优先采用现有 validator，并以 requirements 表达模型差异。

多个 GPU consumer 还重复 `(sci_numbers, packed_partitions)` grid 和 `(cluster_size, cluster_size)` block 的构造。可以抽一个显式接收 partition 数的 launch-config helper，但优先级低于 traversal 和 CSR；不能把 8-way pair traversal 与 16-way center builder 隐式固定成同一个常量。

建议 launch helper 只返回 grid/block，不包装 `Launch_Device_Kernel`。这样保留每个 consumer 的 stream、shared-memory、模板实例和参数顺序，也便于比较重构前后的 launch shape。

### P3：endpoint reduction

EAM、ReaxFF VDW 和 regular LJ 都有 subgroup reduction 与 endpoint atomic accumulation。可以抽取标量/vector subgroup reduce、atomic vector add 等底层 primitive，但 ownership、energy 分配、virial 布局和 force target 不应统一。

这是低优先级项目：代码重复存在，但错误抽象容易增加寄存器压力或阻碍模板 specialization。

## 3. 明确不抽取的边界

- 不创建跨 LJ、EAM、SITS、ReaxFF 的通用势函数 evaluator。
- 不统一 force-only/full 之外的物理输出语义。
- 不把 EDIP、Tersoff、SW 的 cutoff predicate 合并；三者的参数布局、严格 `<`/`<=` 语义和 row key 不同。
- 不把 ReaxFF bond CSR、EEQ H-matrix CSR 和 hydrogen endpoint-incidence 混成单一“通用邻居表”。
- 不因名称中出现 `Full_Neighbor` 就删除 clustered 派生的模型私有 CSR。

以下重复也暂不作为抽取理由：

- `deviceMemset`、timer start/stop 等少量 host boilerplate；
- 势函数内部相似的 cutoff 或距离判断；
- energy/virial 最终写回；
- 不同模型中名称相似但 ownership 不同的 atomic add；
- CPU 与 GPU 为保持并行语义而存在的结构差异。

primitive 的成功标准不是最大限度减少代码行数，而是让 packed layout、generation 和 CSR 生命周期只有一个权威实现，同时不牺牲 kernel specialization。

## 4. 建议批次

### C1：纯解码 primitive

- [x] 在 contract test 中增加 split/exclusion/effective-mask 的 host cases；
- [x] 增加 host/device inline helper；
- [x] 迁移 EAM、VDW、bond-order、EEQ、EDIP、Tersoff 的主要 GPU 路径；
- [x] 迁移 soft-LJ、SITS，并对 regular LJ 做受限抽取；
- [x] 补齐 CPU mask helper 的 orientation 测试；
- [x] 完成精确父提交的 `sm_89` resource/SASS 对照；
- [x] 确认 register、stack、shared、local 资源记录与 launch site 不变；
- [x] 增加独立 CUDA/host primitive parity；
- [x] 用 NCU 解释代表目标 kernel 的 SASS body 漂移并完成 C1 验收。

停止条件：helper 导致额外 global load、register 增长、spill、occupancy limit 下降且没有实测吞吐收益。

### C2：CPU iterator

- [x] 抽取 SCI/CJ/lane 解码 iterator；
- [x] visitor 保留模型语义；
- [x] 迁移 regular/soft LJ、SITS、EAM、SW/EDIP/Tersoff 与 ReaxFF
  VDW/bond-order/EEQ/HB；
- [x] contract/oracle 覆盖 empty、partial exclusion、local/ghost、self、
  periodic shift、动态 stride 和 directed cutoff；
- [x] CPU/CUDA 构建与 clustered contract/manybody oracle；
- [x] EAM 精确父版本 CPU byte-exact A/B；
- [x] soft-LJ、SITS、ReaxFF 扩大 CPU 场景回放并形成 C2 独立验收记录。

迁移时每次只处理一个 consumer；不在同一提交调整 OpenMP scheduling、浮点累加顺序或 pair ownership。

### C3：CSR storage 与 device scan

- [x] 统一 EDIP/Tersoff storage 与 stamp；
- [x] SW 仅复用与 row-key 无关的 storage/stamp；
- [x] 保持原 count/fill kernel，单独将 host scan 改为 device scan；
- [x] 评估 ReaxFF bond-order/EEQ：已有 device scan，异构并行 payload 不适合直接
  套用当前 `int items` storage；
- [x] CPU/CUDA build、contract/manybody oracle 与 SW/EDIP/Tersoff 9 组 LAMMPS
  对照；
- [x] 补齐 zero-row、稀疏 counts、tail total、负 count 与 overflow 的
  device-scan 边界测试；
- [x] 使用 64 位总和并在超过 `INT_MAX` 时失败，不弱化旧 host 累加检查；
- [x] 完成目标 builder 的 normalized SASS/resource 对照并作出 C3 验收决定；
- [x] 不与三体 force kernel 重写混在本批。

第一步只抽 storage/stamp 并保持原 count/fill；第二步才把 host scan 改为 device scan。这样可以分别验证代码共享与性能变化。

### C4：profile-driven kernel 实验

- [x] 测量实际 rebuild interval；
- [x] 比较 host scan 与 device scan；
- [x] 利用 SW 已有 cursor fallback 做 direct-vs-CSR 对照；
- [x] 单独评估 Tersoff 双 `k` 扫描和 ReaxFF HB lane/control-flow；
- [x] 以端到端 throughput 决定是否保留，不引入 production probe 或双路径
  fallback；本批没有保留新的 force-kernel 改写。

## 5. 最小验证矩阵

- `ClusteredSpatialViewContract`：合法空 payload、partial payload、generation、pair-shift、endpoint incidence；
- 新增 host primitive cases：全部 27 shift、两个 split、zero/full/partial exclusion、active marker absent/present、local-local/local-ghost/self ownership；
- 新增 CUDA 与 host primitive parity；
- `ManybodyClusteredOracle`：directed relation、非对称 cutoff、periodic image identity、EDIP/Tersoff 三元组；
- ReaxFF EEQ、bond-order、HB oracle；
- NBNXM canonical pair oracle 的 force-only/full；
- 涉及 device code 时检查 normalized SASS、register、spill、occupancy、launch；
- 最终运行 replay 36/36、production 36/36 和 3% migration gate。

## 6. 审查与验收规则

每个批次必须记录：

```text
精确父提交
涉及的 consumer 和模板实例
行为保持项
新增/复用的 primitive
CPU/CUDA correctness
source/candidate SASS 和资源差异
NCU 指标及测量 workload
replay/production 结果
接受、拆分或回退决定
```

具体门槛：

- 合法空 pair list 仍为 no-op，partial payload 仍被拒绝；
- pair ownership、exclusion、非对称 cutoff 和 periodic image identity 不变；
- 不增加 legacy/native fallback 或第二套 provider；
- 不以虚函数、运行时 `std::function` 或跨模型分支进入 device hot loop；
- 无新增 local-memory spill；occupancy 或 register 变化必须由 SASS/NCU 解释；
- 单 kernel 变快但端到端 throughput 无收益时不保留额外复杂度；
- 任一 production cell 超出既有 3% gate 时停止合入并拆分定位。

## 7. C1 实施与验证记录

### 7.1 基线和改动范围

- 精确父提交：`ccddac6fb78536b1295b309cb4a582370e23c1bb`；
- 工作分支：`backport/clustered-lj-mainline`；
- 公共 primitive：`neighbor_list/contract/traversal.cuh`；
- 迁移 consumer：regular/soft LJ、SITS、EAM、ReaxFF VDW/bond-order/EEQ、
  EDIP、Tersoff；
- C1 检查点明确保留：SW 与 ReaxFF HB 的 center-cursor 解码、regular LJ 的特化
  active 判定，以及当时仍手写遍历骨架的 CPU consumer；后者已由后续 C2 收口。

### 7.2 已完成验证

CPU 构建：

```text
cmake --build build-b6-candidate-cpu \
  --target SPONGE CLUSTERED_SPATIAL_VIEW_TEST \
  MANYBODY_CLUSTERED_ORACLE_TEST -j24
结果：通过，46/46 build steps
```

CPU 测试：

```text
ctest --test-dir build-b6-candidate-cpu --output-on-failure \
  -R '^(ClusteredSpatialViewContract|ManybodyClusteredOracle)$'
结果：2/2 通过
```

CUDA 构建使用 CUDA 13.0.88 与 pixi GCC 11.4 host compiler：

```text
cmake --build build-b6-candidate-cuda-sm89 \
  --target SPONGE CLUSTERED_SPATIAL_VIEW_TEST \
  MANYBODY_CLUSTERED_ORACLE_TEST -j24
结果：通过，137/137 build steps
```

CUDA 测试：

```text
ctest --test-dir build-b6-candidate-cuda-sm89 --output-on-failure \
  -R '^(ClusteredSpatialViewContract|ManybodyClusteredOracle)$'
结果：2/2 通过
```

CUDA 测试必须在可访问 GPU 的执行环境中运行；受限 sandbox 中出现的 device
allocation failure 已由同一 binary 的 GPU 环境复测排除，不属于源码或显存容量问题。

独立 host/device parity 使用同一组输入分别调用 host inline primitive 和 CUDA kernel，
覆盖 null exclusion pointer、两个 split、active marker absent/present、exclusion
authoritative case，以及 regular LJ 所需的动态 `i_cluster_size=4`。CPU stub 和 CUDA
实现均完成编译；CUDA `ClusteredSpatialViewContract` 在 GPU 环境中通过（1/1，约
0.13 秒）。这项测试验证的是 primitive 输出一致性，不替代 consumer oracle。

### 7.3 Device code 对照

父提交与 candidate 均使用 CUDA 13.0.88、pixi GCC 11.4 和 `CUDA_ARCH=89`。
这里必须设置项目 cache 变量 `CUDA_ARCH`；只设置标准
`CMAKE_CUDA_ARCHITECTURES` 会被 `cmake/parallel/cuda.cmake` 覆盖成 `native`，不能
形成有效对照。

对 `cuobjdump --dump-resource-usage` 结果 demangle 后，按 function record 排序：

```text
parent function records:    937
candidate function records: 937
parent SHA-256:    24ae897eda4e719b34f31e3dc2709af575554819ef112edb45f89aab9df8ce02
candidate SHA-256: 24ae897eda4e719b34f31e3dc2709af575554819ef112edb45f89aab9df8ce02
```

因此所有 device function 的 register、stack、shared、local 和 constant resource
记录一致；没有新增 local-memory spill。源码 diff 也没有改变目标 kernel 的签名或
launch site。

对 SASS 只归一化 anonymous-namespace translation-unit hash 后，父/candidate 均有
937 个 function，其中本批目标 kernel 41 个；41 个 body 均非精确等价。各族实例数
和指令文本行数变化如下：

| Kernel 族 | 实例数 | candidate − parent 行数分布 |
|---|---:|---|
| regular LJ | 24 | `0 × 19`、`-16 × 2`、`+16 × 2`、`+112 × 1` |
| soft-LJ | 2 | `0 × 2` |
| SITS | 3 | `0 × 2`、`+16 × 1` |
| EAM rho/force | 3 | `0 × 2`、`-16 × 1` |
| ReaxFF VDW | 2 | `+32 × 1`、`+64 × 1` |
| ReaxFF bond-order | 1 | `0 × 1` |
| ReaxFF EEQ | 2 | `0 × 1`、`+16 × 1` |
| EDIP builder | 2 | `0 × 1`、`+16 × 1` |
| Tersoff builder | 2 | `0 × 1`、`+16 × 1` |

“行数不变”不代表指令等价；它只说明代码体积没有粗粒度变化。完整原始结果与审计
脚本位于 `.tmp/c1-primitive-refactor/sass/`，不作为源码提交内容。

### 7.4 NCU 与 timing 结果

NCU 使用 Nsight Compute 2026.1.1、`sm_89`，父/candidate 二进制由同一 CUDA
13.0.88 与 pixi GCC 11.4 工具链构建。regular LJ 使用 160k-water production
gmxpacked snapshot，warmup 20 次后捕获一个 kernel instance；所有比较均固定相同
模板实例与 launch 形态。

| 指标 | force-only parent | force-only candidate | full-output parent | full-output candidate |
|---|---:|---:|---:|---:|
| duration | 299.200 us | 299.584 us | 736.352 us | 737.408 us |
| SM throughput | 44.95% | 44.96% | 33.61% | 33.56% |
| memory throughput | 35.57% | 35.51% | 66.98% | 66.96% |
| achieved occupancy | 41.18% | 41.39% | 33.13% | 33.05% |
| registers/thread | 72 | 72 | 96 | 96 |
| global-load requests | 4,206,953 | 4,206,953 | 3,655,170 | 3,655,170 |
| global-load sectors | 5,106,748 | 5,106,748 | 4,554,965 | 4,554,965 |
| local-store sectors | 0 | 0 | 18,214,494 | 18,214,494 |
| executed instructions | 169,721,907 | 170,079,989 | 313,480,308 | 313,838,390 |

force-only 的 duration 变化为 `+0.128%`，full-output 为 `+0.143%`；L1/L2、
coalescing、occupancy limit 和主要 warp-stall 排名均稳定。full-output 的 local-store
traffic 是父版本已有流量，candidate 没有增加。instruction count 分别增加 `0.211%`
和 `0.114%`，足以解释非逐字相同的 SASS，但未形成资源或吞吐回退。

为降低单次 NCU 时序噪声，又执行五组父/candidate 交错的 2,000-iteration
force-only timing：父版本均值 `0.298668 ms`、candidate `0.300084 ms`，变化
`+0.474%`；中位数变化 `+0.360%`。所有运行 `sanity=ok`。full-output 的 reference
比较为 `matched=1`，容差 `2e-5`。

EAM 作为直接多体 consumer 的代表，分别捕获 rho 与 force kernel。candidate 的单次
duration 相对父版本为 `-3.85%` 和 `-1.45%`；两者 registers、shared memory、
occupancy limits 和 local-store sectors 不变，global-load request 仅少 8、sector 少
32，instruction count 增加约 `0.11%`/`0.10%`。`frc.dat` 与 `mdout.txt` 在父/candidate
间逐字节一致。单次改善不作为性能收益结论，只用于排除 helper 迁移造成的退化。

原始 `.ncu-rep`、CSV、交错 timing 和解析脚本保存在
`.tmp/c1-primitive-refactor/ncu/`，不作为源码提交内容。

### 7.5 C1 验收决定

C1 接受。依据是 correctness、host/device parity、937 条 device resource record、
代表 kernel NCU 和交错 timing 同时通过；没有触发“新增 global load、register 增长、
spill 或 occupancy limit 下降”的停止条件。下一批 C2 只处理 CPU iterator，不夹带
device kernel 表达式或 CSR 生命周期变化。

## 8. C2 实施与验证记录

### 8.1 Contract 与迁移范围

C2 没有改变 provider/builder payload，只在 CPU consumer 侧把重复的 layout decode
替换为静态模板 adapter。candidate record 只携带 identity、mask、local fact 与已有
shift metadata；visitor 内的坐标访问、距离计算和物理输出保持原位。

迁移按原循环顺序分为三组：

| Adapter | Consumer | 保持的关键顺序/生命周期 |
|---|---|---|
| J-major pair | EAM、ReaxFF VDW/BO/EEQ/HB | J endpoint accumulator 与 begin/end-J flush |
| I-tile builder | SW、EDIP、Tersoff | directed dual insertion、模型 cutoff 与 CSR row key |
| local-I/J-tile | regular/soft LJ、SITS | I aggregation、每 tile J flush、consumer-specific shift |

其中 regular LJ 继续使用既有 SCI shift；soft-LJ 继续使用 SCI shift 加 min-image；
SITS 继续从 pair-shift metadata 取 per-pair shift。这是三个 consumer 原有的语义边界，
C2 没有借 iterator 迁移顺便统一它们。

### 8.2 验证与验收结果

当前源码状态已完成：

```text
CPU build：SPONGE + ClusteredSpatialViewContract + ManybodyClusteredOracle
结果：18/18 build steps，通过；两个测试 2/2 通过

CUDA sm_89 build：同三个 target
结果：13/13 build steps，通过；最终 GPU 测试 2/2 通过
```

EAM 使用精确父提交与 candidate 的 CPU binary 对同一 fixture 执行 A/B，二者退出码均为
0，且以下输出逐字节一致：

```text
frc.dat    69a11104d80816a6f241ccb044b860e534ba8315147c17fa65a56bfd58c10933
mdout.txt  db14a618e7c6b9ad5b3fbe61749397299a338bd63725fdb1ca75efb8808fbe50
```

扩大 CPU 场景结果：

- soft-LJ point-energy：1/1 通过；最大误差项 PM，绝对误差 `0.28`，门槛 `2.0`；
- SITS facade smoke 与 clustered sparse repeat：2/2 通过；
- ReaxFF PETN 16,240 原子单帧：1/1 通过；相对势能差 `3.176842e-4`、最大电荷差
  `4.96e-4`、最大力差 `0.721789`，均低于已有 fixture 门槛。

最后又将 J-tile 的 pair-shift 解码改为按需 accessor：regular/soft LJ 不再为未使用的
pair shift 执行分支/读取，SITS 显式请求该 metadata。修改后 CPU 与 CUDA target 均重新
编译，contract/manybody oracle 分别 2/2 通过，SITS 2/2 复测通过。

C2 接受。残留搜索确认目标 CPU consumer 中不存在第二套手写 SCI/CJ/lane 解码；
剩余手写 packed traversal 属于 GPU kernel 或 custom-pair/JIT ABI，不在 CPU iterator
批次范围内。

## 9. C3 实施与验证记录

### 9.1 Storage/stamp 边界

新增 `SPONGE/manybody/clustered_csr.h`，提供两个 consumer-side 基元：

- `ClusteredPayloadStamp` 只比较、捕获或重置
  `provider_incarnation + gmxpacked_payload_generation`；
- `ClusteredCSRStorage` 统一管理 `counts`、`offsets`、`items`、capacity、item count，
  GPU 构建时额外持有可复用的 scan workspace 与 device total。

SW、EDIP、Tersoff 的类成员已从八组独立 pointer/count/capacity 字段迁到上述对象。
各模型仍负责何时 rebuild、row key、directed cutoff、count/fill kernel 和 force kernel；
公共 storage 不读取坐标或模型参数，也不进入 provider。`Clear()` 仍由模型 owner 的既有
释放路径显式调用，结构体没有增加隐式析构或复制语义，避免改变现有生命周期。

SW 的 row 是 sorted atom，EDIP/Tersoff 的 row 是 atom id。本批因此没有继续抽取
EDIP/Tersoff/SW 的统一 enumerator：三者共享的是内存和 generation 生命周期，不是
模型关系定义。

### 9.2 Device scan

GPU 路径现在由 `Clustered_CSR_Device_Exclusive_Scan()` 调用 CUB/hipCUB
`DeviceScan::ExclusiveSum`，随后用单线程 finalize kernel 写入 `offsets[row_count]`
和 device total，最后只把一个 `int` total 拷回 host 以决定 items capacity。与旧路径
相比，每次 CSR rebuild 消除了：

```text
atom_numbers × sizeof(int) counts D2H
host exclusive-prefix 循环
(atom_numbers + 1) × sizeof(int) offsets H2D
```

保留下来的标量 D2H 会同步 stream；在不改变现有动态分配接口的前提下，这是 fill 前
获得 items capacity 所需的边界。CPU 路径继续使用原 host vector/prefix 逻辑，行为和
插入顺序未改变。

device scan 的 offsets 仍为 `int`，但 scan 前先由一个 device validation/reduction
kernel 计算 64 位 item total 和最小 count，再把 16 B stats 拷回 host。任一 count
为负或 total 超过 `INT_MAX` 都在 CUB scan 前失败，因而不会让 32 位 prefix 发生
正向溢出；这恢复了旧 host `long long` 累加检查的失败语义。zero-row 则直接写入
`offsets[0]=0`，不启动 validation 或 CUB。

### 9.3 NCU 结果

按 CUDA 性能流程先对修改前 SW builder 做 NCU，再实施 device scan。环境为 RTX 4090
`sm_89`、CUDA 13.0.88、Nsight Compute 2026.1.1；workload 为
`benchmarks/comparison/tests/lammps/outputs/sw/0/sponge` 的 10,648 原子、
`step_limit=0` rebuild。基线报告位于 `.tmp/c3-csr/ncu-baseline/sw_builder.ncu-rep`，
candidate 报告位于 `.tmp/c3-csr/ncu-device-scan/`。

SW 原有 count/fill kernel 在本批没有源码改动。用与基线相同的 kernel filter
重新捕获后：

| Kernel | 指标 | host-scan 基线 | device-scan candidate | 变化 |
|---|---|---:|---:|---:|
| count | duration | 260.256 us | 264.416 us | +1.598% |
| count | SM throughput | 69.86% | 68.81% | -1.05 pp |
| count | achieved occupancy | 72.91% | 72.99% | +0.08 pp |
| count | registers/thread | 47 | 47 | 0 |
| fill | duration | 270.880 us | 270.592 us | -0.106% |
| fill | SM throughput | 63.80% | 64.03% | +0.23 pp |
| fill | achieved occupancy | 59.86% | 59.84% | -0.01 pp |
| fill | registers/thread | 55 | 55 | 0 |

表中变化均低于 3%，两个 kernel 的 spill 都为 0。基线显示 count/fill 分别偏
混合型/SM 型而非 DRAM 带宽受限，所以本批没有趁机改写两次 packed decode。

最终 validation NCU 为 `4.096 us`、36 registers、4 KiB launch shared、0 spill；
SM/MEM/DRAM throughput 分别为 `0.068%/1.258%/1.258%`，achieved occupancy
`16.92%`。warp sample 以 long-scoreboard `14/25`、barrier `5/25` 为主，线程执行
比 `31.46/32`，无 shared-bank conflict。它是只有一个 CTA 的 launch/latency-bound
rebuild helper，不值得为提高 occupancy 增加并行归约复杂度。finalize 为
`6.272 us`、16 registers、0 spill；这些极短 kernel 的 NCU replay duration 噪声较大，
因此最终接受决定还使用下述 Nsight Systems 时间线。

以 C1 后 candidate SASS 作为 C3 前基线，最终 binary 的 SW、EDIP、Tersoff count/fill
共 6 个模板实例 normalized body 全部逐项相等，line delta 全为 0，resource multiset
也完全相等。新增 validation 的静态资源记录为 36 registers、3,072 B shared、
0 stack/local；finalize 为 8 registers、0 stack/shared/local。完整 dump 和审计脚本在
`.tmp/c3-csr/sass-final/`。

Nsight Compute 2026.1.1 CSV 使用 `Kernel Name` 列，当前自动分析脚本仍期待
`Function Name`，因此该批数据按相同 kernel name、launch 和 metric key 手工比对；
原始 CSV 和 `.ncu-rep` 均保留以供复核。

### 9.4 Correctness 与当前决定

storage/stamp 迁移后以及 device scan 迁移后均重新构建 CPU/CUDA target。当前记录为：

```text
CPU：SPONGE + ClusteredSpatialViewContract + ManybodyClusteredOracle，构建通过
CUDA sm_89：同三个 target，构建通过
CPU contract/manybody oracle：2/2 通过
GPU contract/manybody oracle：2/2 通过
SW/EDIP/Tersoff CPU LAMMPS 对照：9/9 通过
SW/EDIP/Tersoff GPU LAMMPS 对照：9/9 通过
```

边界测试同时覆盖 stamp capture/mismatch/reset、zero-row、稀疏 counts
`{2,0,3,1}` 对应 offsets `{0,2,2,5,6}`、负 count，以及 `{INT_MAX,1}` overflow。

Nsight Systems 2026.1.1 使用同一 10,648 原子 SW fixture、`refresh_interval=1` 强制
21 次 CSR rebuild。父/candidate 均为 CUDA 13.0.88、`sm_89`：

| count→fill 区间 | host scan 父版本 | device scan candidate |
|---|---:|---:|
| D2H | 21 次 / 894,432 B | 21 次 / 336 B |
| H2D | 21 次 / 894,516 B | 0 |
| GPU copy time | 173.089 us | 19.997 us |
| 新增 scan kernels total | 0 | 145.405 us |
| warm interval median | 29.674 us | 26.442 us |
| warm interval filtered mean | 32.886 us | 26.758 us |

candidate 有一个 233.486 us 的 host 调度离群点，因此未采用原始 mean 宣称收益；稳定
样本的 interval 中位数改善约 `10.9%`，但相对于约 `0.55 ms` 的完整 CSR rebuild
只约 `0.6%`。结论是保留 device scan：它消除了随 atom count 线性增长的双向全量
传输，当前规模已有小幅正收益，同时不把该结果外推成生产 step throughput 加速。

ReaxFF bond-order 的 GPU CSR 已使用 `thrust::exclusive_scan`，row end 由
`bond_count` 而不是 `offsets[i+1]` 表达；EEQ 已使用 device exclusive scan + reduce。
两者还持有与 `items` 不同的并行数组和容量规则。结论是暂不迁入
`ClusteredCSRStorage`，后续若抽取，应先把 scan-total/overflow helper 与 typed
parallel-array storage 分开设计。

C3 接受。依据是 overflow/negative failure semantics、CPU/CUDA correctness、18 组
LAMMPS 对照、6 个目标模板实例的 SASS/resource exact、NCU 和强制 rebuild 时间线
同时闭合。C4 才评估三体 force kernel，本批不混入 direct-vs-CSR 或数学 kernel 重写。

## 10. C4 profile-driven kernel 评估

### 10.1 实际 rebuild interval

使用最终 candidate、160k-water NVT production 输入、2,000 steps，在 RTX 4090 上用
Nsight Systems 2026.1.1 记录完整 CUDA 时间线。`Check_Clustered_Rebuild` 执行 2,000
次，payload builder 的代表 count kernel 执行 5 次：一次初始化构建，以及完成第
`533`、`1032`、`1434`、`1839` 次 check 后的四次重建。四个稳态间隔为
`533/499/402/405` steps，平均 `459.75`、中位数 `452` steps。

这不是 SW/Tersoff 专属轨迹，但 provider generation 是所有 clustered consumer 的
共同失效边界，可用于判断 derived CSR 的摊销量级。该结果说明“每步直接遍历以省掉
偶发 CSR rebuild”不成立；C3 的 device scan 主要是消除 rebuild 时的线性 host
传输，而不是改变每步 force 成本。原始 report、SQLite 与查询结果位于
`.tmp/c4-profile/wat160k-nvt-rebuild-trace.*`。

### 10.2 SW direct-vs-CSR/cached A/B

SW 使用 10,648-atom diamond fixture，full-output kernel、相同 launch
`666 × (32,16,1)`。基线保持 production capacity 64；实验版本只把
`kSWClusteredCachedNeighborCapacity` 临时降到 8，使相同 center rows 进入既有
gmxpacked center-cursor fallback。采样后已恢复 64 并重新构建最终 binary。

| 指标 | CSR row + shared cache（64） | cursor fallback（8） |
|---|---:|---:|
| duration | `101.824 us` | `6.688832 ms` |
| SM throughput | `74.54%` | `49.97%` |
| achieved occupancy | `54.35%` | `46.98%` |
| registers/thread | 64 | 64 |
| executed instructions | `64.06 M` | `4,261.51 M` |
| local-store sectors | `1.53 M` | `39.04 M` |
| L1/L2 hit | `92.37%/87.83%` | `98.42%/96.82%` |

fallback 慢约 `65.7×`。较高 cache hit 不能抵消重复 cursor 控制流、pair decode、周期
位移和二次邻居扫描；因此不删除 SW center CSR，也不新增 production 分流。容量 8
只是 profile 实验，不属于最终源码。

### 10.3 Tersoff 双 `k` 扫描

同一 10,648-atom fixture 的最终 full-output `Tersoff_Force_CUDA<true>` 为
`1.119776 ms`，launch `333 × (32,32,1)`，56 registers/thread、0 local-store
sector、47.14% achieved occupancy。SM/memory/DRAM throughput 分别为
`49.74%/42.46%/0.89%`，L1/L2 hit 为 `98.54%/83.65%`，已执行约 `706.32 M`
instructions，branch-target uniformity 为 `99.24%`。

证据指向数学指令、长 live range 与原子累加，而不是 DRAM 或 cache miss。把第一次
`k` 扫描的几何量保留到导数扫描会增加 per-`j` storage/register pressure；当前
1024-thread block 已同时受 register 与 warp 上限约束。没有先验端到端收益足以覆盖
这一风险，因此 C4 不实施双扫描融合，也不建立按 degree 分流的第二套 kernel。

### 10.4 ReaxFF HB lane/control-flow

复核 B5.7 保留的 PETN 16,240 原子原始 NCU report，结果如下：

| 方案 | duration | achieved occupancy | registers | executed instructions |
|---|---:|---:|---:|---:|
| legacy HB | `444.384 us` | `7.04%` | 80 | `26.74 M` |
| clustered warp/H | `508.256 us` | `35.07%` | 80 | `211.01 M` |
| clustered thread/H | `9.389632 ms` | `2.26%` | 96 | `337.06 M` |
| clustered 8 lanes/H | `714.176 us` | `14.93%` | 80 | `131.30 M` |
| clustered 16 lanes/H | `440.544 us` | `29.00%` | 80 | `145.06 M` |

16-lane 主核加 `1.952 us` atom-to-sorted map 约 `442.496 us`，仍是已测 clustered
方案中唯一与 legacy 持平并略快的粒度；所有方案 local-store sector 均为 0。C4
因此保持 16 lanes/H，不再追加 4-lane、32-lane或运行时 lane dispatch。

### 10.5 最终 replay、production 与决定

最终 candidate 在恢复 SW capacity 64 后重新构建。replay 使用 3 systems × 2 output
modes × parent/current × 3 runs、每次 warmup 200 + 2,000 iterations；production
使用 3 systems × NVT/NPT × parent/current × 3 runs、每次 10,000 steps。冻结父
binary 为 B6 的 `336d501f`，两套矩阵都清除未声明的 `SPONGE_*` 环境变量，并保持
5% pre/post idle-SM 门槛。

| system | production NVT speed delta | production NPT speed delta | replay force-only | replay full |
|---|---:|---:|---:|---:|
| wat160k | `+0.030%` | `+0.230%` | `+0.476%` | `+0.305%` |
| wat600k | `+0.180%` | `-0.408%` | `+0.394%` | `+0.370%` |
| DNA_COU | `+0.469%` | `-0.798%` | `+0.334%` | `+0.976%` |

production 列为 candidate speed delta，正值更快；replay 列为 candidate duration
delta，正值更慢。36/36 replay 与 36/36 production 均 valid，六个 production cell
与六个 replay cell 全部通过 3% conjunctive gate。第一次 replay/production 尝试分别
因一次 post-run idle SM 为 10%/8% 被 runner 拒绝；最终结果来自保持 5% 门槛后的
完整重跑，不复用 invalid row，也未放宽协议。

C4 接受“无新增 kernel 改写”的决定。最终保留的源码变化仍只有 C1 traversal
primitive、C2 CPU adapters 与 C3 CSR storage/device scan；C4 的容量 8 实验已回退，
没有新增 production probe、degree dispatch、dual path 或环境变量。

## 11. 源码布局与规模

相关实现按职责分为五层：

```text
SPONGE/neighbor_list/
    neighbor_list.*               主程序 facade
    provider/                     owner、lifecycle、view/metadata 发布
    builder/                      CPU/GPU geometry/candidate/payload builder
    contract/
        types.h、view.*           公共 payload/view 与 validation
        traversal.cuh             host/device 位布局 primitive
        cpu_traversal.h           CPU iterator 与 candidate record

SPONGE/Lennard_Jones_force/        regular/soft LJ direct consumer
SPONGE/Selective_Interaction/      SITS sparse-view consumer

SPONGE/manybody/
    clustered_gmxpacked_cpu.h      旧 include 路径的 forwarding header
    clustered_csr.h                derived CSR storage/stamp/device scan
    eam.cpp                        direct two-pass consumer
    sw.cpp                         sorted-center CSR/cursor consumer
    edip.cpp                       atom-id directed CSR consumer
    tersoff.cpp                    atom-id directed CSR consumer
    reaxff/                        VDW、bond-order、EEQ、HB

tools/
    clustered_spatial_view_test/   contract/primitive tests
    manybody_clustered_oracle_test/多体行为 oracle
```

provider/builder/contract 目录的当前 `wc -l -c` 汇总如下。builder 包含 `.cpp`、
`.h/.cuh` 与 `detail/*.inc*`；数字是审查面快照，不是代码质量指标：

| 层 | 行数 | 字节数 |
|---|---:|---:|
| `neighbor_list/neighbor_list.{h,cpp}` facade | 1,069 | 43,806 |
| `neighbor_list/provider/` | 1,977 | 82,967 |
| `neighbor_list/builder/` | 12,155 | 558,141 |
| `neighbor_list/contract/` | 1,602 | 59,556 |
| **合计** | **16,803** | **744,470** |

以下为当前 C1+C2+C3 工作树中主要 contract、consumer 与测试文件的明细：

| 文件 | 行数 | 字节数 |
|---|---:|---:|
| `neighbor_list/contract/traversal.cuh` | 403 | 14,285 |
| `neighbor_list/contract/cpu_traversal.h` | 372 | 14,354 |
| `manybody/clustered_gmxpacked_cpu.h` | 5 | 216 |
| `manybody/clustered_csr.h` | 258 | 7,407 |
| `Lennard_Jones_force/clustered_lj_warp_record_kernel.cuh` | 680 | 38,589 |
| `Lennard_Jones_force/Lennard_Jones_force.cpp` | 1,292 | 50,799 |
| `Lennard_Jones_force/LJ_soft_core.cpp` | 1,322 | 56,696 |
| `Selective_Interaction/SITS.cpp` | 2,858 | 111,758 |
| `manybody/eam.cpp` | 1,053 | 42,046 |
| `manybody/sw.cpp` | 1,643 | 69,371 |
| `manybody/edip.cpp` | 865 | 37,092 |
| `manybody/tersoff.cpp` | 1,146 | 45,103 |
| `manybody/reaxff/vdw.cpp` | 747 | 29,457 |
| `manybody/reaxff/bond_order.cpp` | 1,294 | 54,948 |
| `manybody/reaxff/eeq.cpp` | 1,241 | 48,329 |
| `manybody/reaxff/hydrogen_bond.cpp` | 733 | 30,471 |
| `clustered_spatial_view_test.cpp` | 1,302 | 55,587 |
| `manybody_clustered_oracle_test.cpp` | 1,212 | 48,794 |
| **合计** | **18,426** | **755,302** |

最大的实现面是 builder（约 12.2k 行），最大的单 consumer 文件是 SITS。C3 新增
258 行公共 CSR header，同时令 SW/EDIP/Tersoff 三个 `.cpp` 合计减少 113 行；加入
60 行 stamp/scan 边界测试后，选定文件合计为 18,426 行、755,302 B。该数字只反映
重复生命周期代码收口和新增验证，不作为性能或可维护性的单独验收依据。

## 12. 代码证据索引

| 主题 | 主要位置 |
|---|---|
| provider owner/lifecycle/view 发布 | `SPONGE/neighbor_list/provider/` |
| geometry/candidate/payload builder | `SPONGE/neighbor_list/builder/` |
| payload/view contract 与 validation | `SPONGE/neighbor_list/contract/types.h`、`view.*` |
| 公共 lane/shift/active-mask/cursor primitive | `SPONGE/neighbor_list/contract/traversal.cuh` |
| CPU pair mask 与三类 iterator | `SPONGE/neighbor_list/contract/cpu_traversal.h` |
| 旧 CPU helper include 兼容 | `SPONGE/manybody/clustered_gmxpacked_cpu.h` |
| derived CSR storage/stamp/device scan | `SPONGE/manybody/clustered_csr.h` |
| regular LJ traversal/reduction | `SPONGE/Lennard_Jones_force/clustered_lj_warp_record_kernel.cuh` |
| soft-LJ traversal | `SPONGE/Lennard_Jones_force/LJ_soft_core.cpp` |
| SITS traversal | `SPONGE/Selective_Interaction/SITS.cpp` |
| EAM rho/force traversal | `SPONGE/manybody/eam.cpp` |
| ReaxFF VDW traversal | `SPONGE/manybody/reaxff/vdw.cpp` |
| ReaxFF bond-order CSR | `SPONGE/manybody/reaxff/bond_order.cpp` |
| ReaxFF EEQ CSR | `SPONGE/manybody/reaxff/eeq.cpp` |
| ReaxFF HB center cursor | `SPONGE/manybody/reaxff/hydrogen_bond.cpp` |
| SW center CSR/cursor | `SPONGE/manybody/sw.cpp` |
| EDIP directed CSR | `SPONGE/manybody/edip.cpp` |
| Tersoff directed CSR | `SPONGE/manybody/tersoff.cpp` |
| contract tests | `tools/clustered_spatial_view_test/clustered_spatial_view_test.cpp` |
| manybody oracle | `tools/manybody_clustered_oracle_test/manybody_clustered_oracle_test.cpp` |
| 已有 NCU/SASS 数据 | `docs/clustered-lj-mainline-backport/03-validation-and-acceptance.md` |
| C1/C3 NCU/SASS 工作记录 | `.tmp/c1-primitive-refactor/`、`.tmp/c3-csr/` |
