---
name: sponge-manager-remd
description: >
  询问 SPONGE_MANAGER、scheduler/worker_protocol、manager.toml、T/H/HT-REMD、
  REST2-REMD、RuntimeState 交换、worker transport 或相关重构时使用。
---

本技能记录 SPONGE 多 worker 调度与 REMD/REST2-REMD 的当前实现约定、设计边界和后续开发注意事项。
面向 Codex 开发协作使用；用户可见配置说明应放在 `docs/manager-config.md`。

## 当前架构

`SPONGE` 仍是单副本 MD runtime，普通用户直接运行 `SPONGE` 的方式不变。
`SPONGE_MANAGER` 是外部调度器，负责批量启动多个常驻 `SPONGE` child-process worker，并按 epoch/block 调度运行。

核心分层：

- `SpongeScheduler`：单个 SPONGE runtime 的生命周期包装，支持初始化、block 运行、RuntimeState 导出/导入、observable 收集、速度缩放、近邻表失效。
- `worker_protocol`：manager 与 worker 的请求/响应协议，屏蔽 TCP、shm、file 等传输细节。
- `manager/core`：维护 schedule、worker session、block dispatch、state routing、exchange log。
- `manager/algorithms/remd`：实现 T-REMD、H-REMD、HT-REMD、REST2-REMD 的接受率与交换策略。

重要边界：

- manager 不应成为温度、压力、lambda、force-field 参数的物理源头。
- 物理参数仍来自 mdin 或 `schedules.inputs` 覆写后的 worker 命令行。
- manager 只负责调度、交换、日志、RuntimeState 路由和必要的交换后维护。
- 当前 runtime 是 process-global 状态，不支持一个进程内并行常驻多个 SPONGE worker；manager 路径只保留 child-process worker。

## manager.toml 约定

推荐入口：

```bash
SPONGE_MANAGER --config manager.toml
```

推荐结构：

```toml
[manager]
block_steps = 1000
epochs = 100
transport = "tcp"
log_path = "manager_exchange.log"

[exchange]
enabled = true
mode = "tremd"

[worker_defaults]
working_directory_root = "replicas"
args = ["-dont_check_input", "1"]

[worker_defaults.inputs]
mode = "NVT"
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
default_out_file_prefix = "mdout"
thermostat = "middle_langevin"

[schedules]
ids = [0, 1, 2]

[schedules.inputs]
target_temperature = [300.0, 310.0, 320.0]
device = 0
```

约定：

- `worker_defaults.mdin` 是可选公共 mdin；设置时会转换成 `-mdin <path>`；不要再在 `worker_defaults.args` 中重复写 `-mdin`。
- `worker_defaults.inputs` 是所有 schedule 的公共 SPONGE 参数，既可以覆写 mdin，也可以在没有 mdin 时完整构造 worker 命令行。
- `schedules.inputs` 是每个 schedule 的差异参数，优先级高于 `worker_defaults.inputs`。
- `schedules.inputs` 中的普通键通常都会转成 worker 命令行 `-key value`。
- `hamiltonian_id` 是 manager-only input，用于 H/HT-REMD 的 Hamiltonian slot 标识，不传给 worker。
- 支持批量 `[schedules] ids = [...]`，其中 list input 按下标展开，scalar input 广播到所有 schedule。
- 批量 `[schedules]` 禁止与逐项 `[[schedules]]` 混用。
- schedule id 推荐固定使用 `0, 1, 2, ...`。
- `working_directory_root` 只在 `worker_defaults` 里设置；每个 schedule 默认工作目录为 `<root>/<schedule_id>`。
- 如果用户设置 `default_out_file_prefix = "mdout"`，manager 应自动传入 `mdout_0`、`mdout_1` 等 schedule-local 前缀。
- 启动前必须检查输出文件冲突，避免多个 schedule 覆盖同一组 mdout、trajectory、restart 或 manager log。
- 默认自动推断 worker executable：优先使用 `SPONGE_MANAGER` 同目录下的 `SPONGE` / `SPONGE.exe`，再查找 `PATH`。
- `working_directory_root` 可相对 `manager.toml`，不必强制绝对路径。
- `inputs` 中的字符串路径 key 应由 manager 相对 `manager.toml` 解析成绝对路径，包括 `default_in_file_prefix`、`*_in_file`、`*_out_file`、`*_file`、`*_path`、`*_directory`。
- `default_out_file_prefix` 不解析成绝对路径；它是输出前缀，由 manager 按 worker 工作目录隔离并自动追加 schedule id。

`mdin` 缺省是明确支持的。此时 manager 不注入 `-mdin`，只把 `worker_defaults.args`、`worker_defaults.inputs` 和 `schedules.inputs` 合并成 worker 命令行。这与直接运行 `SPONGE -mode NVT -dt 0.002 ...` 的语义一致。

`SPONGE_MANAGER` 的正式 block 调度永远要求 worker 正常打印。不要在 manager config 中设计 `emit_output` 开关；输出频率必须通过 SPONGE 自身的 `write_*_interval` 参数控制。内部 foreign-state probe 仍可走不打印路径，但这是 worker protocol 内部细节，不暴露给用户。

## RuntimeState 交换语义

SPONGE_MANAGER 采用“交换完整 runtime state”而不是“交换参数标签”。

最小 RuntimeState 应包含：

- coordinates
- velocities
- box/cell
- step/time
- 当前实现已序列化的必要 continuation metadata

REMD 接受交换后，manager 将两个 schedule 的 RuntimeState 对调。这样 schedule-local 输出轨迹天然对应固定 thermodynamic/Hamiltonian slot，后续 rerun 和分析更直接。

需要注意：

- T/HT-REMD 接受交换时应按温度比缩放速度。
- H/HT/REST2 需要 foreign-state observable/cross evaluation。
- SITS 这类带历史偏置且 RuntimeState 未完整序列化的 selective interaction 目前应视为 probe-unsafe。
- REST2 是静态 Hamiltonian scaling，当前设计为 probe-safe。

## REMD 算法约定

当前配对策略是 odd-even 邻居交换：

```text
round 0: (0,1), (2,3), ...
round 1: (1,2), (3,4), ...
round 2: (0,1), (2,3), ...
```

交换模式：

- `tremd`：要求 `target_temperature`，接受后缩放速度。
- `hremd`：要求 `target_temperature` 和 `hamiltonian_id`，使用 Hamiltonian cross evaluation。
- `htremd`：同时考虑温度和 Hamiltonian cross evaluation，接受后缩放速度。
- `rest2`：要求 `target_temperature` 和 `REST2_lambda_m`，基于 REST2 Hamiltonian cross evaluation。

随机数注意事项：

- exchange acceptance 的随机数必须随 exchange attempt 更新。
- 不要在每轮复用固定随机数，否则接受率会出现非物理模式。
- correctness test 应检查接受率、round/pair 统计、temperature index walk、state continuity 和能量/温度分布。

## Transport 约定

当前面向用户只暴露 child-process worker，不再暴露 in-process worker 选项。

transport：

- `tcp`：默认；loopback TCP 控制与 payload。
- `shm`：TCP 控制消息 + shared-memory payload，适合单节点大 RuntimeState。
- `file`：persistent worker + file request/response，主要作为兼容/调试 fallback，性能通常较差。

实现原则：

- manager core 不直接依赖 TCP socket、shared-memory 对象或文件路径。
- 新 transport 应放在 `worker_protocol` 层，通过统一 session/transport 接口接入。
- 多节点通信暂不作为当前目标；一般集群上节点 IP/host 发现需要调度系统配合，先避免扩大设计面。

## Selective Interaction 与 REST2

REST2 与 SITS 走统一 `SPONGE/Selective_Interaction/` facade。

当前设计意图：

- `Selective_Interaction` 对 `main.cpp` 暴露统一 hook。
- `SITS` 和 `REST2` 分别管理自己的初始化、局部 atom mapping、energy/print 行为。
- REST2 复用 selective atom/pair classification 思路。
- REST2 对 direct short-range LJ 和 direct Coulomb 做 pair scaling：
  - hot-hot: `lambda_m`
  - hot-cold: `sqrt(lambda_m)`
  - cold-cold: `1.0`
- 当前 REST2 不应声称已覆盖 reciprocal PME、bonded terms 或 1-4 terms，除非源码已明确实现并验证。
- REST2 的用户输入可在 mdin 中设置，也可由 `schedules.inputs.REST2_lambda_m` 覆写。

REST2 单副本 mdin 示例：

```text
REST2_mode = on
REST2_atom_numbers = 22
REST2_lambda_m = 1.0
```

REST2-REMD manager 示例：

```toml
[exchange]
enabled = true
mode = "rest2"

[worker_defaults.inputs]
target_temperature = 300.0
default_out_file_prefix = "rest2"

[schedules]
ids = [0, 1, 2]

[schedules.inputs]
REST2_lambda_m = [1.0, 0.9, 0.8]
```

## 测试建议

改 manager/config 时至少运行：

```bash
pixi run -e dev-cuda13 format-check
pixi run -e dev-cuda13 cmake --build <build-dir> --target SPONGE_MANAGER --parallel 4
pixi run -e dev-cuda13 cmake --build <build-dir> --target SPONGE --parallel 4
```

建议覆盖：

- `benchmarks/validation/rest2/tests/test_rest2.py`
- `benchmarks/performance/remd/tests/test_ice_tremd_perf.py`
- 新语法 bulk `[schedules]`
- duplicate schedule id rejection
- output path conflict rejection
- executable auto inference
- `step_limit` 被 manager 托管逻辑覆盖

性能测试时优先比较 `tcp` 与 `shm`，`file` 可只做 fallback smoke，不应作为主要性能路径。
