# SPONGE_MANAGER Config

本文档是面向用户的 `SPONGE_MANAGER` 配置说明。开发计划、迁移背景和 REMD 实现细节不放在 `docs/`，而放在 `skills/sponge-manager-remd/` 供 Codex 协作时使用。

## 启动方式

推荐通过 `manager.toml` 启动：

```bash
SPONGE_MANAGER --config manager.toml
```

`SPONGE_MANAGER` 会启动多个常驻 `SPONGE` worker 子进程。普通单副本模拟仍然直接运行 `SPONGE`，不需要 manager。

## 基本结构

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
target_temperature = 300.0

[schedules]
ids = [0, 1, 2]

[schedules.inputs]
target_temperature = [300.0, 310.0, 320.0]
device = 0
```

## `[manager]`

`[manager]` 控制调度行为。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `block_steps` | int | `1000` | 每个 epoch 中每个 worker 运行的 MD 步数。 |
| `epochs` | int | `1` | 总调度 epoch 数。总步数为 `block_steps * epochs`。 |
| `transport` | string | `"tcp"` | 通信方式，可选 `"tcp"`、`"shm"`、`"file"`。 |
| `log_path` | string | `manager_exchange.log` | exchange log 输出路径。 |

## `[exchange]`

`[exchange]` 控制是否执行副本交换。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | 根据 `mode` 推断 | 是否启用交换。 |
| `mode` | string | 空 | 可选 `"tremd"`、`"hremd"`、`"htremd"`、`"rest2"`。 |
| `start_round` | int | `0` | 初始 exchange round。 |

当前交换配对为 odd-even 邻居交换：

```text
round 0: (0,1), (2,3), ...
round 1: (1,2), (3,4), ...
```

如果只想批量运行多个 worker，不做 REMD：

```toml
[exchange]
enabled = false
```

## `[worker_defaults]`

`[worker_defaults]` 提供所有 schedule 共用的 worker 设置。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mdin` | string | 空 | 可选公共 mdin 文件。如果设置，会转换成 worker 参数 `-mdin <path>`。 |
| `args` | string array | 空 | 传给 SPONGE 的公共命令行参数。不要在这里重复写 `-mdin`。 |
| `working_directory_root` | string | 空 | 所有 schedule 工作目录的根目录。 |
| `executable` | string | 自动推断 | SPONGE worker 可执行文件路径。 |
| `executable_path` | string | 自动推断 | `executable` 的等价字段。 |

如果不设置 `executable`，manager 会优先查找 `SPONGE_MANAGER` 同目录下的 `SPONGE` / `SPONGE.exe`，然后查找 `PATH`。

`mdin` 可以缺省。缺省时，manager 不会给 worker 注入 `-mdin`，用户可以完全通过 `worker_defaults.inputs` 和 `schedules.inputs` 构造 SPONGE 命令行参数。这与直接运行 `SPONGE -key value ...` 的语义一致。

如果设置了 `mdin`，不要再在 `args` 里重复写 `-mdin`。

## `[worker_defaults.inputs]`

`worker_defaults.inputs` 是所有 schedule 共享的 SPONGE 参数，会转换成 worker 命令行：

```toml
[worker_defaults.inputs]
mode = "NPT"
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
target_pressure = 1.0
default_out_file_prefix = "mdout"
```

因此有两种合法用法：

```toml
[worker_defaults]
mdin = "mdin.spg.toml"

[worker_defaults.inputs]
target_temperature = 300.0
```

或完全不使用 mdin：

```toml
[worker_defaults]
args = ["-dont_check_input", "1"]

[worker_defaults.inputs]
mode = "NVT"
dt = 0.002
cutoff = 8.0
default_in_file_prefix = "WAT"
target_temperature = 300.0
thermostat = "middle_langevin"
thermostat_tau = 1.0
thermostat_seed = 2026
```

## `[schedules]` 批量写法

规则化副本推荐使用批量写法：

```toml
[schedules]
ids = [0, 1, 2, 3]

[schedules.inputs]
target_temperature = [300.0, 310.0, 320.0, 330.0]
device = 0
```

list 值会按 `ids` 下标展开，scalar 值会广播到所有 schedule。批量 `[schedules]` 不能和逐项 `[[schedules]]` 混用。

## `[[schedules]]` 逐项写法

需要为每个副本设置不同参数时，可以使用逐项写法：

```toml
[[schedules]]
schedule_id = 0
name = "replica_300K"

[schedules.inputs]
target_temperature = 300.0
device = 0

[[schedules]]
schedule_id = 1
name = "replica_310K"

[schedules.inputs]
target_temperature = 310.0
device = 1
```

`schedule_id` 必须唯一，推荐使用 `0, 1, 2, ...`。

## 输出目录与输出前缀

推荐只设置：

```toml
[worker_defaults]
working_directory_root = "replicas"

[worker_defaults.inputs]
default_out_file_prefix = "mdout"
```

schedule `0` 的工作目录会自动解析为 `replicas/0`，schedule `1` 为 `replicas/1`。

manager 会自动给 `default_out_file_prefix` 添加 schedule 后缀，避免输出互相覆盖：

```text
mdout_0
mdout_1
...
```

如果用户没有设置 `default_out_file_prefix`，默认基底是 `"mdout"`。

启动前 manager 会检查常见输出路径冲突，包括 mdout、info、trajectory、box、restart 以及 manager log。

### 路径解析规则

`working_directory_root` 可以写绝对路径，也可以写相对路径。相对路径会按 `manager.toml` 所在目录解析：

```toml
[worker_defaults]
working_directory_root = "replicas"
```

如果 `manager.toml` 位于 `/data/run/manager.toml`，则 schedule `0` 的工作目录是：

```text
/data/run/replicas/0
```

如果工作目录不存在，`SPONGE_MANAGER` 会在启动 worker 前自动创建。

`worker_defaults.inputs` 和 `schedules.inputs` 中的字符串路径参数会由 manager 解析。相对路径会按 `manager.toml` 所在目录解析成绝对路径，再传给 worker。例如：

```toml
[worker_defaults.inputs]
default_in_file_prefix = "WAT"
coordinate_in_file = "init_coordinate.txt"
```

如果 `manager.toml` 位于 `/data/run/manager.toml`，manager 实际传给 worker 的是：

```text
default_in_file_prefix = /data/run/WAT
coordinate_in_file = /data/run/init_coordinate.txt
```

当前会自动解析的 key 包括：

- `default_in_file_prefix`
- 以 `_in_file`、`_out_file`、`_file`、`_path`、`_directory` 结尾的字符串参数

`default_out_file_prefix` 不会被解析成绝对路径。它仍然是输出前缀，由 manager 按 worker 工作目录隔离并自动追加 schedule id。

因此纯 `manager.toml` 写法可以保持简洁：

```toml
[worker_defaults]
working_directory_root = "replicas"

[worker_defaults.inputs]
default_in_file_prefix = "system/WAT"
coordinate_in_file = "system/init_coordinate.txt"
velocity_in_file = "system/init_velocity.txt"
default_out_file_prefix = "mdout"
```

## T-REMD 示例

```toml
[manager]
block_steps = 1000
epochs = 100
transport = "tcp"

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
default_out_file_prefix = "tremd"
thermostat = "middle_langevin"

[schedules]
ids = [0, 1, 2]

[schedules.inputs]
target_temperature = [300.0, 310.0, 320.0]
```

## H-REMD / HT-REMD 示例

H-REMD 和 HT-REMD 需要 `hamiltonian_id`：

```toml
[exchange]
enabled = true
mode = "hremd"

[schedules]
ids = [0, 1]

[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = [0, 1]
lambda_lj = [0.0, 0.5]
```

`hamiltonian_id` 是 manager-only input，不会传给 worker；其它 input 会作为 SPONGE 参数覆写传入。

## REST2-REMD 示例

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

公共 mdin 中可以写：

```text
REST2_mode = on
REST2_atom_numbers = 22
REST2_lambda_m = 1.0
```

不同 schedule 的 `REST2_lambda_m` 会通过 `schedules.inputs` 覆写。

## FEP + REST2 NPT 批量平衡示例

下面示例展示如何用 `SPONGE_MANAGER` 调度 4 个 FEP 双拓扑 NPT 平衡 worker，并打开 REST2 副本交换。每个 schedule 使用不同的 soft-core `lambda_lj`，同时使用不同的 REST2 `lambda_m`。

```toml
[manager]
block_steps = 1000
epochs = 100
transport = "tcp"
log_path = "manager_exchange.log"

[exchange]
enabled = true
mode = "rest2"

[worker_defaults]
args = ["-dont_check_input", "1"]
working_directory_root = "fep_rest2_replicas"

[worker_defaults.inputs]
md_name = "FEP NPT REST2 manager run"
mode = "NPT"
dt = 0.002
cutoff = 8.0
constrain_mode = "SHAKE"
barostat = "andersen_barostat"
thermostat = "middle_langevin"
thermostat_tau = 0.1
thermostat_seed = 2026
target_temperature = 300.0
velocity_max = 20
REST2_mode = "on"
REST2_atom_numbers = 55
default_out_file_prefix = "fep_rest2"
write_information_interval = 1000
write_mdout_interval = 1000
write_trajectory_interval = 1000
write_restart_file_interval = 1000

[schedules]
ids = [0, 1, 2, 3]

[schedules.inputs]
lambda_lj = [0.0, 0.1, 0.2, 0.3]
REST2_lambda_m = [1.0, 0.9, 0.8, 0.7]
default_in_file_prefix = [
  "benchmarks/performance/rest2/data/fep_test_for_remd/0/TMP",
  "benchmarks/performance/rest2/data/fep_test_for_remd/1/TMP",
  "benchmarks/performance/rest2/data/fep_test_for_remd/2/TMP",
  "benchmarks/performance/rest2/data/fep_test_for_remd/3/TMP",
]
coordinate_in_file = [
  "benchmarks/performance/rest2/data/fep_test_for_remd/0/TMP_coordinate.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/1/TMP_coordinate.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/2/TMP_coordinate.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/3/TMP_coordinate.txt",
]
velocity_in_file = [
  "benchmarks/performance/rest2/data/fep_test_for_remd/0/TMP_velocity.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/1/TMP_velocity.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/2/TMP_velocity.txt",
  "benchmarks/performance/rest2/data/fep_test_for_remd/3/TMP_velocity.txt",
]
```

这里的相对路径假设 `manager.toml` 放在仓库根目录运行；manager 会按 `manager.toml` 所在目录解析这些输入路径。`REST2_atom_numbers = 55` 表示把该双拓扑测试体系中开头的配体/突变区域作为 REST2 hot region。若换成其它体系，应改成对应 hot atoms 数量或使用 REST2 atom file。

## 通信方式

```toml
[manager]
transport = "tcp"
```

可选值：

- `"tcp"`：默认模式，本机 loopback TCP。
- `"shm"`：控制消息走 TCP，大 payload 走 shared memory，适合单节点大 RuntimeState。
- `"file"`：文件 request/response fallback，通常只建议用于调试。

## step_limit

在 manager 管理模式下，总步数由：

```text
block_steps * epochs
```

决定。manager 会覆盖 worker 的托管 step limit，并移除 worker args 中的 `-step_limit`，避免 mdin/CLI 的旧 step limit 提前终止 worker。

## 输出控制

`SPONGE_MANAGER` 总是要求 worker 走正常 SPONGE 输出路径。是否每步打印、多久写一次 mdout、trajectory、restart 或 box，由 mdin 或 `worker_defaults.inputs` / `schedules.inputs` 中的 `write_*_interval` 参数控制。

manager 托管 worker 时，worker 的 stdout/stderr 不直接打印到屏幕，而是追加到各自工作目录下的 `worker_<id>_manager.log`。屏幕上只保留 manager 自己的摘要和错误信息。

因此 `manager.toml` 不提供 `emit_output` 开关。如果想减少文件 I/O，应把对应输出 interval 调大，或把不需要的输出 interval 设为 `0`。
