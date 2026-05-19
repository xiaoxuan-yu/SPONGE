# SPONGE_MANAGER 调度语法说明

本文档说明当前 `SPONGE_MANAGER` 的任务调度语法和运行模型。当前推荐方式是使用
一个 `manager.toml` 文件统一描述多个 SPONGE worker、每个 worker 的运行目录、
mdin 参数覆写，以及是否执行 REMD/REST2-REMD 交换。

## 基本概念

`SPONGE_MANAGER` 是外部调度器，`SPONGE` 是实际执行 MD 的 worker。

一次 manager 运行中有三层概念：

- `manager`：全局调度器，决定每个 block 跑多少步、跑多少个 epoch、用什么通信方式。
- `schedule`：一个 Hamiltonian/temperature slot，也就是一个被 manager 管理的副本槽位。
- `worker`：一个常驻的 SPONGE 子进程，负责执行对应 schedule 的 MD block、导出/导入 runtime state、做 foreign-state probe。

当前的调度模型是：

```text
SPONGE_MANAGER
  -> schedule[0] -> worker[0] -> SPONGE
  -> schedule[1] -> worker[1] -> SPONGE
  -> schedule[2] -> worker[2] -> SPONGE
  ...
```

每个 epoch 的流程是：

```text
1. manager 并行要求所有 worker 各跑一个 block
2. worker 返回 observable 和完整 RuntimeState
3. 如果启用 REMD，manager 做 odd-even 邻居交换尝试
4. 如果交换接受，manager 交换两个 schedule 的 RuntimeState
5. 写 exchange log 和 schedule state
6. 进入下一个 epoch
```

注意：当前 worker 是 child-process persistent 模式。worker 子进程会常驻，block 之间不会重复初始化 SPONGE。

## 启动方式

推荐命令：

```bash
SPONGE_MANAGER --config manager.toml
```

如果不提供 `--config`，当前只会进入一个很小的内置 skeleton/demo 路径，不适合作为正式调度入口。

## manager.toml 总体结构

一个典型文件如下：

```toml
[manager]
block_steps = 1000
epochs = 100
transport = "tcp"
log_path = "manager_exchange.log"

[exchange]
enabled = true
mode = "tremd"
start_round = 0

[worker_defaults]
mdin = "mdin.spg.toml"
emit_output = true
args = ["-dont_check_input", "1"]
working_directory_root = "replicas"

[worker_defaults.inputs]
default_out_file_prefix = "mdout"

[schedules]
ids = [0, 1]

[schedules.inputs]
target_temperature = [300.0, 310.0]
```

## `[manager]`

`[manager]` 控制全局调度行为。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `block_steps` | int | `1000` | 每个 epoch 中，每个 worker 运行的 MD 步数。 |
| `epochs` | int | `1` | manager 总共执行多少个调度 epoch。 |
| `emit_output` | bool | 不推荐新写 | 旧式字段；新语法推荐写在 `[worker_defaults]`。 |
| `transport` | string | `"tcp"` | worker 通信方式，可选 `"tcp"`、`"shm"`、`"file"`。 |
| `log_path` | string | `manager_exchange.log` | exchange log 输出路径。 |
| `exchange_log_path` | string | `manager_exchange.log` | `log_path` 的旧别名。 |
| `remd_mode` | string | 空 | 旧式字段，推荐改用 `[exchange].mode`。 |
| `exchange_round` | int | `0` | 旧式字段，推荐改用 `[exchange].start_round`。 |

路径规则：

- `log_path` 如果是相对路径，会相对于 `manager.toml` 所在目录解析。
- `block_steps` 和 `epochs` 必须为正数。

### `emit_output`

新语法推荐把 `emit_output` 写在 `[worker_defaults]`，默认值为 `true`。

`emit_output = false` 时，worker 仍会运行 MD、返回 observable 和 RuntimeState，但不会每步调用完整输出路径。这样适合 benchmark 或 REMD 调度。

`emit_output = true` 时，会让 worker 在 block 内按 mdin 的输出 interval 写文件。需要注意多副本输出前缀，避免互相覆盖。

## `[exchange]`

`[exchange]` 控制是否做副本交换，以及使用哪种交换算法。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | bool | 根据 `mode` 推断 | 是否启用交换。 |
| `mode` | string | 空 | 可选 `"tremd"`、`"hremd"`、`"htremd"`、`"rest2"`。 |
| `start_round` | int | `0` | 初始 exchange round。 |
| `pairing` | string | 暂未使用 | 预留字段。当前实现使用 odd-even 邻居交换。 |

如果 `enabled = false`，即使设置了 mode，也不会做 REMD。

当前交换配对是 odd-even：

```text
round 0: (0,1), (2,3), ...
round 1: (1,2), (3,4), ...
round 2: (0,1), (2,3), ...
```

## `[worker_defaults]`

`[worker_defaults]` 给所有 schedule 提供默认 worker 配置。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `executable` | string | 自动推断 | SPONGE worker 路径。 |
| `executable_path` | string | 自动推断 | `executable` 的等价字段。 |
| `mdin` | string | 空 | 公共 mdin 文件路径，会转换成 `-mdin <path>`。 |
| `args` | string array | 空 | 传给 SPONGE 的基础命令行参数。 |
| `working_directory_root` | string | 空 | schedule 相对工作目录的公共根目录。 |
| `emit_output` | bool | `true` | 是否让 worker 写常规 mdout/trajectory/restart。 |
| `inputs` | table | 空 | 所有 schedule 共享的 SPONGE 参数覆写。 |

推荐写法：

```toml
[worker_defaults]
mdin = "mdin.spg.toml"
args = ["-dont_check_input", "1"]
working_directory_root = "replicas"
emit_output = true

[worker_defaults.inputs]
default_out_file_prefix = "mdout"
```

如果不设置 `executable`，manager 会优先查找 `SPONGE_MANAGER` 同目录下的 `SPONGE` / `SPONGE.exe`，然后查找 `PATH`。
如果 `executable` 是相对路径，会相对于 `manager.toml` 所在目录解析。

## `[[schedules]]`

每个 `[[schedules]]` 定义一个副本槽位。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `schedule_id` | int | 数组下标 | schedule 的唯一 id，不能重复，推荐使用 `0, 1, 2, ...`。 |
| `name` | string | `schedule_<id>` | schedule 名称。 |
| `label` | string | 无 | `name` 的别名；如果存在，优先使用 `label`。 |
| `working_directory` | string | `<working_directory_root>/<schedule_id>` | 旧式手动工作目录字段；新语法默认按 id 自动推断。 |
| `inputs` | table | 空 | 传给 worker 的参数覆写。 |
| `worker` | table | 空 | 当前 schedule 专属的 worker 设置。 |

### 工作目录解析规则

新语法推荐只设置一次 `worker_defaults.working_directory_root`，每个 schedule 的工作目录自动变成：

```text
<working_directory_root>/<schedule_id>
```

例如：

```toml
[worker_defaults]
working_directory_root = "replicas"

[[schedules]]
schedule_id = 0
```

实际目录是：

```text
<manager.toml 所在目录>/replicas/0
```

## `[schedules.inputs]`

`schedules.inputs` 是 manager 调度语法中最重要的部分。

这里的键值会被转换成 SPONGE 命令行参数：

```toml
[schedules.inputs]
target_temperature = 300.0
REST2_lambda_m = 0.8
```

会变成 worker 命令行上的：

```bash
-target_temperature 300.0 -REST2_lambda_m 0.8
```

也就是说，`schedules.inputs` 可以覆写 mdin 里的同名参数。这个设计使得用户可以：

- 用一份公共 `mdin` 定义体系和大部分 MD 参数。
- 用多个 schedule 的 `inputs` 定义温度、lambda、REST2 lambda 等副本差异。

支持的值类型：

- integer
- float
- bool
- string

逐项 `[[schedules]]` 写法中不支持数组或嵌套表作为 input value。

批量 `[schedules]` 写法中，`schedules.inputs` 支持 list 值按 `ids` 下标展开，也支持 scalar 值广播到所有 schedule：

```toml
[schedules]
ids = [0, 1, 2, 3]

[schedules.inputs]
target_temperature = [300.0, 310.0, 320.0, 330.0]
device = 0
```

批量 `[schedules]` 不能和逐项 `[[schedules]]` 在同一个 `manager.toml` 中混用。

### 输出前缀自动隔离

manager 会自动处理 `default_out_file_prefix`，避免多个 schedule 输出互相覆盖。

规则是：

```text
实际 default_out_file_prefix = 用户给定前缀 + "_" + schedule_id
```

推荐把前缀写在公共区域：

```toml
[worker_defaults.inputs]
default_out_file_prefix = "rest2"
```

schedule 0 实际传给 worker：

```text
default_out_file_prefix = rest2_0
```

schedule 1 实际传给 worker：

```text
default_out_file_prefix = rest2_1
```

如果用户没有设置 `default_out_file_prefix`，默认基底是 `"mdout"`，所以会得到：

```text
mdout_0
mdout_1
...
```

### manager-only input

当前 `hamiltonian_id` 是 manager-only input：它用于 H-REMD/HT-REMD 的调度校验和 Hamiltonian 标识，但不会传给 worker。

其它 input 通常都会作为 `-key value` 传给 SPONGE worker。

## `[schedules.worker]`

每个 schedule 可以覆盖或追加自己的 worker 设置。常规使用不推荐设置这些字段，只有调试不同 worker 可执行文件或追加特殊参数时才需要。

```toml
[[schedules]]
schedule_id = 0

[schedules.worker]
name = "worker_0"
executable_path = "/path/to/SPONGE"
args = ["-extra_flag", "value"]
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | worker 名称。 |
| `executable_path` | string | 当前 schedule 专用的 SPONGE 路径。 |
| `working_directory` | string | 当前 schedule 专用工作目录；也可写在 `[[schedules]]` 顶层。 |
| `args` | string array | 追加到 `worker_defaults.args` 后面的参数。 |

注意：`schedules.worker.args` 是追加，不是替换 `worker_defaults.args`。

## 通信方式 `transport`

当前 `transport` 在 `[manager]` 中统一设置，对所有 schedule 生效。

可选值：

- `"tcp"`：默认模式。manager 与 worker 通过本机 loopback TCP 通信。
- `"shm"`：控制通道仍基于 TCP，RuntimeState 等大 payload 走 shared memory。
- `"file"`：通过临时文件目录传 request/response。当前也是 persistent worker，但通常比 TCP/shm 慢。

示例：

```toml
[manager]
transport = "shm"
```

如果不设置，默认是 `"tcp"`。

## step_limit 如何处理

manager 会根据：

```text
managed_step_limit = block_steps * epochs
```

给 worker 设置托管步数上限。

同时，manager 会从 worker args 中移除 `-step_limit`，避免 mdin/CLI 自己的 `step_limit` 提前让 worker 停掉。

因此在 manager 管理模式下，应把模拟总长度理解为：

```text
总步数 = block_steps * epochs
```

而不是由 mdin 里的 `step_limit` 决定。

## 普通多 worker 调度示例

不做 REMD，只批量运行多个 SPONGE worker：

```toml
[manager]
block_steps = 1000
epochs = 1
transport = "tcp"

[exchange]
enabled = false

[worker_defaults]
mdin = "mdin.spg.toml"
emit_output = true
args = ["-dont_check_input", "1"]
working_directory_root = "runs"

[worker_defaults.inputs]
default_out_file_prefix = "md"

[schedules]
ids = [0, 1]

[schedules.inputs]
target_temperature = [300.0, 310.0]
```

运行：

```bash
SPONGE_MANAGER --config manager.toml
```

## T-REMD 示例

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
mdin = "mdin.spg.toml"
emit_output = false
args = ["-dont_check_input", "1"]
working_directory_root = "replicas"

[worker_defaults.inputs]
default_out_file_prefix = "tremd"

[schedules]
ids = [0, 1]

[schedules.inputs]
target_temperature = [300.0, 310.0]
```

T-REMD 要求每个 schedule 提供：

```toml
target_temperature = ...
```

接受率使用温度 REMD 公式。接受交换时会交换完整 RuntimeState，并按温度比缩放速度。

## H-REMD 示例

```toml
[exchange]
enabled = true
mode = "hremd"

[[schedules]]
schedule_id = 0
[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = 0
lambda_lj = 0.0

[[schedules]]
schedule_id = 1
[schedules.inputs]
target_temperature = 300.0
hamiltonian_id = 1
lambda_lj = 0.5
```

H-REMD 要求：

```toml
target_temperature = ...
hamiltonian_id = ...
```

其中 `hamiltonian_id` 不传给 worker，只是 manager 用来确认这是 Hamiltonian slot。具体物理参数仍通过其它 input 或 worker mdin 控制。

## HT-REMD 示例

```toml
[exchange]
enabled = true
mode = "htremd"
```

HT-REMD 同时要求：

```toml
target_temperature = ...
hamiltonian_id = ...
```

接受交换时会同时考虑温度和 Hamiltonian cross-evaluation，并在接受交换时按温度比缩放速度。

## REST2-REMD 示例

REST2-REMD 使用 `exchange.mode = "rest2"`。所有 schedule 通常使用相同的物理 thermostat 温度，只改变 `REST2_lambda_m`。

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
mdin = "mdin.spg.toml"
emit_output = false
args = ["-dont_check_input", "1"]
working_directory_root = "replicas"

[worker_defaults.inputs]
target_temperature = 300.0
default_out_file_prefix = "rest2"

[schedules]
ids = [0, 1, 2]

[schedules.inputs]
REST2_lambda_m = [1.0, 0.9, 0.8]
```

REST2-REMD 要求每个 schedule 提供：

```toml
target_temperature = ...
REST2_lambda_m = ...
```

`REST2_lambda_m` 会传给 worker，覆写 mdin 中的同名设置。因此可以用一份公共 mdin：

```text
REST2_mode = on
REST2_atom_numbers = 22
REST2_lambda_m = 1.0
```

然后由 `manager.toml` 中不同 schedule 的 `REST2_lambda_m` 决定每个副本槽位的 Hamiltonian。

## exchange log

manager 会写 CSV 格式 exchange log。默认文件名：

```text
manager_exchange.log
```

字段包括：

```text
record_type,mode,epoch,exchange_round,pair_index,schedule_id,walker_id,
left_schedule,right_schedule,step,time_ps,potential_energy,effective_potential,
temperature,volume,log_acceptance,acceptance_probability,random_value,accepted
```

两类记录：

- `exchange_attempt`：一次交换尝试，包括配对、log acceptance、接受概率、随机数、是否接受。
- `schedule_state`：当前 epoch 后每个 schedule 持有的 walker、step、能量、温度、体积。

可以用这个 log 分析：

- acceptance ratio
- random stream 是否推进
- walker diffusion
- 每个 schedule 的状态流动

## 当前实现边界

当前 manager 调度框架已经支持：

- 多个 persistent child-process worker。
- worker block 并行运行。
- `tcp` / `shm` / `file` 三种 transport。
- manager 通过完整 RuntimeState 交换坐标、速度、box、thermostat/barostat 状态、RNG 状态等。
- `schedules.inputs` 覆写 mdin 参数。
- `tremd`、`hremd`、`htremd`、`rest2` 四种 exchange mode。

当前需要注意的限制：

- `transport` 目前是 manager 全局设置，不是每个 schedule 单独设置。
- `pairing` 字段已解析但暂未使用；当前固定 odd-even 邻居交换。
- 逐项 `[[schedules]]` 的 `schedules.inputs` 只支持 scalar；批量 `[schedules]` 的 `schedules.inputs` 支持 list 展开。
- REST2 第一版只缩放 short-range direct LJ/Coulomb，不缩放 PME reciprocal、bonded、1-4。
- SITS 仍被标记为 foreign-state probe unsafe，因此不能直接用于 H-REMD/REST2 风格 cross-evaluation。

## 推荐实践

- 使用 `manager.toml` 作为 manager 唯一入口。
- mdin 保留体系和通用 MD 参数；副本差异放在 `schedules.inputs`。
- 只设置一次 `worker_defaults.working_directory_root`，让 manager 按数字 schedule id 自动生成工作目录。
- 在 `worker_defaults.inputs` 中设置一次 `default_out_file_prefix`，让 manager 自动追加 schedule id。
- REST2-REMD 中保持所有 schedule 的 `target_temperature` 相同，只改变 `REST2_lambda_m`。
- 长时间测试前先用小 `block_steps`、小 `epochs` 做 smoke test，确认 exchange log 和输出路径正确。
