# SPONGE_MANAGER 下一版 TOML 调度语法草案

本文档记录下一版 `manager.toml` 调度语法的设计草案。它暂时不是当前实现的完整说明，
而是后续重构 `SPONGE_MANAGER` 配置解析、路径管理和输出冲突检查时的讨论基准。

## 设计目标

下一版语法希望把 `manager.toml` 从“多个 SPONGE 进程启动脚本”收敛为“多副本实验描述文件”。

核心目标：

- 默认保留 worker 输出，方便 FEP、REMD、REST2 等后续 rerun、轨迹分析和 restart。
- 公共 SPONGE 输入参数集中写在 `worker_defaults.inputs`。
- 每个 schedule 只描述自己的差异参数，例如温度、lambda、REST2 lambda、GPU device。
- 路径由 `working_directory_root` 和 schedule id 自动推断，避免用户手动维护大量目录。
- manager 启动前检查所有 schedule 的实际输出路径，发现冲突立即报错。

## 推荐 TOML 结构

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
mdin = "mdin.txt"
working_directory_root = "outputs/ice_ih_tremd"
emit_output = true

[worker_defaults.inputs]
mode = "NPT"
dt = 0.001
default_out_file_prefix = "mdout"

[[schedules]]
schedule_id = 0
name = "replica_220K"

[schedules.inputs]
target_temperature = 220.0
device = 0

[[schedules]]
schedule_id = 1
name = "replica_221K"

[schedules.inputs]
target_temperature = 221.0
device = 1
```

对于规则化的副本序列，也希望支持批量写法，避免重复书写多个 `[[schedules]]`：

```toml
[schedules]
ids = [0, 1, 2, 3, 4]

[schedules.inputs]
target_temperature = [220.0, 221.0, 222.0, 223.0, 224.0]
device = [0, 0, 0, 0, 0]
REST2_lambda_m = [1.0, 0.95, 0.90, 0.85, 0.80]
```

该写法等价于显式展开 5 个 `[[schedules]]`，每个 schedule 使用对应下标的输入值。
批量 `[schedules]` 写法禁止与逐项 `[[schedules]]` 写法混用。

## 字段职责

### `[manager]`

`[manager]` 只负责 manager 自己的调度行为。

建议保留：

- `block_steps`：每个 epoch 内每个 worker 运行多少步。
- `epochs`：总共运行多少个 manager epoch。
- `transport`：worker 通信方式，例如 `tcp`、`shm`、`file`。
- `log_path`：manager exchange log 输出路径。

建议迁移：

- `emit_output` 不再放在 `[manager]`，改放入 `[worker_defaults]`。因为它本质上控制 worker 是否写常规 SPONGE 输出，而不是 exchange manager 算法本身。

### `[exchange]`

`[exchange]` 只描述是否做副本交换，以及采用哪种交换算法。

建议字段：

- `enabled`：是否启用 exchange。
- `mode`：可选 `tremd`、`hremd`、`htremd`、`rest2`。
- `start_round`：初始 exchange round。
- `pairing`：未来预留，用于配置 odd-even、all-neighbor 或其它配对策略。

### `[worker_defaults]`

`[worker_defaults]` 描述所有 worker 共享的启动配置、公共输入和路径根目录。

建议字段：

- `executable`：SPONGE worker 可执行文件路径。默认情况下可以省略，由 manager 自动推断。
- `mdin`：公共 mdin 文件路径。
- `args`：保留为高级 escape hatch，用于少数无法被结构化字段覆盖的参数。
- `working_directory_root`：所有 schedule 工作目录的公共根目录。
- `emit_output`：是否让 worker 输出常规 mdout、轨迹、restart 等文件。默认值建议为 `true`。

### SPONGE 可执行文件自动推断

一般情况下，`SPONGE_MANAGER` 和 `SPONGE` 会安装或构建在同一个 pixi 环境中，因此用户不应该被要求手动填写 `worker_defaults.executable`。

默认推断规则建议为：

```text
1. 如果用户显式设置 worker_defaults.executable，则使用该路径。
2. 否则优先在 SPONGE_MANAGER 可执行文件所在目录查找同名平台可执行文件：
   - Linux/macOS: SPONGE
   - Windows: SPONGE.exe
3. 如果同目录不存在，则从 PATH 中查找 SPONGE。
4. 如果仍然找不到，manager 报错并提示用户设置 worker_defaults.executable。
```

在 pixi 环境中，常规运行方式应简化为：

```bash
pixi run SPONGE_MANAGER --config manager.toml
```

对应的 `manager.toml` 通常不需要写：

```toml
[worker_defaults]
executable = "./SPONGE"
```

只有在以下场景才建议显式设置 `executable`：

- 调试不同构建目录里的 SPONGE。
- 对比 baseline 与开发版 SPONGE。
- 集群环境中 manager 与 worker 可执行文件路径不一致。
- 用户希望固定使用某个绝对路径，避免 PATH 或环境差异。

推荐把公共 SPONGE 参数写入：

```toml
[worker_defaults.inputs]
dt = 0.001
mode = "NPT"
default_out_file_prefix = "mdout"
```

不推荐命名为 `[worker_defaults.mdin]`，因为这不是完整 mdin 文件，而是 manager 转换为 SPONGE CLI 覆写参数的一组 key-value。
因此建议与 `schedules.inputs` 保持一致，统一叫 `inputs`。

### `[[schedules]]`

每个 `[[schedules]]` 描述一个副本槽位。

建议字段：

- `schedule_id`：schedule 唯一 id。推荐并默认使用 `0, 1, 2, 3, 4, ...` 这种连续整数形式。
- `name`：用户可读名称。如果省略，可自动设为 `schedule_<id>`。
- `inputs`：当前 schedule 对公共输入的覆写。
- `worker`：保留为可选高级配置，但常规使用不应该依赖它。

建议移除或弱化：

- `working_directory`：不推荐用户逐个 schedule 手写。默认由 `working_directory_root` 和 schedule id 自动推断。

### `[schedules]` 批量写法

除 `[[schedules]]` 逐个声明外，下一版语法建议支持 `[schedules]` 批量写法。

适用场景：

- T-REMD 温度序列。
- REST2 lambda 序列。
- FEP lambda windows。
- 一组只在少数 scalar 参数上不同的普通 worker。

推荐语法：

```toml
[schedules]
ids = [0, 1, 2, 3, 4]

[schedules.inputs]
target_temperature = [220.0, 221.0, 222.0, 223.0, 224.0]
device = [0, 0, 0, 0, 0]
```

展开规则：

- `ids` 必须是整数列表。
- `ids` 推荐使用连续整数 `0, 1, 2, 3, 4, ...`。
- `schedules.inputs` 中如果某个值是列表，则列表长度必须等于 `ids` 长度。
- `schedules.inputs` 中如果某个值是 scalar，则广播到所有 schedule。
- 展开后，每个 schedule 的 `schedule_id` 取 `ids[i]`，每个 list input 取第 `i` 项。

例如上面的配置等价于：

```toml
[[schedules]]
schedule_id = 0
[schedules.inputs]
target_temperature = 220.0
device = 0

[[schedules]]
schedule_id = 1
[schedules.inputs]
target_temperature = 221.0
device = 0
```

为了避免语法歧义，不允许在同一个 `manager.toml` 中同时使用 `[schedules]` 批量写法和多个 `[[schedules]]` 逐项写法。若两者同时出现，manager 应直接报错。

## 参数合并规则

最终传给 worker 的输入参数建议按如下优先级合并：

```text
mdin 文件 < worker_defaults.inputs < schedules.inputs < manager 内部托管参数
```

含义：

- mdin 文件定义体系和大部分通用 MD 参数。
- `worker_defaults.inputs` 定义本次 manager 任务的公共覆写参数。
- `schedules.inputs` 定义每个副本槽位的差异参数。
- manager 内部托管参数用于保证调度正确，例如托管 `step_limit`，不能被用户参数覆盖。

示例：

```toml
[worker_defaults.inputs]
target_temperature = 220.0
default_out_file_prefix = "mdout"

[[schedules]]
schedule_id = 3

[schedules.inputs]
target_temperature = 223.0
device = 1
```

则 schedule 3 的有效输入中：

```text
target_temperature = 223.0
default_out_file_prefix = mdout_<schedule_id>
device = 1
```

## 输出与路径规则

### 工作目录

用户只需要设置一次：

```toml
[worker_defaults]
working_directory_root = "outputs/remd"
```

每个 schedule 的工作目录默认自动推断：

```text
<working_directory_root>/<schedule_id>
```

例如：

```text
outputs/remd/0
outputs/remd/1
outputs/remd/2
```

如果设置了 `name`，可以考虑使用更可读的目录名：

```text
outputs/remd/replica_220K
outputs/remd/replica_221K
```

确定规则：工作目录默认始终使用数字 `schedule_id`，不使用 `name`。

也就是说，即使用户设置：

```toml
[[schedules]]
schedule_id = 3
name = "replica_223K"
```

默认工作目录仍然是：

```text
<working_directory_root>/3
```

这样可以保证路径稳定、容易脚本化，并避免 `name` 中特殊字符影响文件系统路径。

### 输出前缀

`default_out_file_prefix` 推荐只在 `worker_defaults.inputs` 中设置一次。每个 schedule 的实际输出前缀由 manager 自动追加数字 schedule id。

如果用户没有显式设置 `default_out_file_prefix`，manager 默认使用：

```text
mdout
```

然后按 schedule 自动追加后缀：

```text
mdout_<schedule_id>
```

例如：

```text
outputs/remd/0/mdout_0.out
outputs/remd/1/mdout_1.out
```

如果用户在 `worker_defaults.inputs` 中设置：

```toml
default_out_file_prefix = "tremd"
```

则 schedule 0 和 schedule 1 的实际前缀为：

```text
tremd_0
tremd_1
```

这意味着用户通常只需要在公共区域写：

```toml
[worker_defaults.inputs]
default_out_file_prefix = "mdout"
```

而不需要在每个 `[[schedules]]` 或批量 `[schedules.inputs]` 中重复设置。

常规情况下，不推荐在 `schedules.inputs` 中设置 `default_out_file_prefix`。schedule 的输出隔离应该由 manager 按数字 id 自动完成，而不是由用户手动维护。

如果用户确实在某个 `schedules.inputs` 中显式设置 `default_out_file_prefix`，manager 可以把它作为高级覆写处理，但仍然必须自动追加该 schedule 的数字 id，并执行输出冲突检查。

### `emit_output`

`emit_output` 默认建议改为：

```toml
emit_output = true
```

原因：

- FEP、REMD、REST2 等通常需要轨迹、restart 或能量输出做后续 rerun 与分析。
- 默认写输出更符合普通用户预期。
- benchmark 或调试通信开销时，可以显式设置 `emit_output = false`。

## 启动前输出冲突检查

manager 在启动任何 worker 前，应根据最终解析结果构建完整的输出路径表。

至少检查：

- 不同 schedule 的实际 `working_directory` 是否相同。
- 不同 schedule 的实际 `default_out_file_prefix` 是否会生成相同输出文件。
- `log_path` 是否和任意 worker 输出文件冲突。
- 如果未来支持 restart、trajectory、energy、force dump 等更多输出类型，也应该纳入同一套检查。

建议行为：

- 如果检测到冲突，manager 直接报错并退出。
- 报错信息需要列出冲突的 schedule id、文件类型和完整路径。
- 不应启动部分 worker 后才发现冲突。

示例错误信息：

```text
Output path conflict:
  schedules 0 and 1 both write outputs/remd/mdout.out
Please set worker_defaults.working_directory_root or distinct default_out_file_prefix values.
```

## 与现有语法的迁移关系

当前实现中常见写法：

```toml
[manager]
emit_output = false

[worker_defaults]
args = ["-mdin", "mdin.txt", "-dont_check_input", "1"]
working_directory_root = "replicas"

[[schedules]]
schedule_id = 0
working_directory = "replica_0"

[schedules.inputs]
target_temperature = 220.0
default_out_file_prefix = "mdout"
```

下一版建议写法：

```toml
[worker_defaults]
mdin = "mdin.txt"
working_directory_root = "replicas"
emit_output = true

[worker_defaults.inputs]
default_out_file_prefix = "mdout"

[[schedules]]
schedule_id = 0

[schedules.inputs]
target_temperature = 220.0
```

兼容策略建议：

- 初期继续接受旧字段，但打印 deprecation warning。
- `worker_defaults.args = ["-mdin", "..."]` 继续保留，避免破坏已有脚本。
- 如果同时设置 `worker_defaults.mdin` 和 `worker_defaults.args` 中的 `-mdin`，应报错或明确规定优先级。
- 如果用户仍显式设置 `schedules.working_directory`，初期可以继续支持，但推荐警告并迁移到自动目录规则。

## 当前实现决策

这些规则已经按第一版实现落地：

- `emit_output` 推荐写在 `[worker_defaults]`，默认值为 `true`；`[manager].emit_output` 暂时保留为旧式兼容别名。
- `worker_defaults.mdin` 会转换为 `-mdin <path>`；如果 `worker_defaults.args` 中同时出现 `-mdin`，manager 直接报错。
- `worker_defaults.inputs` 先合并到每个 schedule，随后由 `schedules.inputs` 覆写。
- 批量 `[schedules]` 支持 list input 按 `ids` 展开，scalar input 广播到所有 schedule。
- 所有 `schedule_id` 必须唯一。
- 批量 `[schedules]` 与逐项 `[[schedules]]` 禁止混用；TOML 解析阶段或 manager 配置解析阶段应拒绝这种配置。
- 第一版输出冲突检查覆盖常规 `default_out_file_prefix` 派生路径：prefix 本身、`.info`、`.out`、`.dat`、`.box`，并检查 `manager.log_path` 是否与这些路径冲突。

后续仍可继续扩展：

- 更完整覆盖用户显式指定的 trajectory/restart/force trajectory 文件名。
- 对旧字段输出更清晰的 deprecation warning。
- 增加更多配置错误路径单元测试。
