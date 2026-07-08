# SatelliteMARS

SatelliteMARS 是一个面向卫星任务调度的多智能体强化学习实验项目。项目使用 STK 11 计算地面观测任务与卫星之间的可见时间窗口，并将可见性结果整理为离线训练数据，再使用 MADDPG 为多颗卫星学习任务接受/拒绝策略。

## 项目流程

项目分为数据生成、数据整理、离线训练和结果绘制四个阶段。

```text
mission.py
定义随机任务属性
    |
    v
create_mission.py
生成任务数据 data/missions*.csv
    |
    v
compute_access.py
连接 STK 11 场景，计算任务点与卫星的可见窗口
生成 data/access*.csv
    |
    v
handle_csv.py
对齐任务数据和可见窗口，筛选可观测任务
生成 data/MRL_data*.csv
    |
    v
sort.py
按批次内到达时间排序
生成 data/MRL_data_sorted*.csv
    |
    v
augment_data.py / merge_csv.py
构造训练用实验数据
生成 data/lab/*.csv
    |
    v
no_similate_train.py
使用离线任务集训练 MADDPG 调度策略
生成 models_D/<lab_name>/*.pth 和 data/reward/*_rewards.csv
    |
    v
plot_training_rewards.py
绘制 reward 曲线并保存到 pic/
```

STK 只参与访问窗口计算阶段。训练阶段读取已经整理好的 CSV 数据，不再实时调用 STK。

## 运行环境

### STK 访问窗口计算

重新生成 `data/access*.csv` 时，需要准备：

- Windows
- STK 11
- 已打开的 STK 场景：`scenario/RLSTAR.sc`
- Python 包：`pywin32`、`tqdm`

`compute_access.py` 通过 `win32com.client.GetActiveObject('STK11.Application')` 连接当前正在运行的 STK 11 应用，并读取场景中的 `Satellite1` 到 `Satellite7`。

### 离线训练与绘图

训练、数据处理和绘图阶段按脚本使用以下 Python 包：

- `no_similate_train.py` / `MADDPG.py` / `no_similate_env.py` / `no_similate_utils.py`：`numpy`、`torch`、`tqdm`
- `merge_csv.py`：`pandas`
- `plot_training_rewards.py`：`pandas`、`matplotlib`

已有的 `data/lab/` 数据可以直接用于训练；已有的 `data/reward/` 数据可以直接用于绘制曲线。

## 数据依赖

### 任务数据

`create_mission.py` 生成任务 CSV。每条任务包含：

- `batch_id`：任务批次
- `task_id`：批次内任务编号
- `latitude` / `longitude`：任务位置
- `arrival_time`：任务到达秒数
- `observation_duration`：观测时长
- `profit`：任务收益
- `memory_usage`：存储消耗

### 可见窗口数据

`compute_access.py` 读取任务 CSV，在 STK 中为每个任务位置创建 Place，并计算各卫星对该 Place 的 access interval。

输出字段：

- `batch_id`
- `task_id`
- `satellite`
- `intervals`

### 强化学习数据

`handle_csv.py` 将任务数据和可见窗口数据合并。脚本以 `18 Aug 2018 03:45:00.000` 作为 STK 场景基准时间，把任务的相对到达秒数转换为场景内绝对时间，并保留任务完整落在卫星可见窗口内的记录。

输出字段：

- `batch_id`
- `task_id`
- `arrival_time_seconds`
- `observable_by_satellites`
- `reward`
- `observation_duration`
- `memory_usage`
- `latitude`
- `longitude`

`no_similate_train.py` 使用的实验数据需要保持这个字段结构。

## 核心脚本

### `mission.py`

定义 `Mission` 类，用于随机生成任务位置、到达时间、观测时长、收益和存储消耗。

### `create_mission.py`

批量生成随机任务，并写入 `data/missions.csv` 或指定输出文件。

### `compute_access.py`

读取任务列表，连接已打开的 STK 11 场景，计算任务点对各卫星的可见时间窗口，并写入 access CSV。

### `handle_csv.py`

将任务到达秒数映射到 STK 场景时间，检查任务开始到结束是否完整处于可见窗口内，并生成强化学习使用的任务数据。

### `sort.py`

按 `batch_id` 分组，并在每个批次内按照 `arrival_time_seconds` 排序。

### `generate_data.py`

串联完整数据生成流程：

1. 生成任务
2. 计算 access interval
3. 合并任务与 access 结果
4. 按到达时间排序

可通过 `batch_size`、`batch_num`、`prefix` 和 `visible` 控制生成规模和输出位置。

### `augment_data.py`

从已有 `MRL_data` 格式数据中抽样任务，并随机扰动 reward、观测时长、存储消耗和到达时间，用于构造多批次训练数据。

### `merge_csv.py`

合并多个实验 CSV，并重新编号 `batch_id`，用于构造组合实验集。

### `no_similate_env.py`

定义离线多智能体环境 `MultiEnv`。环境维护每颗卫星的剩余时间、剩余存储容量、已接受任务时间段和当前观测状态。

每颗卫星的观测状态包括：

- 剩余时间比例
- 剩余存储比例
- 当前任务观测时长
- 当前任务存储消耗
- 当前任务收益

### `no_similate_utils.py`

提供训练辅助结构：

- `Task`：将 CSV 行转换为训练任务对象，并记录可观测该任务的卫星集合。
- `ReplayBuffer`：经验回放池。
- `check_time_window`：检查任务时间段是否与已安排任务冲突。
- `moving_average`：reward 曲线平滑。

### `MADDPG.py`

实现 MADDPG 训练所需的 actor、critic、target network、Gumbel-Softmax 离散动作采样和多智能体参数更新。

### `no_similate_train.py`

训练入口脚本。默认读取：

```text
data/lab/<lab_name>.csv
```

训练输出：

```text
data/reward/<lab_name>_rewards.csv
models_D/<lab_name>/agent_*_actor_*.pth
```

`data/reward/<lab_name>_rewards.csv` 会在训练开始时创建。Actor 参数会在 episode 总 reward 超过当前最高值时保存到 `models_D/<lab_name>/`。

训练参数在脚本顶部设置，包括 `lab_name`、`EPOCH_NUM`、`STEP_NUM`、学习率、折扣因子、batch size 和 replay buffer 大小。

### `plot_training_rewards.py`

读取 reward CSV，绘制总收益和各卫星收益曲线，并保存到 `pic/`。

## 调度约束

训练过程中，智能体会先为每颗卫星生成接受/拒绝动作，随后使用硬约束过滤不可执行动作。任务被接受前需要满足：

- 当前任务在该卫星的可见列表中。
- 卫星剩余观测时间足够。
- 卫星剩余存储容量足够。
- 当前任务时间段不与该卫星已接受任务冲突，并保留 `check_time_window` 中设置的 5 秒任务切换间隔。
- 卫星尚未进入 `done` 状态。

通过约束过滤后，环境执行剩余动作并计算 reward。也就是说，MADDPG 学习的是可行动作集合内的调度偏好，物理可见性和资源约束由规则层保证。

## 实验数据

`实验大纲.md` 记录了四组实验设置：

- 实验一：验证数据增强方法的有效性。
- 实验二：比较不同智能体数量对收益的影响。
- 实验三：比较不同任务规模对收益的影响。
- 实验四：在逐步增加任务规模的设置下验证学习能力。

相关输入数据保存在 `data/lab/`，训练 reward 保存在 `data/reward/`，曲线图片保存在 `pic/`。

## 运行示例

### 生成任务

```bash
python create_mission.py
```

默认生成 `data/missions.csv`，每批 30 个任务，共 2 个批次。

### 计算 STK 可见窗口

运行前先在 STK 11 中打开：

```text
scenario/RLSTAR.sc
```

然后执行：

```bash
python compute_access.py
```

默认读取 `data/missions.csv`，写入 `data/access.csv`，并以非可视化方式连接当前 STK 应用。

### 一键生成数据

`generate_data.py` 会依次运行任务生成、access 计算、CSV 合并和排序：

```bash
python generate_data.py
```

直接运行脚本时，当前入口会生成 10 组 `data/true/` 数据，每组调用 `generate_data(200, 500, visible=1, prefix=...)`。这一流程包含 `compute_access.py`，因此运行前同样需要打开 STK 11 场景。

### 训练模型

修改 `no_similate_train.py` 顶部的 `lab_name`，确认对应的 `data/lab/<lab_name>.csv` 存在，然后执行：

```bash
python no_similate_train.py
```

默认 `lab_name` 为 `lab4`，训练读取 `data/lab/lab4.csv`，reward 写入 `data/reward/lab4_rewards.csv`。

### 绘制 reward 曲线

修改 `plot_training_rewards.py` 中的 `lab_name`，然后执行：

```bash
python plot_training_rewards.py
```

默认 `lab_name` 为 `lab4`，脚本读取 `data/reward/lab4_rewards.csv` 并输出 `pic/lab4_*_reward_plot.png`。

## 目录结构

```text
.
├── data/                 # 任务数据、access 数据、实验数据和 reward 记录
│   ├── augment/          # 数据增强中间结果
│   ├── lab/              # 训练使用的实验数据集
│   ├── reward/           # 训练 reward CSV
│   └── true/             # STK 计算得到的真实访问数据
├── models_D/             # 训练得到的 actor 参数
├── pic/                  # reward 曲线图
├── scenario/             # STK 11 场景、卫星和传感器配置
├── MADDPG.py             # MADDPG 实现
├── no_similate_env.py    # 离线多智能体环境
├── no_similate_train.py  # 训练入口
└── 实验大纲.md           # 实验设置说明
```

## 默认设置

- 场景时间窗口：3600 秒
- 默认卫星数量：7
- 卫星命名：`Satellite1` 到 `Satellite7`
- 单星初始存储容量：150
- 任务收益范围：1 到 10
- 任务观测时长范围：3 到 6
- 任务存储消耗范围：3 到 6

这些设置分布在 `mission.py`、`no_similate_env.py`、`no_similate_train.py` 和 `实验大纲.md` 中，可按实验需要调整。
