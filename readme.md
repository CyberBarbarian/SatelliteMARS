# README

## 参数设置

星机多智能体默认配置为5个卫星，2个WRJ，场景时间设置为3600s，任务奖励（1-10）、任务所需观测时间（3-6）均在一定范围内随机生成，每个场景默认训练1000批次，仿真软件场景存放在`scenario/RLSTAR.sc`中；在默认配置下，共设计有四个实验， 分别为`lab_50`，`lab_100`，`lab_200`，`lab_400`，每批次分别传入50、100、200、400个观测任务。

## 运行实验

运行`no_similate_train.py`，读取场景下的指定实验文件，记录训练过程中不同批次的单个智能体奖励函数收益和总奖励函数收益，以CSV格式存放到`data/reward`/中，并将训练过程中奖励函数收益最高的模型存储到`models/`中；

> 输入：`data/lab/lab_50.csv`，`data/lab/lab_100.csv`，`data/lab/lab_200.csv`，`data/lab/lab_400.csv`
>
> 输出：`data/reward/lab_50_rewards.csv`，`data/reward/lab_100_rewards.csv`，`data/reward/lab_200_rewards.csv`，`data/reward/lab_400_rewards.csv`

运行`plot_training_rewards.py`，读取`data/reward`/中的收益文件并绘制模型奖励函数曲线。

> 输入：`data/reward/lab_50_rewards.csv`，`data/reward/lab_100_rewards.csv`，`data/reward/lab_200_rewards.csv`，`data/reward/lab_400_rewards.csv`
>
> 输出：`pic/lab/lab_50_Agent1(1-7) Reward_reward_plot.png`，`pic/lab/lab_50_total_reward_plot.png`，
>
> `pic/lab/lab_100_Agent1(1-7) Reward_reward_plot.png`，`pic/lab/lab_100_total_reward_plot.png`，
>
> `pic/lab/lab_200_Agent1(1-7) Reward_reward_plot.png`，`pic/lab/lab_200_total_reward_plot.png`，
>
> `pic/lab/lab_400_Agent1(1-7) Reward_reward_plot.png`，`pic/lab/lab_400_total_reward_plot.png`