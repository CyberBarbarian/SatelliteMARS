
"""
单纯用FCFS算法的话，依然：xx轮*xx个
任务带着可观测卫星约束的条件到达
到达的话，如果当前卫星处于空闲状态，那么直接让卫星进行观测，如果处于忙碌状态，那么看看有没有其余可进行观测的卫星，如果有，直接进行观测，如果没有，放弃该任务
"""
import csv
import os.path
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from no_similate_env import MultiEnv
from no_similate_utils import Task, check_time_window

start = time.perf_counter()

EPOCH_NUM = 1000
STEP_NUM = 50  # 相当于50个目标，一步生成一个目标

scenario = "FCFS_xxx"
# 模型保存地址
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models_D/" + scenario + "/"
timestamp = time.strftime("%Y%m%d%H%M%S")

env = MultiEnv(agent_num=7)
total_step = 0
return_list = []
total_reward_list = []

# 需要任务的id，任务到达时间，任务奖励，任务内存消耗，任务时间消耗，任务结束时间 任务可被哪些卫星观测到

task_set = [[] for _ in range(1001)]
# 打开文件并创建csv阅读器 TODO:生成数据集csv文件
with open \
        (r'C:\Users\xts\PyProject\maddpg\graduate_project\generate_task\data\hot_MRL_data_50_1000_augmented_sorted.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    for row in reader:
        task = Task(name=f"tar_{row[0]}_{row[1]}",
                    longitude=float(row[8]), latitude=float(row[7]), arrival_time=float(row[2]),
                    exist_time=float(row[5]), reward=float(row[4]), cost_stor=float(row[6]),
                    observed_satellite=row[3], env=env)
        task_set[int(row[0])].append(task)
print("ok")
for episode in range(EPOCH_NUM):
    cur_task_set = task_set[episode +1]

    ep_returns = 0
    total_reward = 0
    env.reset()
    for step, task in enumerate(cur_task_set):

        # 对于每个到来的任务，先找到其满足条件的卫星，然后判断该卫星的状态，如果满足，就接收

        for i, name in enumerate(env.agents):
            if (task.observed_satellite[name] is True) and \
                    (check_time_window(task.arrival_time, task.end_time, env.time_window[name])
                    and (env.done[name] is False) and (env.remain_time[name] - task.exist_time) >= 0
                    and (env.remain_stor[name] - task.cost_stor) >= 0):
                env.reward[name] += task.reward
                env.remain_time[name] -= task.exist_time
                env.remain_stor[name] -= task.cost_stor
                total_reward += task.reward
                env.time_window[name].append((task.arrival_time, task.end_time))

        for name in env.agents:
            if env.remain_time[name] <= 10 or env.remain_stor[name] <= 0:
                env.done[name] = True

        if (all(done_value is True for done_value in env.done)):
            break

    total_reward_list.append(total_reward)

end = time.perf_counter()
runTime = end - start

print("运行时间：", runTime)

# TODO:再添一段记录每个episode中的total_reward
total_reward_array = np.array(total_reward_list)

writer = SummaryWriter("logs_fcfs")
# 查看tensorboard: 1.cd到logs所在的目标下  2.tensorboard --logdir=logs_v1

for i, reward in enumerate(total_reward_array):
    writer.add_scalar('total_reward', reward, i)

writer.close()
