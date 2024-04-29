import copy
import csv
import os.path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from MADDPG import MADDPG, no_similate_evaluate
# from graduate_project import utils
from no_similate_env import MultiEnv
from no_similate_utils import ReplayBuffer, Task, check_time_window

start = time.perf_counter()

# 超参数
EPOCH_NUM = 1000
STEP_NUM = 50  # 相当于50个目标，一步生成一个目标
BUFFERSIZE = 1000000
BATCH_SIZE = 512
HIDDEN_DIM = 64
UPDATE_INTERVAL = 10
MINIMAL_SIZE = 512  # 可以避免在缓冲区还不够大时就开始进行采样，从而确保训练的稳定性和有效性。
ACTOR_LR = 1e-2
CRITIC_LR = 1e-2
GAMMA = 0.95
TAU = 1e-2

scenario = "STK_EPISODE_100_noprint_no_similate_test2"
# 模型保存地址
current_path = os.path.dirname(os.path.realpath(__file__))
agent_path = current_path + "/models_D/" + scenario + "/"
timestamp = time.strftime("%Y%m%d%H%M%S")

# TODO：设置观测量，对观测结果进行绘图表示
# TODO:再设置一个仿真周期
env = MultiEnv(agent_num=7)
replay_buffer = ReplayBuffer(capacity=BUFFERSIZE)
states = env.reset()
states_dim = []
action_dim = []

for state in states.values():
    states_dim.append(len(state))
for action in env.actions.values():
    action_dim.append(len(action))

critic_input_dim = sum(states_dim) + sum(action_dim)
# TODO:应该把任务的状态也作为环境的一部分
agent = MADDPG(env=env, states_dim=states_dim, actions_dim=action_dim, hidden_dim_1=128, hidden_dim_2=100,
               critic_input_dim=critic_input_dim,
               actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU)
total_step = 0
return_list = []
total_reward_list = []

# 需要任务的id，任务到达时间，任务奖励，任务内存消耗，任务时间消耗，任务结束时间 任务可被哪些卫星观测到
# TODO: 一次性把csv里的数据都导给task，后续每一步直接用task，需要修改task类的输入
task_set = [[] for _ in range(1001)]
# 打开文件并创建csv阅读器
with open(r'data/augment/MRL_data_50_1000_augmented.csv',
          newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    for row in reader:
        task = Task(name=f"tar_{row[0]}_{row[1]}",
                    longitude=float(row[8]), latitude=float(row[7]), arrival_time=float(row[2]),
                    exist_time=float(row[5]), reward=float(row[4]), cost_stor=float(row[6]),
                    observed_satellite=row[3], env=env)
        task_set[int(row[0])].append(task)

for episode in range(EPOCH_NUM):
    cur_task_set = task_set[episode + 1]
    state_dict = env.reset()
    state = []
    for state_value in state_dict.values():
        state.append(state_value)
    ep_returns = 0
    total_reward = 0
    for step, task in enumerate(cur_task_set):
        # task = Task(f"tar_{episode}_{step}", step)  # 这里的生成一个目标，理解为进行一个step
        # 将任务状态填入环境状态
        for i in range(env.agent_num):
            state[i][2] = task.exist_time
            state[i][3] = task.cost_stor
            state[i][4] = task.reward

        # 1.产生动作,是两维的
        actions = agent.take_action(state, explore=True)
        # 判断硬约束条件：（把原来在env.step里的搬过来）  认为：应该在根据状态产生动作之前就进行硬约束条件的判断
        for i, name in enumerate(env.agents):
            if ((task.observed_satellite[name] is False)  # 是否可被当前卫星观测
                    or (env.done[name] is True)  # 当前卫星是否已经不可再工作
                    or (env.remain_time[name] - task.exist_time) < 0  # 当前卫星的剩余时间是否足以支持该任务的消耗
                    or (env.remain_stor[name] - task.cost_stor) < 0  # 当前卫星的剩余容量是否足以支持该任务的消耗
                    # 因为任务的到达时间没有被排过序，所以每个任务之间的关系都要被判断一遍
                    or not (
                            check_time_window(task.arrival_time, task.end_time,
                                              env.time_window[name]))):  # 当前任务的到达时间是否在可执行范围内
                actions[i][0][0] = 0

        # 2.环境更新
        next_obs_dict, reward_dict, done_dict = env.step(actions, task)
        # 3.存储经验
        # 3.1 处理数据格式
        next_obs = []
        reward = []
        done = []
        for next_obs_, reward_, done_ in zip(next_obs_dict.values(), reward_dict.values(), done_dict.values()):
            next_obs.append(next_obs_)
            reward.append(reward_)
            done.append(done_)  # 用next_obs_copy解决了
        next_obs_copy = copy.deepcopy(next_obs)

        replay_buffer.add(states=state, actions=actions, reward=reward, next_state=next_obs_copy, done=done)

        # 3.2 更新下一步状态
        state = next_obs_copy
        total_reward += sum(reward)
        total_step += 1
        # TODO:如果所有agent的done都为True，才终止， 如果有agent的done为True，那么应该冻结该agent，使其action变为[0,1]
        if all(done_value == True for done_value in done):
            break

        if (replay_buffer.size() >= MINIMAL_SIZE and total_step % UPDATE_INTERVAL == 0):
            sample = replay_buffer.sample(BATCH_SIZE)  # 满足采样条件才从缓冲池中进行采样


            def stack_sample(x):  # x 表示state/action/reward/next_state/done中的一个
                recover = [[sub_x[i] for sub_x in x]  # sub_x 表示其中的一个样本（包含好多agent）, sub_x[i]表示第i个agent的内容
                           for i in range(len(x[0]))]  # len(x[0]) 表明有几个样本 # recover按agent为核心进行划分维度，x是以经验的条数进行划分
                return [torch.FloatTensor(np.vstack(i))
                        for i in recover]


            # action的类型是array
            sample = [stack_sample(x) for x in sample]
            # 更新agent参数
            for agent_idx in range(env.agent_num):
                agent.update(sample, agent_idx)
            # 更新目标网络参数
            agent.update_all_target()
    total_reward_list.append(total_reward)
    print(f"{episode}完成！")
    # 达到一定episode后，进行评估
    if (episode + 1) % 1 == 0:
        # 评估的话，需要专门的评估数据，这里暂时先用下一批次的代替
        ep_returns = no_similate_evaluate(env=env, task_set=task_set, agent=agent)
        return_list.append(ep_returns)
    # 保存训练参数
    if episode == 0:
        highest_reward = total_reward

    if total_reward > highest_reward:
        highest_reward = total_reward
        print(f"Highest reward updated at episode {episode}:{round(highest_reward, 2)}")
        for agent_i in range(env.agent_num):
            cur_agent = agent.agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(cur_agent.actor.state_dict(),
                       f"{agent_path}" + f"agent_{agent_i}_actor_{scenario}_{timestamp}.pth")
        print("保存模型成功！")
end = time.perf_counter()
runTime = end - start

print("运行时间：", runTime)

# TODO:再添一段记录每个episode中的total_reward
total_reward_array = np.array(total_reward_list)
np.save('4_25_total_reward_array_test3.npy', total_reward_array)

return_array = np.array(return_list)
np.save('4_25_total_return_array_test3.npy', return_array)

writer = SummaryWriter("logs_v2_add_time")
# 查看tensorboard: 1.cd到logs所在的目标下  2.tensorboard --logdir=logs_v1

for i, reward in enumerate(total_reward_array):
    writer.add_scalar('total_reward_v2', reward, i)

for j in range(env.agent_num):
    for i, reward in enumerate(return_array[:, j]):
        writer.add_scalar(f'Yaogan_{j + 1}_reward_v2', reward, i)

writer.close()

# plt.figure()
# plt.plot(
#     np.arange(total_reward_array.shape[0]),  # [0, 100]
#     utils.moving_average(total_reward_array, 9)  # return_array[:, i]: [2, ]
# )
# plt.xlabel("Episodes")
# plt.ylabel("total_reward")
# plt.title("total_reward by MADDPG")
# plt.show()


# for i, agent_name in enumerate(env.agents):
#     plt.figure()
#     plt.plot(
#         np.arange(return_array.shape[0]) * 2,  # [0, 100]
#         utils.moving_average(return_array[:, i], 9)  # return_array[:, i]: [2, ]
#     )
#     plt.xlabel("Episodes")
#     plt.ylabel("Returns")
#     plt.title(f"{agent_name} by MADDPG")
# plt.show()
