import copy
import csv
import os.path
import time

import numpy as np
import torch
from tqdm import tqdm

from MADDPG import MADDPG, no_similate_evaluate
from no_similate_env import MultiEnv
from no_similate_utils import ReplayBuffer, Task, check_time_window

start = time.perf_counter()

lab_name = "lab_200"
EPOCH_NUM = 1500
STEP_NUM = 400
BUFFERSIZE = 1000000
BATCH_SIZE = 64
HIDDEN_DIM = 64
UPDATE_INTERVAL = 10
MINIMAL_SIZE = 512
ACTOR_LR = 1e-2
CRITIC_LR = 1e-2
GAMMA = 0.95
TAU = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent_path = "models/" + lab_name + "/"
timestamp = time.strftime("%Y%m%d%H%M%S")

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
agent = MADDPG(env=env, states_dim=states_dim, actions_dim=action_dim, hidden_dim_1=128, hidden_dim_2=100,
               critic_input_dim=critic_input_dim,
               actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU, device=device)
total_step = 0
return_list = []
total_reward_list = []

task_set = [[] for _ in range(1501)]

reward_name = 'data/reward/' + lab_name + '_rewards.csv'
data_name = 'data/lab/' + lab_name + '.csv'

with open(reward_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Epoch', 'Total Reward', 'Agent1 Reward', 'Agent2 Reward', 'Agent3 Reward', 'Agent4 Reward', 'Agent5 Reward',
         'Agent6 Reward', 'Agent7 Reward'])

with open(data_name, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    i = 0
    for row in tqdm(reader):
        i = i + 1
        task = Task(name=f"tar_{row[0]}_{row[1]}",
                    longitude=float(row[8]), latitude=float(row[7]), arrival_time=float(row[2]),
                    exist_time=float(row[5]), reward=float(row[4]), cost_stor=float(row[6]),
                    observed_satellite=row[3], env=env)
        try:
            task_set[int(row[0])].append(task)
        except:
            print(row[0])
            print(int(row[0]))
            print(i)

print("ok")

for episode in tqdm(range(EPOCH_NUM)):
    cur_task_set = task_set[episode + 1]
    state_dict = env.reset()
    state = []
    for state_value in state_dict.values():
        state.append(state_value)
    ep_returns = 0
    total_reward = 0
    total_step = 0
    agent_rewards = [0] * env.agent_num
    for step, task in enumerate(cur_task_set):
        for i in range(env.agent_num):
            state[i][2] = task.exist_time
            state[i][3] = task.cost_stor
            state[i][4] = task.reward

        actions = agent.take_action(state, explore=True)

        for i, name in enumerate(env.agents):
            if ((task.observed_satellite[name] is False)
                    or (env.done[name] is True)
                    or (env.remain_time[name] - task.exist_time) < 0
                    or (env.remain_stor[name] - task.cost_stor) < 0
                    or not (
                            check_time_window(task.arrival_time, task.end_time, env.time_window[name]))):
                actions[i][0][0] = 0

        next_obs_dict, reward_dict, done_dict = env.step(actions, task)

        next_obs = []
        reward = []
        done = []
        for next_obs_, reward_, done_ in zip(next_obs_dict.values(), reward_dict.values(), done_dict.values()):
            next_obs.append(next_obs_)
            reward.append(reward_)
            done.append(done_)
        next_obs_copy = copy.deepcopy(next_obs)

        replay_buffer.add(states=state, actions=actions, reward=reward, next_state=next_obs_copy, done=done)
        state = next_obs_copy
        total_reward += sum(reward)
        for i, agent_reward in enumerate(reward):
            agent_rewards[i] += agent_reward
        total_step += 1
        if all(done_value is True for done_value in done):
            break
        if replay_buffer.size() >= MINIMAL_SIZE and total_step % UPDATE_INTERVAL == 0:
            sample = replay_buffer.sample(BATCH_SIZE)


            def stack_sample(x):
                recover = [[sub_x[i] for sub_x in x]
                           for i in range(len(x[0]))]
                return [torch.FloatTensor(np.vstack(i)).to(device)
                        for i in recover]


            sample = [stack_sample(x) for x in sample]
            for agent_idx in range(env.agent_num):
                agent.update(sample, agent_idx)
            agent.update_all_target()
    total_reward_list.append(total_reward)
    with open(reward_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, total_reward] + agent_rewards)
    if (episode + 1) % 1 == 0:
        ep_returns = no_similate_evaluate(env=env, task_set=task_set, agent=agent)
        return_list.append(ep_returns)
    if episode == 0:
        highest_reward = total_reward

    if total_reward > highest_reward:
        highest_reward = total_reward
        for agent_i in range(env.agent_num):
            cur_agent = agent.agents[agent_i]
            flag = os.path.exists(agent_path)
            if not flag:
                os.makedirs(agent_path)
            torch.save(cur_agent.actor.state_dict(),
                       f"{agent_path}" + f"agent_{agent_i}_actor_{lab_name}_{timestamp}.pth")
end = time.perf_counter()
runTime = end - start

print("运行时间：", runTime)
writer.close()
