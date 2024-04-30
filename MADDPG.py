import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from no_similate_utils import check_time_window


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = tens_type(*shape).uniform_()  # uniform_()用于在指定范围内生成随机数的张量方法
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def onehot_from_logits(logits, eps=0.01):
    """生成最优动作的独热形式"""
    # 要求输入logits至少为两维的[[]]
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()  # [1, 0]  # [10,2]

    rand_acs = torch.eye(logits.shape[1])[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ].clone().detach().to(logits.device)  # device

    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def gumbel_softmax(logits, temperature=0.1):
    """从Gumbel-Softmax分布中采样,并进行离散化, 目的是为了让离散分布的采样可导"""
    y = gumbel_softmax_sample(logits, temperature)  # logits就是self.actor输出的值
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y


class TwoLayerFC(torch.nn.Module):
    # 修改神经元数目：128，100
    def __init__(self, num_in, hidden_dim_1,hidden_dim_2, num_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)  # 加一层试试
        self.fc3 = torch.nn.Linear(hidden_dim_2, num_out)

    def forward(self, x):  # 要求x输入就是两维的
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2, critic_input_dim, actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(num_in=state_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, num_out=action_dim).to(device)
        self.target_actor = TwoLayerFC(num_in=state_dim, hidden_dim_1=hidden_dim_1,hidden_dim_2=hidden_dim_2, num_out=action_dim).to(device)
        self.critic = TwoLayerFC(num_in=critic_input_dim, hidden_dim_1=hidden_dim_1,hidden_dim_2=hidden_dim_2, num_out=1).to(device)
        self.target_critic = TwoLayerFC(num_in=critic_input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, num_out=1).to(device)

        # 初始化目标网络的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 设置优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action_DDPG(self, state, explore=False):
        # 一步就能获取到动作，接下来使离散分布的动作具有可导性
        action = self.actor(state)
        if explore:
            # 增大探索概率
            action = gumbel_softmax(action)
        else:
            # 增大确定性
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()

    def soft_update(self, net, target_net, tau):
        """更新目标网络参数，软更新，用于MADDPG的update_all_target函数中"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class MADDPG:
    def __init__(self, env, states_dim, actions_dim, hidden_dim_1,hidden_dim_2, critic_input_dim, actor_lr, critic_lr, gamma, tau, device):
        self.agents = []
        for i in range(env.agent_num):
            self.agents.append(
                DDPG(state_dim=states_dim[i], action_dim=actions_dim[i], hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                     critic_input_dim=critic_input_dim, actor_lr=actor_lr, critic_lr=critic_lr, device=device)
            )
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    def take_action(self, states, explore):
        states = [  # 这里states[i]再加一个[]，表示将其变为两维的
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(self.agents))
        ]  # array->tensor
        return [
            agent.take_action_DDPG(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    @property
    def policies(self):
        """返回每个agent的目标策略网络，在update中更新参数时用"""
        return [agt.actor for agt in self.agents]
    @property
    def target_policies(self):
        """返回每个agent的目标策略网络，在update中更新参数时用"""
        return [agt.target_actor for agt in self.agents]

    def update(self, sample, agent_idx):
        """
        用经验池中的样本更新参数
        :param sample: 卫星的状态[]，卫星执行的动作，reward，卫星的下一个状态，done
        :param i_agent: 卫星agent的编号
        :return:
        """
        observation, action, reward, next_observation, done = sample
        cur_agent = self.agents[agent_idx]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_observation)
        ]
        # 由每个agent的一批次的某个经验组成 其维度为[采样经验数，agent数量*(next_obs维度+action维度)]
        target_critic_input = torch.cat((*next_observation, *all_target_act), dim=1)
        # 贝尔曼公式
        target_critic_ouput = reward[agent_idx].view(-1, 1) + self.gamma*cur_agent.target_critic(target_critic_input) * (1 - done[agent_idx].view(-1, 1))

        critic_input = torch.cat((*observation, *action), dim=1)
        critic_output = cur_agent.critic(critic_input)

        critic_loss = self.critic_criterion(critic_output, target_critic_ouput.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_output = cur_agent.actor(observation[agent_idx])
        cur_actor_to_critic_input = gumbel_softmax(cur_actor_output)
        all_actor_actions = []
        for i, (pi, _obs) in enumerate(zip(self.policies, observation)):
            if i == agent_idx:
                all_actor_actions.append(cur_actor_to_critic_input)
            else:
                all_actor_actions.append(onehot_from_logits(pi(_obs)))
        actor_to_critic_input = torch.cat((*observation, *all_actor_actions), dim=1)
        actor_loss = -cur_agent.critic(actor_to_critic_input).mean()
        # 在这里，乘以1e-3的目的是为了对actor网络的权重衰减（weight decay）。
        # 乘以一个小的数值（比如1e-3）可以有效地控制神经网络权重的大小，它有助于提高模型的泛化能力，减少过拟合的风险。
        # 这种方法通常被称为L2正则化或权重衰减
        actor_loss += (cur_actor_output ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_target(self):
        """
        函数是用于更新每个智能体的目标模型参数。在MADDPG算法中，每个智能体都有对应的目标actor和critic网络，
        这些目标网络的参数需要定期地与本地网络的参数同步，以保持它们与本地网络的一致性。
        因此，update_all_targets(self)函数的作用就是遍历所有智能体，然后分别将其本地网络的参数以一定的速率（由参数self.tau控制）更新到对应的目标网络中，
        以实现目标网络的参数慢慢地向本地网络的参数靠近。
        :return:
        """
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


# def evaluate(env, sce, agent, cur_episode, n_episode=2, episode_length=50):
#     """对学习的策略进行评估(仿真测试),此时不会进行探索!!!"""
#     returns = np.zeros(env.agent_num)
#     for episode in range(n_episode):
#         # # 创建新的STK环境
#         # name = f"stk_{cur_episode}_{episode}_evaluate"
#         # print(f"创建场景{name}中...")
#         # root_iaf = root.QueryInterface(STKObjects.IAgStkObject)
#         # sce = root_iaf.Children.New(19, name)
#         # create_satellite(sce=sce, lines=lines, tle_path=tle_path)
#
#         states_dict = env.reset()
#         states = []
#         for st in states_dict.values():
#             states.append(st)
#         for step in range(episode_length):
#             task = Task(sce, f"tar_{cur_episode}_{episode}_{step}_eva", step)
#             # 将任务状态填入环境状态
#             for i in range(env.agent_num):
#                 states[i][2] = task.exist_time
#                 states[i][3] = task.cost_stor
#                 states[i][4] = task.reward
#
#             actions = agent.take_action(states, explore=False)
#             next_obs_dict, reward_dict, done_dict = env.step(actions, task, sce)
#             next_obs = []
#             reward = []
#             done = []
#             for next_obs_, reward_, done_ in zip(next_obs_dict.values(), reward_dict.values(), done_dict.values()):
#                 next_obs.append(next_obs_)
#                 reward.append(reward_)
#                 done.append(done_)
#             next_obs_copy = copy.deepcopy(next_obs)
#             states = next_obs_copy  # 加入了更新状态的语句
#             reward = np.array(reward)
#             returns += reward / n_episode
#             if all(done_value == True for done_value in done):
#                 break
#         # SAVE_PATH = rf"C:\Users\xts\PyProject\RL\STK\scenarios\{name}.sc"
#         # root.SaveAs(SAVE_PATH)
#         # root.CloseScenario()
#     return returns.tolist()


def no_similate_evaluate(env, agent, task_set, n_episode=3):
    """对学习的策略进行评估(仿真测试),此时不会进行探索!!!"""
    returns = np.zeros(env.agent_num)
    for episode in range(n_episode):
        random_idx = random.randint(1, 20)
        cur_task_set = task_set[random_idx]
        states_dict = env.reset()
        states = []
        for st in states_dict.values():
            states.append(st)
        for step, task in enumerate(cur_task_set):
            # task = Task(f"tar_{cur_episode}_{episode}_{step}_eva", step)
            # 将任务状态填入环境状态
            for i in range(env.agent_num):
                states[i][2] = task.exist_time
                states[i][3] = task.cost_stor
                states[i][4] = task.reward

            actions = agent.take_action(states, explore=False)
            # 判断硬约束条件：（把原来在env.step里的搬过来）  认为：应该在根据状态产生动作之前就进行硬约束条件的判断
            for i, name in enumerate(env.agents):
                if ((task.observed_satellite[name] is False)  # 是否可被当前卫星观测
                        or (env.done[name] is True)  # 当前卫星是否已经不可再工作
                        or (env.remain_time[name] - task.exist_time) < 0  # 当前卫星的剩余时间是否足以支持该任务的消耗
                        or (env.remain_stor[name] - task.cost_stor) < 0  # 当前卫星的剩余容量是否足以支持该任务的消耗
                        # 因为任务的到达时间没有被排过序，所以每个任务之间的关系都要被判断一遍
                        or not (check_time_window(task.arrival_time, task.end_time,
                                                  env.time_window[name]))):  # 当前任务的到达时间是否在可执行范围内
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
            states = next_obs_copy  # 加入了更新状态的语句
            reward = np.array(reward)
            returns += reward / n_episode
            if all(done_value is True for done_value in done):
                break
    return returns.tolist()




