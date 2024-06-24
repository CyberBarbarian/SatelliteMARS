import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from no_similate_utils import check_time_window


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = tens_type(*shape).uniform_()  # uniform_()用于在指定范围内生成随机数的张量方法
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def onehot_from_logits(logits, eps=0.01):
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    rand_acs = torch.eye(logits.shape[1])[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ].clone().detach().to(logits.device)

    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def gumbel_softmax(logits, temperature=0.1):
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y


class TwoLayerFC(torch.nn.Module):

    def __init__(self, num_in, hidden_dim_1, hidden_dim_2, num_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim_1)
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = torch.nn.Linear(hidden_dim_2, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim_1, hidden_dim_2, critic_input_dim, actor_lr, critic_lr,
                 device):
        self.actor = TwoLayerFC(num_in=state_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                                num_out=action_dim).to(device)
        self.target_actor = TwoLayerFC(num_in=state_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                                       num_out=action_dim).to(device)
        self.critic = TwoLayerFC(num_in=critic_input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                                 num_out=1).to(device)
        self.target_critic = TwoLayerFC(num_in=critic_input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2,
                                        num_out=1).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action_DDPG(self, state, explore=False):

        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class MADDPG:
    def __init__(self, env, states_dim, actions_dim, hidden_dim_1, hidden_dim_2, critic_input_dim, actor_lr, critic_lr,
                 gamma, tau, device):
        self.agents = []
        for i in range(env.agent_num):
            self.agents.append(
                DDPG(state_dim=states_dim[i], action_dim=actions_dim[i], hidden_dim_1=hidden_dim_1,
                     hidden_dim_2=hidden_dim_2,
                     critic_input_dim=critic_input_dim, actor_lr=actor_lr, critic_lr=critic_lr, device=device)
            )
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(self.agents))
        ]
        return [
            agent.take_action_DDPG(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    @property
    def policies(self):

        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):

        return [agt.target_actor for agt in self.agents]

    def update(self, sample, agent_idx):

        observation, action, reward, next_observation, done = sample
        cur_agent = self.agents[agent_idx]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_observation)
        ]

        target_critic_input = torch.cat((*next_observation, *all_target_act), dim=1)

        target_critic_ouput = reward[agent_idx].view(-1, 1) + self.gamma * cur_agent.target_critic(
            target_critic_input) * (1 - done[agent_idx].view(-1, 1))

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

        actor_loss += (cur_actor_output ** 2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_target(self):

        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def no_similate_evaluate(env, agent, task_set, n_episode=3):
    returns = np.zeros(env.agent_num)
    for episode in range(n_episode):
        random_idx = random.randint(1, 20)
        cur_task_set = task_set[random_idx]
        states_dict = env.reset()
        states = []
        for st in states_dict.values():
            states.append(st)
        for step, task in enumerate(cur_task_set):

            for i in range(env.agent_num):
                states[i][2] = task.exist_time
                states[i][3] = task.cost_stor
                states[i][4] = task.reward

            actions = agent.take_action(states, explore=False)
            for i, name in enumerate(env.agents):
                if ((task.observed_satellite[name] is False)
                        or (env.done[name] is True)
                        or (env.remain_time[name] - task.exist_time) < 0
                        or (env.remain_stor[name] - task.cost_stor) < 0
                        or not (check_time_window(task.arrival_time, task.end_time,
                                                  env.time_window[name]))):
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
            states = next_obs_copy
            reward = np.array(reward)
            returns += reward / n_episode
            if all(done_value is True for done_value in done):
                break
    return returns.tolist()
