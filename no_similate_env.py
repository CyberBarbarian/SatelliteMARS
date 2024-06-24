from datetime import datetime

import numpy as np


class TimePeriod:

    def __init__(self, start_time, end_time, is_available=True):
        self.start_time = start_time
        self.end_time = end_time
        self.is_available = is_available  # 如果某个区域内有任务安排了，则设置其为False，需要判断某个时段段内是True还是False

    def check_available(self):
        pass

    # 判断某个时间点是否在可用时间段内
    def is_within_time_period(time, start_time, end_time):
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        return start <= time <= end

    def modify_time_period(start_time, end_time, start_new_hour, start_new_minute, start_new_second, end_new_hour,
                           end_new_minute, end_new_second):
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # 修改开始时间的时分秒
        start = start.replace(hour=start_new_hour, minute=start_new_minute, second=start_new_second)

        # 修改结束时间的时分秒
        end = end.replace(hour=end_new_hour, minute=end_new_minute, second=end_new_second)

        return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


class MultiEnv:
    STOR_MAX = 150.0
    TIME_MAX = 3600.0

    def __init__(self, agent_num=7):

        self.agents = [f"Satellite{i + 1}" for i in range(agent_num)]
        self.agent_num = agent_num
        self.action_space = [1, 1]
        self.actions = {i: self.action_space for i in self.agents}

        self.satellite_time_unique = {i: 1 for i in self.agents}
        self.stor_max = MultiEnv.STOR_MAX
        self.remain_stor = {i: MultiEnv.STOR_MAX for i in self.agents}
        self.time_max = MultiEnv.TIME_MAX
        self.remain_time = {i: MultiEnv.TIME_MAX for i in self.agents}
        self.observation = {i: [1.0, 1.0, 0.0, 0.0, 0.0] for i in self.agents}
        self.next_observation = {
            i: [self.remain_time[i] / self.time_max, self.remain_stor[i] / self.stor_max, 0.0, 0.0, 0.0] for i in
            self.agents}
        self.reward = {i: 0 for i in self.agents}
        self.done = {i: False for i in self.agents}
        self.render_mode = "human"
        self.time_window = {i: [] for i in self.agents}

    def reset(self):
        self.stor_max = MultiEnv.STOR_MAX
        self.remain_stor = {i: MultiEnv.STOR_MAX for i in self.agents}
        self.time_max = MultiEnv.TIME_MAX
        self.remain_time = {i: MultiEnv.TIME_MAX for i in self.agents}
        self.observation = {i: [1.0, 1.0, 0.0, 0.0, 0.0] for i in self.agents}
        self.next_observation = {
            i: [self.remain_time[i] / self.time_max, self.remain_stor[i] / self.stor_max, 0.0, 0.0, 0.0] for i in
            self.agents}
        self.reward = {i: 0 for i in self.agents}
        self.done = {i: False for i in self.agents}
        self.time_window = {i: [] for i in self.agents}
        return self.observation

    def step(self, actions, task):

        ALPHA = 1
        BETA = 0.2
        # ENDING_NUM = 50
        num_of_accept = 0
        last_max_reward = 0
        sum_reward = 0

        for i, name in enumerate(self.agents):
            if actions[i][0][0] == 1:
                num_of_accept += 1

        for i, name in enumerate(self.agents):
            if (actions[i][0][0] == 1):
                self.reward[name] = np.log((task.reward / np.exp(num_of_accept * ALPHA)) + (
                        task.reward / np.exp(num_of_accept * ALPHA) ** 2) + (
                                                   task.reward / np.exp(num_of_accept * ALPHA) ** 3) - (
                                                   task.cost_stor / task.reward) * BETA + 10)
                self.remain_time[name] = self.remain_time[name] - task.exist_time
                self.remain_stor[name] = self.remain_stor[name] - task.cost_stor
                self.next_observation[name][0] = self.remain_time[
                                                     name] / self.time_max
                self.next_observation[name][1] = self.remain_stor[name] / self.stor_max
                self.next_observation[name][2] = task.exist_time
                self.next_observation[name][3] = task.cost_stor
                self.next_observation[name][4] = task.reward
                self.time_window[name].append((task.arrival_time, task.end_time))
            else:
                self.reward[name] = 0
                self.next_observation[name][2] = task.exist_time
                self.next_observation[name][3] = task.cost_stor
                self.next_observation[name][4] = task.reward

        for name in self.agents:
            if self.next_observation[name][0] <= 0.1 or self.next_observation[name][1] <= 0.1:
                self.done[name] = True

        return self.next_observation, self.reward, self.done
