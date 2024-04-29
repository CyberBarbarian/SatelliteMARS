# from STK.integration.create_scenario_integration import ScenarioClass
# from STK.target_class_xts import TargetClass
from datetime import datetime

import numpy as np

# 任务是不断出现在环境中的，用STK生成任务，

"""
准备加入时间约束：考虑使用硬约束，即，先全部筛选出可用的卫星，然后进行决策，筛选的过程可用mask掩码来把可用的卫星留下
决策过程：
卫星先根据任务的经纬度位置，得到对某个位置的访问时间窗口（过境时段）
将过境时段与任务的到达时间和结束时间进行比较，若满足条件，则输入网络进行判断，看是否接收，

判断任务是否满足时间窗口
满足后进行决策调度
"""


class TimePeriod:
    """
    时间段类
    """

    def __init__(self, start_time, end_time, is_available=True):
        """
        示例用法
        start_time = datetime.datetime(2024, 4, 11, 8, 0, 0)
        end_time = datetime.datetime(2024, 4, 11, 12, 0, 0)
        :param start_time:
        :param end_time:
        :param is_available:
        """
        self.start_time = start_time
        self.end_time = end_time
        self.is_available = is_available  # 如果某个区域内有任务安排了，则设置其为False，需要判断某个时段段内是True还是False

    def check_available(self):
        pass

    # 判断某个时间点是否在可用时间段内
    def is_within_time_period(time, start_time, end_time):
        """
        示例用法
        time = "2024-04-11 10:30:00"
        is_within = is_within_time_period(time, "2024-04-11 08:00:00", "2024-04-11 12:00:00")
        :param start_time:
        :param end_time:
        :return:
        """
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        return start <= time <= end

    def modify_time_period(start_time, end_time, start_new_hour, start_new_minute, start_new_second, end_new_hour,
                           end_new_minute, end_new_second):
        """
        示例用法
        new_start, new_end = modify_time_period("2024-04-11 08:00:00", "2024-04-11 12:00:00", 3, 30, 0)
        :param end_time:
        :param start_new_hour:
        :param start_new_minute:
        :param start_new_second:
        :param end_new_hour:
        :param end_new_minute:
        :param end_new_second:
        :return:
        """
        start = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # 修改开始时间的时分秒
        start = start.replace(hour=start_new_hour, minute=start_new_minute, second=start_new_second)

        # 修改结束时间的时分秒
        end = end.replace(hour=end_new_hour, minute=end_new_minute, second=end_new_second)

        return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


"""
卫星：状态为【空余时间段（TimePeriod类），剩余存储量】
任务：状态为【到达次序（暂不考虑），开始时间和结束时间（TimePeriod类）】,

"""


# 启发点：加入到达时间，时间窗口，结束时间的约束
# 对各个数值的归一化处理

class MultiEnv:
    STOR_MAX = 150.0
    TIME_MAX = 3600.0

    def __init__(self, agent_num=7):
        # 初始化智能体
        """
        输入状态：（被归一化为[0,1]）
        任务的到达时刻(暂时忽略)
        任务的收益（ok）
        任务的存储消耗（）
        卫星当前时刻的剩余存储容量的比例（stor/STOR）
        卫星当前时刻的剩余空闲时间区间的比例
        """
        self.agents = [f"Satellite{i + 1}" for i in range(agent_num)]  # 这里只是设置agent的名字
        self.agent_num = agent_num
        self.action_space = [1, 1]  # 接受 拒绝  2个动作   TODO: 4_26:突然感觉动作纬度一维就行
        self.actions = {i: self.action_space for i in self.agents}  # 动作：接受任务 不接受任务

        self.satellite_time_unique = {i: 1 for i in self.agents}  # 表示卫星的唯一性观测，为0表示不满足，为1表示满足
        self.stor_max = MultiEnv.STOR_MAX
        self.remain_stor = {i: MultiEnv.STOR_MAX for i in self.agents}
        self.time_max = MultiEnv.TIME_MAX
        self.remain_time = {i: MultiEnv.TIME_MAX for i in self.agents}
        # observation包括 [剩余时间，剩余容量，任务持续时间，任务消耗容量，任务奖励]--> [剩余时间，剩余容量，任务持续时间，任务消耗容量，任务奖励]
        self.observation = {i: [1.0, 1.0, 0.0, 0.0, 0.0] for i in self.agents}
        self.next_observation = {
            i: [self.remain_time[i] / self.time_max, self.remain_stor[i] / self.stor_max, 0.0, 0.0, 0.0] for i in
            self.agents}
        # 其余观测
        self.reward = {i: 0 for i in self.agents}
        self.done = {i: False for i in self.agents}
        self.render_mode = "human"  # 可视化：human 非可视化：None
        self.time_window = {i: [] for i in self.agents}  # 时间窗口，用来存放已经接受的任务
        # 新增mask
        # self.mask = {i: 1 for i in self.agents}  # 1 表示接受和拒绝都能执行，当卫星不满足约束条件时，self.mask变为：0 其实只看第一个就行self.mask * self.action[name][0]

    def reset(self):
        """给出每个卫星的状态即可"""
        self.stor_max = MultiEnv.STOR_MAX
        self.remain_stor = {i: MultiEnv.STOR_MAX for i in self.agents}
        self.time_max = MultiEnv.TIME_MAX
        self.remain_time = {i: MultiEnv.TIME_MAX for i in self.agents}
        self.observation = {i: [1.0, 1.0, 0.0, 0.0, 0.0] for i in self.agents}
        self.next_observation = {
            i: [self.remain_time[i] / self.time_max, self.remain_stor[i] / self.stor_max, 0.0, 0.0, 0.0] for i in
            self.agents}
        # 其余观测
        self.reward = {i: 0 for i in self.agents}  # 初始奖励均为0
        self.done = {i: False for i in self.agents}
        self.time_window = {i: [] for i in self.agents}  # 时间窗口，用来存放已经接受的任务
        # self.mask = {i: 1 for i in self.agents}
        return self.observation

    def step(self, actions, task):
        """
        :param actions: 输入为列表，每个agent的动作 0 or 1
        :param task: 输入为task类，要求含有属性：idx:表示到达次序； reward：表示任务奖励； cost：表示任务消耗
        :param sce: STK仿真场景的接口
        :return:
        """
        ALPHA = 1
        BETA = 0.2
        # ENDING_NUM = 50
        num_of_accept = 0
        last_max_reward = 0
        sum_reward = 0

        # 需要先判断任务是否在当前卫星的时间窗口内，如果在的话对相应的卫星进行执行动作，那这个过程应该在产生动作的函数里面？
        for i, name in enumerate(self.agents):
            # if actions[i][0][0] == 1 and (self.remain_time[name] - task.exist_time) >= 0 and (self.remain_stor[name] - task.cost_stor) >= 0:  # 动作空间的第一个位置为1表示接受，第二个位置为1表示拒绝 # TODO:改一下
            #     num_of_accept += 1
            if actions[i][0][0] == 1:  # 动作空间的第一个位置为1表示接受，第二个位置为1表示拒绝 # TODO:改一下
                num_of_accept += 1

        # for i, name in enumerate(self.agents):
        #     if (actions[i][0][0] == 1) and (self.done[name] == False) and (self.remain_time[name] - task.exist_time) >= 0 and (self.remain_stor[name] - task.cost_stor) >= 0:  # 需要加个等于0
        #         self.reward[name] = np.log((task.reward / np.exp(num_of_accept*ALPHA)) + (task.reward / np.exp(num_of_accept*ALPHA)**2) + (task.reward / np.exp(num_of_accept*ALPHA)**3) - (task.cost_stor / task.reward) * BETA + 10)
        #         self.remain_time[name] = self.remain_time[name] - task.exist_time
        #         self.remain_stor[name] = self.remain_stor[name] - task.cost_stor
        #         self.next_observation[name][0] = self.remain_time[name] / self.time_max  # 这个observation变了，会导致state也跟着一起变
        #         self.next_observation[name][1] = self.remain_stor[name] / self.stor_max
        #         # 添加当前任务的状态
        #         self.next_observation[name][2] = task.exist_time
        #         self.next_observation[name][3] = task.cost_stor
        #         self.next_observation[name][4] = task.reward  # observation中的reward指的是任务本身的reward，而不是最后每个卫星获得的reward
        #
        #     else:
        #         # 可能是action=0，可能是done=true，可能是剩余时间不足，也可能是任务的剩余存储容量不足，在这些情况下，next_obs的后3项只需变为当前任务的属性值即可，前两项不用改变
        #         self.reward[name] = 0
        #         self.next_observation[name][2] = task.exist_time
        #         self.next_observation[name][3] = task.cost_stor
        #         self.next_observation[name][4] = task.reward
        #
        for i, name in enumerate(self.agents):
            if (actions[i][0][0] == 1):  # 需要加个等于0
                self.reward[name] = np.log((task.reward / np.exp(num_of_accept * ALPHA)) + (
                        task.reward / np.exp(num_of_accept * ALPHA) ** 2) + (
                                                   task.reward / np.exp(num_of_accept * ALPHA) ** 3) - (
                                                   task.cost_stor / task.reward) * BETA + 10)
                self.remain_time[name] = self.remain_time[name] - task.exist_time
                self.remain_stor[name] = self.remain_stor[name] - task.cost_stor
                self.next_observation[name][0] = self.remain_time[
                                                     name] / self.time_max  # 这个observation变了，会导致state也跟着一起变
                self.next_observation[name][1] = self.remain_stor[name] / self.stor_max
                # 添加当前任务的状态
                self.next_observation[name][2] = task.exist_time
                self.next_observation[name][3] = task.cost_stor
                self.next_observation[name][4] = task.reward  # observation中的reward指的是任务本身的reward，而不是最后每个卫星获得的reward
                # 将任务添加到当前卫星的时间窗口, 假设任务到达后就要开始进行观测
                self.time_window[name].append((task.arrival_time, task.end_time))
            else:
                # 可能是action=0，可能是done=true，可能是剩余时间不足，也可能是任务的剩余存储容量不足，在这些情况下，next_obs的后3项只需变为当前任务的属性值即可，前两项不用改变
                self.reward[name] = 0
                self.next_observation[name][2] = task.exist_time
                self.next_observation[name][3] = task.cost_stor
                self.next_observation[name][4] = task.reward

        # if task.idx > ENDING_NUM:
        #     for name in self.agents:
        #         self.done[name] = True
        # TODO: 控制结束条件，当只剩下一点时，就近似的将其当做是已经结束
        for name in self.agents:
            if self.next_observation[name][0] <= 0.1 or self.next_observation[name][1] <= 0.1:
                self.done[name] = True

        return self.next_observation, self.reward, self.done
