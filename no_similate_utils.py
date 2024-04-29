import collections
import random

import numpy as np


class Task:
    def __init__(self, name, longitude, latitude, arrival_time, exist_time, reward, cost_stor, observed_satellite, env):
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.arrival_time = arrival_time
        self.exist_time = exist_time
        self.end_time = self.arrival_time + self.exist_time
        self.reward = reward
        self.cost_stor = cost_stor

        satellite_list = observed_satellite.split(', ')
        satellite_set = set(satellite_list)
        self.observed_satellite = {i: i in satellite_set for i in env.agents}
        # self.longitude = random.uniform(-30, 30)
        # self.latitude = random.uniform(-60, 60)
        # # 任务的3个状态：时间消耗，内存消耗，奖励
        # self.exist_time = random.uniform(180.0, 480.0)   # 单位为：s
        # self.reward = self.exist_time * 0.04  # 14.4~24
        # self.cost_stor = self.exist_time * 0.025  # cost在9~15之间
        # self.name = name
        # # 新加的, 假设仿真时间为1h
        # self.arrival_time = random.randint(0, 3600)
        # self.end_time = self.arrival_time + self.exist_time
        # self.observed_satellite = {'YAOGAN-1': 1, 'YAOGAN-2': 1, 'YAOGAN-3': 1}  # 如要修改卫星个数，可手动修改。如果为0，表示不可被观测


class ReplayBuffer:
    """
    经验池：
    实现功能：添加经验， 拿出经验，经验池的动态长度
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, states, actions, reward, next_state, done):
        self.buffer.append((states, actions, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, reward, next_state, done = zip(*transitions)
        return np.array(states), actions, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def check_time_window(task_start, task_end, unaccessible_ranges):
    # unaccessible_ranges 是一个包含不可访问时间段的列表，每个元素是一个元组 (start, end)
    for unac_start, unac_end in unaccessible_ranges:
        # 如果待检查的起始时间和结束时间都在当前不可访问时间段的开始之前
        if task_end <= unac_start:
            continue
        # 如果待检查的起始时间和结束时间都在当前不可访问时间段的结束之后
        if task_start >= unac_end + 5:  # 设置5为卫星执行不同任务时的转换时间
            continue
        # 否则，如果待检查时间段与当前不可访问时间段有交集，则认为不可访问
        return False
    # 如果没有找到交集，则待检查时间段是可访问的
    return True


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 头部加了个0，再进行累加和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
    # cumulative_sum[window_size:] 表示从window_size下标开始一直到结束
    # cumulative_sum[:-window_size] 表示从开头开始，一直到倒数第window_size个停止
    # np.arange(1, window_size-1, 2) 生成一个从 1 到 window-1 的整数序列，步长为 2
    # [::2] 是 Python 切片操作，表示从数组的起始位置开始，不断进行索引下标+2，并取出相应的值。
    # 5-9  10-18
