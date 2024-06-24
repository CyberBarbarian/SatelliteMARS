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


class ReplayBuffer:
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
    for unac_start, unac_end in unaccessible_ranges:
        if task_end <= unac_start:
            continue
        if task_start >= unac_end + 5:
            continue
        return False
    return True


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))  # 头部加了个0，再进行累加和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))
