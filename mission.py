# mission.py
import random


class Mission:
    def __init__(self):
        # 随机生成经纬度，这里假设经度范围(-180, 180)，纬度范围(-90, 90)
        self.latitude = random.uniform(3, 53)
        self.longitude = random.uniform(74, 133)
        # 随机生成到达时刻，假设是在某两小时内的任意秒（3600秒内）
        self.arrival_time = random.randint(0, 3600)
        # 随机生成观测时长
        self.observation_duration = random.randint(3, 6)
        # 随机生成收益
        self.profit = random.randint(1, 10)
        # 随机生成内存消耗
        self.memory_usage = random.randint(3, 6)
