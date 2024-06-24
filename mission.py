# mission.py
import random


class Mission:
    def __init__(self):
        self.latitude = random.uniform(3, 53)
        self.longitude = random.uniform(74, 133)

        self.arrival_time = random.randint(0, 3600)

        self.observation_duration = random.randint(3, 6)

        self.profit = random.randint(1, 10)

        self.memory_usage = random.randint(3, 6)
