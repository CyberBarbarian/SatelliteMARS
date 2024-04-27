# generate_tasks.py
import csv

from mission import Mission


def generate_missions(batch_size, num_batches, filename):
    # 创建 CSV 文件并写入任务数据
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入 CSV 头部
        writer.writerow(
            ['batch_id', 'task_id', 'latitude', 'longitude', 'arrival_time', 'observation_duration', 'profit',
             'memory_usage'])

        for batch_id in range(1, num_batches + 1):
            for task_id in range(1, batch_size + 1):
                mission = Mission()
                # 将任务数据及其批次和编号信息写入 CSV
                writer.writerow([batch_id, task_id, mission.latitude, mission.longitude,
                                 mission.arrival_time, mission.observation_duration,
                                 mission.profit, mission.memory_usage])


# 设置参数并调用函数
generate_missions(batch_size=30, num_batches=2, filename='data/missions.csv')
