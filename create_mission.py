# generate_tasks.py
import csv

from mission import Mission


def generate_missions(batch_size, num_batches, filename='data/missions.csv'):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['batch_id', 'task_id', 'latitude', 'longitude', 'arrival_time', 'observation_duration', 'profit',
             'memory_usage'])

        for batch_id in range(1, num_batches + 1):
            for task_id in range(1, batch_size + 1):
                mission = Mission()
                writer.writerow([batch_id, task_id, mission.latitude, mission.longitude,
                                 mission.arrival_time, mission.observation_duration,
                                 mission.profit, mission.memory_usage])
    print(f"Missions have been generated and saved to {filename}")


if __name__ == '__main__':
    generate_missions(batch_size=30, num_batches=2, filename='data/missions.csv')
