import csv
from datetime import datetime, timedelta


def read_csv(filename):
    with open(filename, newline='') as file:
        return list(csv.DictReader(file))


def parse_time(time_str):
    return datetime.strptime(time_str, '%d %b %Y %H:%M:%S.%f')


def check_within_interval(arrival_time, intervals):
    for start, end in intervals:
        print(parse_time(start), parse_time(end), arrival_time)
        if parse_time(start) <= arrival_time <= parse_time(end):
            return True
    return False


def integrate_csv(missions_file, access_file, output_file):
    missions = read_csv(missions_file)
    accesses = read_csv(access_file)

    base_time = datetime.strptime("18 Aug 2018 04:00:00.000", "%d %b %Y %H:%M:%S.%f")

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'arrival_time_seconds', 'observable_by_satellite', 'reward'])

        for access in accesses:
            mission = next(
                (m for m in missions if m['batch_id'] == access['batch_id'] and m['task_id'] == access['task_id']),
                None)
            if mission:
                arrival_time_seconds = int(mission['arrival_time'])
                actual_arrival_time = base_time + timedelta(seconds=arrival_time_seconds)
                intervals = eval(access['intervals'])

                if check_within_interval(actual_arrival_time, intervals):
                    writer.writerow([access['batch_id'], access['task_id'], arrival_time_seconds,
                                     f"{access['satellite']} observed by {access['sensor']}", mission['profit']])


# 设置文件路径并调用函数
integrate_csv('data/missions.csv', 'data/access.csv', 'data/integrated_results.csv')
