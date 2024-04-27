import csv
from datetime import datetime, timedelta


def read_csv(filename):
    with open(filename, newline='') as file:
        return list(csv.DictReader(file))


def parse_time(time_str):
    return datetime.strptime(time_str, '%d %b %Y %H:%M:%S.%f')


def parse_intervals(intervals):
    parsed_intervals = []
    for start, end in intervals:
        parsed_start = parse_time(start)
        parsed_end = parse_time(end)
        parsed_intervals.append((parsed_start, parsed_end))
    return parsed_intervals


def check_within_interval(arrival_time, end_time, parsed_intervals):
    for start, end in parsed_intervals:
        parsed_start = parse_time(start)
        parsed_end = parse_time(end)
        if parsed_start <= arrival_time <= parsed_end and parsed_start <= end_time <= parsed_end:
            return True
    return False


def integrate_csv(missions_file, access_file, output_file):
    missions = read_csv(missions_file)
    accesses = read_csv(access_file)

    base_time = datetime.strptime("18 Aug 2018 04:00:00.000", "%d %b %Y %H:%M:%S.%f")
    task_data = {}

    for access in accesses:
        mission = next(
            (m for m in missions if m['batch_id'] == access['batch_id'] and m['task_id'] == access['task_id']), None)
        if mission:
            arrival_time_seconds = int(mission['arrival_time'])
            actual_arrival_time = base_time + timedelta(seconds=arrival_time_seconds)
            end_time = actual_arrival_time + timedelta(seconds=int(mission['observation_duration']))
            intervals = eval(access['intervals'])

            if check_within_interval(actual_arrival_time, end_time, intervals):
                key = (access['batch_id'], access['task_id'])
                if key not in task_data:
                    task_data[key] = {
                        'batch_id': access['batch_id'],
                        'task_id': access['task_id'],
                        'arrival_time_seconds': arrival_time_seconds,
                        'satellites': set(),
                        'reward': mission['profit'],
                        'observation_duration': mission['observation_duration'],
                        'memory_usage': mission['memory_usage'],
                        'latitude': mission['latitude'],
                        'longitude': mission['longitude']
                    }
                task_data[key]['satellites'].add(access['satellite'])

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'arrival_time_seconds', 'observable_by_satellites', 'reward',
                         'observation_duration', 'memory_usage', 'latitude', 'longitude'])
        for key, data in task_data.items():
            writer.writerow([
                data['batch_id'], data['task_id'], data['arrival_time_seconds'],
                ', '.join(data['satellites']), data['reward'],
                data['observation_duration'], data['memory_usage'],
                data['latitude'], data['longitude']
            ])


# Setting file paths and calling the function
integrate_csv('data/missions.csv', 'data/access.csv', 'data/MRL_data.csv')
