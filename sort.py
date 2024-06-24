import csv
from datetime import datetime


def read_csv(filename):
    with open(filename, newline='') as file:
        return list(csv.DictReader(file))


def write_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def parse_time(time_str):
    return datetime.strptime(time_str, '%d %b %Y %H:%M:%S.%f')


def sort_missions(_missions):
    batches = {}
    for task in _missions:
        batch_id = task['batch_id']
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(task)

    _sorted_missions = []
    for batch_id in batches:
        batches[batch_id].sort(key=lambda x: int(x['arrival_time_seconds']))
        _sorted_missions.extend(batches[batch_id])

    return _sorted_missions


def sort_csv(input_file='data/MRL_data.csv', output_file='data/MRL_data_sorted.csv'):
    missions = read_csv(input_file)

    sorted_missions = sort_missions(missions)

    write_csv(output_file, sorted_missions)

    print(f"Mission have been sorted and written to {output_file}")


if __name__ == "__main__":
    sort_csv()
