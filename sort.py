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


def sort_tasks(_tasks):
    # Group tasks by batch_id
    batches = {}
    for task in _tasks:
        batch_id = task['batch_id']
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(task)

    # Sort tasks within each batch by arrival time
    _sorted_tasks = []
    for batch_id in batches:
        batches[batch_id].sort(key=lambda x: int(x['arrival_time_seconds']))
        _sorted_tasks.extend(batches[batch_id])

    return _sorted_tasks


# Load data from the integrated results CSV
tasks = read_csv('data/MRL_data.csv')

# Sort the tasks by arrival time within each batch
sorted_tasks = sort_tasks(tasks)

# Write the sorted tasks back to a new CSV
write_csv('data/MRL_data_sorted.csv', sorted_tasks)

print("Tasks have been sorted and written to 'data/MRL_data_sorted.csv'")
