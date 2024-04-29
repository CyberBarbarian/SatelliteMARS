import csv
import random

import sort


def load_data(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data


def write_data(filename, data, fieldnames):
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for batch_id, batch in enumerate(data, start=1):
            for task_id, task in enumerate(batch, start=1):
                task['batch_id'] = batch_id
                task['task_id'] = task_id
                writer.writerow(task)


def randomize_task(task):
    # 重置任务收益，观测时长和内存消耗
    task['reward'] = random.randint(1, 10)
    task['observation_duration'] = random.randint(3, 6)
    task['memory_usage'] = random.randint(3, 6)
    # 调整到达时间
    arrival_time = int(task['arrival_time_seconds'])
    delta_seconds = random.randint(-60, 60)
    new_arrival_time = max(0, min(3600, arrival_time + delta_seconds))
    task['arrival_time_seconds'] = new_arrival_time
    return task


def augment_data(original_data, num_samples, num_batches):
    all_batches = []
    for _ in range(num_batches):
        batch = random.sample(original_data, num_samples)  # Select random subset
        augmented_batch = [randomize_task(task.copy()) for task in batch]  # Augment each task in the batch
        all_batches.append(augmented_batch)
    return all_batches


if __name__ == "__main__":
    num_samples = 50  # Number of tasks to select for each batch
    num_batches = 1000  # Number of batches to create
    input_filename = 'data/augment/MRL_data_1000_1.csv'
    output_filename = f'data/augment/MRL_data_{num_samples}_{num_batches}_augmented.csv'
    original_data = load_data(input_filename)
    fieldnames = list(original_data[0].keys())  # Assume all dicts have the same keys

    # Ensure 'batch_id' and 'task_id' are included in fieldnames
    if 'batch_id' not in fieldnames:
        fieldnames.insert(0, 'batch_id')
    if 'task_id' not in fieldnames:
        fieldnames.insert(1, 'task_id')

    augmented_batches = augment_data(original_data, num_samples, num_batches)
    write_data(output_filename, augmented_batches, fieldnames)
    print("Data augmentation complete and saved to", output_filename)
    sort.sort_csv(input_file=output_filename, output_file=output_filename.replace('.csv', '_sorted.csv'))
