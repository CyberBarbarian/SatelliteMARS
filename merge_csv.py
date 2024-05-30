import pandas as pd


def merge_csv_files(file_list, output_file):
    combined_df = pd.DataFrame()
    current_batch_id = 1

    for file in file_list:
        df = pd.read_csv(file)
        # 找到该文件中所有不同的batch_id，并为它们生成新的batch_id
        unique_batch_ids = df['batch_id'].unique()
        new_batch_id_map = {id: i for i, id in enumerate(unique_batch_ids, current_batch_id)}
        current_batch_id += len(unique_batch_ids)

        # 更新df中的batch_id
        df['batch_id'] = df['batch_id'].map(new_batch_id_map)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved as {output_file}")


def main():
    # List of CSV files to merge
    csv_files = ['data/lab/lab1_augment.csv',
                 'data/lab/lab1_true.csv',
                 ]  # Update this list as necessary
    output_file_name = 'data/lab/lab1.csv'  # Define your output file name

    merge_csv_files(csv_files, output_file_name)


if __name__ == '__main__':
    main()
