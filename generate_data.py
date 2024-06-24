from compute_access import compute_access
from create_mission import generate_missions
from handle_csv import integrate_csv
from sort import sort_csv


# 生成指定批次、数量的数据，并对其进行一系列处理，存放在指定目录下
def generate_data(batch_size, batch_num, visible=1, prefix=None):
    if prefix is not None:
        mission_file = f'data/{prefix}missions_{batch_size}_{batch_num}.csv'
        access_file = f'data/{prefix}access_{batch_size}_{batch_num}.csv'
        MRL_file = f'data/{prefix}MRL_data_{batch_size}_{batch_num}.csv'
        MRL_file_sorted = f'data/{prefix}MRL_data_sorted_{batch_size}_{batch_num}.csv'

    else:
        mission_file = f'data/missions_{batch_size}_{batch_num}.csv'
        access_file = f'data/access_{batch_size}_{batch_num}.csv'
        MRL_file = f'data/MRL_data_{batch_size}_{batch_num}.csv'
        MRL_file_sorted = f'data/MRL_data_sorted_{batch_size}_{batch_num}.csv'

    generate_missions(batch_size=batch_size, num_batches=batch_num, filename=mission_file)

    compute_access(_missions_filename=mission_file, _access_filename=access_file, visible=visible)

    integrate_csv(missions_file=mission_file, access_file=access_file, output_file=MRL_file)

    sort_csv(input_file=MRL_file, output_file=MRL_file_sorted)


if __name__ == '__main__':
    for i in range(10):
        generate_data(200, 500, visible=1, prefix=f'true/{i + 1}_')
