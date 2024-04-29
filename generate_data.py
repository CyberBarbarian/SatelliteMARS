import compute_access
import create_mission
import handle_csv
import sort


# 生成指定批次、数量的数据，并对其进行一系列处理，存放在指定目录下
def generate_data(batch_size, batch_num, visible=1, prefix=None):
    '''

    :param batch_size: 每批次任务数量
    :param batch_num: 任务批次
    :param visible: 是否开启 STK 可视化窗口，默认为 1
    :param prefix: 是否添加数据集前缀，默认为 None
    :return:
    '''
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

    # 生成任务数据
    create_mission.generate_missions(batch_size=batch_size, num_batches=batch_num, filename=mission_file)
    # 计算任务的覆盖情况
    compute_access.compute_access(_missions_filename=mission_file, _access_filename=access_file, visible=visible)
    # 整合任务数据和覆盖情况数据
    handle_csv.integrate_csv(missions_file=mission_file, access_file=access_file, output_file=MRL_file)
    # 对任务数据进行排序
    sort.sort_csv(input_file=MRL_file, output_file=MRL_file_sorted)


if __name__ == '__main__':
    # for i in range(1, 20):
    #     generate_data(100, 100, visible=1, prefix=f'True/{i}')
    generate_data(100, 1, visible=1, prefix=f'augment/')

