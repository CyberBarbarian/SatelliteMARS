import compute_access
import create_mission
import handle_csv
import sort


# 生成指定批次、数量的数据，并对其进行一系列处理，存放在指定目录下
def generate_data(batch_size, batch_num, visible=1):
    # 生成任务数据
    mission_file = f'data/missions_{batch_size}_{batch_num}.csv'
    create_mission.generate_missions(batch_size=batch_size, num_batches=batch_num, filename=mission_file)
    # 计算任务的覆盖情况
    access_file = f'data/access_{batch_size}_{batch_num}.csv'
    compute_access.compute_access(mission_file, access_file,visible=visible)
    # 整合任务数据和覆盖情况数据
    MRL_file = f'data/MRL_data_{batch_size}_{batch_num}.csv'
    handle_csv.integrate_csv(missions_file='data/missions.csv', access_file='data/access.csv', output_file=MRL_file)
    # 对任务数据进行排序
    MRL_file_sorted = f'data/MRL_data_sorted_{batch_size}_{batch_num}.csv'
    sort.sort_csv(input_file=MRL_file, output_file=MRL_file_sorted)

if __name__ == '__main__':
    generate_data(30, 10,1)
    generate_data(20, 10,0)
    generate_data(50, 500,0)
    generate_data(50, 1000,0)
    generate_data(100, 500,0)
    generate_data(100, 1000,0)
    generate_data(200, 500,0)
    generate_data(200, 1000,0)
    generate_data(400, 500,0)
    generate_data(400, 1000,0)
