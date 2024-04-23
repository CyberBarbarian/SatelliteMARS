import win32com.client
import csv

# 连接到正在运行的STK实例
uiApplication = win32com.client.GetActiveObject('STK11.Application')
uiApplication.Visible =1
root = uiApplication.Personality2

# 从CSV文件中加载任务
def load_missions(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        missions = list(reader)
    return missions

# 在STK中创建地面点
def create_place(name, latitude, longitude):
    place = root.CurrentScenario.Children.New(32, name)  # ePlace
    place.Position.AssignGeodetic(latitude, longitude, 0)
    place.UseTerrain = True
    place.HeightAboveGround = 0.05  # in km
    return place

# 计算卫星对地面点的覆盖情况
def compute_access_for_place(place, satellite_names, sensor_names):
    access_results = []
    for satellite_name, sensor_name in zip(satellite_names, sensor_names):
        satellite = root.GetObjectFromPath(f"Satellite/{satellite_name}")
        sensor = satellite.Children.Item(sensor_name)
        # access = sensor.GetAccessToObject(place)
        access=satellite.GetAccessToObject(place)
        access.ComputeAccess()
        intervalCollection = access.ComputedAccessIntervalTimes
        try:
            if intervalCollection.Count > 0:
                intervals = intervalCollection.ToArray(0, -1)
                access_results.append((satellite_name, sensor_name, intervals))
            else:
                pass
        except Exception as e:
            print(f"Failed to retrieve intervals for {sensor_name} and {place.InstanceName}: {str(e)}")
    return access_results

# 将覆盖结果保存到CSV文件中
def save_access_results(filename, results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'latitude', 'longitude', 'satellite', 'sensor', 'intervals'])
        for result in results:
            writer.writerow(result)

# 加载任务
missions = load_missions('data/missions.csv')
# 定义卫星和传感器的名称
satellite_names = [f"Satellite{i+1}" for i in range(7)]
sensor_names = [f"Sensor{i+1}" for i in range(7)]

batch_results = []

# 对每个任务进行覆盖计算
for mission in missions:
    place_name = f"Place_{mission['batch_id']}_{mission['task_id']}"
    place = create_place(place_name, float(mission['latitude']), float(mission['longitude']))
    access_results = compute_access_for_place(place, satellite_names, sensor_names)
    for satellite_name, sensor_name, intervals in access_results:
        batch_results.append([
            mission['batch_id'], mission['task_id'], mission['latitude'], mission['longitude'],
            satellite_name, sensor_name, intervals
        ])
    place.Unload()

# 保存覆盖结果
save_access_results('data/access.csv', batch_results)