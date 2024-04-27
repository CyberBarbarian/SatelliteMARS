import csv

import win32com.client

# 连接到正在运行的STK实例
uiApplication = win32com.client.GetActiveObject('STK11.Application')
uiApplication.Visible = 0
root = uiApplication.Personality2


# 从CSV文件中加载任务
def load_missions(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        _missions = list(reader)
    return _missions


# 在STK中创建地面点
def create_place(name, latitude, longitude):
    _place = root.CurrentScenario.Children.New(32, name)  # ePlace
    _place.Position.AssignGeodetic(latitude, longitude, 0)
    _place.UseTerrain = True
    _place.HeightAboveGround = 0.05  # in km
    return _place


# 计算卫星对地面点的覆盖情况
def compute_access_for_place(_place, _satellite_names):
    _access_results = []
    for _satellite_name in _satellite_names:
        satellite = root.GetObjectFromPath(f"Satellite/{_satellite_name}")
        access = satellite.GetAccessToObject(_place)
        access.ComputeAccess()
        interval_collection = access.ComputedAccessIntervalTimes
        try:
            if interval_collection.Count > 0:
                _intervals = interval_collection.ToArray(0, -1)
                _access_results.append((_satellite_name, _intervals))
            else:
                pass
        except Exception as e:
            print(f"Failed to retrieve intervals for {_satellite_name} and {_place.InstanceName}: {str(e)}")
    return _access_results


# 将覆盖结果保存到CSV文件中
def save_access_results(filename, results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'latitude', 'longitude', 'satellite', 'intervals'])
        for result in results:
            writer.writerow(result)


# 加载任务
missions = load_missions('data/missions.csv')
# 定义卫星和传感器的名称
satellite_names = [f"Satellite{i + 1}" for i in range(7)]

batch_results = []

# 对每个任务进行覆盖计算
for mission in missions:
    place_name = f"Place_{mission['batch_id']}_{mission['task_id']}"
    place = create_place(place_name, float(mission['latitude']), float(mission['longitude']))
    access_results = compute_access_for_place(place, satellite_names)
    for satellite_name, intervals in access_results:
        batch_results.append([
            mission['batch_id'], mission['task_id'], mission['latitude'], mission['longitude'],
            satellite_name, intervals
        ])
    place.Unload()

# 保存覆盖结果
save_access_results('data/access.csv', batch_results)
