import csv
import win32com.client

# Connect to the running STK instance
uiApplication = win32com.client.GetActiveObject('STK11.Application')
uiApplication.Visible = 1
root = uiApplication.Personality2


# Load tasks from a CSV file
def load_missions(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        missions = list(reader)
    return missions


# Create a place in STK
def create_place(name, latitude, longitude):
    _place = root.CurrentScenario.Children.New(32, name)  # ePlace
    _place.Position.AssignGeodetic(latitude, longitude, 0)
    _place.UseTerrain = True
    _place.HeightAboveGround = 0.05  # in km
    return _place


# Calculate coverage of satellites over a place
def compute_access_for_place(_place, _satellite_names):
    _access_results = {}
    for satellite_name in _satellite_names:
        satellite = root.GetObjectFromPath(f"Satellite/{satellite_name}")
        access = satellite.GetAccessToObject(_place)
        access.ComputeAccess()
        interval_collection = access.ComputedAccessIntervalTimes
        if interval_collection.Count > 0:
            if satellite_name in _access_results:
                _access_results[satellite_name].append(True)
            else:
                _access_results[satellite_name] = True
    return _access_results


# Save coverage results to a CSV file
def save_access_results(filename, results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'latitude', 'longitude', 'observable_satellites', 'arrival_time',
                         'observation_duration', 'profit', 'memory_usage'])
        for result in results:
            writer.writerow([
                result['batch_id'], result['task_id'], result['latitude'], result['longitude'],
                ', '.join(result['satellites']), result['arrival_time'], result['observation_duration'],
                result['profit'], result['memory_usage']
            ])


# Load missions
missions = load_missions('data/missions.csv')
# Define satellite names
satellite_names = [f"Satellite{i + 1}" for i in range(7)]
mission_access = {}

# Calculate coverage for each task
for mission in missions:
    place_name = f"Place_{mission['batch_id']}_{mission['task_id']}"
    place = create_place(place_name, float(mission['latitude']), float(mission['longitude']))
    access_results = compute_access_for_place(place, satellite_names)
    key = (mission['batch_id'], mission['task_id'], mission['latitude'], mission['longitude'])
    if key not in mission_access:
        mission_access[key] = {
            'satellites': set(),
            'arrival_time': mission['arrival_time'],
            'observation_duration': mission['observation_duration'],
            'profit': mission['profit'],
            'memory_usage': mission['memory_usage']
        }
    mission_access[key]['satellites'].update(access_results.keys())
    place.Unload()

# Prepare the results for saving
batch_results = [{
    'batch_id': k[0],
    'task_id': k[1],
    'latitude': k[2],
    'longitude': k[3],
    'satellites': list(v['satellites']),
    'arrival_time': v['arrival_time'],
    'observation_duration': v['observation_duration'],
    'profit': v['profit'],
    'memory_usage': v['memory_usage']
} for k, v in mission_access.items()]

# Save the coverage results
save_access_results('data/access.csv', batch_results)
