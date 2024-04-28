import csv

import win32com.client
from tqdm import tqdm


def load_missions(filename):
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        return list(reader)

def create_places(missions, visible=1):
    uiApplication = win32com.client.GetActiveObject('STK11.Application')
    uiApplication.Visible = visible
    root = uiApplication.Personality2
    places = []
    for mission in missions:
        place_name = f"Place_{mission['batch_id']}_{mission['task_id']}"
        place = root.CurrentScenario.Children.New(32, place_name)  # ePlace
        place.Position.AssignGeodetic(float(mission['latitude']), float(mission['longitude']), 0)
        place.UseTerrain = True
        place.HeightAboveGround = 0.05  # in km
        places.append(place)
    return places

def compute_access_for_places(places, satellite_names, visible=1):
    uiApplication = win32com.client.GetActiveObject('STK11.Application')
    uiApplication.Visible = visible
    root = uiApplication.Personality2
    results = []
    for place in places:
        for satellite_name in satellite_names:
            satellite = root.GetObjectFromPath(f"Satellite/{satellite_name}")
            access = satellite.GetAccessToObject(place)
            access.ComputeAccess()
            interval_collection = access.ComputedAccessIntervalTimes
            if interval_collection.Count > 0:
                intervals = interval_collection.ToArray(0, -1)
                results.append((place.InstanceName, satellite_name, intervals))
    return results

def save_access_results(filename, results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['batch_id', 'task_id', 'latitude', 'longitude', 'satellite', 'intervals'])
        for result in results:
            place_name, satellite_name, intervals = result
            batch_id, task_id = place_name.split('_')[1], place_name.split('_')[2]
            writer.writerow([batch_id, task_id, '', '', satellite_name, intervals])  # Assume latitude and longitude are fetched elsewhere

def unload_places(places):
    for place in places:
        place.Unload()

def compute_access(_missions_filename='data/missions.csv', _access_filename='data/access.csv', visible=1):
    missions = load_missions(_missions_filename)
    satellite_names = [f"Satellite{i + 1}" for i in range(7)]
    batch_results = []
    for i in tqdm(range(0, len(missions), 50), desc="Computing access in batches", unit="batch"):
        batch_missions = missions[i:i+50]
        places = create_places(batch_missions, visible)
        access_results = compute_access_for_places(places, satellite_names, visible)
        batch_results.extend(access_results)
        unload_places(places)
    save_access_results(_access_filename, batch_results)
    print(f"Access finished! Results saved to {_access_filename}")

if __name__ == '__main__':
    compute_access(visible=1)
