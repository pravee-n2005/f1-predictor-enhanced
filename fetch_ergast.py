import requests
import pandas as pd

def fetch_season(season):
    url = f"https://ergast.com/api/f1/{season}/results.json?limit=2000"
    resp = requests.get(url)
    data = []
    if resp.status_code == 200:
        results = resp.json()['MRData']['RaceTable']['Races']
        for race in results:
            for result in race['Results']:
                constructor = result['Constructor']['name']
                driver = result['Driver']
                driver_name = f"{driver['givenName']} {driver['familyName']}"
                grid = int(result['grid'])
                position = result['position']
                position = int(position) if position.isdigit() else None

                data.append({
                    "season": season,
                    "round": race['round'],
                    "circuit": race['Circuit']['circuitName'],
                    "constructor": constructor,
                    "driver": driver_name,
                    "grid": grid,
                    "position": position
                })
    return pd.DataFrame(data)

# Fetch multiple seasons
all_seasons = []
for season in range(2018, 2024):
    print(f"Fetching season {season}...")
    df = fetch_season(season)
    all_seasons.append(df)

df_all = pd.concat(all_seasons, ignore_index=True)

# Add winner column
df_all['winner'] = df_all['position'] == 1

# Save the dataset
df_all.to_csv("race_data_full.csv", index=False)
print("✅ Data saved to race_data_full.csv")
