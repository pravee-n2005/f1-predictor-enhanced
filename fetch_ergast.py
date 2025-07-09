import requests
import pandas as pd

def fetch_season(season):
    url = f"https://ergast.com/api/f1/{season}/results.json?limit=2000"
    resp = requests.get(url)
    races = resp.json()['MRData']['RaceTable']['Races']
    rows = []
    for race in races:
        for r in race['Results']:
            rows.append({
                'season': int(season),
                'raceName': race['raceName'],
                'driver': r['Driver']['familyName'],
                'constructor': r['Constructor']['name'],
                'grid': int(r['grid']),
                'position': int(r['position']),
                'date': race['date']
            })
    return pd.DataFrame(rows)

# Fetch all seasons
df_all = pd.concat([fetch_season(s) for s in range(2018, 2025)], ignore_index=True)
df_all['winner'] = (df_all['position'] == 1).astype(int)
df_all.to_csv('race_data_full.csv', index=False)
print("Saved race_data_full.csv with", len(df_all), "rows")
