"""Download ALL cities / towns / suburbs (and optionally villages) of Tamil Nadu
from OpenStreetMap and save them to data/tamil_nadu_area_locations.csv

    python fetch_areas.py              # cities, towns, suburbs   (~ 1-2 thousand)
    python fetch_areas.py --villages   # + villages               (~ 10 thousand+, slower map)

Needs internet. Takes 1-3 minutes. Run once, then restart app.py.
NOTE: 'district' is the nearest district centre -> approximate near borders.
"""
import sys
import numpy as np
import pandas as pd
import requests

PLACES = "city|town|suburb|neighbourhood" + ("|village" if "--villages" in sys.argv else "")
QUERY = f"""
[out:json][timeout:300];
area["ISO3166-2"="IN-TN"]->.tn;
node["place"~"^({PLACES})$"]["name"](area.tn);
out;
"""
SERVERS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter"]


def fetch():
    for url in SERVERS:
        try:
            print("Trying", url)
            r = requests.post(url, data={"data": QUERY}, timeout=320,
                              headers={"User-Agent": "accident-hotspot-student-project"})
            r.raise_for_status()
            return r.json()["elements"]
        except Exception as e:
            print("  failed:", e)
    raise SystemExit("All Overpass servers failed - check internet and retry.")


def to_dataframe(elements):
    rows = [{"area": e["tags"].get("name:en") or e["tags"]["name"],
             "type": e["tags"]["place"], "latitude": e["lat"], "longitude": e["lon"]}
            for e in elements if "lat" in e and "tags" in e]
    return pd.DataFrame(rows).drop_duplicates(["area", "latitude", "longitude"])


def add_district(df):
    d = pd.read_csv("data/preprocessed_tamil_nadu_accidents.csv")[["district", "latitude", "longitude"]]
    d = pd.concat([d, pd.read_csv("data/extra_districts.csv")], ignore_index=True)
    la, lo = np.radians(df["latitude"].values[:, None]), np.radians(df["longitude"].values[:, None])
    dl, dn = np.radians(d["latitude"].values[None, :]), np.radians(d["longitude"].values[None, :])
    a = np.sin((dl - la) / 2) ** 2 + np.cos(la) * np.cos(dl) * np.sin((dn - lo) / 2) ** 2
    df["district"] = d["district"].values[a.argmin(axis=1)]
    return df


if __name__ == "__main__":
    df = add_district(to_dataframe(fetch()))
    df = df[["district", "area", "type", "latitude", "longitude"]].sort_values(["district", "area"])
    df.to_csv("data/tamil_nadu_area_locations.csv", index=False)
    print(f"Saved {len(df)} areas")
    print(df["type"].value_counts().to_string())
