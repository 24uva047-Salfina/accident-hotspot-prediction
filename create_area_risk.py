import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

AREA_FILE = "data/tamil_nadu_area_locations.csv"
ACCIDENT_FILE = "data/preprocessed_tamil_nadu_accidents.csv"
OUTPUT_FILE = "data/tamil_nadu_area_risk.csv"

print("\n========================================")
print("AREA RISK DATASET CREATION")
print("========================================")

# 1. LOAD AREA DATA
print("\nLoading area data...")

areas = pd.read_csv(AREA_FILE)

print("Area data shape:", areas.shape)
print("Area columns:", areas.columns.tolist())


# 2. LOAD ACCIDENT DATA
print("\nLoading accident data...")

accidents = pd.read_csv(ACCIDENT_FILE)

print("Accident data shape:", accidents.shape)
print("Accident columns:", accidents.columns.tolist())


# 3. CLEAN COORDINATES
print("\nCleaning coordinates...")

for col in ["latitude", "longitude"]:
    areas[col] = pd.to_numeric(
        areas[col],
        errors="coerce"
    )

    accidents[col] = pd.to_numeric(
        accidents[col],
        errors="coerce"
    )


areas = areas.dropna(
    subset=["latitude", "longitude"]
).copy()

accidents = accidents.dropna(
    subset=["latitude", "longitude"]
).copy()


# 4. TAMIL NADU BOUNDARY
areas = areas[
    areas["latitude"].between(8, 14)
    &
    areas["longitude"].between(76, 81)
].copy()

accidents = accidents[
    accidents["latitude"].between(8, 14)
    &
    accidents["longitude"].between(76, 81)
].copy()


print("\nValid areas:", len(areas))
print("Valid accident locations:", len(accidents))


# 5. FIND NEAREST ACCIDENT LOCATION
print("\nFinding nearest accident location for each area...")

earth_radius_km = 6371

area_coords = np.radians(
    areas[["latitude", "longitude"]].values
)

accident_coords = np.radians(
    accidents[["latitude", "longitude"]].values
)

tree = BallTree(
    accident_coords,
    metric="haversine"
)

distances, indices = tree.query(
    area_coords,
    k=1
)

areas["nearest_distance_km"] = (
    distances[:, 0] * earth_radius_km
)

nearest_index = indices[:, 0]


# 6. COPY ACCIDENT STATISTICS
print("\nCopying accident statistics...")

copy_columns = [
    "district",
    "total__2020",
    "total__2021",
    "2020_fatal",
    "2021_fatal",
    "2020_nonfatal",
    "2021_nonfatal"
]

for col in copy_columns:
    if col in accidents.columns:
        areas[col] = accidents.iloc[
            nearest_index
        ][col].values


# 7. CALCULATE ACCIDENT COUNT
areas["accident_count"] = (
    areas["total__2020"].fillna(0)
    +
    areas["total__2021"].fillna(0)
)


# 8. RISK CLASSIFICATION
def get_risk(value):

    if value >= 100:
        return "High"

    elif value >= 40:
        return "Medium"

    else:
        return "Low"


areas["risk_level"] = areas[
    "accident_count"
].apply(get_risk)


# 9. FINAL DATASET
print("\nCreating final dataset...")

columns = [
    "district",
    "area",
    "latitude",
    "longitude",
    "nearest_distance_km",
    "accident_count",
    "risk_level"
]

result = areas[columns].copy()


# 10. SAVE
result.to_csv(
    OUTPUT_FILE,
    index=False
)


# 11. DISPLAY RESULT
print("\n========================================")
print("AREA RISK DATASET CREATED")
print("========================================")

print("Output:", OUTPUT_FILE)
print("Total areas:", len(result))

print("\nRisk distribution:")
print(
    result["risk_level"].value_counts()
)

print("\nSample:")
print(
    result.head(10).to_string(
        index=False
    )
)

print("\nDONE!")