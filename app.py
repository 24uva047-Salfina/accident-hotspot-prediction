"""Accident Hotspot Prediction - Flask backend."""
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from features import build_features, label_from_score, risk_score

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

app = Flask(__name__, template_folder="templates")

print("TomTom key loaded:", bool(TOMTOM_API_KEY))

MODEL_PATH = os.path.join(BASE_DIR, "model", "accident_risk_model.pkl")
DATA_PATH = os.path.join(
    BASE_DIR, "data", "preprocessed_tamil_nadu_accidents.csv"
)
EXTRA_PATH = os.path.join(BASE_DIR, "data", "extra_districts.csv")
AREAS_PATH = os.path.join(BASE_DIR, "data", "tamil_nadu_area_locations.csv")

# ------------------------------------------------------------
# Load model and accident dataset
# ------------------------------------------------------------

print("Loading accident risk model...")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Accident dataset not found: {DATA_PATH}")

bundle = joblib.load(MODEL_PATH)

if not isinstance(bundle, dict) or "model" not in bundle or "classes" not in bundle:
    raise SystemExit("Model file must contain 'model' and 'classes'.")

model = bundle["model"]
CLASSES = list(bundle["classes"])

districts = pd.read_csv(DATA_PATH)

required_columns = [
    "district", "latitude", "longitude", "coordinate_status",
    "2021_fatal", "total__2021", "total__2020", "2020_fatal",
    "2020_nonfatal", "2021_nonfatal", "accident_change",
    "accident_change_percent"
]
missing_columns = [c for c in required_columns if c not in districts.columns]

if missing_columns:
    raise SystemExit(f"Dataset missing columns: {missing_columns}")

districts = districts[
    districts["coordinate_status"].astype(str).str.strip().str.lower() == "valid"
].copy()

districts = districts.dropna(
    subset=["latitude", "longitude", "total__2021", "2021_fatal"]
).reset_index(drop=True)

if districts.empty:
    raise SystemExit("No valid district rows found in the accident dataset.")

X_now = build_features(districts, "2021_fatal", "total__2021")
proba = model.predict_proba(X_now)

districts["base_idx"] = proba.argmax(axis=1)
districts["base_conf"] = proba.max(axis=1) * 100
districts["observed_idx"] = label_from_score(
    risk_score(districts["total__2021"])
).values

districts["ev"] = proba @ np.array([1.0, 2.0, 3.0])
districts["severity"] = (
    0.5 * districts["ev"]
    + 0.5 * (1 + 2 * risk_score(districts["total__2021"]) / 100)
)

D_EV = districts["severity"].to_numpy(dtype=float)
D_LAT = districts["latitude"].to_numpy(dtype=float)
D_LON = districts["longitude"].to_numpy(dtype=float)

print("Accident risk model loaded successfully!")
print("Valid district records:", len(districts))

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        np.radians, (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_district_idx(lats, lons):
    lats, lons = np.atleast_1d(lats), np.atleast_1d(lons)
    distances = haversine_km(
        lats[:, None], lons[:, None],
        D_LAT[None, :], D_LON[None, :]
    )
    return distances.argmin(axis=1)


def idw_risk(lats, lons, k=3):
    """Estimate continuous risk from the nearest district centres."""
    lats, lons = np.atleast_1d(lats), np.atleast_1d(lons)
    distances = haversine_km(
        lats[:, None], lons[:, None],
        D_LAT[None, :], D_LON[None, :]
    )
    k = min(k, len(D_EV))
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, indices, axis=1)
    weights = 1.0 / np.maximum(nearest_distances, 1.0) ** 2
    values = (weights * D_EV[indices]).sum(axis=1) / weights.sum(axis=1)
    return values, distances.argmin(axis=1)


def in_tamil_nadu(lat, lon):
    return 8 <= lat <= 13.6 and 76 <= lon <= 80.4


def in_region(lat, lon):
    """Wider box used for routes: a road between two Tamil Nadu places can
    briefly cross the state border (Hosur, Gudalur, north of Chennai ...)."""
    return 7 <= lat <= 15 and 75 <= lon <= 82


_weather_cache = {}


def get_weather(lat, lon):
    """Fetch current weather; cache for 10 minutes per approximate grid cell."""
    key = (round(lat, 1), round(lon, 1))
    cached = _weather_cache.get(key)

    if cached and time.time() - cached[0] < 600:
        return cached[1]

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "timezone": "auto",
                "current": (
                    "temperature_2m,precipitation,"
                    "wind_speed_10m,weather_code"
                )
            },
            timeout=10
        )
        response.raise_for_status()
        current = response.json()["current"]

        weather = {
            "temperature": float(current["temperature_2m"]),
            "precipitation": float(current["precipitation"]),
            "wind_speed": float(current["wind_speed_10m"]),
            "weather_code": int(current.get("weather_code", 0))
        }
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        print("Weather error:", error)
        weather = None

    _weather_cache[key] = (time.time(), weather)
    return weather


HEAVY_RAIN_CODES = {65, 67, 82, 95, 96, 99}
FOG_CODES = {45, 48}


def weather_severity(weather):
    if not weather:
        return 0, []

    reasons = []

    if (
        weather["precipitation"] >= 2.5
        or weather["weather_code"] in HEAVY_RAIN_CODES
    ):
        reasons.append("Heavy rain / thunderstorm - slippery road")

    if weather["weather_code"] in FOG_CODES:
        reasons.append("Fog - low visibility")

    if weather["wind_speed"] >= 40:
        reasons.append("Strong wind")

    if reasons:
        return 1, reasons

    if weather["precipitation"] > 0:
        return 0, ["Light rain - wet road"]

    return 0, []


def weather_adjust(index, weather):
    bump, reasons = weather_severity(weather)
    return min(2, index + bump), reasons


def recommendations(level, reasons):
    tips = {
        "Low": [
            "Normal driving conditions - follow speed limits and wear a seatbelt/helmet."
        ],
        "Medium": [
            "Reduce speed and keep a safe following distance.",
            "Stay alert at junctions and near bus stops."
        ],
        "High": [
            "High-risk zone - drive slowly and avoid overtaking.",
            "Avoid night travel here if an alternative route exists.",
            "Prefer the safer alternative route suggested on the map."
        ]
    }.get(level, ["Drive carefully and follow road safety rules."]).copy()

    for reason in reasons:
        if "rain" in reason.lower():
            tips.append("Wet road: brake early and double your following distance.")
        if "fog" in reason.lower():
            tips.append("Use fog lamps/low beam and slow down.")
        if "wind" in reason.lower():
            tips.append("Hold the steering firmly, especially two-wheelers.")

    return tips


def level_from_value(value):
    return "Low" if value < 1.5 else "Medium" if value < 2.5 else "High"


def safe_float(value):
    try:
        return float(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def clip_bbox(west, south, east, north, max_km2=9000.0):
    """TomTom accepts at most 10,000 km2 per incident request.
    If the map view is bigger, use the centre part of it."""
    mid_lat = (south + north) / 2
    width_km = (east - west) * 111.32 * np.cos(np.radians(mid_lat))
    height_km = (north - south) * 110.57
    area = width_km * height_km

    if area <= max_km2:
        return west, south, east, north, False

    scale = (max_km2 / area) ** 0.5
    cx, cy = (west + east) / 2, mid_lat
    half_w, half_h = (east - west) / 2 * scale, (north - south) / 2 * scale
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h, True


# ------------------------------------------------------------
# Website pages
# ------------------------------------------------------------

@app.route("/")
@app.route("/landing")
def landing():
    return render_template("landing.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/risk-map")
def risk_map():
    return render_template("tamil_nadu_accident_risk_map.html")


@app.route("/logout")
def logout():
    return render_template("index.html")


@app.route("/all-areas")
def all_areas_page():
    return render_template("all_areas.html")


# ------------------------------------------------------------
# Basic APIs
# ------------------------------------------------------------

@app.route("/api/test")
def test_backend():
    return jsonify(
        status="success",
        message="Accident Hotspot Backend is connected!"
    )


@app.route("/api/weather")
def api_weather():
    lat = request.args.get("latitude", type=float)
    lon = request.args.get("longitude", type=float)

    if lat is None or lon is None:
        return jsonify(
            status="error",
            message="Latitude and longitude are required."
        ), 400

    if not in_tamil_nadu(lat, lon):
        return jsonify(
            status="error",
            message="Location is outside Tamil Nadu."
        ), 400

    weather = get_weather(lat, lon)

    if weather is None:
        return jsonify(
            status="error",
            message="Weather service unavailable."
        ), 502

    return jsonify(
        status="success",
        location={"latitude": lat, "longitude": lon},
        weather=weather
    )


@app.route("/api/traffic")
def api_traffic():
    lat = request.args.get("latitude", type=float)
    lon = request.args.get("longitude", type=float)

    if lat is None or lon is None:
        return jsonify(
            status="error",
            message="Latitude and longitude are required."
        ), 400

    if not in_tamil_nadu(lat, lon):
        return jsonify(
            status="error",
            message="Location is outside Tamil Nadu."
        ), 400

    if not TOMTOM_API_KEY:
        return jsonify(
            status="error",
            message="TomTom API key is not configured."
        ), 503

    try:
        response = requests.get(
            "https://api.tomtom.com/traffic/services/4/"
            "flowSegmentData/absolute/10/json",
            params={
                "point": f"{lat},{lon}",
                "key": TOMTOM_API_KEY
            },
            timeout=10
        )

        if response.status_code != 200:
            print("TomTom traffic error:", response.status_code, response.text[:500])
            return jsonify(
                status="error",
                message="TomTom traffic request failed.",
                code=response.status_code
            ), response.status_code

        return jsonify(
            status="success",
            location={"latitude": lat, "longitude": lon},
            traffic=response.json()
        )

    except requests.RequestException as error:
        print("TomTom traffic request exception:", error)
        return jsonify(
            status="error",
            message="Traffic service temporarily unavailable."
        ), 502


# ------------------------------------------------------------
# TomTom live traffic incidents
# ------------------------------------------------------------

@app.route("/api/incidents")
def api_incidents():
    """Fetch current traffic incidents for the visible map area."""

    if not TOMTOM_API_KEY:
        return jsonify(
            status="error",
            message="TomTom API key is not configured."
        ), 503

    try:
        west = request.args.get("west", type=float)
        south = request.args.get("south", type=float)
        east = request.args.get("east", type=float)
        north = request.args.get("north", type=float)

        if None in (west, south, east, north):
            return jsonify(
                status="error",
                message="Map bounds are required."
            ), 400

        if not (
            76 <= west < east <= 81
            and 8 <= south < north <= 14
        ):
            return jsonify(
                status="error",
                message="Bounds must be within Tamil Nadu."
            ), 400

        # TomTom allows max 10,000 km2 per request -> use the centre of the view
        west, south, east, north, clipped = clip_bbox(west, south, east, north)

        url = (
            "https://api.tomtom.com/traffic/services/5/"
            "incidentDetails"
        )

        params = {
            "key": TOMTOM_API_KEY,
            "bbox": f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}",
            "fields": (
                "{incidents{type,geometry{type,coordinates},"
                "properties{id,iconCategory,magnitudeOfDelay,"
                "events{description},from,to,startTime,lastReportTime}}}"
            ),
            "language": "en-GB",
            "timeValidityFilter": "present"
        }

        response = requests.get(url, params=params, timeout=15)

        if response.status_code != 200:
            # The detailed error is logged locally; the key is not printed.
            print("TomTom incidents HTTP status:", response.status_code)
            print("TomTom incidents error:", response.text[:1000])

            return jsonify(
                status="error",
                message="TomTom incident request failed.",
                code=response.status_code,
                details=response.text[:1000]
            ), response.status_code

        result = response.json()
        incidents = result.get("incidents", [])

        return jsonify(
            status="success",
            source="TomTom Traffic Incidents",
            count=len(incidents),
            clipped_to_centre=clipped,
            incidents=incidents
        )

    except requests.RequestException as error:
        print("TomTom incident request exception:", error)
        return jsonify(
            status="error",
            message="Live incident service is temporarily unavailable."
        ), 502

    except (ValueError, TypeError) as error:
        print("TomTom incident response error:", error)
        return jsonify(
            status="error",
            message="Unable to process the live incident response."
        ), 502


# ------------------------------------------------------------
# Accident risk prediction API
# ------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify(
            status="error",
            message="No JSON data received."
        ), 400

    try:
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))
    except (TypeError, ValueError):
        return jsonify(
            status="error",
            message="Latitude and longitude are required."
        ), 400

    if not in_tamil_nadu(lat, lon):
        return jsonify(
            status="error",
            message="Location is outside Tamil Nadu."
        ), 400

    # 1) the district named by the map (OpenStreetMap) wins
    # 2) otherwise the nearest of ALL 38 district centres
    hint = resolve_district(data.get("district_hint"))
    row = None
    estimated_district = None

    if hint in _ROW:
        row = districts.iloc[_ROW[hint]]
    elif hint in NO_DATA:
        estimated_district = hint
    else:
        nearest_all = int(
            np.argmin(haversine_km(lat, lon, ALL_LAT, ALL_LON))
        )
        if str(ALL_NAMES[nearest_all]) in NO_DATA:
            estimated_district = str(ALL_NAMES[nearest_all])
        else:
            row = districts.iloc[int(nearest_district_idx(lat, lon)[0])]

    weather = get_weather(lat, lon)

    weather_json = {
        "temperature": weather["temperature"],
        "precipitation": weather["precipitation"],
        "wind_speed": weather["wind_speed"]
    } if weather else None

    # Chennai / Mayiladuthurai: no accident figures -> clearly labelled estimate
    if row is None:
        values, near = idw_risk(lat, lon)
        base_idx = CLASSES.index(level_from_value(float(values[0])))
        final_idx, reasons = weather_adjust(base_idx, weather)
        level = str(CLASSES[final_idx])
        helper_row = districts.iloc[int(near[0])]

        return jsonify(
            status="success",
            location=data.get("location", "Selected Tamil Nadu Location"),
            district=estimated_district,
            nearest_district=str(helper_row["district"]),
            risk_basis=(
                "Estimated from nearby district data; "
                "local accident figures unavailable"
            ),
            estimated=True,
            area_specific_accident_data_available=False,
            latitude=lat,
            longitude=lon,
            risk_level=level,
            base_risk_level=str(CLASSES[base_idx]),
            confidence=None,
            weather_available=weather is not None,
            weather_factors=reasons,
            recommendations=recommendations(level, reasons),
            weather=weather_json,
            nearest_location={
                "latitude": float(helper_row["latitude"]),
                "longitude": float(helper_row["longitude"])
            },
            accident_data={
                "total_2020": None, "total_2021": None,
                "fatal_2020": None, "fatal_2021": None,
                "nonfatal_2020": None, "nonfatal_2021": None,
                "accident_change": None, "accident_change_percent": None
            }
        )

    final_idx, reasons = weather_adjust(int(row["base_idx"]), weather)
    level = str(CLASSES[final_idx])

    return jsonify(
        status="success",
        location=data.get("location", "Selected Tamil Nadu Location"),
        district=str(row["district"]),
        nearest_district=str(row["district"]),
        risk_basis="District-level historical accident data (2020-2021)",
        estimated=False,
        area_specific_accident_data_available=False,
        latitude=lat,
        longitude=lon,
        risk_level=level,
        base_risk_level=str(CLASSES[int(row["base_idx"])]),
        confidence=round(float(row["base_conf"]), 2),
        weather_available=weather is not None,
        weather_factors=reasons,
        recommendations=recommendations(level, reasons),
        weather=weather_json,
        nearest_location={
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"])
        },
        accident_data={
            "total_2020": safe_float(row["total__2020"]),
            "total_2021": safe_float(row["total__2021"]),
            "fatal_2020": safe_float(row["2020_fatal"]),
            "fatal_2021": safe_float(row["2021_fatal"]),
            "nonfatal_2020": safe_float(row["2020_nonfatal"]),
            "nonfatal_2021": safe_float(row["2021_nonfatal"]),
            "accident_change": safe_float(row["accident_change"]),
            "accident_change_percent": safe_float(
                row["accident_change_percent"]
            )
        }
    )


# ------------------------------------------------------------
# Route risk API
# ------------------------------------------------------------

@app.route("/api/route-risk", methods=["POST"])
def api_route_risk():
    """Analyse the risk along a route using district-level estimates."""

    data = request.get_json(silent=True) or {}
    coordinates = data.get("coordinates")

    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return jsonify(
            status="error",
            message="Send coordinates: [[lat, lon], ...]"
        ), 400

    if len(coordinates) > 50000:
        return jsonify(
            status="error",
            message="Route too long."
        ), 400

    try:
        points = np.asarray(coordinates, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Invalid coordinate shape")

        if not np.isfinite(points).all():
            raise ValueError("Coordinates must be finite")

    except (ValueError, TypeError):
        return jsonify(
            status="error",
            message="Invalid coordinates."
        ), 400

    # Only reject nonsense. A real road may cross the state border for a while.
    inside = (
        (points[:, 0] >= 7) & (points[:, 0] <= 15)
        & (points[:, 1] >= 75) & (points[:, 1] <= 82)
    )
    if not inside.all():
        return jsonify(
            status="error",
            message="Route coordinates must be in South India."
        ), 400

    segment_km = haversine_km(
        points[:-1, 0], points[:-1, 1],
        points[1:, 0], points[1:, 1]
    )

    mid_lat = (points[:-1, 0] + points[1:, 0]) / 2
    mid_lon = (points[:-1, 1] + points[1:, 1]) / 2

    values, nearest = idw_risk(mid_lat, mid_lon)
    values = values.copy()

    count = len(segment_km)

    # weather at start / middle / end; each segment uses the nearest third
    sample_positions = [0, count // 2, count - 1]
    weather_samples = [
        get_weather(float(mid_lat[i]), float(mid_lon[i]))
        for i in sample_positions
    ]
    third = np.minimum(2, (np.arange(count) * 3) // max(count, 1))

    reasons = []

    for part in range(3):
        bump, sample_reasons = weather_severity(weather_samples[part])
        mask = third == part
        values[mask] = np.minimum(3.0, values[mask] + bump)

        for reason in sample_reasons:
            if reason not in reasons:
                reasons.append(reason)

    classes = np.where(
        values < 1.5, 0,
        np.where(values < 2.5, 1, 2)
    )

    low_km = float(segment_km[classes == 0].sum())
    medium_km = float(segment_km[classes == 1].sum())
    high_km = float(segment_km[classes == 2].sum())
    total_km = low_km + medium_km + high_km

    score = (
        float((values * segment_km).sum() / total_km)
        if total_km > 0 else 1.0
    )

    overall = (
        "High" if score >= 2.5
        else "Medium" if score >= 1.7
        else "Low"
    )

    district_names = districts["district"].to_numpy()[nearest]
    crossed = list(dict.fromkeys(district_names.tolist()))
    risky = list(dict.fromkeys(district_names[classes == 2].tolist()))

    tips = recommendations(overall, reasons)
    if risky:
        tips.insert(0, "Extra care needed in: " + ", ".join(risky))

    return jsonify(
        status="success",
        analysis={
            "lowKm": low_km,
            "mediumKm": medium_km,
            "highKm": high_km,
            "totalKm": total_km,
            "riskScore": score,
            "overall": overall,
            "districts": crossed,
            "weather_factors": reasons,
            "tips": tips,
            "weather_available": any(
                sample is not None for sample in weather_samples
            )
        }
    )


# ------------------------------------------------------------
# Dashboard API
# ------------------------------------------------------------

@app.route("/api/dashboard-data")
def api_dashboard():
    df = districts.copy()
    df["risk_level"] = [str(CLASSES[int(i)]) for i in df["base_idx"]]
    counts = df["risk_level"].value_counts()

    return jsonify(
        status="success",
        summary={
            "total_2020": int(df["total__2020"].sum()),
            "total_2021": int(df["total__2021"].sum()),
            "district_count": len(df),
            "high_risk_count": int(counts.get("High", 0))
        },
        risk_distribution={
            str(level): int(counts.get(str(level), 0))
            for level in CLASSES
        },
        model_metrics=bundle.get("metrics", {}),
        districts=[
            {
                "district": str(row["district"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "total_2020": int(row["total__2020"]),
                "total_2021": int(row["total__2021"]),
                "fatal_2020": int(row["2020_fatal"]),
                "fatal_2021": int(row["2021_fatal"]),
                "accident_change": safe_float(row["accident_change"]),
                "risk_level": str(row["risk_level"]),
                "observed_risk": str(
                    CLASSES[int(row["observed_idx"])]
                )
            }
            for _, row in df.iterrows()
        ]
    )


# ------------------------------------------------------------
# District and area datasets
# ------------------------------------------------------------

if os.path.exists(EXTRA_PATH):
    extra_districts = pd.read_csv(EXTRA_PATH)
else:
    extra_districts = pd.DataFrame(
        columns=["district", "latitude", "longitude"]
    )

required_extra = {"district", "latitude", "longitude"}
if not extra_districts.empty and not required_extra.issubset(
    extra_districts.columns
):
    raise SystemExit(
        "extra_districts.csv must contain district, latitude and longitude."
    )

ALL_NAMES = np.concatenate([
    districts["district"].astype(str).to_numpy(),
    extra_districts["district"].astype(str).to_numpy()
])

ALL_LAT = np.concatenate([
    D_LAT,
    extra_districts["latitude"].to_numpy(dtype=float)
])

ALL_LON = np.concatenate([
    D_LON,
    extra_districts["longitude"].to_numpy(dtype=float)
])

NO_DATA = set(extra_districts["district"].astype(str).tolist())


# ------------------------------------------------------------
# District-name matching (OpenStreetMap / common spellings -> dataset names)
# ------------------------------------------------------------

_ALIASES = {
    "kanchipuram": "Kancheepuram", "kancheepuram": "Kancheepuram",
    "kanyakumari": "Kanniyakumari", "kanniyakumari": "Kanniyakumari",
    "ranipet": "Ranipettai", "ranipettai": "Ranipettai",
    "nilgiris": "The Nilgiris", "thenilgiris": "The Nilgiris",
    "nilgiri": "The Nilgiris",
    "thiruvallur": "Tiruvallur", "tiruvallur": "Tiruvallur",
    "tiruvarur": "Thiruvarur", "thiruvarur": "Thiruvarur",
    "tirupathur": "Thirupathur", "tirupattur": "Thirupathur",
    "thirupathur": "Thirupathur",
    "tiruppur": "Tirupur", "tirupur": "Tirupur",
    "tiruvannamalai": "Thiruvannamalai", "thiruvannamalai": "Thiruvannamalai",
    "viluppuram": "Villupuram", "villupuram": "Villupuram",
    "tuticorin": "Thoothukudi", "thoothukkudi": "Thoothukudi",
    "thoothukudi": "Thoothukudi",
    "trichy": "Tiruchirappalli", "tiruchirapalli": "Tiruchirappalli",
    "tiruchchirappalli": "Tiruchirappalli", "tiruchirappalli": "Tiruchirappalli",
    "sivaganga": "Sivagangai", "sivagangai": "Sivagangai",
    "pudukottai": "Pudukkottai", "pudukkottai": "Pudukkottai",
    "tanjore": "Thanjavur", "thanjavur": "Thanjavur",
    "thirunelveli": "Tirunelveli", "tirunelveli": "Tirunelveli",
    "mayiladudurai": "Mayiladuthurai", "mayiladuthurai": "Mayiladuthurai",
    "madras": "Chennai", "chennai": "Chennai",
}


def _name_key(name):
    key = re.sub(r"[^a-z]", "", str(name or "").lower())
    return key[:-8] if key.endswith("district") else key


_CANON = {
    _name_key(n): n
    for n in list(districts["district"]) + list(extra_districts["district"])
}
_CANON.update({_name_key(a): c for a, c in _ALIASES.items()})
_ROW = {str(r["district"]): i for i, r in districts.iterrows()}


def resolve_district(hint):
    """Map a district name from the map to the dataset spelling (or None)."""
    return _CANON.get(_name_key(hint)) if hint else None


@app.route("/api/districts-all")
def api_districts_all():
    """Return district locations and clearly label estimates."""

    output = []

    for _, row in districts.iterrows():
        output.append({
            "district": str(row["district"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "risk": str(CLASSES[int(row["base_idx"])]),
            "risk_basis": "Historical district accident data (2020-2021)",
            "estimated": False,
            "total_2021": int(row["total__2021"]),
            "fatal_2021": int(row["2021_fatal"])
        })

    if not extra_districts.empty:
        extra_lat = extra_districts["latitude"].to_numpy(dtype=float)
        extra_lon = extra_districts["longitude"].to_numpy(dtype=float)
        estimates, nearest = idw_risk(extra_lat, extra_lon)

        for i, (_, row) in enumerate(extra_districts.iterrows()):
            output.append({
                "district": str(row["district"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "risk": level_from_value(float(estimates[i])),
                "risk_basis": (
                    "Estimated from nearby district data; "
                    "local accident figures unavailable"
                ),
                "estimated": True,
                "nearest_data_district": str(
                    districts.iloc[int(nearest[i])]["district"]
                ),
                "total_2021": None,
                "fatal_2021": None
            })

    return jsonify(
        status="success",
        count=len(output),
        districts=output
    )


@app.route("/api/areas")
def api_areas():
    """Return configured area locations with estimated risk."""

    if not os.path.exists(AREAS_PATH):
        return jsonify(status="success", count=0, areas=[])

    areas_df = pd.read_csv(AREAS_PATH)

    required_area_columns = {"area", "latitude", "longitude"}
    if not required_area_columns.issubset(areas_df.columns):
        return jsonify(
            status="error",
            message="Area dataset is missing required columns."
        ), 500

    areas_df = areas_df[
        areas_df["latitude"].between(8, 13.6)
        & areas_df["longitude"].between(76, 80.4)
    ].reset_index(drop=True)

    if areas_df.empty:
        return jsonify(status="success", count=0, areas=[])

    latitudes = areas_df["latitude"].to_numpy(dtype=float)
    longitudes = areas_df["longitude"].to_numpy(dtype=float)

    values, _ = idw_risk(latitudes, longitudes)

    distances = haversine_km(
        latitudes[:, None], longitudes[:, None],
        ALL_LAT[None, :], ALL_LON[None, :]
    )
    nearest_all = distances.argmin(axis=1)

    output = []

    for i in range(len(areas_df)):
        if "district" in areas_df and pd.notna(areas_df["district"].iloc[i]):
            district_name = str(areas_df["district"].iloc[i])
        else:
            district_name = str(ALL_NAMES[nearest_all[i]])

        # same rule as /api/districts-all: Chennai-side areas are estimates
        near_no_data = str(ALL_NAMES[nearest_all[i]]) in NO_DATA

        output.append({
            "area": str(areas_df["area"].iloc[i]),
            "district": district_name,
            "type": (
                str(areas_df["type"].iloc[i])
                if "type" in areas_df else "area"
            ),
            "latitude": float(latitudes[i]),
            "longitude": float(longitudes[i]),
            "risk": level_from_value(float(values[i])),
            "score": round(float(values[i]), 2),
            "estimated": True,
            "risk_basis": (
                "Estimated from nearby districts; this district has no "
                "local accident figures"
                if near_no_data
                else "Estimated from nearby district-level data"
            ),
            "area_specific_accident_data_available": False
        })

    return jsonify(
        status="success",
        count=len(output),
        areas=output
    )


# ------------------------------------------------------------
# Start Flask
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1"
    )