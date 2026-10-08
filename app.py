from flask import Flask, render_template, jsonify, request
import joblib
import pandas as pd
import os
import requests
from dotenv import load_dotenv


# =========================================================
# FLASK APP
# =========================================================

app = Flask(
    __name__,
    template_folder="templates"
)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# FILE PATHS
# =========================================================

OLD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "accident_lightgbm_model.pkl"
)

WEATHER_MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "accident_weather_model.pkl"
)

WEATHER_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "weather_risk_label_encoder.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "preprocessed_tamil_nadu_accidents.csv"
)


# =========================================================
# LOAD OLD MODEL
# Used for dashboard
# =========================================================

print("\nLoading LightGBM model...")

model = joblib.load(
    OLD_MODEL_PATH
)

print("LightGBM model loaded successfully!")


# =========================================================
# LOAD WEATHER-AWARE MODEL
# Used for real-time prediction
# =========================================================

print("\nLoading weather-aware model...")

weather_model = joblib.load(
    WEATHER_MODEL_PATH
)

weather_encoder = joblib.load(
    WEATHER_ENCODER_PATH
)

print(
    "Weather-aware model loaded successfully!"
)


# =========================================================
# LOAD ACCIDENT DATA
# =========================================================

print("\nLoading Tamil Nadu accident dataset...")

accident_data = pd.read_csv(
    DATA_PATH
)

print(
    "Accident dataset shape:",
    accident_data.shape
)

print(
    "Accident dataset columns:",
    accident_data.columns.tolist()
)


# =========================================================
# OLD MODEL FEATURES
# =========================================================

FEATURES = [
    "2020_fatal",
    "2021_fatal",
    "2020_nonfatal",
    "2021_nonfatal",
    "total__2020",
    "total__2021",
    "accident_change",
    "accident_change_percent",
    "latitude",
    "longitude"
]


# =========================================================
# WEATHER MODEL FEATURES
# =========================================================

WEATHER_FEATURES = [
    "2020_fatal",
    "2021_fatal",
    "2020_nonfatal",
    "2021_nonfatal",
    "total__2020",
    "total__2021",
    "accident_change",
    "accident_change_percent",
    "latitude",
    "longitude",
    "avg_temperature",
    "total_precipitation",
    "avg_wind_speed"
]


# =========================================================
# CHECK DATA FEATURES
# =========================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in accident_data.columns
]

if missing_features:

    raise ValueError(
        "Missing model features in dataset: "
        + str(missing_features)
    )

print(
    "All accident model features are available."
)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "landing.html"
    )


# =========================================================
# BACKEND TEST
# =========================================================

@app.route("/api/test")
def test_backend():

    return jsonify({

        "status": "success",

        "message":
            "Accident Hotspot Backend is connected!"

    })


# =========================================================
# LIVE WEATHER API
# OPEN-METEO
# =========================================================

@app.route("/api/weather")
def get_weather():

    latitude = request.args.get(
        "latitude",
        type=float
    )

    longitude = request.args.get(
        "longitude",
        type=float
    )

    if (
        latitude is None
        or longitude is None
    ):

        return jsonify({

            "status": "error",

            "message":
                "Latitude and longitude are required."

        }), 400


    url = (
        "https://api.open-meteo.com/"
        "v1/forecast"
    )


    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "wind_speed_10m,"
            "weather_code"
        ),

        "timezone": "auto"
    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        weather_data = response.json()


        return jsonify({

            "status": "success",

            "location": {

                "latitude": latitude,

                "longitude": longitude

            },

            "weather":
                weather_data.get(
                    "current",
                    {}
                )

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =========================================================
# LIVE TRAFFIC API
# TOMTOM
# =========================================================

@app.route("/api/traffic")
def get_traffic():

    latitude = request.args.get(
        "latitude",
        type=float
    )

    longitude = request.args.get(
        "longitude",
        type=float
    )


    if (
        latitude is None
        or longitude is None
    ):

        return jsonify({

            "status": "error",

            "message":
                "Latitude and longitude are required."

        }), 400


    if not TOMTOM_API_KEY:

        return jsonify({

            "status": "error",

            "message":
                "TomTom API key is not configured."

        }), 500


    url = (
        "https://api.tomtom.com/"
        "traffic/services/4/"
        "flowSegmentData/"
        "absolute/10/json"
    )


    params = {

        "point":
            f"{latitude},{longitude}",

        "key":
            TOMTOM_API_KEY

    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )


        if response.status_code != 200:

            return jsonify({

                "status": "error",

                "message":
                    "TomTom API request failed.",

                "code":
                    response.status_code,

                "details":
                    response.text

            }), response.status_code


        traffic_data = response.json()


        return jsonify({

            "status": "success",

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude

            },

            "traffic":
                traffic_data

        })


    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# =========================================================
# REAL-TIME ACCIDENT RISK PREDICTION
# WEATHER-AWARE LIGHTGBM
# =========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        # -------------------------------------------------
        # CHECK JSON
        # -------------------------------------------------

        if data is None:

            return jsonify({

                "status": "error",

                "message":
                    "No JSON data received."

            }), 400


        # -------------------------------------------------
        # GET LOCATION
        # -------------------------------------------------

        location = data.get(
            "location",
            "Selected Tamil Nadu Location"
        )

        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )


        # -------------------------------------------------
        # VALIDATE COORDINATES
        # -------------------------------------------------

        if (
            latitude is None
            or longitude is None
        ):

            return jsonify({

                "status": "error",

                "message":
                    "Latitude and longitude are required."

            }), 400


        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )


        # -------------------------------------------------
        # TAMIL NADU BOUNDARY CHECK
        # -------------------------------------------------

        if not (
            8 <= latitude <= 14
            and
            76 <= longitude <= 81
        ):

            return jsonify({

                "status": "error",

                "message":
                    "Location is outside Tamil Nadu."

            }), 400


        # -------------------------------------------------
        # FIND NEAREST HISTORICAL DISTRICT
        # -------------------------------------------------

        temp_data = accident_data.copy()


        temp_data = temp_data[
            temp_data["coordinate_status"]
            == "Valid"
        ].copy()


        temp_data["distance"] = (

            (
                temp_data["latitude"]
                - latitude
            ) ** 2

            +

            (
                temp_data["longitude"]
                - longitude
            ) ** 2

        )


        nearest_index = (
            temp_data["distance"]
            .idxmin()
        )


        row = temp_data.loc[
            nearest_index
        ]


        district = str(
            row["district"]
        )


        # -------------------------------------------------
        # CURRENT WEATHER
        # OPEN-METEO
        # -------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/"
            "v1/forecast"
        )


        weather_params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "current": [
                "temperature_2m",
                "precipitation",
                "wind_speed_10m"
            ],

            "timezone": "auto"
        }


        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=15
        )


        weather_response.raise_for_status()


        weather_data = (
            weather_response.json()
        )


        current_weather = (
            weather_data["current"]
        )


        temperature = float(
            current_weather[
                "temperature_2m"
            ]
        )


        precipitation = float(
            current_weather[
                "precipitation"
            ]
        )


        wind_speed = float(
            current_weather[
                "wind_speed_10m"
            ]
        )


        # -------------------------------------------------
        # CREATE WEATHER MODEL INPUT
        # -------------------------------------------------

        input_data = pd.DataFrame([{

            "2020_fatal":
                float(
                    row["2020_fatal"]
                ),

            "2021_fatal":
                float(
                    row["2021_fatal"]
                ),

            "2020_nonfatal":
                float(
                    row["2020_nonfatal"]
                ),

            "2021_nonfatal":
                float(
                    row["2021_nonfatal"]
                ),

            "total__2020":
                float(
                    row["total__2020"]
                ),

            "total__2021":
                float(
                    row["total__2021"]
                ),

            "accident_change":
                float(
                    row["accident_change"]
                ),

            "accident_change_percent":
                float(
                    row[
                        "accident_change_percent"
                    ]
                ),

            "latitude":
                latitude,

            "longitude":
                longitude,

            "avg_temperature":
                temperature,

            "total_precipitation":
                precipitation,

            "avg_wind_speed":
                wind_speed

        }])


        # -------------------------------------------------
        # WEATHER-AWARE PREDICTION
        # -------------------------------------------------

        prediction = (
            weather_model.predict(
                input_data[
                    WEATHER_FEATURES
                ]
            )
        )


        risk_level = (
            weather_encoder
            .inverse_transform(
                prediction
            )[0]
        )


        # -------------------------------------------------
        # MODEL CONFIDENCE
        # -------------------------------------------------

        probabilities = (
            weather_model
            .predict_proba(
                input_data[
                    WEATHER_FEATURES
                ]
            )[0]
        )


        confidence = round(

            float(
                probabilities.max()
            ) * 100,

            2

        )


        # -------------------------------------------------
        # DEBUG
        # -------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "REAL-TIME RISK PREDICTION"
        )

        print(
            "=========================================="
        )

        print(
            "Location:",
            location
        )

        print(
            "District:",
            district
        )

        print(
            "Latitude:",
            latitude
        )

        print(
            "Longitude:",
            longitude
        )

        print(
            "Temperature:",
            temperature
        )

        print(
            "Precipitation:",
            precipitation
        )

        print(
            "Wind Speed:",
            wind_speed
        )

        print(
            "Risk:",
            risk_level
        )

        print(
            "Confidence:",
            confidence
        )

        print(
            "==========================================\n"
        )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status":
                "success",

            "location":
                location,

            "district":
                district,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "risk_level":
                str(
                    risk_level
                ),

            "confidence":
                confidence,

            "weather": {

                "temperature":
                    temperature,

                "precipitation":
                    precipitation,

                "wind_speed":
                    wind_speed

            },

            "nearest_location": {

                "latitude":
                    float(
                        row["latitude"]
                    ),

                "longitude":
                    float(
                        row["longitude"]
                    )

            },

            "accident_data": {

                "total_2020":
                    float(
                        row["total__2020"]
                    ),

                "total_2021":
                    float(
                        row["total__2021"]
                    ),

                "fatal_2020":
                    float(
                        row["2020_fatal"]
                    ),

                "fatal_2021":
                    float(
                        row["2021_fatal"]
                    ),

                "nonfatal_2020":
                    float(
                        row["2020_nonfatal"]
                    ),

                "nonfatal_2021":
                    float(
                        row["2021_nonfatal"]
                    ),

                "accident_change":
                    float(
                        row["accident_change"]
                    ),

                "accident_change_percent":
                    float(
                        row[
                            "accident_change_percent"
                        ]
                    )

            }

        })


    except Exception as e:

        print(
            "Prediction error:",
            str(e)
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# DASHBOARD API
# =========================================================

@app.route(
    "/api/dashboard-data"
)
def dashboard_data():

    try:

        df = accident_data.copy()


        # -------------------------------------------------
        # TOTAL ACCIDENTS
        # -------------------------------------------------

        total_2020 = int(
            df["total__2020"].sum()
        )

        total_2021 = int(
            df["total__2021"].sum()
        )


        # -------------------------------------------------
        # DISTRICT COUNT
        # -------------------------------------------------

        district_count = len(df)


        # -------------------------------------------------
        # OLD MODEL PREDICTIONS
        # -------------------------------------------------

        X = df[FEATURES]

        predictions = model.predict(
            X
        )


        risk_levels = []


        for prediction in predictions:

            if isinstance(
                prediction,
                str
            ):

                risk = prediction

            else:

                prediction = int(
                    prediction
                )

                if prediction == 0:
                    risk = "Low"

                elif prediction == 1:
                    risk = "Medium"

                elif prediction == 2:
                    risk = "High"

                else:
                    risk = "Unknown"


            risk_levels.append(
                risk
            )


        df["risk_level"] = (
            risk_levels
        )


        # -------------------------------------------------
        # RISK COUNTS
        # -------------------------------------------------

        low_count = int(
            (
                df["risk_level"]
                == "Low"
            ).sum()
        )

        medium_count = int(
            (
                df["risk_level"]
                == "Medium"
            ).sum()
        )

        high_count = int(
            (
                df["risk_level"]
                == "High"
            ).sum()
        )


        # -------------------------------------------------
        # DISTRICT DATA
        # -------------------------------------------------

        districts = []


        for index, row in df.iterrows():

            districts.append({

                "district":
                    str(
                        row["district"]
                    ),

                "latitude":
                    float(
                        row["latitude"]
                    ),

                "longitude":
                    float(
                        row["longitude"]
                    ),

                "total_2020":
                    int(
                        row["total__2020"]
                    ),

                "total_2021":
                    int(
                        row["total__2021"]
                    ),

                "fatal_2020":
                    int(
                        row["2020_fatal"]
                    ),

                "fatal_2021":
                    int(
                        row["2021_fatal"]
                    ),

                "accident_change":
                    float(
                        row["accident_change"]
                    ),

                "risk_level":
                    str(
                        row["risk_level"]
                    )

            })


        # -------------------------------------------------
        # RETURN DASHBOARD DATA
        # -------------------------------------------------

        return jsonify({

            "status":
                "success",

            "summary": {

                "total_2020":
                    total_2020,

                "total_2021":
                    total_2021,

                "district_count":
                    district_count,

                "high_risk_count":
                    high_count

            },

            "risk_distribution": {

                "Low":
                    low_count,

                "Medium":
                    medium_count,

                "High":
                    high_count

            },

            "districts":
                districts

        })


    except Exception as e:

        print(
            "Dashboard error:",
            str(e)
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# =========================================================
# RISK MAP
# =========================================================

@app.route("/risk-map")
def risk_map():

    return render_template(
        "tamil_nadu_accident_risk_map.html"
    )


# =========================================================
# LANDING PAGE
# =========================================================

@app.route("/landing")
def landing():

    return render_template(
        "landing.html"
    )


# =========================================================
# DASHBOARD PAGE
# =========================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# =========================================================
# PREDICTION PAGE
# =========================================================

@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html"
    )


# =========================================================
# MAP PAGE
# =========================================================

@app.route("/map")
def map_page():

    return render_template(
        "map.html"
    )


# =========================================================
# DATASET PAGE
# =========================================================

@app.route("/dataset")
def dataset():

    return render_template(
        "dataset.html"
    )


# =========================================================
# CONTACT PAGE
# =========================================================

@app.route("/contact")
def contact():

    return render_template(
        "contact.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    return render_template(
        "index.html"
    )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        " ACCIDENT HOTSPOT PREDICTION SYSTEM"
    )

    print(
        "======================================"
    )

    print(
        "Old Model:",
        OLD_MODEL_PATH
    )

    print(
        "Weather Model:",
        WEATHER_MODEL_PATH
    )

    print(
        "Dataset:",
        DATA_PATH
    )

    print(
        "Old Features:",
        len(FEATURES)
    )

    print(
        "Weather Features:",
        len(WEATHER_FEATURES)
    )

    print(
        "TomTom API:",
        "Configured"
        if TOMTOM_API_KEY
        else "Not configured"
    )

    print(
        "======================================\n"
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )