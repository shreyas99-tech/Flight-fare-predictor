"""
=============================================================================
Flight Ticket Fare Prediction - Scalable Flask Web Application
=============================================================================
This backend server loads the trained machine learning pipeline from
'flight_model.pkl' and exposes:
1. GET  /        -> Serves the multi-tier flight booking dashboard.
2. POST /predict -> Validates search parameters across 30 Tier 1, 2, and 3 cities,
                    computes spatial flight distance, runs the trained
                    HistGradientBoostingRegressor pipeline, and returns the
                    predicted fare in INR with comprehensive route insights.
3. GET  /info    -> Returns metadata on supported tiers, routes, and features.
=============================================================================
"""

import os
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(BASE_DIR, "flight_model.pkl")

app = Flask(
    __name__,
    root_path=BASE_DIR,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Cached model pipeline
model_pipeline = None


def get_model():
    """
    Lazy loads or retrieves the cached trained pipeline.
    """
    global model_pipeline
    if model_pipeline is None:
        if not os.path.exists(MODEL_FILE_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_FILE_PATH}' not found. "
                "Please run 'python train_model.py' first to train and serialize the model."
            )
        model_pipeline = joblib.load(MODEL_FILE_PATH)
    return model_pipeline


# Scalable City Registry with Geographical Coordinates
CITY_REGISTRY = {
    # Tier 1 Metros
    "Delhi": {"tier": "Tier 1 Metros", "lat": 28.61, "lon": 77.20},
    "Mumbai": {"tier": "Tier 1 Metros", "lat": 19.07, "lon": 72.87},
    "Bangalore": {"tier": "Tier 1 Metros", "lat": 12.97, "lon": 77.59},
    "Hyderabad": {"tier": "Tier 1 Metros", "lat": 17.38, "lon": 78.48},
    "Kolkata": {"tier": "Tier 1 Metros", "lat": 22.57, "lon": 88.36},
    "Chennai": {"tier": "Tier 1 Metros", "lat": 13.08, "lon": 80.27},

    # Tier 2 Commercial Capitals & High-Volume Cities
    "Pune": {"tier": "Tier 2 Commercial Hubs", "lat": 18.52, "lon": 73.85},
    "Ahmedabad": {"tier": "Tier 2 Commercial Hubs", "lat": 23.02, "lon": 72.57},
    "Goa": {"tier": "Tier 2 Commercial Hubs", "lat": 15.38, "lon": 73.83},
    "Jaipur": {"tier": "Tier 2 Commercial Hubs", "lat": 26.91, "lon": 75.78},
    "Lucknow": {"tier": "Tier 2 Commercial Hubs", "lat": 26.84, "lon": 80.94},
    "Kochi": {"tier": "Tier 2 Commercial Hubs", "lat": 9.93, "lon": 76.26},
    "Chandigarh": {"tier": "Tier 2 Commercial Hubs", "lat": 30.73, "lon": 76.77},
    "Guwahati": {"tier": "Tier 2 Commercial Hubs", "lat": 26.14, "lon": 91.73},
    "Bhubaneswar": {"tier": "Tier 2 Commercial Hubs", "lat": 20.29, "lon": 85.82},
    "Indore": {"tier": "Tier 2 Commercial Hubs", "lat": 22.71, "lon": 75.85},
    "Patna": {"tier": "Tier 2 Commercial Hubs", "lat": 25.59, "lon": 85.13},
    "Srinagar": {"tier": "Tier 2 Commercial Hubs", "lat": 34.08, "lon": 74.79},
    "Coimbatore": {"tier": "Tier 2 Commercial Hubs", "lat": 11.01, "lon": 76.95},
    "Visakhapatnam": {"tier": "Tier 2 Commercial Hubs", "lat": 17.68, "lon": 83.21},

    # Tier 3 Regional Hubs & Gateways
    "Varanasi": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 25.31, "lon": 82.97},
    "Dehradun": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 30.31, "lon": 78.03},
    "Amritsar": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 31.63, "lon": 74.87},
    "Ranchi": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 23.34, "lon": 85.30},
    "Raipur": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 21.25, "lon": 81.62},
    "Bagdogra": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 26.68, "lon": 88.32},
    "Udaipur": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 24.58, "lon": 73.71},
    "Madurai": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 9.92, "lon": 78.11},
    "Surat": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 21.17, "lon": 72.83},
    "Port Blair": {"tier": "Tier 3 Regional & Tourist Hubs", "lat": 11.66, "lon": 92.74},
}

ALL_CITIES = list(CITY_REGISTRY.keys())
TIER_1_CITIES = [c for c, d in CITY_REGISTRY.items() if d["tier"] == "Tier 1 Metros"]
TIER_2_CITIES = [c for c, d in CITY_REGISTRY.items() if d["tier"] == "Tier 2 Commercial Hubs"]
TIER_3_CITIES = [c for c, d in CITY_REGISTRY.items() if d["tier"] == "Tier 3 Regional & Tourist Hubs"]

ALLOWED_AIRLINES = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air", "Alliance Air"]
ALLOWED_DEPARTURE_SLOTS = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
ALLOWED_CABIN_CLASSES = ["Economy", "Business"]
ALLOWED_DAY_TYPES = ["Weekday (Mon-Thu)", "Weekend (Fri-Sun)"]
ALLOWED_SEASONS = ["Regular", "Peak / Festive", "Off-Peak / Monsoon"]
ALLOWED_BOOKING_TYPES = ["Non-Refundable (Standard)", "Flexible / Free Cancellation"]
ALLOWED_STOPS = [0, 1, 2]


def calculate_flight_distance(city1: str, city2: str) -> float:
    """
    Computes geographical great-circle distance (km) between two Indian cities.
    """
    if city1 not in CITY_REGISTRY or city2 not in CITY_REGISTRY:
        return 1200.0
    c1, c2 = CITY_REGISTRY[city1], CITY_REGISTRY[city2]
    lat1, lon1 = math.radians(c1["lat"]), math.radians(c1["lon"])
    lat2, lon2 = math.radians(c2["lat"]), math.radians(c2["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(6371.0 * c, 1)


@app.route("/")
def index():
    """
    Renders the multi-tier flight fare dashboard.
    """
    return render_template(
        "index.html",
        tier1_cities=TIER_1_CITIES,
        tier2_cities=TIER_2_CITIES,
        tier3_cities=TIER_3_CITIES,
        all_cities=ALL_CITIES,
        airlines=ALLOWED_AIRLINES,
        departure_slots=ALLOWED_DEPARTURE_SLOTS,
        cabin_classes=ALLOWED_CABIN_CLASSES,
        day_types=ALLOWED_DAY_TYPES,
        seasons=ALLOWED_SEASONS,
        booking_types=ALLOWED_BOOKING_TYPES,
        stops_options=ALLOWED_STOPS,
    )


@app.route("/info", methods=["GET"])
def info():
    """
    Returns available flight search options and model metadata.
    """
    return jsonify({
        "status": "success",
        "model_type": "HistGradientBoostingRegressor with ColumnTransformer",
        "cities_count": len(ALL_CITIES),
        "tier1_metros": TIER_1_CITIES,
        "tier2_cities": TIER_2_CITIES,
        "tier3_cities": TIER_3_CITIES,
        "allowed_airlines": ALLOWED_AIRLINES,
        "allowed_departure_slots": ALLOWED_DEPARTURE_SLOTS,
        "allowed_cabin_classes": ALLOWED_CABIN_CLASSES,
        "allowed_day_types": ALLOWED_DAY_TYPES,
        "allowed_seasons": ALLOWED_SEASONS,
        "allowed_booking_types": ALLOWED_BOOKING_TYPES,
        "allowed_stops": ALLOWED_STOPS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Processes flight parameters across Tier 1/2/3 networks and returns fare predictions.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid request: No JSON payload received."
            }), 400

        # 1. Source & Destination Validation
        source_city = str(data.get("source_city", "")).strip()
        destination_city = str(data.get("destination_city", "")).strip()

        if not source_city or source_city not in ALL_CITIES:
            return jsonify({
                "status": "error",
                "message": f"Invalid origin city '{source_city}'. Please select from supported Indian cities."
            }), 400

        if not destination_city or destination_city not in ALL_CITIES:
            return jsonify({
                "status": "error",
                "message": f"Invalid destination city '{destination_city}'. Please select from supported Indian cities."
            }), 400

        # Critical Validation: Distinct cities
        if source_city.lower() == destination_city.lower():
            return jsonify({
                "status": "error",
                "message": f"Departure and arrival cities cannot be identical ({source_city}). Please select distinct cities."
            }), 400

        # 2. Categorical Validations
        airline = str(data.get("airline", "")).strip()
        if not airline or airline not in ALLOWED_AIRLINES:
            return jsonify({
                "status": "error",
                "message": f"Invalid airline '{airline}'. Must be one of: {ALLOWED_AIRLINES}."
            }), 400

        departure_time = str(data.get("departure_time", "")).strip()
        if not departure_time or departure_time not in ALLOWED_DEPARTURE_SLOTS:
            return jsonify({
                "status": "error",
                "message": f"Invalid departure slot '{departure_time}'. Must be one of: {ALLOWED_DEPARTURE_SLOTS}."
            }), 400

        cabin_class = str(data.get("cabin_class", "")).strip()
        if not cabin_class or cabin_class not in ALLOWED_CABIN_CLASSES:
            return jsonify({
                "status": "error",
                "message": f"Invalid cabin class '{cabin_class}'. Must be one of: {ALLOWED_CABIN_CLASSES}."
            }), 400

        day_type = str(data.get("day_type", ALLOWED_DAY_TYPES[0])).strip()
        if day_type not in ALLOWED_DAY_TYPES:
            day_type = ALLOWED_DAY_TYPES[0]

        season = str(data.get("season", ALLOWED_SEASONS[0])).strip()
        if season not in ALLOWED_SEASONS:
            season = ALLOWED_SEASONS[0]

        booking_type = str(data.get("booking_type", ALLOWED_BOOKING_TYPES[0])).strip()
        if booking_type not in ALLOWED_BOOKING_TYPES:
            booking_type = ALLOWED_BOOKING_TYPES[0]

        # 3. Numerical Inputs
        try:
            stops = int(data.get("stops", 0))
            if stops not in ALLOWED_STOPS:
                return jsonify({
                    "status": "error",
                    "message": f"Stops must be 0 (non-stop), 1, or 2. Received: {stops}."
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid integer value for 'stops'."
            }), 400

        try:
            duration_hours = float(data.get("duration_hours", 2.0))
            if duration_hours < 0.5 or duration_hours > 35.0:
                return jsonify({
                    "status": "error",
                    "message": "Duration must be between 0.5 and 35.0 hours."
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid numeric format for 'duration_hours'."
            }), 400

        try:
            days_left = int(data.get("days_left", 15))
            if days_left < 1 or days_left > 120:
                return jsonify({
                    "status": "error",
                    "message": "Days left until departure must be between 1 and 120."
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid integer format for 'days_left'."
            }), 400

        # Calculate exact geographical distance
        distance_km = calculate_flight_distance(source_city, destination_city)

        # 4. Single-row DataFrame matching the 12 features
        input_df = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "departure_time": departure_time,
            "stops": stops,
            "cabin_class": cabin_class,
            "day_type": day_type,
            "season": season,
            "booking_type": booking_type,
            "duration_hours": duration_hours,
            "days_left": days_left,
            "distance_km": distance_km,
        }])

        # 5. Predict using trained pipeline
        pipeline = get_model()
        predicted_fare = float(pipeline.predict(input_df)[0])
        predicted_fare = max(2000.0, round(predicted_fare, 2))

        # 6. Route Tier Classification
        src_tier = CITY_REGISTRY[source_city]["tier"].split()[0] + " " + CITY_REGISTRY[source_city]["tier"].split()[1]
        dst_tier = CITY_REGISTRY[destination_city]["tier"].split()[0] + " " + CITY_REGISTRY[destination_city]["tier"].split()[1]
        route_tier_label = f"{src_tier} ➔ {dst_tier}"

        # 7. Travel Statistics & Advice
        fare_per_hour = round(predicted_fare / duration_hours, 2) if duration_hours > 0 else 0
        fare_per_km = round(predicted_fare / distance_km, 2) if distance_km > 0 else 0

        # Dynamic booking demand & surge indicator
        if days_left <= 3:
            booking_advice = "🚨 High last-minute emergency surge. Booking 20+ days ahead typically cuts fares by 40-55%."
            demand_level = "High Surge"
        elif season == "Peak / Festive":
            booking_advice = "🎉 Festive/Holiday season demand peak. Seats sell fast with premium dynamic pricing."
            demand_level = "Peak Season"
        elif day_type == "Weekend (Fri-Sun)":
            booking_advice = "🏖️ Weekend leisure travel surge active. Tuesday/Wednesday flights offer better bargains."
            demand_level = "Weekend Surge"
        elif days_left > 25:
            booking_advice = "🌟 Optimal early-bird booking window with the lowest base fares."
            demand_level = "Early Saver"
        else:
            booking_advice = "✅ Standard advance purchase window with fair market rates."
            demand_level = "Moderate"

        lower_bound = max(1800.0, round(predicted_fare * 0.95 / 50) * 50)
        upper_bound = round(predicted_fare * 1.05 / 50) * 50
        stops_label = "Non-Stop" if stops == 0 else f"{stops} Stop{'s' if stops > 1 else ''}"

        return jsonify({
            "status": "success",
            "prediction": {
                "raw_fare": predicted_fare,
                "formatted_fare": f"₹{predicted_fare:,.0f}",
                "fare_per_hour": f"₹{fare_per_hour:,.0f}/hr",
                "fare_per_km": f"₹{fare_per_km:,.1f}/km",
                "fare_range": f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}",
                "distance_km": f"{distance_km:,.0f} km",
                "stops_label": stops_label,
                "route_tier": route_tier_label,
                "booking_advice": booking_advice,
                "demand_level": demand_level,
            },
            "inputs": {
                "airline": airline,
                "route": f"{source_city} → {destination_city}",
                "source_city": source_city,
                "destination_city": destination_city,
                "departure_time": departure_time,
                "cabin_class": cabin_class,
                "day_type": day_type,
                "season": season,
                "booking_type": booking_type,
                "stops": stops,
                "duration_hours": duration_hours,
                "days_left": days_left,
                "distance_km": distance_km,
            },
        })

    except FileNotFoundError as fnf_err:
        return jsonify({
            "status": "error",
            "message": str(fnf_err)
        }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server processing error: {str(e)}"
        }), 500


if __name__ == "__main__":
    try:
        get_model()
        print("Flight Fare model pipeline loaded successfully into Flask.")
    except Exception as err:
        print(f"Warning: {err}")

    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Flight Fare Predictor at http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
