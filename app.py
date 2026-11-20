"""
=============================================================================
Flight Ticket Fare Prediction - Flask Web Application
=============================================================================
This backend server loads the trained machine learning pipeline from
'flight_model.pkl' and exposes:
1. GET  /        -> Serves the flight booking & fare estimation dashboard.
2. POST /predict -> Validates search parameters (ensuring source != destination),
                    invokes the HistGradientBoostingRegressor pipeline, and
                    returns the predicted fare in INR with flight insights.
3. GET  /info    -> Returns supported routes, airlines, and metadata.
=============================================================================
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(BASE_DIR, "flight_model.pkl")

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


# Accepted categories
ALLOWED_AIRLINES = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air"]
ALLOWED_CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata", "Chennai"]
ALLOWED_DEPARTURE_SLOTS = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
ALLOWED_CABIN_CLASSES = ["Economy", "Business"]
ALLOWED_STOPS = [0, 1, 2]


@app.route("/")
def index():
    """
    Renders the flight fare predictor user interface.
    """
    return render_template(
        "index.html",
        airlines=ALLOWED_AIRLINES,
        cities=ALLOWED_CITIES,
        departure_slots=ALLOWED_DEPARTURE_SLOTS,
        cabin_classes=ALLOWED_CABIN_CLASSES,
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
        "allowed_airlines": ALLOWED_AIRLINES,
        "allowed_cities": ALLOWED_CITIES,
        "allowed_departure_slots": ALLOWED_DEPARTURE_SLOTS,
        "allowed_cabin_classes": ALLOWED_CABIN_CLASSES,
        "allowed_stops": ALLOWED_STOPS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint that processes flight parameters and returns fare prediction.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "status": "error",
                "message": "Invalid request: No JSON payload received."
            }), 400

        # 1. Extract and Validate Source & Destination
        source_city = str(data.get("source_city", "")).strip()
        destination_city = str(data.get("destination_city", "")).strip()

        if not source_city or source_city not in ALLOWED_CITIES:
            return jsonify({
                "status": "error",
                "message": f"Invalid source city '{source_city}'. Must be one of: {ALLOWED_CITIES}."
            }), 400

        if not destination_city or destination_city not in ALLOWED_CITIES:
            return jsonify({
                "status": "error",
                "message": f"Invalid destination city '{destination_city}'. Must be one of: {ALLOWED_CITIES}."
            }), 400

        # CRITICAL CHECK: Source and destination must be distinct
        if source_city.lower() == destination_city.lower():
            return jsonify({
                "status": "error",
                "message": f"Departure and arrival cities cannot be the same ({source_city}). Please select distinct cities."
            }), 400

        # 2. Validate Airline
        airline = str(data.get("airline", "")).strip()
        if not airline or airline not in ALLOWED_AIRLINES:
            return jsonify({
                "status": "error",
                "message": f"Invalid airline '{airline}'. Must be one of: {ALLOWED_AIRLINES}."
            }), 400

        # 3. Validate Departure Time
        departure_time = str(data.get("departure_time", "")).strip()
        if not departure_time or departure_time not in ALLOWED_DEPARTURE_SLOTS:
            return jsonify({
                "status": "error",
                "message": f"Invalid departure time '{departure_time}'. Must be one of: {ALLOWED_DEPARTURE_SLOTS}."
            }), 400

        # 4. Validate Cabin Class
        cabin_class = str(data.get("cabin_class", "")).strip()
        if not cabin_class or cabin_class not in ALLOWED_CABIN_CLASSES:
            return jsonify({
                "status": "error",
                "message": f"Invalid cabin class '{cabin_class}'. Must be one of: {ALLOWED_CABIN_CLASSES}."
            }), 400

        # 5. Validate Numerical Inputs
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
                "message": "Invalid value for 'stops'. Must be an integer (0, 1, or 2)."
            }), 400

        try:
            duration_hours = float(data.get("duration_hours", 0))
            if duration_hours < 0.5 or duration_hours > 30.0:
                return jsonify({
                    "status": "error",
                    "message": "Duration must be between 0.5 and 30.0 hours."
                }), 400
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid numeric format for 'duration_hours'."
            }), 400

        try:
            days_left = int(data.get("days_left", 0))
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

        # 6. Build single-row DataFrame with exact columns expected by pipeline
        input_df = pd.DataFrame([{
            "airline": airline,
            "source_city": source_city,
            "destination_city": destination_city,
            "departure_time": departure_time,
            "stops": stops,
            "cabin_class": cabin_class,
            "duration_hours": duration_hours,
            "days_left": days_left,
        }])

        # 7. Predict using pipeline
        pipeline = get_model()
        predicted_fare = float(pipeline.predict(input_df)[0])
        predicted_fare = max(1800.0, round(predicted_fare, 2))

        # 8. Compute Helpful Travel Statistics & Advice
        fare_per_hour = round(predicted_fare / duration_hours, 2) if duration_hours > 0 else 0

        # Booking advice based on days_left
        if days_left <= 3:
            booking_advice = "🚨 High last-minute surge pricing active. Booking 15+ days ahead typically cuts fares in half."
            demand_level = "High Surge"
        elif days_left <= 10:
            booking_advice = "⚡ Moderate booking demand. Fares are climbing as departure nears."
            demand_level = "Moderate"
        elif days_left <= 25:
            booking_advice = "✅ Standard advance purchase window. Fair market rates."
            demand_level = "Normal"
        else:
            booking_advice = "🌟 Optimal early-bird booking window with the lowest base fares."
            demand_level = "Low / Saver"

        # Reasonable market price range (+/- 4%)
        lower_bound = max(1600.0, round(predicted_fare * 0.96 / 50) * 50)
        upper_bound = round(predicted_fare * 1.04 / 50) * 50

        stops_label = "Non-Stop" if stops == 0 else f"{stops} Stop{'s' if stops > 1 else ''}"

        return jsonify({
            "status": "success",
            "prediction": {
                "raw_fare": predicted_fare,
                "formatted_fare": f"₹{predicted_fare:,.0f}",
                "fare_per_hour": f"₹{fare_per_hour:,.0f}/hr",
                "fare_range": f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}",
                "stops_label": stops_label,
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
                "stops": stops,
                "duration_hours": duration_hours,
                "days_left": days_left,
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

    print("Starting Flight Fare Predictor at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
