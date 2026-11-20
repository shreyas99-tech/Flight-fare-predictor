"""
=============================================================================
Flight Ticket Fare Prediction - Scalable Machine Learning Training Pipeline
=============================================================================
This script:
1. Generates and saves a scalable domestic flight dataset ('flights.csv')
   spanning Tier 1, Tier 2, and Tier 3 Indian cities with comprehensive
   pricing factors (spatial distance, seasonality, day type, booking flexibility).
2. Preprocesses categorical and numerical columns using ColumnTransformer and
   OneHotEncoder(sparse_output=False, handle_unknown='ignore').
3. Trains a HistGradientBoostingRegressor model inside a scikit-learn Pipeline.
4. Evaluates performance (MAE, R², RMSE) and serializes the complete trained
   pipeline to 'flight_model.pkl' using joblib.
=============================================================================
"""

import os
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "flights.csv")
MODEL_FILE_PATH = os.path.join(BASE_DIR, "flight_model.pkl")

# City Registry: Tier 1 Metros, Tier 2 Commercial Hubs, Tier 3 Regional Centers
CITY_REGISTRY = {
    # Tier 1 Metros
    "Delhi": {"tier": "Tier 1", "lat": 28.61, "lon": 77.20},
    "Mumbai": {"tier": "Tier 1", "lat": 19.07, "lon": 72.87},
    "Bangalore": {"tier": "Tier 1", "lat": 12.97, "lon": 77.59},
    "Hyderabad": {"tier": "Tier 1", "lat": 17.38, "lon": 78.48},
    "Kolkata": {"tier": "Tier 1", "lat": 22.57, "lon": 88.36},
    "Chennai": {"tier": "Tier 1", "lat": 13.08, "lon": 80.27},

    # Tier 2 Popular Commercial & State Capitals
    "Pune": {"tier": "Tier 2", "lat": 18.52, "lon": 73.85},
    "Ahmedabad": {"tier": "Tier 2", "lat": 23.02, "lon": 72.57},
    "Goa": {"tier": "Tier 2", "lat": 15.38, "lon": 73.83},
    "Jaipur": {"tier": "Tier 2", "lat": 26.91, "lon": 75.78},
    "Lucknow": {"tier": "Tier 2", "lat": 26.84, "lon": 80.94},
    "Kochi": {"tier": "Tier 2", "lat": 9.93, "lon": 76.26},
    "Chandigarh": {"tier": "Tier 2", "lat": 30.73, "lon": 76.77},
    "Guwahati": {"tier": "Tier 2", "lat": 26.14, "lon": 91.73},
    "Bhubaneswar": {"tier": "Tier 2", "lat": 20.29, "lon": 85.82},
    "Indore": {"tier": "Tier 2", "lat": 22.71, "lon": 75.85},
    "Patna": {"tier": "Tier 2", "lat": 25.59, "lon": 85.13},
    "Srinagar": {"tier": "Tier 2", "lat": 34.08, "lon": 74.79},
    "Coimbatore": {"tier": "Tier 2", "lat": 11.01, "lon": 76.95},
    "Visakhapatnam": {"tier": "Tier 2", "lat": 17.68, "lon": 83.21},

    # Tier 3 Regional Hubs, Pilgrim & Tourist Gateways
    "Varanasi": {"tier": "Tier 3", "lat": 25.31, "lon": 82.97},
    "Dehradun": {"tier": "Tier 3", "lat": 30.31, "lon": 78.03},
    "Amritsar": {"tier": "Tier 3", "lat": 31.63, "lon": 74.87},
    "Ranchi": {"tier": "Tier 3", "lat": 23.34, "lon": 85.30},
    "Raipur": {"tier": "Tier 3", "lat": 21.25, "lon": 81.62},
    "Bagdogra": {"tier": "Tier 3", "lat": 26.68, "lon": 88.32},
    "Udaipur": {"tier": "Tier 3", "lat": 24.58, "lon": 73.71},
    "Madurai": {"tier": "Tier 3", "lat": 9.92, "lon": 78.11},
    "Surat": {"tier": "Tier 3", "lat": 21.17, "lon": 72.83},
    "Port Blair": {"tier": "Tier 3", "lat": 11.66, "lon": 92.74},
}

ALL_CITIES = list(CITY_REGISTRY.keys())
AIRLINES = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air", "Alliance Air"]
DEPARTURE_SLOTS = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
DAY_TYPES = ["Weekday (Mon-Thu)", "Weekend (Fri-Sun)"]
SEASONS = ["Regular", "Peak / Festive", "Off-Peak / Monsoon"]
BOOKING_TYPES = ["Non-Refundable (Standard)", "Flexible / Free Cancellation"]


def calculate_flight_distance(city1: str, city2: str) -> float:
    """
    Computes geographical great-circle distance (km) between two Indian cities.
    """
    c1, c2 = CITY_REGISTRY[city1], CITY_REGISTRY[city2]
    # Haversine formula
    lat1, lon1 = math.radians(c1["lat"]), math.radians(c1["lon"])
    lat2, lon2 = math.radians(c2["lat"]), math.radians(c2["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(6371.0 * c, 1)


def generate_flight_dataset(file_path: str, num_records: int = 12000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic, highly scalable Indian domestic flight fare dataset.
    Features:
      - airline
      - source_city (Tier 1, Tier 2, Tier 3)
      - destination_city (distinct from source)
      - departure_time
      - stops (0, 1, 2)
      - cabin_class (Economy, Business)
      - day_type (Weekday vs Weekend)
      - season (Regular, Peak/Festive, Off-Peak)
      - booking_type (Non-Refundable vs Flexible)
      - duration_hours
      - days_left
      - distance_km
      - price (Target in INR)
    """
    random.seed(random_seed)

    airline_premiums = {
        "Vistara": 1800,
        "Air India": 900,
        "IndiGo": 0,
        "Akasa Air": -350,
        "SpiceJet": -500,
        "Alliance Air": 250,  # Regional carrier on remote routes
    }

    time_premiums = {
        "Morning": 850,
        "Evening": 750,
        "Afternoon": 150,
        "Early Morning": 50,
        "Night": -400,
    }

    data = []

    for _ in range(num_records):
        # Pick distinct origin and destination
        source = random.choice(ALL_CITIES)
        dest_candidates = [c for c in ALL_CITIES if c != source]
        destination = random.choice(dest_candidates)

        src_tier = CITY_REGISTRY[source]["tier"]
        dst_tier = CITY_REGISTRY[destination]["tier"]

        # Airline selection: Alliance Air flies primarily Tier 2 and Tier 3 routes
        if "Tier 3" in [src_tier, dst_tier]:
            airline = random.choices(AIRLINES, weights=[0.38, 0.22, 0.12, 0.14, 0.08, 0.06])[0]
        else:
            airline = random.choices(AIRLINES[:5], weights=[0.42, 0.24, 0.20, 0.08, 0.06])[0]

        departure_time = random.choice(DEPARTURE_SLOTS)
        day_type = random.choices(DAY_TYPES, weights=[0.60, 0.40])[0]
        season = random.choices(SEASONS, weights=[0.65, 0.20, 0.15])[0]
        booking_type = random.choices(BOOKING_TYPES, weights=[0.82, 0.18])[0]

        # Business class only on Vistara and Air India
        if airline in ["Vistara", "Air India"] and src_tier == "Tier 1" and dst_tier in ["Tier 1", "Tier 2"]:
            cabin_class = random.choices(["Economy", "Business"], weights=[0.80, 0.20])[0]
        else:
            cabin_class = "Economy"

        # Distance calculation
        distance_km = calculate_flight_distance(source, destination)

        # Stops probability based on route tiers & distance
        if src_tier == "Tier 1" and dst_tier == "Tier 1" and distance_km < 2200:
            stops = random.choices([0, 1, 2], weights=[0.80, 0.18, 0.02])[0]
        elif "Tier 3" in [src_tier, dst_tier] and distance_km > 1000:
            stops = random.choices([0, 1, 2], weights=[0.20, 0.72, 0.08])[0]
        else:
            stops = random.choices([0, 1, 2], weights=[0.50, 0.45, 0.05])[0]

        # Duration based on distance and stops
        base_flight_time = 0.55 + (distance_km / 680.0)
        if stops == 0:
            duration_hours = round(max(0.8, base_flight_time + random.uniform(-0.15, 0.25)), 1)
        elif stops == 1:
            duration_hours = round(base_flight_time + random.uniform(2.5, 7.5), 1)
        else:
            duration_hours = round(base_flight_time + random.uniform(6.0, 13.0), 1)

        # Days left (1 to 60 days)
        days_left = random.randint(1, 60)

        # ----------------- BASE PRICING FORMULA -----------------
        # Base fare proportional to distance + airline + time slot
        base_rate_per_km = 3.65
        base_fare = 1900 + (distance_km * base_rate_per_km)
        fare = base_fare + airline_premiums[airline] + time_premiums[departure_time]

        # Tier route monopoly / regional cost premium:
        # Tier 3 and regional routes have fewer flights and higher airport handling/fuel surcharges
        if src_tier == "Tier 3" or dst_tier == "Tier 3":
            fare += 750
        if src_tier == "Tier 3" and dst_tier == "Tier 3":
            fare += 1100

        # Non-stop direct flight convenience premium
        if stops == 0:
            fare += 700
        elif stops == 2:
            fare -= 450  # Discount for double layover hassle

        # Day of week factor: Weekend surge
        if day_type == "Weekend (Fri-Sun)":
            fare *= random.uniform(1.08, 1.18)

        # Seasonality factor: Festivals/Vacations vs Monsoon/Off-peak
        if season == "Peak / Festive":
            fare *= random.uniform(1.25, 1.45)
        elif season == "Off-Peak / Monsoon":
            fare *= random.uniform(0.88, 0.94)

        # Booking type: Flexible cancellation fee
        if booking_type == "Flexible / Free Cancellation":
            fare += random.uniform(1200, 1800)

        # Advance booking dynamic pricing curve:
        if days_left <= 2:
            fare *= random.uniform(1.90, 2.35)  # Emergency last-minute surge
        elif days_left <= 7:
            fare *= random.uniform(1.40, 1.70)  # High surge
        elif days_left <= 15:
            fare *= random.uniform(1.12, 1.28)  # Moderate surge
        elif days_left <= 30:
            fare *= random.uniform(0.98, 1.05)  # Standard baseline
        else:
            fare *= random.uniform(0.86, 0.94)  # Early-bird saver

        # Cabin Class multiplier
        if cabin_class == "Business":
            fare = (fare * random.uniform(3.0, 3.8)) + 14000

        # Realistic Gaussian market noise (std dev ₹380)
        noise = random.gauss(0, 380)
        final_price = max(2200.0, round((fare + noise) / 50) * 50)

        data.append({
            "airline": airline,
            "source_city": source,
            "destination_city": destination,
            "departure_time": departure_time,
            "stops": stops,
            "cabin_class": cabin_class,
            "day_type": day_type,
            "season": season,
            "booking_type": booking_type,
            "duration_hours": duration_hours,
            "days_left": days_left,
            "distance_km": distance_km,
            "price": final_price,
        })

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Generated scalable flight dataset with {len(df)} records across 30 cities and saved to '{file_path}'.")
    return df


def load_or_create_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads existing flights.csv or generates a fresh scalable dataset.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Check if dataset has the scalable schema (e.g. distance_km and day_type)
        if "distance_km" in df.columns and len(df) >= 8000:
            print(f"Loading existing scalable flight dataset ({len(df)} rows) from '{file_path}'...")
            return df

    print(f"Generating a new scalable domestic flight dataset with Tier 1/2/3 cities...")
    return generate_flight_dataset(file_path)


def build_and_train_model():
    """
    Builds the ColumnTransformer + HistGradientBoostingRegressor Pipeline,
    trains it on the scalable dataset, evaluates metrics, and saves flight_model.pkl.
    """
    print("\n" + "=" * 70)
    print(" SCALABLE FLIGHT TICKET FARE PREDICTION - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load / Generate Dataset
    df = load_or_create_dataset(CSV_FILE_PATH)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Total Unique Airports/Cities: {df['source_city'].nunique()}")
    print("\nSample dataset preview (first 3 rows):")
    print(df.head(3).to_string())

    # 2. Features & Target
    target_column = "price"
    feature_columns = [col for col in df.columns if col != target_column]

    numeric_features = [
        "days_left",
        "duration_hours",
        "stops",
        "distance_km",
    ]

    categorical_features = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "cabin_class",
        "day_type",
        "season",
        "booking_type",
    ]

    X = df[feature_columns]
    y = df[target_column]

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)} flights")
    print(f"Testing set size:  {len(X_test)} flights")

    # 4. Preprocessing: ColumnTransformer with OneHotEncoder(sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    # 5. Full Pipeline with HistGradientBoostingRegressor
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=220,
                    max_depth=12,
                    learning_rate=0.07,
                    min_samples_leaf=20,
                    l2_regularization=0.15,
                    random_state=42,
                ),
            ),
        ]
    )

    # 6. Fit the model
    print("\nTraining HistGradientBoostingRegressor pipeline across multi-tier network...")
    pipeline.fit(X_train, y_train)
    print("Training finished successfully!")

    # 7. Model Evaluation on Test Data
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n" + "-" * 55)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("-" * 55)
    print(f"  R-squared (R2) Score : {r2:.4f} (Explains {r2 * 100:.1f}% of fare variance)")
    print(f"  Mean Absolute Error  : ₹{mae:,.2f} (Average prediction deviation)")
    print(f"  Root Mean Sq. Error  : ₹{rmse:,.2f}")
    print("-" * 55)

    # 8. Save Pipeline using joblib
    joblib.dump(pipeline, MODEL_FILE_PATH)
    print(f"\nTrained pipeline successfully saved to:\n  -> '{MODEL_FILE_PATH}'")

    # 9. Verification sample predictions across different tiers:
    # Test A: Tier 1 to Tier 1 Metro flight (DEL -> BOM)
    sample_metro = pd.DataFrame([{
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Economy",
        "day_type": "Weekday (Mon-Thu)",
        "season": "Regular",
        "booking_type": "Non-Refundable (Standard)",
        "duration_hours": 2.2,
        "days_left": 15,
        "distance_km": calculate_flight_distance("Delhi", "Mumbai"),
    }])

    # Test B: Tier 2 to Tier 3 Regional flight (Goa -> Dehradun)
    sample_regional = pd.DataFrame([{
        "airline": "IndiGo",
        "source_city": "Goa",
        "destination_city": "Dehradun",
        "departure_time": "Morning",
        "stops": 1,
        "cabin_class": "Economy",
        "day_type": "Weekend (Fri-Sun)",
        "season": "Peak / Festive",
        "booking_type": "Flexible / Free Cancellation",
        "duration_hours": 6.5,
        "days_left": 3,
        "distance_km": calculate_flight_distance("Goa", "Dehradun"),
    }])

    loaded_pipeline = joblib.load(MODEL_FILE_PATH)
    fare_metro = float(loaded_pipeline.predict(sample_metro)[0])
    fare_regional = float(loaded_pipeline.predict(sample_regional)[0])

    print("\nVerification Test Predictions:")
    print(f"  1. Metro Flight (DEL ➔ BOM, 15 days out, Regular): ₹{fare_metro:,.2f}")
    print(f"  2. Regional Tier Flight (GOI ➔ DED, 3 days out, Weekend & Peak): ₹{fare_regional:,.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    build_and_train_model()
