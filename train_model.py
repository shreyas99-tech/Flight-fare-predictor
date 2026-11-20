"""
=============================================================================
Flight Ticket Fare Prediction - Machine Learning Training Pipeline
=============================================================================
This script:
1. Generates and saves a realistic domestic flight dataset ('flights.csv').
2. Preprocesses categorical and numerical columns using ColumnTransformer and
   OneHotEncoder(sparse_output=False, handle_unknown='ignore').
3. Trains a HistGradientBoostingRegressor model inside a scikit-learn Pipeline.
4. Evaluates performance (MAE, R², RMSE) and serializes the complete trained
   pipeline to 'flight_model.pkl' using joblib.
=============================================================================
"""

import os
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


def generate_flight_dataset(file_path: str, num_records: int = 2400, random_seed: int = 42) -> pd.DataFrame:
    """
    Creates a realistic dataset reflecting Indian domestic aviation pricing dynamics.
    Features:
      - airline: IndiGo, Air India, Vistara, SpiceJet, Akasa Air
      - source_city: Delhi, Mumbai, Bangalore, Hyderabad, Kolkata, Chennai
      - destination_city: (guaranteed distinct from source)
      - departure_time: Early Morning, Morning, Afternoon, Evening, Night
      - stops: 0 (non-stop), 1, 2
      - cabin_class: Economy, Business
      - duration_hours: Flight length in hours
      - days_left: Days remaining before departure (1 to 50)
      - price: Ticket fare in Indian Rupees (INR)
    """
    random.seed(random_seed)

    airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air"]
    cities = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata", "Chennai"]
    departure_slots = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]

    # Base typical non-stop duration and distance factor between major hubs
    route_profiles = {
        ("Delhi", "Mumbai"): (2.2, 3800),
        ("Mumbai", "Delhi"): (2.2, 3800),
        ("Delhi", "Bangalore"): (2.8, 4400),
        ("Bangalore", "Delhi"): (2.8, 4400),
        ("Delhi", "Kolkata"): (2.2, 3900),
        ("Kolkata", "Delhi"): (2.2, 3900),
        ("Delhi", "Hyderabad"): (2.2, 3700),
        ("Hyderabad", "Delhi"): (2.2, 3700),
        ("Delhi", "Chennai"): (2.9, 4500),
        ("Chennai", "Delhi"): (2.9, 4500),
        ("Mumbai", "Bangalore"): (1.7, 3200),
        ("Bangalore", "Mumbai"): (1.7, 3200),
        ("Mumbai", "Hyderabad"): (1.5, 3000),
        ("Hyderabad", "Mumbai"): (1.5, 3000),
        ("Mumbai", "Chennai"): (2.0, 3600),
        ("Chennai", "Mumbai"): (2.0, 3600),
        ("Mumbai", "Kolkata"): (2.6, 4200),
        ("Kolkata", "Mumbai"): (2.6, 4200),
        ("Bangalore", "Hyderabad"): (1.2, 2600),
        ("Hyderabad", "Bangalore"): (1.2, 2600),
        ("Bangalore", "Chennai"): (1.0, 2400),
        ("Chennai", "Bangalore"): (1.0, 2400),
        ("Bangalore", "Kolkata"): (2.5, 4100),
        ("Kolkata", "Bangalore"): (2.5, 4100),
        ("Hyderabad", "Chennai"): (1.3, 2700),
        ("Chennai", "Hyderabad"): (1.3, 2700),
        ("Hyderabad", "Kolkata"): (2.0, 3500),
        ("Kolkata", "Hyderabad"): (2.0, 3500),
        ("Chennai", "Kolkata"): (2.3, 3800),
        ("Kolkata", "Chennai"): (2.3, 3800),
    }

    # Airline brand premiums / adjustments
    airline_premiums = {
        "Vistara": 1800,
        "Air India": 900,
        "IndiGo": 0,
        "Akasa Air": -300,
        "SpiceJet": -450,
    }

    # Time slot premiums (rush hours vs off-peak)
    time_premiums = {
        "Morning": 900,
        "Evening": 800,
        "Afternoon": 200,
        "Early Morning": 100,
        "Night": -350,
    }

    data = []

    for _ in range(num_records):
        # Pick distinct source and destination
        source = random.choice(cities)
        dest_candidates = [c for c in cities if c != source]
        destination = random.choice(dest_candidates)

        airline = random.choice(airlines)
        departure_time = random.choice(departure_slots)

        # Cabin class distribution: Vistara & Air India offer Business Class
        if airline in ["Vistara", "Air India"]:
            cabin_class = random.choices(["Economy", "Business"], weights=[0.78, 0.22])[0]
        else:
            cabin_class = "Economy"  # Low-cost carriers fly economy

        # Stops
        stops = random.choices([0, 1, 2], weights=[0.55, 0.40, 0.05])[0]

        # Base route properties
        base_dur, base_route_fare = route_profiles.get((source, destination), (2.3, 3800))

        # Adjust duration based on stops
        if stops == 0:
            duration_hours = round(max(0.9, base_dur + random.uniform(-0.2, 0.3)), 1)
        elif stops == 1:
            duration_hours = round(base_dur + random.uniform(3.0, 8.5), 1)
        else:
            duration_hours = round(base_dur + random.uniform(7.5, 14.0), 1)

        # Days left until departure (1 to 50 days)
        days_left = random.randint(1, 50)

        # Base economic fare formula
        fare = base_route_fare + airline_premiums[airline] + time_premiums[departure_time]

        # Stop adjustments
        if stops == 0:
            fare += 600  # Convenience bonus for non-stop direct
        elif stops == 2:
            fare -= 400  # Inconvenience discount

        # Dynamic pricing curve based on days_left:
        # Last-minute surge vs early booking discount
        if days_left <= 2:
            fare *= random.uniform(1.85, 2.30)  # Extreme emergency surge
        elif days_left <= 7:
            fare *= random.uniform(1.35, 1.65)  # High surge
        elif days_left <= 15:
            fare *= random.uniform(1.10, 1.25)  # Moderate surge
        elif days_left <= 30:
            fare *= random.uniform(0.98, 1.05)  # Baseline
        else:
            fare *= random.uniform(0.88, 0.95)  # Early bird saver

        # Cabin Class Multiplier (Business class typically 3x - 4x economy)
        if cabin_class == "Business":
            fare = fare * random.uniform(3.2, 4.0) + 12000

        # Add realistic minor market fluctuations
        noise = random.gauss(0, 350)
        final_price = max(2400.0, round((fare + noise) / 50) * 50)

        data.append({
            "airline": airline,
            "source_city": source,
            "destination_city": destination,
            "departure_time": departure_time,
            "stops": stops,
            "cabin_class": cabin_class,
            "duration_hours": duration_hours,
            "days_left": days_left,
            "price": final_price,
        })

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Generated realistic flight dataset with {len(df)} records and saved to '{file_path}'.")
    return df


def load_or_create_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads existing flights.csv or generates a new one if missing.
    """
    if os.path.exists(file_path):
        print(f"Loading existing flight dataset from '{file_path}'...")
        df = pd.read_csv(file_path)
    else:
        print(f"'{file_path}' not found. Generating fresh flight dataset...")
        df = generate_flight_dataset(file_path)
    return df


def build_and_train_model():
    """
    Builds the ColumnTransformer + HistGradientBoostingRegressor Pipeline,
    trains it on flights.csv, evaluates metrics, and saves flight_model.pkl.
    """
    print("\n" + "=" * 65)
    print(" FLIGHT TICKET FARE PREDICTION - TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load dataset
    df = load_or_create_dataset(CSV_FILE_PATH)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDataset preview (first 3 rows):")
    print(df.head(3).to_string())

    # 2. Separate Features and Target
    target_column = "price"
    feature_columns = [col for col in df.columns if col != target_column]

    numeric_features = [
        "days_left",
        "duration_hours",
        "stops",
    ]

    categorical_features = [
        "airline",
        "source_city",
        "destination_city",
        "departure_time",
        "cabin_class",
    ]

    X = df[feature_columns]
    y = df[target_column]

    # 3. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)} flights")
    print(f"Testing set size:  {len(X_test)} flights")

    # 4. Preprocessing Pipeline:
    # ColumnTransformer with OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    # Setting sparse_output=False is required for HistGradientBoostingRegressor to receive dense arrays.
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
                    max_iter=180,
                    max_depth=9,
                    learning_rate=0.08,
                    min_samples_leaf=15,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
        ]
    )

    # 6. Fit the model
    print("\nTraining HistGradientBoostingRegressor pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training finished successfully!")

    # 7. Model Evaluation on Test Data
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print("\n" + "-" * 50)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("-" * 50)
    print(f"  R-squared (R2) Score : {r2:.4f} (Explains {r2 * 100:.1f}% of fare variance)")
    print(f"  Mean Absolute Error  : ₹{mae:,.2f} (Average prediction deviation)")
    print(f"  Root Mean Sq. Error  : ₹{rmse:,.2f}")
    print("-" * 50)

    # 8. Save Pipeline to disk using joblib
    joblib.dump(pipeline, MODEL_FILE_PATH)
    print(f"\nTrained pipeline successfully saved to:\n  -> '{MODEL_FILE_PATH}'")

    # 9. Verification sample prediction
    sample_flight = pd.DataFrame([{
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Economy",
        "duration_hours": 2.2,
        "days_left": 3,
    }])

    loaded_pipeline = joblib.load(MODEL_FILE_PATH)
    test_fare = float(loaded_pipeline.predict(sample_flight)[0])
    print(f"\nVerification Test Prediction (Delhi -> Mumbai, Vistara, 3 days out):")
    print(f"  Predicted Fare: ₹{test_fare:,.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    build_and_train_model()
