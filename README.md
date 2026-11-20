# ✈️ SkyFare AI - Domestic Flight Ticket Fare Predictor

Welcome! This is an end-to-end Machine Learning web application designed to predict domestic airline ticket prices (in Indian Rupees, ₹) based on real-world travel factors like advance booking days, flight duration, number of stops, departure time, cabin class, and airline brand.

Built strictly using **Python, pandas, scikit-learn, joblib, Flask, HTML, CSS, and JavaScript**—without any external ML frameworks.

---

## 🌟 The Real-World Analogy: How This Works

If you are brand new to coding and Machine Learning, here is how the entire system works using everyday analogies:

1. **The Airline Revenue Manager (The Problem)**:
   Airlines don't set ticket prices randomly. They use a system called **Dynamic Yield Management**.
   * If you book a flight 2 days before departure, you are likely in a hurry, so the algorithm charges more.
   * If you book 30 days in advance, airlines offer discounted early-bird seats to fill planes early.
   * Business class provides lie-flat seats and gourmet meals, costing roughly 3x to 4x more than Economy.
   * Flying non-stop saves hours compared to layovers, commanding a premium.

2. **The Flight History Logbook (`flights.csv`)**:
   Imagine an airport flight ledger recording **2,400 past flight bookings** across major Indian hubs (Delhi, Mumbai, Bangalore, Hyderabad, Kolkata, Chennai). It notes the airline, route, departure slot, class, days left, and the actual fare paid.

3. **The Machine Learning Brain (`HistGradientBoostingRegressor`)**:
   Instead of a human memorizing 2,400 rows, our computer trains a **Histogram Gradient Boosting** model. Think of this as a team of fast mathematical analysts who study past flights, group similar flight patterns into bins, and continuously refine their estimates by correcting past mistakes.

4. **Dense One-Hot Encoding (`OneHotEncoder(sparse_output=False)`)**:
   Computers only understand math, not words like `"IndiGo"` or `"Delhi"`.
   `OneHotEncoder` translates these words into binary 1s and 0s (like switches). Setting `sparse_output=False` gives the algorithm a clean, complete number grid (dense array) that `HistGradientBoostingRegressor` requires.

5. **Saving the Brain (`flight_model.pkl`)**:
   Once training completes, we "freeze" the trained pipeline into a file called `flight_model.pkl` using `joblib`. This means the web app can answer booking queries instantly without having to retrain each time.

6. **The Flight Dispatcher (`app.py` / Flask)**:
   Flask is like the flight desk agent. When you choose a route on the website, Flask receives the details, ensures you didn't accidentally select the same departure and arrival city, asks the trained model for an appraisal, and returns the fare in milliseconds.

7. **The Boarding Pass UI (`index.html`, `style.css`, `script.js`)**:
   The interactive website designed like a modern digital boarding pass with rolling price counters, instant demo routes, and smart booking advice.

---

## 📁 Project Directory Structure

```text
flight-fare-prediction/
│
├── flights.csv              # Historical dataset (2,400 flight records)
├── train_model.py           # Training script for HistGradientBoosting pipeline
├── flight_model.pkl         # Saved, trained machine learning model
├── app.py                   # Flask backend server & /predict JSON API
├── test_app.py              # Automated test suite
├── requirements.txt         # Minimal dependency list
│
├── templates/
│   └── index.html           # Travel dashboard & boarding pass UI
│
└── static/
    ├── style.css            # Modern travel styling & responsive layout
    └── script.js            # Route swapping, validation, fetch API & counter
```

---

## 🧠 Core Features & Mechanics

| Feature | Type | Options / Range | Impact on Price |
| :--- | :--- | :--- | :--- |
| `airline` | Categorical | IndiGo, Air India, Vistara, SpiceJet, Akasa Air | Full-service airlines (Vistara, Air India) have higher baselines than budget carriers. |
| `source_city` | Categorical | Delhi, Mumbai, Bangalore, Hyderabad, Kolkata, Chennai | Sets base distance and route traffic. |
| `destination_city` | Categorical | (Distinct from origin) | Sets flight distance and duration. |
| `departure_time` | Categorical | Early Morning, Morning, Afternoon, Evening, Night | Morning & evening business peak slots cost more than off-peak. |
| `cabin_class` | Categorical | Economy, Business | Business class carries a 3x to 4x multiplier. |
| `stops` | Numerical | 0 (Non-Stop), 1 Stop, 2 Stops | Non-stop flights carry convenience premiums. |
| `duration_hours` | Numerical | 0.8 to 24.0 hours | Direct flights correlate with lower duration; layovers increase duration. |
| `days_left` | Numerical | 1 to 50 days | Exponential surge within 1-3 days; best discounts 25+ days out. |
| **`price` (Target)** | Numerical | ₹2,400 to ₹65,000+ | Predicted ticket fare in Indian Rupees (INR). |

---

## 🚀 Step-by-Step Setup & Running Guide

### Step 1: Open Terminal or Command Prompt
* **Windows**: Press `Win + R`, type `powershell` or `cmd`, and press Enter.
* **Mac / Linux**: Open the `Terminal` app.

### Step 2: Navigate to the project directory
```bash
cd "C:\Users\R Shreyas Karthikeya\.gemini\antigravity\scratch\flight-fare-prediction"
```

### Step 3: Install the required dependencies
```bash
pip install -r requirements.txt
```
*(Installs only `Flask`, `pandas`, `scikit-learn`, and `joblib`)*.

### Step 4: Train the Machine Learning Model
```bash
python train_model.py
```
**Expected Terminal Output:**
```text
=================================================================
 FLIGHT TICKET FARE PREDICTION - TRAINING PIPELINE
=================================================================
Dataset shape: 2400 rows, 9 columns
Training set size: 1920 flights
Testing set size:  480 flights

Training HistGradientBoostingRegressor pipeline...
Training finished successfully!

--------------------------------------------------
 MODEL PERFORMANCE EVALUATION (TEST SET)
--------------------------------------------------
  R-squared (R2) Score : 0.9870 (Explains 98.7% of fare variance)
  Mean Absolute Error  : ₹534.01 (Average prediction deviation)
  Root Mean Sq. Error  : ₹963.48
--------------------------------------------------

Trained pipeline successfully saved to:
  -> '.../flight_model.pkl'

Verification Test Prediction (Delhi -> Mumbai, Vistara, 3 days out):
  Predicted Fare: ₹10,638.70
=================================================================
```

### Step 5: Start the Flask Web App
```bash
python app.py
```
You will see:
```text
Flight Fare model pipeline loaded successfully into Flask.
Starting Flight Fare Predictor at http://127.0.0.1:5000 ...
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Step 6: Open the App in Your Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

* Try the **Quick Flight Scenarios** (e.g. *DEL ➔ BOM Vistara Business*, *BLR ➔ DEL Last-Minute Rush*, etc.) to see the boarding pass update instantly.
* Use the **⇄ Swap button** to invert origin and destination.
* If you set origin and destination to the same city, a friendly notice prevents invalid requests.

### Step 7: How to Stop the Server Cleanly
When you are done testing:
1. Return to the terminal where `app.py` is running.
2. Press **`Ctrl + C`** on your keyboard to terminate the server.

---

## 🧪 Running the Automated Test Suite

To verify all validation checks, endpoints, and predictions in one command:
```bash
python test_app.py
```
This tests:
1. `GET /` serves the HTML interface.
2. `GET /info` returns route metadata.
3. `POST /predict` computes valid Economy fares.
4. `POST /predict` computes realistic Business Class premiums.
5. Rejection of identical origin and destination cities (e.g. Delhi to Delhi).
6. Rejection of invalid airlines and bad inputs.

---

## 🛠️ Technology Stack Compliance

* **Python**: Core programming language.
* **pandas**: Dataset generation, table handling, and DataFrame input structures.
* **scikit-learn**:
  * `ColumnTransformer`: Combines text and numeric processors.
  * `OneHotEncoder`: Encodes airlines and cities (`sparse_output=False` for dense matrix compatibility).
  * `HistGradientBoostingRegressor`: High-performance boosting regressor.
  * `Pipeline`: Encapsulates preprocessing and regression into a single portable object.
* **joblib**: Model serialization to `flight_model.pkl`.
* **Flask**: Web application, template rendering, and JSON REST API.
* **HTML5, CSS3, JavaScript**: Responsive boarding pass interface with no external JS dependencies.
