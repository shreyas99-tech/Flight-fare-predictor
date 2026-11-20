"""
End-to-End Test Suite for Flight Ticket Fare Predictor
Verifies Flask endpoints, input validation, and machine learning pipeline predictions.
"""

import json
from app import app


def run_tests():
    print("Starting Flight Fare Application Tests...")
    client = app.test_client()

    # 1. Test GET /
    print("\n[Test 1] Testing GET / (Main Page)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"SkyFare" in res.data
    print("  -> Passed! Main interface served successfully.")

    # 2. Test GET /info
    print("\n[Test 2] Testing GET /info (Metadata API)...")
    res_info = client.get("/info")
    assert res_info.status_code == 200
    info_data = res_info.get_json()
    assert info_data["status"] == "success"
    print(f"  -> Passed! Model Algorithm: {info_data['model_type']}")

    # 3. Test POST /predict (Valid Economy Flight)
    print("\n[Test 3] Testing POST /predict (Valid Request: Delhi -> Mumbai, IndiGo)...")
    payload = {
        "airline": "IndiGo",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Economy",
        "duration_hours": 2.2,
        "days_left": 15,
    }
    res_pred = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert res_pred.status_code == 200, f"Expected 200, got {res_pred.status_code} - {res_pred.data}"
    data = res_pred.get_json()
    assert data["status"] == "success"
    pred = data["prediction"]
    print(f"  -> Passed! Predicted Fare: {pred['formatted_fare']}")
    print(f"  -> Market Band: {pred['fare_range']}")
    print(f"  -> Hourly Cost: {pred['fare_per_hour']}")
    print(f"  -> Advice: {pred['booking_advice']}")

    # 4. Test POST /predict (Valid Business Class Flight)
    print("\n[Test 4] Testing POST /predict (Valid Request: Vistara Business Class)...")
    biz_payload = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Business",
        "duration_hours": 2.2,
        "days_left": 10,
    }
    res_biz = client.post(
        "/predict",
        data=json.dumps(biz_payload),
        content_type="application/json"
    )
    assert res_biz.status_code == 200
    biz_data = res_biz.get_json()
    assert biz_data["status"] == "success"
    assert biz_data["prediction"]["raw_fare"] > 15000, "Business class should realistically be > ₹15,000"
    print(f"  -> Passed! Business Class Fare: {biz_data['prediction']['formatted_fare']}")

    # 5. Test Critical Validation: Identical Source and Destination
    print("\n[Test 5] Testing Validation: Same Source and Destination (Delhi -> Delhi)...")
    same_city_payload = {
        "airline": "IndiGo",
        "source_city": "Delhi",
        "destination_city": "Delhi",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Economy",
        "duration_hours": 2.2,
        "days_left": 15,
    }
    res_same = client.post(
        "/predict",
        data=json.dumps(same_city_payload),
        content_type="application/json"
    )
    assert res_same.status_code == 400
    err_json = res_same.get_json()
    assert err_json["status"] == "error"
    assert "cannot be the same" in err_json["message"]
    print(f"  -> Passed! Correctly rejected identical cities: {err_json['message']}")

    # 6. Test Validation: Invalid Airline
    print("\n[Test 6] Testing Validation: Unknown Airline...")
    bad_airline_payload = {
        "airline": "SpaceX Airline",
        "source_city": "Delhi",
        "destination_city": "Mumbai",
        "departure_time": "Morning",
        "stops": 0,
        "cabin_class": "Economy",
        "duration_hours": 2.2,
        "days_left": 15,
    }
    res_bad = client.post(
        "/predict",
        data=json.dumps(bad_airline_payload),
        content_type="application/json"
    )
    assert res_bad.status_code == 400
    print(f"  -> Passed! Correctly rejected invalid airline.")

    print("\n" + "=" * 55)
    print("ALL 6 END-TO-END FLIGHT FARE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 55)


if __name__ == "__main__":
    run_tests()
