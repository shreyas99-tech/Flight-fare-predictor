"""
End-to-End Test Suite for Flight Ticket Fare Predictor (v2 - Scaled Up)
Verifies Flask endpoints, input validation, and machine learning pipeline predictions.
Now includes new fields: day_type, season, booking_type
Verifies new response fields: fare_per_km, distance_km, route_tier
Note: distance_km is auto-computed server-side from city coordinates (Haversine).
      Do NOT include distance_km in test payloads.
"""

import json
from app import app


def run_tests():
    print("Starting Flight Fare Application Tests (v2 - Scaled Up)...")
    client = app.test_client()

    # ── Helper payload factories ──────────────────────────────────────────────
    def base_payload(**overrides):
        """Returns a valid default payload with all required v2 fields."""
        default = {
            "airline":          "IndiGo",
            "source_city":      "Delhi",
            "destination_city": "Mumbai",
            "departure_time":   "Morning",
            "stops":            0,
            "cabin_class":      "Economy",
            "duration_hours":   2.2,
            "days_left":        15,
            "day_type":         "Weekday (Mon-Thu)",
            "season":           "Regular",
            "booking_type":     "Non-Refundable (Standard)",
        }
        default.update(overrides)
        return default

    # ── Test 1: GET / ─────────────────────────────────────────────────────────
    print("\n[Test 1] Testing GET / (Main Page)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"SkyFare" in res.data
    print("  -> Passed! Main interface served successfully.")

    # ── Test 2: GET /info ─────────────────────────────────────────────────────
    print("\n[Test 2] Testing GET /info (Metadata API)...")
    res_info = client.get("/info")
    assert res_info.status_code == 200
    info_data = res_info.get_json()
    assert info_data["status"] == "success"
    print(f"  -> Passed! Model Algorithm: {info_data['model_type']}")

    # ── Test 3: Valid Economy prediction (Tier 1 route) ───────────────────────
    print("\n[Test 3] Valid Economy flight: Delhi -> Mumbai (IndiGo, Weekday, Regular season)...")
    res_pred = client.post(
        "/predict",
        data=json.dumps(base_payload()),
        content_type="application/json"
    )
    assert res_pred.status_code == 200, \
        f"Expected 200, got {res_pred.status_code} - {res_pred.data}"
    data = res_pred.get_json()
    assert data["status"] == "success"
    pred = data["prediction"]
    assert "raw_fare"      in pred, "Missing raw_fare in response"
    assert "fare_per_km"   in pred, "Missing fare_per_km in response"
    assert "distance_km"   in pred, "Missing distance_km in response"
    assert "route_tier"    in pred, "Missing route_tier in response"
    # distance_km is returned as a formatted string like "1,148 km"
    assert pred["distance_km"] != "" and pred["distance_km"] != "--", \
        "Distance should be a non-empty value for DEL->BOM"
    print(f"  -> Passed! Fare: {pred['formatted_fare']} | "
          f"Distance: {pred['distance_km']} | "
          f"Per km: {pred['fare_per_km']} | "
          f"Route tier: {pred['route_tier']}")
    print(f"  -> Market Band: {pred['fare_range']}")
    print(f"  -> Advice: {pred['booking_advice']}")

    # ── Test 4: Business Class (Vistara, Weekend, Peak) ───────────────────────
    print("\n[Test 4] Business class: Delhi -> Mumbai (Vistara, Weekend, Peak)...")
    biz_payload = base_payload(
        airline      = "Vistara",
        cabin_class  = "Business",
        days_left    = 10,
        day_type     = "Weekend (Fri-Sun)",
        season       = "Peak / Festive",
        booking_type = "Flexible / Free Cancellation",
    )
    res_biz = client.post(
        "/predict",
        data=json.dumps(biz_payload),
        content_type="application/json"
    )
    assert res_biz.status_code == 200
    biz_data = res_biz.get_json()
    assert biz_data["status"] == "success"
    biz_pred = biz_data["prediction"]
    assert biz_pred["raw_fare"] > 15000, \
        f"Business class should be > ₹15,000; got ₹{biz_pred['raw_fare']:.0f}"
    print(f"  -> Passed! Business Class Fare: {biz_pred['formatted_fare']}")

    # ── Test 5: Tier 2 route (Jaipur -> Chandigarh) ───────────────────────────
    print("\n[Test 5] Tier 2 route: Jaipur -> Chandigarh (Alliance Air)...")
    tier2_payload = base_payload(
        airline          = "Alliance Air",
        source_city      = "Jaipur",
        destination_city = "Chandigarh",
        stops            = 0,
        duration_hours   = 1.5,
        days_left        = 7,
        day_type         = "Weekday (Mon-Thu)",
        season           = "Regular",
        booking_type     = "Non-Refundable (Standard)",
    )
    res_t2 = client.post(
        "/predict",
        data=json.dumps(tier2_payload),
        content_type="application/json"
    )
    assert res_t2.status_code == 200, \
        f"Expected 200, got {res_t2.status_code} - {res_t2.data}"
    t2_data = res_t2.get_json()
    assert t2_data["status"] == "success"
    t2_pred = t2_data["prediction"]
    assert t2_pred["distance_km"] != "" and t2_pred["distance_km"] != "--"
    print(f"  -> Passed! Fare: {t2_pred['formatted_fare']} | "
          f"Route tier: {t2_pred['route_tier']}")

    # ── Test 6: Tier 3 route (Dehradun -> Varanasi) ───────────────────────────
    print("\n[Test 6] Tier 3 route: Dehradun -> Varanasi (IndiGo, Weekend, Peak, Flexible)...")
    tier3_payload = base_payload(
        airline          = "IndiGo",
        source_city      = "Dehradun",
        destination_city = "Varanasi",
        stops            = 1,
        duration_hours   = 4.5,
        days_left        = 2,
        day_type         = "Weekend (Fri-Sun)",
        season           = "Peak / Festive",
        booking_type     = "Flexible / Free Cancellation",
    )
    res_t3 = client.post(
        "/predict",
        data=json.dumps(tier3_payload),
        content_type="application/json"
    )
    assert res_t3.status_code == 200, \
        f"Expected 200, got {res_t3.status_code} - {res_t3.data}"
    t3_data = res_t3.get_json()
    assert t3_data["status"] == "success"
    t3_pred = t3_data["prediction"]
    print(f"  -> Passed! Fare: {t3_pred['formatted_fare']} | "
          f"Route tier: {t3_pred['route_tier']}")

    # ── Test 7: Same city validation ──────────────────────────────────────────
    print("\n[Test 7] Validation: Same Source and Destination (Delhi -> Delhi)...")
    same_city_payload = base_payload(destination_city="Delhi")
    res_same = client.post(
        "/predict",
        data=json.dumps(same_city_payload),
        content_type="application/json"
    )
    assert res_same.status_code == 400
    err_json = res_same.get_json()
    assert err_json["status"] == "error"
    assert "cannot be identical" in err_json["message"] or "cannot be the same" in err_json["message"]
    print(f"  -> Passed! Correctly rejected: {err_json['message']}")

    # ── Test 8: Invalid airline validation ────────────────────────────────────
    print("\n[Test 8] Validation: Unknown Airline...")
    bad_airline_payload = base_payload(airline="SpaceX Airline")
    res_bad = client.post(
        "/predict",
        data=json.dumps(bad_airline_payload),
        content_type="application/json"
    )
    assert res_bad.status_code == 400
    print("  -> Passed! Correctly rejected invalid airline.")

    # ── Test 9: Missing new fields (backward-compat defaults) ─────────────────
    print("\n[Test 9] Backward compat: missing day_type/season/booking_type (should still work)...")
    old_style_payload = {
        "airline":          "SpiceJet",
        "source_city":      "Mumbai",
        "destination_city": "Chennai",
        "departure_time":   "Afternoon",
        "stops":            0,
        "cabin_class":      "Economy",
        "duration_hours":   2.0,
        "days_left":        32,
        # Intentionally omitting day_type, season, booking_type
    }
    res_old = client.post(
        "/predict",
        data=json.dumps(old_style_payload),
        content_type="application/json"
    )
    assert res_old.status_code == 200, \
        f"Should still succeed with missing optional v2 fields; got {res_old.status_code}: {res_old.data}"
    old_data = res_old.get_json()
    assert old_data["status"] == "success"
    print(f"  -> Passed! Fare with defaults: {old_data['prediction']['formatted_fare']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL 9 END-TO-END FLIGHT FARE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
