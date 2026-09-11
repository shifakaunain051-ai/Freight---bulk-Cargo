"""Example request for the /predict endpoint (Model v3).

Route:   Hay Point  ->  East Coast India
Cargo:   Coal
Vessel:  Panamax

Model v3 does NOT use cargo_tonnes - it uses 13 input features.

Usage:
    1. Start the API:
        cd backend
        uvicorn main:app --port 8000
    2. In another terminal run:
        python test_example.py

Uses only the Python standard library (no extra dependencies needed).
"""

import json
import urllib.request

URL = "http://localhost:8000/predict"

# NOTE: no cargo_tonnes - the final model excludes it.
PAYLOAD = {
    "origin": "Hay Point",
    "destination": "East Coast India",
    "commodity": "Coal",
    "vessel_type": "Panamax",
    "current_freight_usd_per_tonne": 16.5,
    "bdi": 1560,
    "vlsfo_usd_per_tonne": 638,
    "coal_price_usd_per_mt": 124,
    "iron_ore_price_usd_per_dmt": 124,
    "wind_kmh": 32,
    "wave_height_m": 2.0,
    "cyclone_risk": 2,
    "weather_delay_days": 0.5,
}


def main():
    body = json.dumps(PAYLOAD).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        status = resp.status
        result = json.loads(resp.read().decode("utf-8"))

    print("Request:", json.dumps(PAYLOAD, indent=2))
    print("\nHTTP status:", status)
    print("Response:", json.dumps(result, indent=2))

    predicted = result.get("predicted_next_month_freight_usd_per_tonne")
    assert isinstance(predicted, (int, float)), "prediction is not numeric!"
    print(f"\nVerified: predicted freight is numeric -> {predicted}")


if __name__ == "__main__":
    main()
