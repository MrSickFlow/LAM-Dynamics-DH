from __future__ import annotations

ARCHIPELAGO_MARITIME_FIXTURE = {
    "ais_vessels": [
        {"mmsi": "230123456", "name": "NORDIC CARRIER", "lat": 60.18, "lon": 21.92},
        {"mmsi": "230234567", "name": "BALTIC EXPRESS", "lat": 60.25, "lon": 22.15},
        {"mmsi": "230345678", "name": "ARCHIPELAGO ONE", "lat": 60.12, "lon": 21.75},
    ],
    "sar_returns": [
        {"lat": 60.18, "lon": 21.92, "confidence": 0.92},
        {"lat": 60.25, "lon": 22.15, "confidence": 0.88},
        {"lat": 60.15, "lon": 21.78, "confidence": 0.91},
        {"lat": 60.22, "lon": 22.05, "confidence": 0.85},
        {"lat": 60.10, "lon": 21.70, "confidence": 0.87},
        {"lat": 60.28, "lon": 22.22, "confidence": 0.90},
        {"lat": 60.14, "lon": 21.85, "confidence": 0.89},
        {"lat": 60.20, "lon": 22.00, "confidence": 0.86},
    ],
}
