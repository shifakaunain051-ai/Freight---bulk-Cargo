"""Weather data fetcher using the free Open-Meteo API (no API key required).

Covers four bulk-cargo ports:
    Hay Point       (-21.37, 149.32)  -- Queensland, Australia (coal)
    Taboneo         (-3.65,  114.85)  -- South Kalimantan, Indonesia (coal)
    Visakhapatnam   (17.68,  83.27)   -- Andhra Pradesh, India
    Paradip         (20.32,  86.70)   -- Odisha, India

Data sources (all Open-Meteo, all free):
    - forecast API (api.open-meteo.com/v1/forecast)     -> current wind, temp
    - marine API   (marine-api.open-meteo.com/v1/marine) -> wave height
    - archive API  (archive-api.open-meteo.com/v1/archive) -> fallback wind
      used only if the forecast API is rate-limited or unreachable.

Derived values (transparent formulas, NOT fabricated):
    cyclone_risk       = clamp(wind_kmh / 30, 0, 5)
        -> maps typical 0-60 km/h wind to a 0-2 risk; storm-force 150 km/h -> 5
    weather_delay_days = max(0, wave_height_m - 1.5) * 0.5
        -> 1.5 m swell adds no delay; each extra metre adds ~0.5 day
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

# Port name -> (latitude, longitude)
PORTS: dict[str, tuple[float, float]] = {
    "Australia West Coast": (-20.32, 118.57),
    "Hay Point": (-21.37, 149.32),
    "Taboneo": (-3.65, 114.85),
    "Visakhapatnam": (17.68, 83.27),
    "Paradip": (20.32, 86.70),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 20


def _get_json(url: str, params: dict) -> Optional[dict]:
    """GET a JSON response. Returns None on any error."""
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            import json
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _derive_cyclone_risk(wind_kmh: Optional[float]) -> Optional[float]:
    """Transparent cyclone-risk score derived from wind speed."""
    if wind_kmh is None:
        return None
    return round(_clamp(wind_kmh / 30.0, 0.0, 5.0), 2)


def _derive_weather_delay(
    wave_height_m: Optional[float], wind_kmh: Optional[float]
) -> Optional[float]:
    """Transparent weather-delay estimate derived from wave height + wind."""
    if wave_height_m is None:
        return None
    delay = max(0.0, wave_height_m - 1.5) * 0.5
    # Strong wind (>40 km/h) adds a little extra delay.
    if wind_kmh is not None and wind_kmh > 40:
        delay += 0.25
    return round(delay, 2)


def _fetch_current_wind_temp(lat: float, lon: float) -> tuple[Optional[float], Optional[float]]:
    """Fetch current wind_speed_10m (km/h) and temperature_2m (C) from the
    forecast API. Falls back to the archive API (most recent hour) if the
    forecast API is rate-limited or unreachable.
    """
    # 1) Primary: forecast API (current conditions)
    data = _get_json(FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": "wind_speed_10m,temperature_2m",
    })
    if data and "current" in data:
        cur = data["current"]
        wind = cur.get("wind_speed_10m")
        temp = cur.get("temperature_2m")
        if wind is not None:
            return float(wind), (float(temp) if temp is not None else None)

    # 2) Fallback: archive API for the last 2 days, take the last non-null hour
    today = datetime.utcnow().date()
    start = (today - timedelta(days=2)).isoformat()
    end = today.isoformat()
    data = _get_json(ARCHIVE_URL, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,temperature_2m",
        "timezone": "GMT",
    })
    if data and "hourly" in data:
        times = data["hourly"].get("time", [])
        winds = data["hourly"].get("wind_speed_10m", [])
        temps = data["hourly"].get("temperature_2m", [])
        wind = temp = None
        for i in range(len(times)):
            if i < len(winds) and winds[i] is not None:
                wind = float(winds[i])
                temp = float(temps[i]) if i < len(temps) and temps[i] is not None else None
        return wind, temp

    return None, None


def _fetch_wave_height(lat: float, lon: float) -> Optional[float]:
    """Fetch current wave_height (m) from the Open-Meteo marine API."""
    data = _get_json(MARINE_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": "wave_height",
    })
    if data and "current" in data:
        wh = data["current"].get("wave_height")
        if wh is not None:
            return float(wh)
    return None


def fetch_port_weather(port: str) -> dict:
    """Fetch the latest weather for a known port.

    Returns a dict with keys: port, latitude, longitude, wind_kmh,
    wave_height_m, cyclone_risk, weather_delay_days, temperature_c,
    fetched_source ('forecast' | 'archive-fallback' | 'partial').
    """
    if port not in PORTS:
        raise ValueError(f"Unknown port '{port}'. Known ports: {list(PORTS)}")
    lat, lon = PORTS[port]

    wind, temp = _fetch_current_wind_temp(lat, lon)
    wave = _fetch_wave_height(lat, lon)

    cyclone_risk = _derive_cyclone_risk(wind)
    delay = _derive_weather_delay(wave, wind)

    # Decide the source label for transparency.
    if wind is not None and wave is not None:
        # Determine whether wind came from forecast or archive fallback.
        # (We re-check the forecast API availability cheaply by testing once more.)
        source = "forecast"
        probe = _get_json(FORECAST_URL, {
            "latitude": lat, "longitude": lon,
            "current": "wind_speed_10m",
        })
        if not (probe and "current" in probe and probe["current"].get("wind_speed_10m") is not None):
            source = "archive-fallback"
    elif wind is not None or wave is not None:
        source = "partial"
    else:
        source = "none"

    return {
        "port": port,
        "latitude": lat,
        "longitude": lon,
        "wind_kmh": wind,
        "wave_height_m": wave,
        "cyclone_risk": cyclone_risk,
        "weather_delay_days": delay,
        "temperature_c": temp,
        "fetched_source": source,
    }


def fetch_all_ports_weather() -> list[dict]:
    """Fetch weather for every port in PORTS."""
    return [fetch_port_weather(port) for port in PORTS]
