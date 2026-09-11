"""SQLite database layer for the freight forecasting backend.

Schema:
    weather_data            one row per (port, fetched_at) weather snapshot
    market_data             one row per (series, fetched_at) market quote
    freight_observations    one row per historical freight rate observation
    data_status             one row per category with a last_updated timestamp
    prediction_logs         one row per model inference event (audit telemetry)

The DB file lives at backend/data/freight.db (git-ignored). It is created
automatically by init_db() on first use with WAL mode enabled.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# backend/data/freight.db  (this file is backend/data/database.py)
DB_PATH = Path(__file__).resolve().parent / "freight.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS weather_data (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    port               TEXT NOT NULL,
    latitude           REAL NOT NULL,
    longitude          REAL NOT NULL,
    wind_kmh           REAL,
    wave_height_m      REAL,
    cyclone_risk       REAL,
    weather_delay_days REAL,
    temperature_c      REAL,
    fetched_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_weather_port_time ON weather_data(port, fetched_at);

CREATE TABLE IF NOT EXISTS market_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series      TEXT NOT NULL,            -- e.g. 'bdi', 'vlsfo_usd_per_tonne'
    value       REAL,
    unit        TEXT,
    source      TEXT,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_series_time ON market_data(series, fetched_at);

CREATE TABLE IF NOT EXISTS freight_observations (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    origin                        TEXT NOT NULL,
    destination                   TEXT NOT NULL,
    commodity                     TEXT NOT NULL,
    vessel_type                   TEXT NOT NULL,
    cargo_tonnes                  REAL,
    current_freight_usd_per_tonne REAL NOT NULL,
    observed_at                   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_freight_route
    ON freight_observations(origin, destination, commodity, vessel_type, observed_at);

CREATE TABLE IF NOT EXISTS data_status (
    category     TEXT PRIMARY KEY,     -- 'weather' | 'market' | 'freight'
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id                                         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at                                 TEXT NOT NULL,
    model_version                              TEXT NOT NULL,
    origin                                     TEXT NOT NULL,
    destination                                TEXT NOT NULL,
    commodity                                  TEXT NOT NULL,
    vessel_type                                TEXT NOT NULL,
    current_freight_usd_per_tonne             REAL NOT NULL,
    predicted_next_month_freight_usd_per_tonne REAL NOT NULL,
    forecast_change_percent                    REAL NOT NULL,
    risk_level                                 TEXT NOT NULL,
    recommendation                             TEXT NOT NULL,
    latency_ms                                 REAL,
    provenance                                 TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_logs_time ON prediction_logs(created_at);
"""


def init_db() -> None:
    """Create all tables and configure WAL mode if they do not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a sqlite connection with WAL mode and 10.0s timeout."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _now_iso() -> str:
    """Return timezone-aware ISO-8601 UTC timestamp ending in 'Z'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _set_status(category: str, last_updated: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO data_status(category, last_updated) VALUES(?, ?) "
            "ON CONFLICT(category) DO UPDATE SET last_updated=excluded.last_updated",
            (category, last_updated),
        )
        conn.commit()


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #
def upsert_weather(
    port: str,
    latitude: float,
    longitude: float,
    wind_kmh: Optional[float],
    wave_height_m: Optional[float],
    cyclone_risk: Optional[float],
    weather_delay_days: Optional[float],
    temperature_c: Optional[float],
) -> None:
    """Insert a new weather snapshot for a port."""
    ts = _now_iso()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO weather_data
               (port, latitude, longitude, wind_kmh, wave_height_m,
                cyclone_risk, weather_delay_days, temperature_c, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (port, latitude, longitude, wind_kmh, wave_height_m,
             cyclone_risk, weather_delay_days, temperature_c, ts),
        )
        conn.commit()
    _set_status("weather", ts)


def get_latest_weather(port: str) -> Optional[dict]:
    """Return the most recent weather row for a port, or None."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM weather_data WHERE port = ?
               ORDER BY fetched_at DESC LIMIT 1""",
            (port,),
        ).fetchone()
    return dict(row) if row else None


def get_all_latest_weather() -> list[dict]:
    """Return the latest weather row for every port."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT w.* FROM weather_data w
               INNER JOIN (
                   SELECT port, MAX(fetched_at) AS max_ts
                   FROM weather_data GROUP BY port
               ) m ON w.port = m.port AND w.fetched_at = m.max_ts
               ORDER BY w.port"""
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Market
# --------------------------------------------------------------------------- #
def upsert_market(series: str, value: float, unit: str, source: str) -> None:
    """Insert a new market quote for a named series."""
    ts = _now_iso()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO market_data(series, value, unit, source, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (series, value, unit, source, ts),
        )
        conn.commit()
    _set_status("market", ts)


def get_latest_market() -> dict[str, dict]:
    """Return {series: {value, unit, source, fetched_at}} for every series."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT m.* FROM market_data m
               INNER JOIN (
                   SELECT series, MAX(fetched_at) AS max_ts
                   FROM market_data GROUP BY series
               ) l ON m.series = l.series AND m.fetched_at = l.max_ts"""
        ).fetchall()
    return {
        r["series"]: {
            "value": r["value"],
            "unit": r["unit"],
            "source": r["source"],
            "fetched_at": r["fetched_at"],
        }
        for r in rows
    }


# --------------------------------------------------------------------------- #
# Freight observations
# --------------------------------------------------------------------------- #
def insert_freight_observation(
    origin: str,
    destination: str,
    commodity: str,
    vessel_type: str,
    current_freight_usd_per_tonne: float,
    cargo_tonnes: Optional[float] = None,
    observed_at: Optional[str] = None,
) -> None:
    ts = observed_at or _now_iso()
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO freight_observations
               (origin, destination, commodity, vessel_type, cargo_tonnes,
                current_freight_usd_per_tonne, observed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (origin, destination, commodity, vessel_type, cargo_tonnes,
             current_freight_usd_per_tonne, ts),
        )
        conn.commit()
    _set_status("freight", ts)


def get_latest_freight_observation(
    origin: str, destination: str, commodity: str, vessel_type: str
) -> Optional[dict]:
    """Return the most recent freight observation matching a route, or None."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM freight_observations
               WHERE origin = ? AND destination = ? AND commodity = ?
                 AND vessel_type = ?
               ORDER BY observed_at DESC LIMIT 1""",
            (origin, destination, commodity, vessel_type),
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Prediction Logs (Telemetry)
# --------------------------------------------------------------------------- #
def insert_prediction_log(
    model_version: str,
    origin: str,
    destination: str,
    commodity: str,
    vessel_type: str,
    current_freight_usd_per_tonne: float,
    predicted_next_month_freight_usd_per_tonne: float,
    forecast_change_percent: float,
    risk_level: str,
    recommendation: str,
    latency_ms: Optional[float] = None,
    provenance: Optional[dict | str] = None,
) -> None:
    """Record a prediction event in prediction_logs. Failures are logged and never raised."""
    try:
        ts = _now_iso()
        prov_str = json.dumps(provenance) if isinstance(provenance, dict) else str(provenance or "")
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO prediction_logs
                   (created_at, model_version, origin, destination, commodity,
                    vessel_type, current_freight_usd_per_tonne,
                    predicted_next_month_freight_usd_per_tonne,
                    forecast_change_percent, risk_level, recommendation,
                    latency_ms, provenance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ts, model_version, origin, destination, commodity,
                 vessel_type, current_freight_usd_per_tonne,
                 predicted_next_month_freight_usd_per_tonne,
                 forecast_change_percent, risk_level, recommendation,
                 latency_ms, prov_str),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Failed to insert prediction telemetry log: %s", exc)


def get_recent_prediction_logs(limit: int = 50) -> list[dict]:
    """Retrieve recent prediction logs for monitoring."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM prediction_logs ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def get_data_status() -> dict:
    """Return last_updated timestamp for each data category."""
    defaults = {"weather": None, "market": None, "freight": None}
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT category, last_updated FROM data_status"
        ).fetchall()
    defaults.update({r["category"]: r["last_updated"] for r in rows})
    return defaults
