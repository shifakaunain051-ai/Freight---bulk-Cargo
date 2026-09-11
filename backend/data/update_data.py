"""CLI script that refreshes the SQLite database with the latest data.

What it does:
    - initialises the database (creates tables if missing)
    - fetches live weather for all supported ports from Open-Meteo and
      stores it in the weather_data table
    - asks the configured market-data provider for the latest BDI / fuel /
      commodity quotes and stores any that are returned (the default
      placeholder provider returns none, so market_data stays empty until a
      real provider is wired up - we never fabricate these values)
    - prints a summary of what was updated

Usage:
    cd backend
    python -m data.update_data            # update everything (default)
    python -m data.update_data --weather  # weather only
    python -m data.update_data --market   # market only (no-op with placeholder)
    python -m data.update_data --ports "Hay Point" "Paradip"

Also runnable directly:
    python data/update_data.py --weather
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running both as a module (-m data.update_data) and as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import database
from data.weather import PORTS, fetch_port_weather
from data.market import MARKET_SERIES, get_default_provider


def update_weather(ports: list[str] | None = None) -> list[dict]:
    """Fetch and store weather for the given ports (default: all)."""
    database.init_db()
    targets = ports or list(PORTS.keys())
    results = []
    for port in targets:
        try:
            w = fetch_port_weather(port)
            database.upsert_weather(
                port=w["port"],
                latitude=w["latitude"],
                longitude=w["longitude"],
                wind_kmh=w["wind_kmh"],
                wave_height_m=w["wave_height_m"],
                cyclone_risk=w["cyclone_risk"],
                weather_delay_days=w["weather_delay_days"],
                temperature_c=w["temperature_c"],
            )
            results.append({"port": port, "status": "ok", "weather": w})
        except Exception as exc:  # noqa: BLE001 - keep updating other ports
            results.append({"port": port, "status": "error", "error": str(exc)})
    return results


def update_market() -> list[dict]:
    """Ask the market provider for every series; store any quotes returned."""
    database.init_db()
    provider = get_default_provider()
    results = []
    for series, meta in MARKET_SERIES.items():
        quote = provider.fetch_series(series)
        if quote is None:
            results.append({
                "series": series,
                "label": meta["label"],
                "status": "unavailable",
                "detail": "no provider configured for this series",
            })
            continue
        database.upsert_market(
            series=quote.series,
            value=quote.value,
            unit=quote.unit,
            source=quote.source,
        )
        results.append({
            "series": series,
            "label": meta["label"],
            "status": "ok",
            "value": quote.value,
            "unit": quote.unit,
            "source": quote.source,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the freight backend SQLite database."
    )
    parser.add_argument(
        "--weather", action="store_true",
        help="Update weather data (default if neither flag is given).",
    )
    parser.add_argument(
        "--market", action="store_true",
        help="Update market data (no-op with the placeholder provider).",
    )
    parser.add_argument(
        "--ports", nargs="*", default=None,
        help="Subset of ports to update (default: all).",
    )
    args = parser.parse_args()

    do_weather = args.weather or (not args.weather and not args.market)
    do_market = args.market or (not args.weather and not args.market)

    database.init_db()
    print("=" * 60)
    print("Freight data update")
    print("=" * 60)

    if do_weather:
        print("\n[weather] fetching Open-Meteo data for ports...")
        results = update_weather(args.ports)
        for r in results:
            if r["status"] == "ok":
                w = r["weather"]
                print(
                    f"  - {r['port']:14} wind={w['wind_kmh']} km/h "
                    f"wave={w['wave_height_m']} m "
                    f"cyclone_risk={w['cyclone_risk']} "
                    f"delay={w['weather_delay_days']}d "
                    f"[{w['fetched_source']}]"
                )
            else:
                print(f"  - {r['port']:14} ERROR: {r['error']}")

    if do_market:
        print("\n[market] querying market-data provider...")
        results = update_market()
        for r in results:
            if r["status"] == "ok":
                print(f"  - {r['series']:26} = {r['value']} {r['unit']} ({r['source']})")
            else:
                print(f"  - {r['series']:26} {r['status']}: {r['detail']}")

    status = database.get_data_status()
    print("\n[data_status]")
    for cat, ts in status.items():
        print(f"  - {cat:8} last_updated = {ts}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
