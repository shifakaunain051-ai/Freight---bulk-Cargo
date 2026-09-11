"""Concurrency test for SQLite WAL mode."""

import concurrent.futures
import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from data import database


class TestSQLiteConcurrency(unittest.TestCase):
    def setUp(self):
        database.init_db()

    def test_concurrent_reads_and_writes_in_wal_mode(self):
        """Simulate concurrent threads writing weather and reading status/telemetry simultaneously."""
        errors = []

        def worker_write(i):
            try:
                database.upsert_weather(
                    port=f"Port_{i % 5}",
                    latitude=10.0,
                    longitude=20.0,
                    wind_kmh=15.0 + i,
                    wave_height_m=1.0,
                    cyclone_risk=1.0,
                    weather_delay_days=0.0,
                    temperature_c=25.0,
                )
                database.insert_prediction_log(
                    model_version="v3",
                    origin=f"Port_{i % 5}",
                    destination="East Coast India",
                    commodity="Coal",
                    vessel_type="Panamax",
                    current_freight_usd_per_tonne=15.0,
                    predicted_next_month_freight_usd_per_tonne=16.0,
                    forecast_change_percent=6.67,
                    risk_level="LOW",
                    recommendation="CHARTER NOW",
                    latency_ms=1.2,
                )
            except Exception as e:
                errors.append(("write", e))

        def worker_read(i):
            try:
                database.get_data_status()
                database.get_all_latest_weather()
                database.get_recent_prediction_logs(limit=10)
            except Exception as e:
                errors.append(("read", e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(40):
                if i % 2 == 0:
                    futures.append(executor.submit(worker_write, i))
                else:
                    futures.append(executor.submit(worker_read, i))
            concurrent.futures.wait(futures)

        self.assertEqual(len(errors), 0, f"Concurrency errors encountered: {errors}")


if __name__ == "__main__":
    unittest.main()
