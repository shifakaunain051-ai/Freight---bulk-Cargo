"""Market data providers.

There is no free, keyless API for the Baltic Dry Index (BDI), VLSFO bunker
prices, coal/iron-ore benchmark prices, or spot freight rates. Those sources
require paid subscriptions (Baltic Exchange, Signal Group, Platts, etc.).

This module defines a clean provider interface so a real adapter can be
plugged in later without touching the rest of the codebase. The default
``PlaceholderMarketDataProvider`` returns None for every series (it never
fabricates values) - callers must treat missing market data as "user must
supply it".

To wire up a real source, subclass ``MarketDataProvider`` and implement
``fetch_series`` / ``fetch_all`` using your subscription's API, then pass an
instance of it to ``update_data.update_market_data(provider=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

# Canonical market series used by the forecasting model. Maps a series key
# (which matches the model feature name) to its human label + unit.
MARKET_SERIES: dict[str, dict] = {
    "bdi": {"label": "Baltic Dry Index", "unit": "index points"},
    "vlsfo_usd_per_tonne": {"label": "VLSFO bunker fuel", "unit": "USD/tonne"},
    "coal_price_usd_per_mt": {"label": "Coal benchmark", "unit": "USD/MT"},
    "iron_ore_price_usd_per_dmt": {"label": "Iron ore benchmark", "unit": "USD/dmt"},
}


@dataclass
class MarketQuote:
    series: str
    value: float
    unit: str
    source: str


class MarketDataProvider(Protocol):
    """Abstract interface every market-data adapter must implement."""

    def fetch_series(self, series: str) -> Optional[MarketQuote]:
        """Return the latest quote for a single series, or None if unavailable."""
        ...

    def fetch_all(self) -> list[MarketQuote]:
        """Return all available quotes (may be a subset of MARKET_SERIES)."""
        ...


class PlaceholderMarketDataProvider:
    """Default no-op provider. Never returns live values.

    Use this while no paid market-data subscription is wired up. Every call
    returns None, which means the caller (forecast_service) will require the
    user to supply the market fields in the /predict request.
    """

    name = "placeholder"

    def fetch_series(self, series: str) -> Optional[MarketQuote]:
        return None

    def fetch_all(self) -> list[MarketQuote]:
        return []


# Module-level default provider instance.
DEFAULT_PROVIDER: MarketDataProvider = PlaceholderMarketDataProvider()


def get_default_provider() -> MarketDataProvider:
    return DEFAULT_PROVIDER


def set_default_provider(provider: MarketDataProvider) -> None:
    """Swap in a real provider at runtime (e.g. after configuring an API key)."""
    global DEFAULT_PROVIDER
    DEFAULT_PROVIDER = provider
