"""
market_data.py — Live market snapshot for the autocallable pricer.

Sources (all free, no API key):
  - Spot: Yahoo Finance via yfinance (^STOXX50E for Euro Stoxx 50)
  - Vol proxy: VSTOXX (^V2TX) — the Euro Stoxx 50 implied vol index,
    directly analogous to VIX for S&P. Used as a flat-vol proxy until
    we build a real surface from Eurex option chains.
  - Div yield: yfinance ticker info (trailing 12m, best-effort)
  - EUR risk-free: ECB €STR via ECB SDW public API (no key)

All fetchers fail soft: on network error they return a fallback value and
log a warning, so the pricer stays runnable offline (important for CI).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import warnings

import yfinance as yf
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    spot: float
    vol: float           # decimal, e.g. 0.18
    rate: float          # decimal, e.g. 0.029
    div_yield: float     # decimal
    as_of: datetime
    source_notes: dict

    def pretty(self) -> str:
        return (
            f"Euro Stoxx 50 snapshot @ {self.as_of:%Y-%m-%d %H:%M UTC}\n"
            f"  Spot       : {self.spot:,.2f}\n"
            f"  Vol (VSTOXX): {self.vol:.2%}\n"
            f"  EUR rate   : {self.rate:.2%}\n"
            f"  Div yield  : {self.div_yield:.2%}\n"
            f"  Sources    : {self.source_notes}"
        )


# ---------- Individual fetchers ----------------------------------------------

def last_valid_close(ticker: str, period: str = "1mo"):
    """Most recent NON-NaN close for a ticker.

    yfinance can return a trailing NaN row for a session that hasn't printed
    yet (e.g. ^N225 just after midnight JST, or ^STOXX50E / ^GSPC outside their
    own session). Dropping NaNs and taking the last valid close gives a robust
    spot for every index. Returns NaN if the ticker yields no data at all, so
    callers can decide whether to fall back or fail.
    """
    try:
        s = yf.Ticker(ticker).history(period=period, auto_adjust=False)["Close"].dropna()
        if len(s):
            return float(s.iloc[-1])
    except Exception as e:
        log.warning("%s close fetch failed: %s", ticker, e)
    return float("nan")


def fetch_spot(ticker: str = "^STOXX50E", fallback: float = 5000.0) -> float:
    px = last_valid_close(ticker, period="1mo")
    if px == px:          # not NaN
        return px
    return fallback


def fetch_realized_vol(ticker: str = "^STOXX50E", window: int = 30,
                       fallback: float = 0.18) -> float:
    """Annualized close-to-close realized vol over `window` trading days.
    Used as a vol proxy; production would calibrate to Eurex option chain.
    (Yahoo delisted ^V2TX so VSTOXX is no longer freely scrapable.)"""
    try:
        import numpy as np
        hist = yf.Ticker(ticker).history(period=f"{window + 10}d", auto_adjust=False)
        if len(hist) > window:
            rets = np.log(hist["Close"]).diff().dropna().tail(window)
            return float(rets.std() * np.sqrt(252))
    except Exception as e:
        log.warning("realized vol fetch failed: %s", e)
    return fallback


def fetch_div_yield(ticker: str = "^STOXX50E", fallback: float = 0.03) -> float:
    try:
        info = yf.Ticker(ticker).info
        dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if dy:
            # yfinance sometimes returns 3.2 (percent) and sometimes 0.032
            return float(dy) / 100 if dy > 1 else float(dy)
    except Exception as e:
        log.warning("div yield fetch failed: %s", e)
    return fallback


def fetch_estr(fallback: float = 0.03) -> float:
    """ECB €STR — the euro risk-free rate. Public SDW API, no key."""
    url = (
        "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT"
        "?lastNObservations=1&format=jsondata"
    )
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        j = r.json()
        series = j["dataSets"][0]["series"]
        key = next(iter(series))
        obs = series[key]["observations"]
        last = obs[next(iter(obs))][0]
        return float(last) / 100.0
    except Exception as e:
        log.warning("€STR fetch failed: %s", e)
    return fallback


# ---------- Combined snapshot -------------------------------------------------

def get_snapshot() -> MarketSnapshot:
    spot = fetch_spot()
    vol = fetch_realized_vol()
    rate = fetch_estr()
    dy = fetch_div_yield()
    return MarketSnapshot(
        spot=spot,
        vol=vol,
        rate=rate,
        div_yield=dy,
        as_of=datetime.now(timezone.utc),
        source_notes={
            "spot": "Yahoo ^STOXX50E",
            "vol": "30d realized vol of ^STOXX50E",
            "rate": "ECB SDW €STR",
            "div_yield": "Yahoo ticker.info",
        },
    )


# ---------- Multi-asset snapshot ---------------------------------------------

@dataclass
class MultiSnapshot:
    assets: list          # list of dicts: name, spot, vol, div_yield
    rate: float
    corr: "np.ndarray"
    as_of: datetime

    def pretty(self) -> str:
        import numpy as np
        lines = [f"Multi-asset snapshot @ {self.as_of:%Y-%m-%d %H:%M UTC}"]
        for a in self.assets:
            lines.append(f"  {a['name']:6s}  spot={a['spot']:>10,.2f}  "
                         f"vol={a['vol']:.2%}  div={a['div_yield']:.2%}")
        lines.append(f"  EUR rate: {self.rate:.2%}")
        lines.append(f"  Corr:\n{np.round(self.corr, 3)}")
        return "\n".join(lines)


def get_multi_snapshot(tickers=(("SX5E", "^STOXX50E"),
                                 ("SPX",  "^GSPC"),
                                 ("NKY",  "^N225"))) -> MultiSnapshot:
    """Fetch spot + realized vol + div yield for each ticker, plus pairwise
    realized correlation from 1Y of joint log returns."""
    import numpy as np
    import pandas as pd

    # Sensible per-index dividend yields (index tickers don't expose these
    # via yfinance, so fetch_div_yield would just return a flat fallback).
    DIV_DEFAULTS = {"SX5E": 0.030, "SPX": 0.013, "NKY": 0.018}

    assets = []
    closes = {}
    for name, tkr in tickers:
        try:
            hist = yf.Ticker(tkr).history(period="1y", auto_adjust=False)
            s = hist["Close"].copy()
            # Strip the time/timezone so the three exchanges align on date.
            # Without this, EU/US/JP close timestamps never match and the
            # joint dropna() wipes out every row -> identity correlation.
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            s = s.dropna()                      # drop forming/empty bars
            closes[name] = s
            if len(s) == 0:
                raise ValueError("no valid closes")
            # last VALID close — yfinance can return a trailing NaN row for a
            # session that hasn't printed yet (e.g. ^N225 just after midnight JST)
            spot = float(s.iloc[-1])
        except Exception as e:
            log.warning("%s history failed: %s", tkr, e)
            spot, closes[name] = float("nan"), None
        assets.append({
            "name": name,
            "spot": spot,
            "vol": fetch_realized_vol(tkr),
            "div_yield": DIV_DEFAULTS.get(name, 0.025),
        })

    # Fail loudly if any spot is still NaN (total fetch failure) rather than
    # letting it cascade into a silent 0-PV downstream.
    bad = [a["name"] for a in assets if not np.isfinite(a["spot"])]
    if bad:
        raise ValueError(
            f"no valid spot for {bad} — data fetch failed for these tickers. "
            f"Retry, or check the ticker symbols."
        )

    # Correlation from date-aligned daily log returns
    n = len(tickers)
    corr = np.eye(n)
    try:
        df = pd.concat({k: v for k, v in closes.items() if v is not None}, axis=1)
        df = df.dropna(how="any")  # keep only dates all exchanges traded
        rets = np.log(df).diff().dropna()
        if len(rets) > 30:
            corr = rets.corr().values
            print(f"  correlation estimated from {len(rets)} joint trading days")
        else:
            log.warning("only %d joint days; using identity correlation", len(rets))
    except Exception as e:
        log.warning("correlation estimation failed: %s", e)

    return MultiSnapshot(
        assets=assets,
        rate=fetch_estr(),
        corr=corr,
        as_of=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_snapshot().pretty())
    print()
    print(get_multi_snapshot().pretty())
