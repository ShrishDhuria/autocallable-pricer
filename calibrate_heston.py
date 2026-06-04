"""
calibrate_heston.py — Multi-index Heston calibration framework.

Calibrates Heston stochastic-vol parameters per underlying index and saves a
single combined parameter file the pricer/dashboard loads automatically.

Indices supported out of the box:
  SPX   — S&P 500. Live calibration via SPY option chain (yfinance). Always works.
  SX5E  — Euro Stoxx 50. Calibration via a Eurex OESX settlement CSV (manual
          download). If no CSV is supplied, falls back to using the SPX-calibrated
          params as a documented proxy (the methodology is index-agnostic).
  NKY   — Nikkei 225. Proxy from SPX params (no free Japanese option chain).

The design is a registry: to add an index, add an entry to INDEX_REGISTRY.

Outputs (all in calibration/):
  heston_params.json        — { "SPX": {...}, "SX5E": {...}, ... } + metadata
  <index>_surface.csv       — the (T,K,iv) surface used for each live calibration
  <index>_fit.png           — market-vs-model scatter for each live calibration

Usage:
  python calibrate_heston.py --index spx
  python calibrate_heston.py --index sx5e --eurex oesx_settlements.csv
  python calibrate_heston.py --all                       # SPX live + others by proxy
  python calibrate_heston.py --all --eurex oesx.csv      # SPX live + SX5E from Eurex
"""

from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from heston import HestonParams, call_price, implied_vol, calibrate
from market_data import last_valid_close

OUT_DIR = Path("calibration")
PARAMS_FILE = OUT_DIR / "heston_params.json"


# ── Index registry ──────────────────────────────────────────────────────────

INDEX_REGISTRY = {
    "SPX": {
        "name": "S&P 500",
        "spot_ticker": "^GSPC",
        "options_ticker": "SPY",     # ETF proxy for the live chain on yfinance
        "rate": 0.04,                # USD risk-free (approx)
        "div_yield": 0.015,
        "calib_mode": "live_yf",
    },
    "SX5E": {
        "name": "Euro Stoxx 50",
        "spot_ticker": "^STOXX50E",
        "options_ticker": None,      # not on yfinance
        "rate": None,                # fetched from ECB €STR
        "div_yield": 0.03,
        "calib_mode": "eurex_csv",   # falls back to proxy if no CSV
    },
    "NKY": {
        "name": "Nikkei 225",
        "spot_ticker": "^N225",
        "options_ticker": None,      # no free chain
        "rate": 0.005,
        "div_yield": 0.018,
        "calib_mode": "proxy",
    },
}

PROXY_SOURCE = "SPX"   # which index's params to reuse for proxy indices


# ── Surface loaders ─────────────────────────────────────────────────────────

def load_yf_surface(options_ticker: str, rate: float, div_yield: float,
                    max_quotes: int = 60, per_expiry: int = 8):
    """Pull a live option-implied-vol surface from yfinance.

    Quality filters (critical — stale data breaks calibration):
      - two-sided quotes only (bid>0 AND ask>0)
      - bid/ask spread <= 30% of mid (rejects illiquid strikes)
      - open interest > 0
    These also act as a market-closed detector: outside US trading hours
    yfinance returns no usable two-sided quotes, so we raise a clear error
    rather than silently calibrating to stale prices.
    """
    import yfinance as yf

    tkr = yf.Ticker(options_ticker)
    S0 = last_valid_close(options_ticker, period="1mo")
    if S0 != S0:          # NaN -> no data at all
        raise ValueError(f"{options_ticker}: no valid spot close (data fetch failed).")
    today = datetime.now(timezone.utc).date()

    candidates = []
    for e in tkr.options:
        try:
            T = (datetime.strptime(e, "%Y-%m-%d").date() - today).days / 365.0
        except Exception:
            continue
        if 0.05 < T < 2.0:
            candidates.append((e, T))
    if len(candidates) > 8:
        step = max(1, len(candidates) // 8)
        candidates = candidates[::step][:8]

    rows = []
    for expiry, T in candidates:
        calls = tkr.option_chain(expiry).calls
        calls = calls[(calls["strike"] > 0.85 * S0) & (calls["strike"] < 1.15 * S0)]
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
        if "openInterest" in calls.columns:
            calls = calls[calls["openInterest"].fillna(0) > 0]
        per = []
        for _, row in calls.iterrows():
            bid, ask = float(row["bid"]), float(row["ask"])
            mid = 0.5 * (bid + ask)
            if mid <= 0 or (ask - bid) / mid > 0.30:   # reject wide/illiquid
                continue
            iv = implied_vol(mid, S0, row["strike"], T, rate, div_yield)
            if np.isfinite(iv) and 0.03 < iv < 1.0:
                per.append([T, float(row["strike"]), iv])
        per.sort(key=lambda x: abs(x[1] - S0))         # keep nearest-ATM
        rows.extend(per[:per_expiry])

    if len(rows) < 10:
        raise ValueError(
            f"{options_ticker}: only {len(rows)} usable two-sided quotes. "
            f"US options markets are likely CLOSED. Run during US hours "
            f"(09:30-16:00 ET = 19:00-01:30 IST) for live quotes."
        )

    if len(rows) > max_quotes:
        rows = rows[:: max(1, len(rows) // max_quotes)][:max_quotes]
    print(f"  {options_ticker} surface: {len(rows)} quotes across "
          f"{len(candidates)} expiries, S0={S0:.2f}")
    return np.array(rows), S0, rate, div_yield


def load_eurex_surface(csv_path: str, spot: float, rate: float, div_yield: float):
    """Parse Eurex OESX settlement CSV (expiry_date, strike, call_settlement)."""
    df = pd.read_csv(csv_path)
    required = {"expiry_date", "strike", "call_settlement"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV needs columns {required}, got {set(df.columns)}")

    today = datetime.now(timezone.utc).date()
    df["T"] = pd.to_datetime(df["expiry_date"]).apply(
        lambda d: (d.date() - today).days / 365.0
    )
    df = df[(df["T"] > 0.05) & (df["T"] < 2.5)]
    df = df[(df["strike"] > 0.70 * spot) & (df["strike"] < 1.30 * spot)]
    df = df[df["call_settlement"] > 0]

    rows = []
    for _, r in df.iterrows():
        iv = implied_vol(r["call_settlement"], spot, r["strike"],
                         r["T"], rate, div_yield)
        if np.isfinite(iv) and 0.05 < iv < 1.5:
            rows.append([r["T"], r["strike"], iv])
    print(f"  Eurex surface: {len(rows)} quotes from {csv_path}, S0={spot:.2f}")
    return np.array(rows), spot, rate, div_yield


# ── Calibration core ────────────────────────────────────────────────────────

def calibrate_surface(surface, S0, r, q, ticker_label: str):
    print(f"  Calibrating Heston on {len(surface)} quotes...")
    params = calibrate(surface, S0, r, q)

    errs = []
    for T, K, iv_m in surface:
        c = call_price(S0, K, T, r, q, params)
        iv_h = implied_vol(c, S0, K, T, r, q)
        if np.isfinite(iv_h):
            errs.append(iv_h - iv_m)
    rmse = float(np.sqrt((np.array(errs) ** 2).mean()))

    print(f"    v0={params.v0:.4f} kappa={params.kappa:.3f} theta={params.theta:.4f} "
          f"xi={params.xi:.3f} rho={params.rho:.3f}")
    print(f"    Feller={params.feller_ok()}  RMSE={rmse:.4f}")

    # Save surface + fit plot
    OUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(surface, columns=["T", "K", "iv"]).to_csv(
        OUT_DIR / f"{ticker_label}_surface.csv", index=False)
    _save_fit_plot(surface, S0, r, q, params, rmse, ticker_label)

    return params, rmse


def _save_fit_plot(surface, S0, r, q, params, rmse, label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for T_val in sorted(np.unique(surface[:, 0])):
            m = surface[:, 0] == T_val
            axes[0].plot(surface[m, 1], surface[m, 2] * 100, "o-",
                         label=f"T={T_val:.2f}y", markersize=3)
        axes[0].set_xlabel("Strike"); axes[0].set_ylabel("Implied Vol (%)")
        axes[0].set_title(f"{label} market IV surface")
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

        model = [implied_vol(call_price(S0, K, T, r, q, params), S0, K, T, r, q) * 100
                 for T, K, _ in surface]
        axes[1].scatter(surface[:, 2] * 100, model, s=8, alpha=0.6)
        lo, hi = surface[:, 2].min() * 100 - 1, surface[:, 2].max() * 100 + 1
        axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
        axes[1].set_xlabel("Market IV (%)"); axes[1].set_ylabel("Heston IV (%)")
        axes[1].set_title(f"{label} fit (RMSE={rmse*100:.2f}%)")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{label}_fit.png", dpi=150)
        plt.close()
        print(f"    saved {label}_surface.csv, {label}_fit.png")
    except Exception as e:
        print(f"    plot skipped: {e}")


# ── Per-index drivers ───────────────────────────────────────────────────────

def calibrate_index(index: str, eurex_csv: str = None):
    """Returns (HestonParams, rmse_or_None, source_str) or (None, ...) for proxy."""
    cfg = INDEX_REGISTRY[index]
    mode = cfg["calib_mode"]
    print(f"\n[{index}] {cfg['name']} — mode: {mode}")

    if mode == "live_yf":
        surface, S0, r, q = load_yf_surface(
            cfg["options_ticker"], cfg["rate"], cfg["div_yield"])
        params, rmse = calibrate_surface(surface, S0, r, q, index)
        return params, rmse, f"yfinance {cfg['options_ticker']} chain", S0, r, q

    if mode == "eurex_csv":
        if eurex_csv and Path(eurex_csv).exists():
            from market_data import fetch_estr, fetch_div_yield
            S0 = last_valid_close(cfg["spot_ticker"], period="1mo")
            if S0 != S0:
                raise ValueError(f"{cfg['spot_ticker']}: no valid spot close.")
            r = fetch_estr()
            q = fetch_div_yield(cfg["spot_ticker"])
            surface, S0, r, q = load_eurex_surface(eurex_csv, S0, r, q)
            params, rmse = calibrate_surface(surface, S0, r, q, index)
            return params, rmse, f"Eurex OESX settlements ({eurex_csv})", S0, r, q
        print(f"  No Eurex CSV — will use {PROXY_SOURCE} params as proxy")
        return None, None, f"proxy from {PROXY_SOURCE}", None, None, None

    # proxy
    print(f"  Proxy index — will use {PROXY_SOURCE} params")
    return None, None, f"proxy from {PROXY_SOURCE}", None, None, None


# ── Orchestration & persistence ─────────────────────────────────────────────

def save_params(results: dict):
    """results: {index: {"params": HestonParams|None, "rmse", "source",
                          "spot","rate","div","realized_vol"}}

    Proxy indices inherit the source index's SHAPE (kappa, xi, rho) but have
    v0 and theta rescaled to their own realized-vol level, so e.g. NKY (high
    vol) and SX5E (lower vol) don't blindly inherit S&P 500's vol magnitude.
    """
    from dataclasses import replace
    OUT_DIR.mkdir(exist_ok=True)

    src = results.get(PROXY_SOURCE)
    proxy_params = src["params"] if (src and src["params"] is not None) else None
    src_rv = src.get("realized_vol") if src else None

    out = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "proxy_source": PROXY_SOURCE,
        "indices": {},
    }
    for idx, res in results.items():
        p = res["params"]
        is_proxy = p is None
        if is_proxy:
            if proxy_params is None:
                print(f"  WARNING: {idx} needs proxy but {PROXY_SOURCE} not calibrated; skipping")
                continue
            p = proxy_params
            # Rescale variance level to this index's realized vol if we have both
            rv = res.get("realized_vol")
            if rv and src_rv and src_rv > 0:
                scale = (rv / src_rv) ** 2
                # cap at the calibration upper bound (0.25 variance = 50% vol)
                p = replace(p,
                            v0=min(p.v0 * scale, 0.25),
                            theta=min(p.theta * scale, 0.25))
                print(f"  {idx}: vol-scaled proxy (realized {rv:.1%} vs "
                      f"{PROXY_SOURCE} {src_rv:.1%}, scale {scale:.2f})")
        out["indices"][idx] = {
            "name": INDEX_REGISTRY[idx]["name"],
            "v0": float(p.v0), "kappa": float(p.kappa), "theta": float(p.theta),
            "xi": float(p.xi), "rho": float(p.rho),
            "feller_ok": bool(p.feller_ok()),
            "rmse": (float(res["rmse"]) if res["rmse"] is not None else None),
            "is_proxy": is_proxy,
            "source": res["source"],
            "spot": (float(res["spot"]) if res["spot"] is not None else None),
            "rate": (float(res["rate"]) if res["rate"] is not None else None),
            "div_yield": (float(res["div"]) if res["div"] is not None else None),
        }
    PARAMS_FILE.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {PARAMS_FILE} with indices: {list(out['indices'].keys())}")
    return out


def load_calibrated_params(path: str = None) -> dict:
    """Load saved params -> {index: HestonParams}. Used by the pricer/dashboard."""
    path = Path(path) if path else PARAMS_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for idx, d in data.get("indices", {}).items():
        out[idx] = HestonParams(d["v0"], d["kappa"], d["theta"], d["xi"], d["rho"])
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _get_eurex_arg():
    if "--eurex" in sys.argv:
        i = sys.argv.index("--eurex")
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


if __name__ == "__main__":
    eurex_csv = _get_eurex_arg()
    results = {}

    if "--all" in sys.argv:
        targets = list(INDEX_REGISTRY.keys())
        # Calibrate PROXY_SOURCE first so proxies can resolve
        targets = [PROXY_SOURCE] + [t for t in targets if t != PROXY_SOURCE]
    elif "--index" in sys.argv:
        i = sys.argv.index("--index")
        targets = [sys.argv[i + 1].upper()]
    else:
        print(__doc__)
        sys.exit(0)

    for idx in targets:
        if idx not in INDEX_REGISTRY:
            print(f"Unknown index {idx}; known: {list(INDEX_REGISTRY)}")
            continue
        try:
            params, rmse, source, S0, r, q = calibrate_index(idx, eurex_csv)
        except Exception as e:
            print(f"  [{idx}] calibration failed: {e}")
            print(f"  [{idx}] will use {PROXY_SOURCE} params as proxy")
            params, rmse, source, S0, r, q = (
                None, None, f"failed, proxy from {PROXY_SOURCE}", None, None, None)
        # Realized vol for this index (used to vol-scale proxies)
        try:
            from market_data import fetch_realized_vol
            rv = fetch_realized_vol(INDEX_REGISTRY[idx]["spot_ticker"])
        except Exception:
            rv = None
        results[idx] = {"params": params, "rmse": rmse, "source": source,
                        "spot": S0, "rate": r, "div": q, "realized_vol": rv}

    summary = save_params(results)

    print("\n" + "=" * 60)
    print("Calibration summary:")
    for idx, d in summary["indices"].items():
        tag = " (proxy)" if d["is_proxy"] else ""
        rmse_s = f"RMSE={d['rmse']:.4f}" if d["rmse"] is not None else ""
        print(f"  {idx:5s} rho={d['rho']:+.3f} {rmse_s}{tag}")
    print("=" * 60)
    print("\nThe pricer/dashboard will auto-load calibration/heston_params.json")
