"""
heston.py — Heston (1993) stochastic volatility model.

Dynamics (risk-neutral):
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2
    <dW1, dW2> = rho dt

Parameters calibrated: v0, kappa, theta, xi, rho.

Provides:
  - call_price(...)          : vanilla European call via Lewis (2001) integral
  - implied_vol(...)         : Brent-solved BS IV
  - calibrate(...)           : least-squares fit to a market IV surface
  - simulate(...)            : Heston MC (full-truncation Euler, correlated)

SX5E option chains are not available via yfinance. To calibrate against a
real SX5E surface, supply a CSV with columns [T, K, iv] (T in years) and
pass it to calibrate(). For a quick live demo, use load_spx_surface().
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# ---------- Parameters --------------------------------------------------------

@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float       # vol of vol
    rho: float

    def as_array(self):
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])

    @classmethod
    def from_array(cls, x):
        return cls(*x)

    def feller_ok(self) -> bool:
        return 2 * self.kappa * self.theta > self.xi ** 2


# ---------- Characteristic function (Heston "little trap" form, Albrecher 2007)

def _char_func(u, T, r, q, p: HestonParams):
    v0, kappa, theta, xi, rho = p.v0, p.kappa, p.theta, p.xi, p.rho
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    exp_dT = np.exp(-d * T)

    # Characteristic function of the DE-DRIFTED log-return ln(S_T / F).
    # The (r-q)·iu·T drift is intentionally omitted: the Lewis (2001) formula
    # below carries the forward in k = ln(F/K), so including the drift here
    # would double-count it and bias every price.
    C = ((kappa * theta / xi ** 2)
         * ((kappa - rho * xi * iu - d) * T
            - 2 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((kappa - rho * xi * iu - d) / xi ** 2) * ((1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * v0)


# ---------- Lewis (2001) call price ------------------------------------------

def call_price(S0, K, T, r, q, p: HestonParams) -> float:
    """European call via Lewis integral. Numerically stable for deep OTM."""
    F = S0 * np.exp((r - q) * T)
    k = np.log(F / K)

    def integrand(u):
        phi = _char_func(u - 0.5j, T, r, q, p)
        return (np.exp(1j * u * k) * phi / (u ** 2 + 0.25)).real

    integral, _ = quad(integrand, 0, 200, limit=200)
    price = S0 * np.exp(-q * T) - np.sqrt(S0 * K) * np.exp(-(r + q) * T / 2) * integral / np.pi
    intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return max(price, intrinsic)


# ---------- Black-Scholes implied vol ----------------------------------------

def bs_call(S0, K, T, r, q, sigma) -> float:
    if sigma <= 0 or T <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S0, K, T, r, q) -> float:
    try:
        return brentq(lambda s: bs_call(S0, K, T, r, q, s) - price, 1e-4, 5.0, xtol=1e-6)
    except Exception:
        return np.nan


# ---------- Calibration -------------------------------------------------------

def calibrate(surface, S0, r, q, x0=None, verbose=False) -> HestonParams:
    """
    surface : iterable of (T, K, iv_market)
    Calibrates Heston to the IV surface with economically sane bounds.

    Bounds reflect equity-index reality:
      v0, theta in [~0, 0.25]  -> spot & long-run vol capped at 50%
      kappa     in [0.1, 15]
      xi        in [0.01, 2.0] -> vol-of-vol capped at 200%
      rho       in [-0.999, 0] -> equity skew is negative; positive rho is
                                  ruled out so the optimiser can't escape to a
                                  nonsensical region.
    The initial guess is derived from the data: v0 from the shortest-maturity
    ATM IV, theta from the longest-maturity ATM IV.
    """
    surface = np.asarray(surface, dtype=float)
    Ts, Ks, ivs_mkt = surface[:, 0], surface[:, 1], surface[:, 2]

    if x0 is None:
        # ATM ~ strike closest to spot, per maturity extreme
        def atm_iv(target_T):
            mask = np.isclose(Ts, target_T)
            sub_K, sub_iv = Ks[mask], ivs_mkt[mask]
            return sub_iv[np.argmin(np.abs(sub_K - S0))]
        try:
            v0_0 = float(atm_iv(Ts.min())) ** 2
            theta_0 = float(atm_iv(Ts.max())) ** 2
        except Exception:
            v0_0, theta_0 = 0.04, 0.04
        v0_0 = min(max(v0_0, 1e-3), 0.24)
        theta_0 = min(max(theta_0, 1e-3), 0.24)
        x0 = [v0_0, 2.0, theta_0, 0.5, -0.6]

    bounds = ([1e-4, 0.1, 1e-4, 1e-2, -0.999],
              [0.25, 15.0, 0.25, 2.0,   0.0])

    def residuals(x):
        p = HestonParams.from_array(x)
        res = []
        for T, K, iv_m in zip(Ts, Ks, ivs_mkt):
            try:
                c = call_price(S0, K, T, r, q, p)
                iv_h = implied_vol(c, S0, K, T, r, q)
            except Exception:
                iv_h = np.nan
            # weight ATM more (where vega is largest and quotes are reliable)
            w = np.exp(-2.0 * abs(np.log(K / S0)))
            res.append(w * (iv_h - iv_m) if np.isfinite(iv_h) else w * 1.0)
        return np.array(res)

    sol = least_squares(residuals, x0, bounds=bounds,
                        method="trf", xtol=1e-8, verbose=2 if verbose else 0)
    return HestonParams.from_array(sol.x)


# ---------- Monte Carlo simulation (full-truncation Euler) -------------------

def simulate(S0, r, q, p: HestonParams, T, n_steps, n_paths, seed=42):
    """Returns spot paths shape (n_paths, n_steps+1)."""
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    half = n_paths // 2
    z1 = rng.standard_normal((half, n_steps))
    z2 = rng.standard_normal((half, n_steps))
    z1 = np.vstack([z1, -z1])
    z2 = np.vstack([z2, -z2])
    w1 = z1
    w2 = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2

    S = np.full((n_paths, n_steps + 1), S0, dtype=float)
    v = np.full(n_paths, p.v0, dtype=float)

    for i in range(n_steps):
        v_pos = np.maximum(v, 0)
        sqrt_v = np.sqrt(v_pos)
        S[:, i + 1] = S[:, i] * np.exp((r - q - 0.5 * v_pos) * dt
                                       + sqrt_v * np.sqrt(dt) * w1[:, i])
        v = v + p.kappa * (p.theta - v_pos) * dt + p.xi * sqrt_v * np.sqrt(dt) * w2[:, i]
    return S


# ---------- Option chain loaders ---------------------------------------------

def load_spx_surface(max_quotes: int = 40):
    """Live SPX surface via yfinance SPY options (US proxy for demo).
    Returns (surface_array, S0, r, q). SX5E chain is not on yfinance."""
    import yfinance as yf
    from datetime import datetime

    tkr = yf.Ticker("SPY")
    S0 = float(tkr.history(period="5d")["Close"].iloc[-1])
    today = datetime.utcnow().date()

    rows = []
    for expiry in tkr.options[:6]:
        T = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days / 365.0
        if T < 0.05 or T > 2.0:
            continue
        chain = tkr.option_chain(expiry).calls
        chain = chain[(chain["strike"] > 0.8 * S0) & (chain["strike"] < 1.2 * S0)]
        chain = chain[chain["bid"] > 0]
        for _, r_ in chain.iterrows():
            mid = 0.5 * (r_["bid"] + r_["ask"])
            iv = implied_vol(mid, S0, r_["strike"], T, 0.04, 0.015)
            if np.isfinite(iv) and 0.05 < iv < 1.5:
                rows.append([T, r_["strike"], iv])
    rows = rows[::max(1, len(rows) // max_quotes)][:max_quotes]
    return np.array(rows), S0, 0.04, 0.015


def load_surface_csv(path: str):
    """Load a [T,K,iv] CSV surface. Use this for SX5E with data from Eurex /
    Refinitiv / Bloomberg."""
    import pandas as pd
    df = pd.read_csv(path)
    return df[["T", "K", "iv"]].values


# ---------- Demo --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--live" in sys.argv:
        print("Fetching SPX (via SPY) surface...")
        surf, S0, r, q = load_spx_surface()
        print(f"  {len(surf)} quotes, S0={S0:.2f}")
    else:
        # Synthetic skew surface for a quick test
        S0, r, q = 100.0, 0.03, 0.02
        true_p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
        Ts = [0.25, 0.5, 1.0, 2.0]
        Ks = [80, 90, 100, 110, 120]
        surf = []
        for T in Ts:
            for K in Ks:
                c = call_price(S0, K, T, r, q, true_p)
                iv = implied_vol(c, S0, K, T, r, q)
                surf.append([T, K, iv])
        surf = np.array(surf)
        print("Calibrating on synthetic surface (true params hidden)...")

    fitted = calibrate(surf, S0, r, q)
    print(f"\nFitted Heston params:")
    print(f"  v0    = {fitted.v0:.4f}")
    print(f"  kappa = {fitted.kappa:.4f}")
    print(f"  theta = {fitted.theta:.4f}")
    print(f"  xi    = {fitted.xi:.4f}")
    print(f"  rho   = {fitted.rho:.4f}")
    print(f"  Feller 2*kappa*theta > xi^2 ? {fitted.feller_ok()}")

    # Residuals
    errs = []
    for T, K, iv_m in surf:
        c = call_price(S0, K, T, r, q, fitted)
        iv_h = implied_vol(c, S0, K, T, r, q)
        errs.append(iv_h - iv_m)
    errs = np.array(errs)
    print(f"\nIV fit: RMSE={np.sqrt((errs**2).mean()):.4f}  MaxAbs={np.abs(errs).max():.4f}")
