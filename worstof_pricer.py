"""
Worst-of Phoenix Autocallable — Monte Carlo pricer.

Payoff triggers on the WORST performer across N underlyings:
    perf_i(t) = S_i(t) / strike_i
    worst(t)  = min_i perf_i(t)

  - Autocall if worst(t) >= AC barrier  -> par + memory coupons
  - Coupon   if worst(t) >= cpn barrier  (else accrue to memory)
  - At T, if never autocalled:
        worst(T) >= KI    -> par
        worst(T) <  KI    -> notional * worst(T)   (equity loss)

Multi-asset GBM simulated via Cholesky-decomposed correlation matrix.
Degrades to single-asset when len(assets) == 1.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


# ---------- Market / product --------------------------------------------------

@dataclass
class Asset:
    name: str
    spot: float
    vol: float
    div_yield: float


@dataclass
class MultiMarket:
    assets: List[Asset]
    rate: float                          # single discount curve (EUR)
    corr: np.ndarray                     # (n, n) correlation matrix

    @property
    def n(self) -> int:
        return len(self.assets)


@dataclass
class WorstOfAutocallable:
    strikes: List[float]                 # one per underlying (initial fixing)
    notional: float = 100.0
    maturity_years: float = 6.0
    obs_per_year: int = 1
    autocall_barrier: float = 1.00
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60
    coupon: float = 0.08


# ---------- Path simulation ---------------------------------------------------

def simulate_paths(mkt: MultiMarket, prod: WorstOfAutocallable,
                   n_paths: int, seed: int = 42,
                   heston_params: list = None) -> np.ndarray:
    """Returns array shape (n_paths, n_obs, n_assets) of spot levels.
    If heston_params is provided (list of HestonParams, one per asset),
    simulates under Heston dynamics with correlated spot shocks.
    Otherwise uses lognormal GBM with asset-level flat vols."""
    n = mkt.n
    n_obs = int(prod.maturity_years * prod.obs_per_year)
    dt = 1.0 / prod.obs_per_year
    rng = np.random.default_rng(seed)

    L = np.linalg.cholesky(mkt.corr)
    spots = np.array([a.spot for a in mkt.assets])
    divs = np.array([a.div_yield for a in mkt.assets])

    if heston_params is None:
        vols = np.array([a.vol for a in mkt.assets])
        drift = (mkt.rate - divs - 0.5 * vols ** 2) * dt
        diff = vols * np.sqrt(dt)
        half = n_paths // 2
        z = rng.standard_normal((half, n_obs, n))
        z = np.concatenate([z, -z], axis=0)
        w = z @ L.T
        log_incr = drift + diff * w
        log_paths = np.cumsum(log_incr, axis=1)
        return spots * np.exp(log_paths)

    # Heston path: sub-step inside each observation period for variance stability
    sub_steps = 20
    h = dt / sub_steps
    sqrt_h = np.sqrt(h)
    half = n_paths // 2
    # Correlated spot Brownians (cross-asset); variance Brownians independent per asset
    z_s = rng.standard_normal((half, n_obs * sub_steps, n))
    z_v = rng.standard_normal((half, n_obs * sub_steps, n))
    z_s = np.concatenate([z_s, -z_s], axis=0)
    z_v = np.concatenate([z_v, -z_v], axis=0)
    w_s = z_s @ L.T  # cross-asset correlation on spot shocks

    S = np.broadcast_to(spots, (n_paths, n)).copy()
    v = np.array([[p.v0 for p in heston_params]] * n_paths)  # (n_paths, n)
    params = heston_params
    rhos = np.array([p.rho for p in params])
    kappas = np.array([p.kappa for p in params])
    thetas = np.array([p.theta for p in params])
    xis = np.array([p.xi for p in params])

    out = np.empty((n_paths, n_obs, n))
    step = 0
    for i in range(n_obs):
        for _ in range(sub_steps):
            v_pos = np.maximum(v, 0)
            sqrt_v = np.sqrt(v_pos)
            # Correlate each asset's variance shock with its own spot shock
            dW1 = w_s[:, step, :]
            dW2 = rhos * dW1 + np.sqrt(1 - rhos ** 2) * z_v[:, step, :]
            S = S * np.exp((mkt.rate - divs - 0.5 * v_pos) * h + sqrt_v * sqrt_h * dW1)
            v = v + kappas * (thetas - v_pos) * h + xis * sqrt_v * sqrt_h * dW2
            step += 1
        out[:, i, :] = S
    return out


# ---------- Pricing -----------------------------------------------------------

def price(mkt: MultiMarket, prod: WorstOfAutocallable,
          n_paths: int = 100_000, seed: int = 42,
          heston_params: list = None):
    paths = simulate_paths(mkt, prod, n_paths, seed, heston_params)  # (P, T, N)
    strikes = np.array(prod.strikes)
    perf = paths / strikes                                      # (P, T, N)
    worst = perf.min(axis=2)                                    # (P, T)

    P, T_obs = worst.shape
    dt = 1.0 / prod.obs_per_year
    obs_times = np.arange(1, T_obs + 1) * dt

    pv = np.zeros(P)
    alive = np.ones(P, dtype=bool)
    memory = np.zeros(P, dtype=int)

    for i in range(T_obs):
        t = obs_times[i]
        df = np.exp(-mkt.rate * t)
        w_i = worst[:, i]

        pay = alive & (w_i >= prod.coupon_barrier)
        pv[pay] += df * (memory[pay] + 1) * prod.coupon * prod.notional
        memory[pay] = 0
        miss = alive & (w_i < prod.coupon_barrier)
        memory[miss] += 1

        ac = alive & (w_i >= prod.autocall_barrier)
        pv[ac] += df * prod.notional
        alive[ac] = False

    T = prod.maturity_years
    df_T = np.exp(-mkt.rate * T)
    w_T = worst[:, -1]
    ki_hit = alive & (w_T < prod.ki_barrier)
    no_ki = alive & (w_T >= prod.ki_barrier)
    pv[no_ki] += df_T * prod.notional
    pv[ki_hit] += df_T * prod.notional * w_T[ki_hit]

    return pv.mean(), pv.std(ddof=1) / np.sqrt(P)


# ---------- Greeks ------------------------------------------------------------

def _bump_spot(mkt: MultiMarket, i: int, h: float) -> MultiMarket:
    assets = [Asset(a.name, a.spot + (h if k == i else 0), a.vol, a.div_yield)
              for k, a in enumerate(mkt.assets)]
    return MultiMarket(assets, mkt.rate, mkt.corr)


def _bump_vol(mkt: MultiMarket, i: int, h: float) -> MultiMarket:
    assets = [Asset(a.name, a.spot, a.vol + (h if k == i else 0), a.div_yield)
              for k, a in enumerate(mkt.assets)]
    return MultiMarket(assets, mkt.rate, mkt.corr)


def _bump_heston_v0(heston_params, i, h):
    """Return a copy of the heston_params list with asset i's v0 bumped."""
    from dataclasses import replace
    return [replace(p, v0=max(p.v0 + (h if k == i else 0), 1e-6))
            for k, p in enumerate(heston_params)]


def greeks(mkt: MultiMarket, prod: WorstOfAutocallable, n_paths: int = 100_000,
           heston_params: list = None):
    base, _ = price(mkt, prod, n_paths, heston_params=heston_params)
    out = {"price": base}
    for i, a in enumerate(mkt.assets):
        h = 0.01 * a.spot
        up, _ = price(_bump_spot(mkt, i, h), prod, n_paths, heston_params=heston_params)
        dn, _ = price(_bump_spot(mkt, i, -h), prod, n_paths, heston_params=heston_params)
        out[f"delta_{a.name}"] = (up - dn) / (2 * h)
        out[f"gamma_{a.name}"] = (up - 2 * base + dn) / (h ** 2)

        if heston_params is not None:
            # Heston vega: bump initial variance v0 by an amount equal to a
            # 1-vol-point move at current level (dv ≈ 2*sqrt(v0)*dσ).
            hp_up = _bump_heston_v0(heston_params, i, 2 * np.sqrt(heston_params[i].v0) * 0.01)
            hp_dn = _bump_heston_v0(heston_params, i, -2 * np.sqrt(heston_params[i].v0) * 0.01)
            vu, _ = price(mkt, prod, n_paths, heston_params=hp_up)
            vd, _ = price(mkt, prod, n_paths, heston_params=hp_dn)
            out[f"vega_{a.name}_1vol"] = (vu - vd) / 2
        else:
            hv = 0.01
            vu, _ = price(_bump_vol(mkt, i, hv), prod, n_paths)
            vd, _ = price(_bump_vol(mkt, i, -hv), prod, n_paths)
            out[f"vega_{a.name}_1vol"] = (vu - vd) / (2 * hv) / 100
    return out


# ---------- Fair-coupon solver -----------------------------------------------

def fair_coupon(mkt: MultiMarket, prod: WorstOfAutocallable,
                n_paths: int = 100_000, seed: int = 42,
                heston_params: list = None,
                tol: float = 1e-4, max_iter: int = 50) -> float:
    """Root-find the coupon that makes PV = notional (par-priced note).

    Uses Brent's method. PV is monotonically increasing in coupon, so
    bracketing is straightforward: [0, 0.5] covers any realistic structure.
    Reuses the same seed across evaluations so MC noise cancels.
    """
    from scipy.optimize import brentq
    from dataclasses import replace

    target = prod.notional

    def pv_minus_par(c: float) -> float:
        trial = replace(prod, coupon=c)
        pv, _ = price(mkt, trial, n_paths=n_paths, seed=seed,
                      heston_params=heston_params)
        return pv - target

    lo, hi = 0.0, 0.50
    f_lo, f_hi = pv_minus_par(lo), pv_minus_par(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"No sign change in [{lo}, {hi}]: PV(0)={f_lo+target:.2f}, "
            f"PV(0.5)={f_hi+target:.2f}. Structure may be infeasible."
        )
    return brentq(pv_minus_par, lo, hi, xtol=tol, maxiter=max_iter)


# ---------- Load calibrated Heston params per underlying ---------------------

def heston_list_for_assets(assets, verbose=True):
    """Build a [HestonParams] list aligned to `assets` from the saved
    calibration file. Returns None if no calibration exists (pricer then
    uses flat-vol GBM). Assets without a calibrated entry fall back to the
    proxy already baked into heston_params.json."""
    try:
        from calibrate_heston import load_calibrated_params
        calib = load_calibrated_params()
    except Exception:
        calib = {}
    if not calib:
        if verbose:
            print("No calibration file found — using flat-vol GBM. "
                  "Run: python calibrate_heston.py --all")
        return None

    hp = []
    for a in assets:
        if a.name in calib:
            hp.append(calib[a.name])
        else:
            # fall back to any available calibrated set
            hp.append(next(iter(calib.values())))
            if verbose:
                print(f"  {a.name}: no entry, using {next(iter(calib))} params")
    if verbose:
        print(f"Loaded Heston params for: {[a.name for a in assets]}")
    return hp


# ---------- Correlation sensitivity ------------------------------------------

def corr_ladder(mkt: MultiMarket, prod: WorstOfAutocallable,
                rhos=(0.3, 0.5, 0.7, 0.9)):
    print(f"\n{'avg_corr':>10} {'PV':>10}")
    n = mkt.n
    for rho in rhos:
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1.0)
        m = MultiMarket(mkt.assets, mkt.rate, C)
        pv, _ = price(m, prod, n_paths=50_000)
        print(f"{rho:>10.2f} {pv:>10.3f}")


# ---------- Main --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    live = "--live" in sys.argv
    use_heston = "--heston" in sys.argv

    if live:
        from market_data import get_multi_snapshot
        snap = get_multi_snapshot()
        print(snap.pretty(), "\n")
        assets = [Asset(a["name"], a["spot"], a["vol"], a["div_yield"])
                  for a in snap.assets]
        mkt = MultiMarket(assets, snap.rate, snap.corr)
        prod = WorstOfAutocallable(
            strikes=[a.spot for a in assets],
            maturity_years=6,
            coupon=0.08,
        )
    else:
        assets = [
            Asset("SX5E", 100, 0.20, 0.03),
            Asset("SPX",  100, 0.18, 0.015),
            Asset("NKY",  100, 0.22, 0.02),
        ]
        corr = np.array([
            [1.0, 0.65, 0.55],
            [0.65, 1.0, 0.50],
            [0.55, 0.50, 1.0],
        ])
        mkt = MultiMarket(assets, rate=0.03, corr=corr)
        prod = WorstOfAutocallable(
            strikes=[100, 100, 100],
            maturity_years=6,
            coupon=0.08,
        )

    # Per-underlying Heston params (loaded from calibration/heston_params.json)
    hp = heston_list_for_assets(mkt.assets) if use_heston else None

    pv, se = price(mkt, prod, n_paths=200_000, heston_params=hp)
    model = "Heston (per-underlying)" if hp else "GBM (flat vol)"
    print(f"\nWorst-of autocall PV: {pv:.4f}  (SE {se:.4f})  [{model}]")
    print(f"Underlyings: {[a.name for a in mkt.assets]}")

    print("\nGreeks:")
    for k, v in greeks(mkt, prod, n_paths=100_000, heston_params=hp).items():
        print(f"  {k:>20}: {v: .5f}")

    corr_ladder(mkt, prod)
