"""
Autocallable Pricer — Monte Carlo
Single-underlying Phoenix autocallable (e.g. on Euro Stoxx 50).

Structure (standard French retail/SG-style):
  - Annual observation dates t_1, ..., t_N
  - Autocall barrier (e.g. 100% of spot): if S(t_i) >= AC barrier, note redeems
    at par + accrued coupons (memory effect)
  - Coupon barrier (e.g. 70%): coupon paid at t_i if S(t_i) >= coupon barrier;
    missed coupons are memorized and paid on next trigger
  - Knock-in put at maturity (e.g. 60%): if never autocalled AND S(T) < KI,
    investor takes equity loss (S(T)/S(0) - 1); else par is returned
  - Notional = 100

Author: prototype for ESSEC MIF project
"""

from dataclasses import dataclass
import numpy as np


# ---------- Product & market data ---------------------------------------------

@dataclass
class Autocallable:
    notional: float = 100.0
    strike: float = 100.0          # initial fixing level (set at trade date)
    maturity_years: float = 6.0
    obs_per_year: int = 1          # annual observations
    autocall_barrier: float = 1.00 # % of strike
    coupon_barrier: float = 0.70
    ki_barrier: float = 0.60       # European knock-in at maturity
    coupon: float = 0.08           # 8% annual, with memory


@dataclass
class Market:
    spot: float = 100.0
    rate: float = 0.03             # EUR risk-free (flat)
    div_yield: float = 0.03        # Euro Stoxx 50 div yield is material
    vol: float = 0.20              # flat vol — extend to surface later


# ---------- Path generation ---------------------------------------------------

def simulate_paths(mkt: Market, prod: Autocallable, n_paths: int, seed: int = 42):
    """GBM under risk-neutral measure. Returns array of shape (n_paths, n_obs)
    containing S at each observation date (excluding t=0)."""
    n_obs = int(prod.maturity_years * prod.obs_per_year)
    dt = 1.0 / prod.obs_per_year
    rng = np.random.default_rng(seed)

    drift = (mkt.rate - mkt.div_yield - 0.5 * mkt.vol ** 2) * dt
    diff = mkt.vol * np.sqrt(dt)

    # Antithetic variates for variance reduction
    half = n_paths // 2
    z = rng.standard_normal((half, n_obs))
    z = np.vstack([z, -z])

    log_returns = drift + diff * z
    log_paths = np.cumsum(log_returns, axis=1)
    return mkt.spot * np.exp(log_paths)


# ---------- Payoff & pricing --------------------------------------------------

def price(mkt: Market, prod: Autocallable, n_paths: int = 100_000, seed: int = 42):
    paths = simulate_paths(mkt, prod, n_paths, seed)
    n_paths_actual, n_obs = paths.shape
    dt = 1.0 / prod.obs_per_year
    obs_times = np.arange(1, n_obs + 1) * dt

    ac_level = prod.autocall_barrier * prod.strike
    cpn_level = prod.coupon_barrier * prod.strike
    ki_level = prod.ki_barrier * prod.strike

    pv = np.zeros(n_paths_actual)
    alive = np.ones(n_paths_actual, dtype=bool)
    memory = np.zeros(n_paths_actual, dtype=int)  # missed coupons held in memory

    for i in range(n_obs):
        t = obs_times[i]
        s_i = paths[:, i]
        df = np.exp(-mkt.rate * t)

        # Pay coupon where alive and above coupon barrier (with memory)
        pay_cpn = alive & (s_i >= cpn_level)
        coupons_paid = (memory[pay_cpn] + 1) * prod.coupon * prod.notional
        pv[pay_cpn] += df * coupons_paid
        memory[pay_cpn] = 0
        # Accrue memory where alive but below coupon barrier
        miss = alive & (s_i < cpn_level)
        memory[miss] += 1

        # Autocall redemption: pay notional, kill path
        ac = alive & (s_i >= ac_level)
        pv[ac] += df * prod.notional
        alive[ac] = False

    # Terminal payoff for paths never autocalled
    T = prod.maturity_years
    df_T = np.exp(-mkt.rate * T)
    s_T = paths[:, -1]
    surv = alive

    ki_hit = surv & (s_T < ki_level)
    no_ki = surv & (s_T >= ki_level)

    # No KI: return par
    pv[no_ki] += df_T * prod.notional
    # KI hit: investor takes equity performance relative to strike
    pv[ki_hit] += df_T * prod.notional * (s_T[ki_hit] / prod.strike)

    price_est = pv.mean()
    stderr = pv.std(ddof=1) / np.sqrt(n_paths_actual)
    return price_est, stderr


# ---------- Greeks via bump & revalue ----------------------------------------

def greeks(mkt: Market, prod: Autocallable, n_paths: int = 100_000):
    base, _ = price(mkt, prod, n_paths, seed=42)

    # Delta: 1% bump
    h = 0.01 * mkt.spot
    up, _ = price(Market(mkt.spot + h, mkt.rate, mkt.div_yield, mkt.vol), prod, n_paths, seed=42)
    dn, _ = price(Market(mkt.spot - h, mkt.rate, mkt.div_yield, mkt.vol), prod, n_paths, seed=42)
    delta = (up - dn) / (2 * h)
    gamma = (up - 2 * base + dn) / (h ** 2)

    # Vega: 1 vol pt
    hv = 0.01
    vu, _ = price(Market(mkt.spot, mkt.rate, mkt.div_yield, mkt.vol + hv), prod, n_paths, seed=42)
    vd, _ = price(Market(mkt.spot, mkt.rate, mkt.div_yield, mkt.vol - hv), prod, n_paths, seed=42)
    vega = (vu - vd) / (2 * hv) / 100  # per 1 vol point

    # Rho: 1bp
    hr = 0.0001
    ru, _ = price(Market(mkt.spot, mkt.rate + hr, mkt.div_yield, mkt.vol), prod, n_paths, seed=42)
    rd, _ = price(Market(mkt.spot, mkt.rate - hr, mkt.div_yield, mkt.vol), prod, n_paths, seed=42)
    rho = (ru - rd) / (2 * hr) / 10000

    return {"price": base, "delta": delta, "gamma": gamma, "vega_1vol": vega, "rho_1bp": rho}


# ---------- Scenario analysis -------------------------------------------------

def spot_ladder(mkt: Market, prod: Autocallable, shocks=(-0.3, -0.2, -0.1, 0, 0.1, 0.2)):
    print(f"\n{'Spot shock':>12} {'Spot':>8} {'PV':>10}")
    for s in shocks:
        m = Market(mkt.spot * (1 + s), mkt.rate, mkt.div_yield, mkt.vol)
        pv, _ = price(m, prod, n_paths=50_000)
        print(f"{s:>11.0%} {m.spot:>8.2f} {pv:>10.3f}")


# ---------- Main --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    live = "--live" in sys.argv

    if live:
        from market_data import get_snapshot
        snap = get_snapshot()
        print(snap.pretty(), "\n")
        mkt = Market(spot=snap.spot, rate=snap.rate,
                     div_yield=snap.div_yield, vol=snap.vol)
        # Strike fixes at today's spot (trade-date convention)
        prod = Autocallable(
            strike=snap.spot,
            maturity_years=6,
            autocall_barrier=1.00,
            coupon_barrier=0.70,
            ki_barrier=0.60,
            coupon=0.08,
        )
    else:
        mkt = Market(spot=100, rate=0.03, div_yield=0.03, vol=0.20)
        prod = Autocallable(
            maturity_years=6,
            autocall_barrier=1.00,
            coupon_barrier=0.70,
            ki_barrier=0.60,
            coupon=0.08,
        )

    pv, se = price(mkt, prod, n_paths=200_000)
    print(f"Autocallable PV: {pv:.4f}  (SE {se:.4f})")
    print(f"Par = {prod.notional} -> fair coupon should make PV = par")

    print("\nGreeks:")
    for k, v in greeks(mkt, prod, n_paths=100_000).items():
        print(f"  {k:>10}: {v: .5f}")

    spot_ladder(mkt, prod)
