"""Property tests for the worst-of autocallable pricer.

These assert the *mathematical* facts a reviewer would probe in an interview,
not that functions merely return floats:

  - Black-Scholes implied vol inverts the BS price (round-trip);
  - the Heston characteristic-function call collapses onto Black-Scholes as
    the vol-of-vol xi -> 0 (with v0 = theta = sigma^2, rho = 0);
  - the Heston Monte Carlo converges to the characteristic-function price;
  - worst-of PV is monotonically increasing in correlation (and equals the
    single-asset PV in the perfectly-correlated, identical-asset limit);
  - autocallable PV is monotonically increasing in the coupon;
  - the fair-coupon solver round-trips: PV at the solved coupon equals par;
  - Greek signs are correct under GBM (delta > 0, vega < 0 — short vol);
  - the Cholesky factor reproduces the correlation matrix (PSD);
  - Monte Carlo is reproducible under a fixed seed.

Floating-point comparisons use ``pytest.approx`` (scalars) or
``np.testing.assert_allclose`` (arrays); ``==`` is never used on floats.
"""
import numpy as np
import pytest

from heston import HestonParams, call_price, bs_call, implied_vol, simulate
from worstof_pricer import (
    Asset, MultiMarket, WorstOfAutocallable, price, greeks, fair_coupon,
)


# --------------------------------------------------------------------------
# Black-Scholes / implied vol
# --------------------------------------------------------------------------
def test_implied_vol_inverts_bs_call(vanilla):
    """implied_vol(bs_call(sigma)) must recover sigma."""
    v = vanilla
    for sigma in (0.10, 0.20, 0.35, 0.50):
        px = bs_call(v["S0"], v["K"], v["T"], v["r"], v["q"], sigma)
        iv = implied_vol(px, v["S0"], v["K"], v["T"], v["r"], v["q"])
        assert iv == pytest.approx(sigma, abs=1e-4)


# --------------------------------------------------------------------------
# Heston -> Black-Scholes limit
# --------------------------------------------------------------------------
def test_heston_reduces_to_bs(vanilla):
    """As xi -> 0 with v0 = theta = sigma^2 and rho = 0, the Heston call
    must collapse onto the Black-Scholes call at vol sigma."""
    v = vanilla
    sigma = 0.20
    p = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=1e-4, rho=0.0)
    c_heston = call_price(v["S0"], v["K"], v["T"], v["r"], v["q"], p)
    c_bs = bs_call(v["S0"], v["K"], v["T"], v["r"], v["q"], sigma)
    assert c_heston == pytest.approx(c_bs, abs=1e-2)


# --------------------------------------------------------------------------
# Heston Monte Carlo convergence
# --------------------------------------------------------------------------
def test_heston_mc_converges_to_charfn(vanilla, heston_params):
    """Vanilla call via the Heston MC must match the characteristic-function
    price within Monte Carlo error."""
    v = vanilla
    p = heston_params
    S = simulate(v["S0"], v["r"], v["q"], p, T=v["T"],
                 n_steps=200, n_paths=200_000, seed=7)
    payoff = np.maximum(S[:, -1] - v["K"], 0.0)
    disc = np.exp(-v["r"] * v["T"])
    mc = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
    ref = call_price(v["S0"], v["K"], v["T"], v["r"], v["q"], p)
    # within 4 standard errors plus a small discretisation allowance
    assert abs(mc - ref) < 4 * se + 0.05 * ref


# --------------------------------------------------------------------------
# Worst-of structure
# --------------------------------------------------------------------------
def test_worstof_pv_rises_with_correlation(market):
    """Higher average correlation -> less worst-of dispersion -> higher PV."""
    prod = market["prod"]
    assets = market["mkt"].assets
    n = len(assets)

    def pv_at_corr(rho):
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1.0)
        m = MultiMarket(assets, market["mkt"].rate, C)
        return price(m, prod, n_paths=40_000, seed=1)[0]

    pv_low = pv_at_corr(0.30)
    pv_high = pv_at_corr(0.90)
    assert pv_high > pv_low


def test_perfect_correlation_matches_single_asset():
    """Three identical assets at correlation ~1 reduce to a single-asset note."""
    assets = [Asset(f"X{i}", 100.0, 0.20, 0.03) for i in range(3)]
    eps = 1e-6
    C = np.full((3, 3), 1.0 - eps)
    np.fill_diagonal(C, 1.0)
    prod = WorstOfAutocallable(strikes=[100.0] * 3, coupon=0.08)

    pv_worstof = price(MultiMarket(assets, 0.03, C), prod, n_paths=40_000, seed=3)[0]

    single = [Asset("X", 100.0, 0.20, 0.03)]
    prod1 = WorstOfAutocallable(strikes=[100.0], coupon=0.08)
    pv_single = price(MultiMarket(single, 0.03, np.array([[1.0]])),
                      prod1, n_paths=40_000, seed=3)[0]

    assert pv_worstof == pytest.approx(pv_single, abs=1.0)


# --------------------------------------------------------------------------
# Coupon monotonicity and fair coupon
# --------------------------------------------------------------------------
def test_pv_monotonic_in_coupon(market):
    """PV must strictly increase with the coupon."""
    mkt = market["mkt"]
    from dataclasses import replace
    base = market["prod"]
    pvs = [price(mkt, replace(base, coupon=c), n_paths=40_000, seed=2)[0]
           for c in (0.02, 0.08, 0.15)]
    assert pvs[0] < pvs[1] < pvs[2]


def test_fair_coupon_round_trips(market):
    """PV at the solved fair coupon must equal par."""
    from dataclasses import replace
    mkt, prod = market["mkt"], market["prod"]
    c = fair_coupon(mkt, prod, n_paths=40_000, seed=5)
    pv = price(mkt, replace(prod, coupon=c), n_paths=40_000, seed=5)[0]
    assert pv == pytest.approx(prod.notional, abs=0.5)


# --------------------------------------------------------------------------
# Greeks
# --------------------------------------------------------------------------
def test_greek_signs_gbm(market):
    """Under GBM: delta > 0 on every leg, vega < 0 (the note is short vol)."""
    g = greeks(market["mkt"], market["prod"], n_paths=60_000)
    for a in market["mkt"].assets:
        assert g[f"delta_{a.name}"] > 0
        assert g[f"vega_{a.name}_1vol"] < 0


# --------------------------------------------------------------------------
# Numerical hygiene
# --------------------------------------------------------------------------
def test_cholesky_reproduces_correlation(market):
    """L @ L.T must reconstruct the correlation matrix (PSD check)."""
    C = market["mkt"].corr
    L = np.linalg.cholesky(C)
    np.testing.assert_allclose(L @ L.T, C, rtol=1e-10, atol=1e-12)


def test_seed_determinism(market):
    """Same seed -> identical PV (antithetic MC is reproducible)."""
    mkt, prod = market["mkt"], market["prod"]
    pv1 = price(mkt, prod, n_paths=20_000, seed=99)[0]
    pv2 = price(mkt, prod, n_paths=20_000, seed=99)[0]
    assert pv1 == pytest.approx(pv2, abs=1e-9)
