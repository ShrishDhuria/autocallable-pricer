"""Regenerate the README results figure from the pricer (offline, no network).

Builds the standard 3-asset worst-of autocallable on fixed synthetic inputs
and a fixed Heston parameter set, then plots the two headline results:

  Left  — the fair coupon under flat-vol GBM vs Heston (the skew premium the
          desk earns by pricing off the surface rather than a flat vol).
  Right — worst-of PV as a function of average correlation (the dominant
          risk axis of a worst-of structure).

    pip install matplotlib
    python make_figures.py        # writes docs/autocallable_results.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heston import HestonParams
from worstof_pricer import (
    Asset, MultiMarket, WorstOfAutocallable, price, fair_coupon,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)


def build():
    assets = [
        Asset("SX5E", 100.0, 0.20, 0.030),
        Asset("SPX",  100.0, 0.18, 0.013),
        Asset("NKY",  100.0, 0.22, 0.018),
    ]
    corr = np.array([
        [1.00, 0.60, 0.50],
        [0.60, 1.00, 0.45],
        [0.50, 0.45, 1.00],
    ])
    mkt = MultiMarket(assets, rate=0.03, corr=corr)
    prod = WorstOfAutocallable(strikes=[100.0] * 3, maturity_years=6, coupon=0.08)
    hp = [HestonParams(0.04, 2.0, 0.04, 0.5, -0.7)] * 3
    return mkt, prod, hp


def main():
    mkt, prod, hp = build()
    n = mkt.n

    # Left: fair coupon GBM vs Heston
    c_gbm = fair_coupon(mkt, prod, n_paths=60_000, seed=11)
    c_hes = fair_coupon(mkt, prod, n_paths=60_000, seed=11, heston_params=hp)
    premium_bps = (c_hes - c_gbm) * 1e4

    # Right: PV vs average correlation (GBM)
    rhos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pvs = []
    for rho in rhos:
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1.0)
        pvs.append(price(MultiMarket(mkt.assets, mkt.rate, C), prod,
                         n_paths=40_000, seed=11)[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(["GBM\n(flat vol)", "Heston\n(skew)"],
                [c_gbm * 100, c_hes * 100],
                color=["#888888", "#1f77b4"])
    axes[0].set_ylabel("Fair coupon (% p.a.)")
    axes[0].set_title(f"Fair coupon: flat vol vs Heston\nskew premium ≈ {premium_bps:+.0f} bps")
    for i, c in enumerate([c_gbm, c_hes]):
        axes[0].text(i, c * 100, f"{c*100:.2f}%", ha="center", va="bottom")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(rhos, pvs, "o-", color="#1f77b4")
    axes[1].axhline(prod.notional, color="grey", ls="--", lw=0.8, label="par")
    axes[1].set_xlabel("Average pairwise correlation")
    axes[1].set_ylabel("PV")
    axes[1].set_title("Worst-of PV vs correlation\n(higher correlation → less dispersion → higher PV)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(DOCS, "autocallable_results.png")
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")
    print(f"  GBM fair coupon    : {c_gbm:.4%}")
    print(f"  Heston fair coupon : {c_hes:.4%}")
    print(f"  skew premium       : {premium_bps:+.0f} bps")


if __name__ == "__main__":
    main()
