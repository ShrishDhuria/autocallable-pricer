"""Runs the worst-of pricer on live data and writes a dated snapshot file."""
from datetime import datetime, timezone
from pathlib import Path

from market_data import get_multi_snapshot
from worstof_pricer import (
    Asset, MultiMarket, WorstOfAutocallable, price, greeks,
)


def main():
    snap = get_multi_snapshot()
    assets = [Asset(a["name"], a["spot"], a["vol"], a["div_yield"])
              for a in snap.assets]
    mkt = MultiMarket(assets, snap.rate, snap.corr)
    prod = WorstOfAutocallable(
        strikes=[a.spot for a in assets],
        maturity_years=6,
        coupon=0.08,
    )

    pv, se = price(mkt, prod, n_paths=200_000)
    g = greeks(mkt, prod, n_paths=100_000)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path("snapshots") / f"{date}.txt"
    out.parent.mkdir(exist_ok=True)

    lines = [
        snap.pretty(),
        "",
        f"PV: {pv:.4f}  (SE {se:.4f})",
        "",
        "Greeks:",
    ]
    lines += [f"  {k:>20}: {v: .5f}" for k, v in g.items()]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
