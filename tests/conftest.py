"""Shared fixtures for the autocallable pricer test-suite.

The project modules (``heston``, ``worstof_pricer``, ``autocallable_pricer``)
live at the project root, so we put that root on ``sys.path`` here rather than
relying on an editable install.

All fixtures are fully synthetic and well-conditioned. Nothing here touches the
network — these tests exercise the *maths* (pricing identities, Monte Carlo
convergence, Greek signs), not the live data layer.
"""
import os
import sys

import numpy as np
import pytest

# --- make the project root importable -------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from heston import HestonParams                       # noqa: E402
from worstof_pricer import Asset, MultiMarket, WorstOfAutocallable  # noqa: E402


@pytest.fixture(scope="session")
def heston_params():
    """A single, sane Heston parameter set (negative skew, moderate vol-of-vol)."""
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)


@pytest.fixture(scope="session")
def market():
    """A 3-asset worst-of market with a PSD correlation matrix.

    Three identical 100-spot underlyings differing only in vol/div, plus a
    well-conditioned correlation matrix. The autocallable struck at the money.
    """
    assets = [
        Asset("AAA", 100.0, 0.20, 0.03),
        Asset("BBB", 100.0, 0.18, 0.02),
        Asset("CCC", 100.0, 0.22, 0.02),
    ]
    corr = np.array([
        [1.00, 0.60, 0.50],
        [0.60, 1.00, 0.45],
        [0.50, 0.45, 1.00],
    ])
    mkt = MultiMarket(assets, rate=0.03, corr=corr)
    prod = WorstOfAutocallable(
        strikes=[100.0, 100.0, 100.0],
        maturity_years=6,
        autocall_barrier=1.00,
        coupon_barrier=0.70,
        ki_barrier=0.60,
        coupon=0.08,
    )
    return {"mkt": mkt, "prod": prod}


@pytest.fixture(scope="session")
def vanilla():
    """Inputs for single-name vanilla / Heston-vs-BS tests."""
    return {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.03, "q": 0.0}
