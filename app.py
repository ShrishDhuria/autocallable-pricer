"""
app.py — Streamlit dashboard for the worst-of autocallable pricer.

Run locally:
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from worstof_pricer import (
    Asset, MultiMarket, WorstOfAutocallable, price, greeks, fair_coupon,
)
from heston import HestonParams

st.set_page_config(page_title="Worst-of Autocallable Pricer", layout="wide")
st.title("Worst-of Phoenix Autocallable")
st.caption("Monte Carlo pricer with Heston stochastic vol — SX5E / SPX / NKY")

PRODUCT_EXPLAINER = """
This note pays an attractive coupon in exchange for the investor taking equity
downside on the **worst-performing** of several underlyings. Everything keys off
the *worst* performer at each annual observation, where each underlying's
performance is its level relative to its strike (the level fixed at trade date):

**performance = spot / strike**, and **worst = min(performance) across all underlyings.**

- **Autocall.** If the worst performer is at or above the *autocall barrier*
  (e.g. 100% of strike) on an observation date, the note redeems early at par
  plus any coupons due. Most notes end this way, early and at par.
- **Coupon (with memory).** If the worst performer is at or above the *coupon
  barrier* (e.g. 70%), a coupon is paid. Missed coupons are remembered and paid
  in full on the next date the barrier is met.
- **Knock-in at maturity.** If the note never autocalled and the worst performer
  finishes below the *knock-in barrier* (e.g. 60%), the investor takes the
  equity loss of that worst performer. Otherwise par is returned.

Because the payoff always tracks the *minimum*, the note is short correlation
(less correlation → more dispersion → a lower worst performer → more knock-in
risk) and short volatility (it has effectively sold a down-and-in put). The
coupon is the price of bearing those risks.
"""

with st.expander("How this product works", expanded=False):
    st.markdown(PRODUCT_EXPLAINER)

# ---------- Sidebar: market inputs -------------------------------------------

st.sidebar.header("Market data")
use_live = st.sidebar.checkbox("Fetch live snapshot", value=False)

default_assets = [
    {"name": "SX5E", "spot": 100.0, "vol": 0.20, "div_yield": 0.030},
    {"name": "SPX",  "spot": 100.0, "vol": 0.18, "div_yield": 0.015},
    {"name": "NKY",  "spot": 100.0, "vol": 0.22, "div_yield": 0.020},
]
default_rate = 0.03
default_corr = np.array([[1.0, 0.65, 0.55],
                         [0.65, 1.0, 0.50],
                         [0.55, 0.50, 1.0]])

if use_live:
    try:
        from market_data import get_multi_snapshot
        snap = get_multi_snapshot()
        default_assets = snap.assets
        default_rate = snap.rate
        default_corr = snap.corr
        st.sidebar.success(f"Snapshot @ {snap.as_of:%Y-%m-%d %H:%M UTC}")
    except Exception as e:
        st.sidebar.error(f"Live fetch failed: {e}")

assets_input = []
strikes_input = []
for a in default_assets:
    st.sidebar.subheader(a["name"])
    spot = st.sidebar.number_input(f"{a['name']} spot (current)", value=float(a["spot"]),
                                    key=f"s_{a['name']}", min_value=0.01)
    strike = st.sidebar.number_input(
        f"{a['name']} strike (fixing)", value=float(a["spot"]),
        key=f"k_{a['name']}", min_value=0.01,
        help="Level fixed at trade date. Leave equal to spot for a fresh trade; "
             "set spot away from strike to price a note already part-way through its life.")
    vol = st.sidebar.slider(f"{a['name']} vol", 0.05, 0.80, float(a["vol"]),
                             0.01, key=f"v_{a['name']}")
    dy = st.sidebar.slider(f"{a['name']} div yield", 0.0, 0.10,
                            float(a["div_yield"]), 0.005, key=f"d_{a['name']}")
    assets_input.append(Asset(a["name"], spot, vol, dy))
    strikes_input.append(strike)

rate = st.sidebar.slider("EUR rate", -0.01, 0.08, float(default_rate), 0.0025)

st.sidebar.subheader("Correlation")
corr_edit = st.sidebar.data_editor(
    pd.DataFrame(default_corr,
                 index=[a.name for a in assets_input],
                 columns=[a.name for a in assets_input]),
    key="corr_edit",
)
corr = corr_edit.values

# ---------- Sidebar: product --------------------------------------------------

st.sidebar.header("Product terms")
maturity = st.sidebar.slider("Maturity (years)", 1, 10, 6)
ac_b = st.sidebar.slider("Autocall barrier", 0.5, 1.2, 1.00, 0.05)
cp_b = st.sidebar.slider("Coupon barrier", 0.3, 1.0, 0.70, 0.05)
ki_b = st.sidebar.slider("KI barrier", 0.3, 0.9, 0.60, 0.05)
coupon = st.sidebar.slider("Coupon (annual)", 0.0, 0.30, 0.08, 0.005)

# ---------- Sidebar: model ----------------------------------------------------

st.sidebar.header("Model")
model = st.sidebar.radio(
    "Dynamics",
    ["GBM (flat vol)", "Heston (calibrated)", "Heston (manual)"],
)
n_paths = st.sidebar.select_slider("MC paths", [10_000, 50_000, 100_000, 200_000],
                                    value=50_000)

heston_params = None
if model == "Heston (calibrated)":
    from calibrate_heston import load_calibrated_params
    calib = load_calibrated_params()
    if not calib:
        st.sidebar.error("No calibration/heston_params.json found. "
                         "Run: python calibrate_heston.py --all")
    else:
        heston_params = []
        for a in assets_input:
            p = calib.get(a.name) or next(iter(calib.values()))
            heston_params.append(p)
        st.sidebar.success(f"Loaded params for {', '.join(a.name for a in assets_input)}")
        with st.sidebar.expander("Calibrated params"):
            for a, p in zip(assets_input, heston_params):
                st.caption(f"{a.name}: v0={p.v0:.3f}, κ={p.kappa:.2f}, "
                           f"θ={p.theta:.3f}, ξ={p.xi:.2f}, ρ={p.rho:.2f}")

elif model == "Heston (manual)":
    st.sidebar.subheader("Heston params (shared)")
    v0 = st.sidebar.slider("v0", 0.001, 0.20, 0.04, 0.005)
    kappa = st.sidebar.slider("kappa", 0.1, 10.0, 2.0, 0.1)
    theta = st.sidebar.slider("theta", 0.001, 0.20, 0.04, 0.005)
    xi = st.sidebar.slider("xi (vol of vol)", 0.05, 2.0, 0.5, 0.05)
    rho = st.sidebar.slider("rho", -0.99, 0.0, -0.7, 0.05)
    hp = HestonParams(v0, kappa, theta, xi, rho)
    heston_params = [hp] * len(assets_input)
    if not hp.feller_ok():
        st.sidebar.warning("Feller condition violated (2κθ < ξ²)")

# ---------- Build market & product -------------------------------------------

try:
    mkt = MultiMarket(assets_input, rate, corr)
    np.linalg.cholesky(corr)
except np.linalg.LinAlgError:
    st.error("Correlation matrix is not positive semi-definite.")
    st.stop()

prod = WorstOfAutocallable(
    strikes=strikes_input,
    maturity_years=maturity,
    autocall_barrier=ac_b,
    coupon_barrier=cp_b,
    ki_barrier=ki_b,
    coupon=coupon,
)

# ---------- Compute ----------------------------------------------------------

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Pricing")
    with st.spinner("Pricing..."):
        pv, se = price(mkt, prod, n_paths=n_paths, heston_params=heston_params)
    st.metric("PV", f"{pv:.3f}", f"vs par {pv - 100:+.2f}")
    st.caption(f"Standard error: ±{se:.3f}  ({n_paths:,} paths)")

    if st.button("Solve fair coupon"):
        with st.spinner("Root-finding..."):
            try:
                fc = fair_coupon(mkt, prod, n_paths=n_paths,
                                 heston_params=heston_params)
                st.success(f"Fair coupon: {fc:.4%}")
                st.caption(f"vs current {coupon:.4%} → "
                           f"{(fc - coupon)*1e4:+.0f} bps")
            except ValueError as e:
                st.error(str(e))

with col2:
    st.subheader("Greeks")
    with st.spinner("Computing Greeks..."):
        g = greeks(mkt, prod, n_paths=n_paths, heston_params=heston_params)
    g_df = pd.DataFrame(
        [{"Underlying": a.name,
          "Delta": g[f"delta_{a.name}"],
          "Gamma": g[f"gamma_{a.name}"],
          "Vega (1 vol pt)": g[f"vega_{a.name}_1vol"]}
         for a in assets_input]
    )
    st.dataframe(g_df.style.format({
        "Delta": "{:.4f}", "Gamma": "{:.5f}", "Vega (1 vol pt)": "{:.4f}"
    }), hide_index=True, use_container_width=True)

# ---------- Worst-of performance panel ---------------------------------------

st.subheader("Worst-of driver — performance by leg")
st.caption("Each leg's performance is spot / strike. The note tracks the **minimum** "
           "(highlighted). Barriers are shown as reference lines; the worst performer "
           "is the one that decides autocall, coupon, and knock-in.")

import altair as alt

perf_rows = []
for a, k in zip(assets_input, strikes_input):
    perf_rows.append({"Underlying": a.name, "Performance": a.spot / k, "Vol": a.vol})
perf_df = pd.DataFrame(perf_rows)
worst_name = perf_df.loc[perf_df["Performance"].idxmin(), "Underlying"]
perf_df["role"] = np.where(perf_df["Underlying"] == worst_name, "worst-of", "other")

pcol1, pcol2 = st.columns([2, 1])

with pcol1:
    bars = alt.Chart(perf_df).mark_bar().encode(
        x=alt.X("Underlying:N", sort=None),
        y=alt.Y("Performance:Q", scale=alt.Scale(zero=False),
                axis=alt.Axis(format="%", title="Performance (spot / strike)")),
        color=alt.Color("role:N",
                        scale=alt.Scale(domain=["worst-of", "other"],
                                        range=["#d62728", "#4c78a8"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("Underlying:N"),
                 alt.Tooltip("Performance:Q", format=".1%"),
                 alt.Tooltip("Vol:Q", format=".1%")],
    )
    rules = alt.Chart(pd.DataFrame({
        "level": [ac_b, cp_b, ki_b],
        "label": ["autocall", "coupon", "knock-in"],
    })).mark_rule(strokeDash=[4, 4], color="grey").encode(
        y="level:Q",
        tooltip=["label:N", alt.Tooltip("level:Q", format=".0%")],
    )
    st.altair_chart(bars + rules, use_container_width=True)

with pcol2:
    show = perf_df[["Underlying", "Performance"]].copy()
    show["Performance"] = show["Performance"].map("{:.1%}".format)
    st.dataframe(
        show.style.apply(
            lambda r: ["background-color: #ffd9d9" if r["Underlying"] == worst_name
                       else "" for _ in r], axis=1),
        hide_index=True, use_container_width=True)
    st.caption(f"Worst-of: **{worst_name}**")
    if perf_df["Performance"].nunique() == 1:
        hv = perf_df.loc[perf_df["Vol"].idxmax(), "Underlying"]
        st.caption(f"All legs at the same performance (fresh trade). The "
                   f"highest-vol leg (**{hv}**) carries the most knock-in risk, "
                   f"as it is most likely to become the worst performer.")

# ---------- Spot ladder ------------------------------------------------------

st.subheader("Worst-spot scenario ladder")
shocks = np.array([-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20])
ladder = []
for s in shocks:
    bumped = [Asset(a.name, a.spot * (1 + s), a.vol, a.div_yield)
              for a in assets_input]
    pv_s, _ = price(MultiMarket(bumped, rate, corr), prod,
                    n_paths=20_000, heston_params=heston_params)
    ladder.append({"Shock": f"{s:+.0%}", "PV": pv_s})
ladder_df = pd.DataFrame(ladder)
st.line_chart(ladder_df.set_index("Shock"), height=250)

# ---------- Correlation ladder -----------------------------------------------

st.subheader("Average-correlation sensitivity")
rhos = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n = len(assets_input)
corr_pvs = []
for r_ in rhos:
    C = np.full((n, n), r_); np.fill_diagonal(C, 1.0)
    pv_c, _ = price(MultiMarket(assets_input, rate, C), prod,
                    n_paths=20_000, heston_params=heston_params)
    corr_pvs.append({"Avg corr": r_, "PV": pv_c})
st.line_chart(pd.DataFrame(corr_pvs).set_index("Avg corr"), height=250)
