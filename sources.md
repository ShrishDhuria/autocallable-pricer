# Data Sources

| Data | Source | Method | Notes |
|---|---|---|---|
| Spot levels (SX5E, SPX, NKY) | Yahoo Finance | `yfinance.Ticker(...).history()` | `^STOXX50E`, `^GSPC`, `^N225`; last close |
| Realized volatility | Yahoo Finance | 30-day close-to-close, annualised | Vol proxy; VSTOXX (`^V2TX`) is delisted from Yahoo |
| Cross-asset correlation | Yahoo Finance | 1y daily log returns, date-aligned | Index normalised to date to align EU/US/JP closes |
| Risk-free rate (€STR) | ECB Data Portal | REST API, dataset `EST.B.EU000A2X2A25.WT` | Reported as % p.a.; converted to decimal |
| Dividend yields | Per-index defaults | SX5E 3.0%, SPX 1.3%, NKY 1.8% | Index tickers don't expose a yield via yfinance |
| SPX vol surface (calibration) | Yahoo Finance | SPY option chain, two-sided liquid quotes only | Live, automatic; requires US market hours |
| SX5E vol surface (calibration) | Eurex OESX settlements | Manual CSV (`expiry_date, strike, call_settlement`) | Optional; falls back to a vol-scaled SPX proxy |
| Heston pricing | Heston (1993); Lewis (2001) | Characteristic-function call via Lewis integral | Full-truncation Euler for the Monte Carlo |
| Autocallable structure | Bouzoubaa & Osseiran (2010) | Standard Phoenix worst-of conventions | SG/BNP-style French retail terms |

## Methodology references
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*.
- Lewis, A. (2001). "A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes." Working paper.
- Bouzoubaa, M. & Osseiran, A. (2010). *Exotic Options and Hybrids*. Wiley.

## Calibration note
S&P 500 is calibrated live to the SPY option chain because SPY is the most
liquid options market in the world and freely accessible. Euro Stoxx 50 uses
the same code path with a Eurex settlement CSV; absent that, it inherits S&P
500's skew shape with its variance level rescaled to its own realized vol.
Production calibration would use a licensed Eurex feed for SX5E directly.

Note: index compositions and listed-option chains change; verify on each refresh.
