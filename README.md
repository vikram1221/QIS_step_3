# QIS Step 3 — Magnificent 7 Momentum Strategy

This project implements a cross-sectional momentum strategy (12–1) on the Magnificent 7 stocks. Signals are constructed using rolling returns, standardized cross-sectionally, and evaluated using performance, information coefficient (IC), and turnover diagnostics.

## Project Structure & Reproducibility

The research pipeline is designed with a clear separation between **research computation** and
**public presentation**.

## Research Code
The core research logic is implemented in Python and is intended to be run locally.

To reproduce all results:

1. Create a `research/` folder
2. Place the Python research script inside it
3. Run the script end-to-end

Running the script automatically:
- downloads price data
- constructs signals
- computes portfolio returns
- generates diagnostics
- exports the final interactive dashboard as `index.html`

## Folder Structure

```text
QIS_step_3/
├── research/
│   └── mag7_prices.py        # Full research pipeline (run locally)
│
├── data/
│   ├── raw/                 # Raw price data (auto-generated)
│   └── processed/           # Cleaned prices, returns, signals (auto-generated)
│
├── output/
│   ├── figures/
│   │   └── index.html       # Exported interactive dashboard
│   └── metrics/             # Portfolio PnL, IC, turnover diagnostics
│
├── index.html               # Deployed dashboard (GitHub Pages)
└── README.md
