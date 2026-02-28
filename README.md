# DA/RT Power Trading Projects (PJM + ERCOT)

This repository contains two market-focused projects for day-ahead (DA) vs real-time (RT) spread analysis and trading research:

- `pjm_da_rt_spread_project`: PJM DA/RT spread modeling plus virtuals strategy backtest with risk layer
- `ercot_da_rt_spread_project`: ERCOT DA/RT spread modeling baseline

## Why this repo

The goal is to show practical desk-style analytics relevant to power trading roles:

- Feature engineering for DA/RT spread behavior
- Predictive baseline modeling
- Strategy rules and risk-aware position sizing (PJM)
- Backtest metrics such as PnL, Sharpe, max drawdown, hit rate, and regime performance (PJM)

## Project Docs

- PJM details: [pjm_da_rt_spread_project/README.md](pjm_da_rt_spread_project/README.md)
- ERCOT details: [ercot_da_rt_spread_project/README.md](ercot_da_rt_spread_project/README.md)

## Quick Start

### PJM (offline demo, no API key required)

```bash
cd pjm_da_rt_spread_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --node "PJM RTO" --mode offline --outdir outputs
```

### PJM (online with fallback)

```bash
cd pjm_da_rt_spread_project
source .venv/bin/activate
export PJM_API_KEY="your_key_here"
python src/main.py --node "PJM RTO" --days 60 --fallback-sample --outdir outputs
```

### ERCOT (offline demo)

```bash
cd ercot_da_rt_spread_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --node "HB_HOUSTON" --mode offline --outdir outputs
```

## Notes

- Generated artifacts are intentionally ignored at repo root to keep version control focused on source code and documentation.
