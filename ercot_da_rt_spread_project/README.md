# ERCOT Day-Ahead vs Real-Time Spread Model

This repo builds a desk-style analysis to understand **DA vs RT spread behavior** using **public ERCOT data**.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt

# Run modes
# - Online: pulls live ERCOT data (slower, depends on data availability)
# - Offline: uses bundled sample data (fast, always works for demos)
# - Fallback: tries live data, then falls back to sample data if fetch fails
# - Demo: skips training/plots for a very fast sanity check

python src/main.py --node "HB_HOUSTON" --days 60 --outdir outputs

# Offline demo (uses bundled sample data)
python src/main.py --node "HB_HOUSTON" --mode offline --outdir outputs

# Fast demo (skips training/plots)
python src/main.py --node "HB_HOUSTON" --mode offline --demo --outdir outputs

# Online with automatic fallback to sample data if live fetch fails
python src/main.py --node "HB_HOUSTON" --days 60 --fallback-sample --outdir outputs
```

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If load forecast endpoints are unavailable in your environment, the script will still run and model spreads using price and time features.
