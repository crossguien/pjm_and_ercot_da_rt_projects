# PJM Day-Ahead vs Real-Time Spread Model

This repo builds a desk-style analysis to understand **DA vs RT spread behavior** using **public PJM data**.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt

# Run modes
# - Online: pulls live PJM data (requires PJM_API_KEY)
# - Offline: uses bundled sample data (fast, always works for demos)
# - Fallback: tries live data, then falls back to sample data if fetch fails or API key is missing
# - Demo: skips training/plots for a very fast sanity check

# Online with API key
export PJM_API_KEY="your_key_here"
python src/main.py --node "PJM RTO" --days 60 --outdir outputs

# Offline demo (uses bundled sample data)
python src/main.py --node "PJM RTO" --mode offline --outdir outputs

# Fast demo (skips training/plots)
python src/main.py --node "PJM RTO" --mode offline --demo --outdir outputs

# Online with automatic fallback to sample data if live fetch fails
python src/main.py --node "PJM RTO" --days 60 --fallback-sample --outdir outputs
```

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If you want a specific hub or node, start by running the script once and inspect the returned location names in the raw data.
