# PJM Day-Ahead vs Real-Time Spread Model

This repo builds a desk-style analysis to understand **DA vs RT spread behavior** using **public PJM data**.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt

python src/main.py --node "PJM RTO" --days 60 --outdir outputs
```

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If you want a specific hub or node, start by running the script once and inspect the returned location names in the raw data.
