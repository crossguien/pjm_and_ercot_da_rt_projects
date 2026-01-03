# ERCOT Day-Ahead vs Real-Time Spread Model

This repo builds a desk-style analysis to understand **DA vs RT spread behavior** using **public ERCOT data**.

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r requirements.txt

python src/main.py --node "HB_HOUSTON" --days 60 --outdir outputs
```

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If load forecast endpoints are unavailable in your environment, the script will still run and model spreads using price and time features.
