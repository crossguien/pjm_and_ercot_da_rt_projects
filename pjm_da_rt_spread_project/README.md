# PJM Day-Ahead vs Real-Time Spread Model

In PJM, DA vs RT spreads reflect a combination of load forecast error, congestion realization, and intraday system conditions. This project evaluates when DA prices structurally misprice real-time outcomes, with emphasis on volatility, skew, and tail behavior rather than average spreads.

The outputs are designed to resemble a trader’s pre-DA risk review and post-trade performance attribution.

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

## Key outputs and how to interpret them

- DA vs RT spread distributions

  Identify asymmetry and tail risk that can materially impact PnL.

- Hour-of-day and day-of-week seasonality

  Reveal structural patterns in spread behavior driven by load shape and congestion.

- Recent window diagnostics

  Highlight regime shifts where historical averages may no longer apply.

For example, widening variance without a change in mean suggests higher risk per MWh even if expected spreads remain flat.

## Example desk workflow

- Review recent DA vs RT spread distributions prior to submitting virtual bids.

- Identify hubs or nodes with increasing volatility or skew.

- Scale exposure based on tail risk rather than expected value alone.

- Use post-settlement results to recalibrate assumptions around congestion and load error.

## Assumptions and limitations

- Live mode requires a PJM API key and depends on public data availability.

- Offline mode uses bundled sample data for fast demos.

- Does not forecast individual constraints or outages.

- Intended for risk evaluation and scenario framing, not deterministic price forecasting.

## Production extensions

- Incorporate constraint-level congestion indicators.

- Integrate probabilistic load and weather inputs.

- Add automated reporting for DA position review meetings.

- Expand to nodal-level analysis for congestion-sensitive strategies.

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If you want a specific hub or node, start by running the script once and inspect the returned location names in the raw data.
