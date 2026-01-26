# ERCOT Day-Ahead vs Real-Time Spread Model

Day-Ahead vs Real-Time spreads are a core driver of discretionary and systematic PnL in ERCOT, where load uncertainty, weather volatility, and scarcity pricing create frequent DA mispricing. This project analyzes when DA prices fail to fully reflect real-time risk, focusing on distributional outcomes rather than point forecasts.

The analysis mirrors how ERCOT desks evaluate spread risk prior to submitting DA positions and how outcomes are reviewed post-settlement.

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
## Key outputs and how to interpret them

* DA vs RT spread distributions
  Used to evaluate skew and tail risk, particularly around scarcity events.

* Hourly and daily seasonality diagnostics
  Highlight periods where DA prices systematically underprice RT volatility, often during peak load or ramp hours.

* Spread summary tables
  Quantify mean, variance, and tail behavior to support position sizing and risk limits.

For example, persistent positive RT-DA skew during summer peak hours indicates upside scarcity risk not fully priced in DA, favoring conservative DA offers or long RT exposure.

## Example desk workflow

- Run the model ahead of the DA close to review recent spread behavior and tail risk.

- Identify hours with asymmetric upside or downside exposure.

- Adjust DA bids or virtual exposure based on forecast uncertainty, not point estimates.

- Post-settlement, review realized spreads versus expected distributions to refine risk assumptions.

## Assumptions and limitations

- Uses publicly available ERCOT market data.

- Offline mode relies on bundled sample data for demonstration purposes.

- Does not incorporate proprietary outage, telemetry, or unit-level data.

- Outputs are intended for risk framing and relative comparisons, not absolute price prediction.

## Production extensions

* Integrate probabilistic load forecasts and weather ensembles.

* Add scarcity and ORDC-specific features.

* Automate daily runs with alerts on regime shifts in spread behavior.

* Extend from hub-level analysis to zonal or nodal resolution.

## Data source
Pulled from public market data via the open-source `gridstatus` library.
If load forecast endpoints are unavailable in your environment, the script will still run and model spreads using price and time features.
