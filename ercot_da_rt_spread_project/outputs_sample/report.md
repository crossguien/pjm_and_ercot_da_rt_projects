# ERCOT DA vs RT Spread Report (HB_HOUSTON)

## Data window
- Days: 60
- Rows (hourly): 24

## Spread stats (DA - RT, $/MWh)
- p01: -0.14
- p05: -0.09
- p50: 0.42
- p95: 0.94
- p99: 0.99

## Baseline model (RandomForest)
- MAE: 0.60
- R2: -71.897
- Train rows: 19
- Test rows: 5
- Saved model: models/rf_da_rt_spread_hb_houston.joblib

## Figures
- spread_timeseries: figures/spread_timeseries.png
- spread_hist: figures/spread_hist.png
- error_vs_spread: figures/error_vs_spread.png

## Resume bullet template
- Built a DA vs RT spread model using public market data, engineered load forecast error features, and trained a baseline model to identify when DA pricing underestimates RT risk.