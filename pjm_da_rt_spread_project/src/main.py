"""
PJM DA vs RT Spread Project
- Downloads PJM day-ahead and real-time LMP plus load (where available)
- Engineers trading-relevant features
- Trains a baseline model to predict DA-RT spread
- Saves datasets, figures, and a short report

Run:
  python src/main.py --node "PJM RTO" --days 90 --outdir outputs
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

try:
    from gridstatus import PJM
except Exception as e:
    raise SystemExit(
        "Missing dependency gridstatus. Install with: pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

UTC = timezone.utc


@dataclass
class Config:
    node: str
    days: int
    outdir: str
    seed: int = 7
    end_date: str | None = None


def ensure_dirs(outdir: str) -> dict:
    paths = {
        "root": outdir,
        "data": os.path.join(outdir, "data"),
        "fig": os.path.join(outdir, "figures"),
        "models": os.path.join(outdir, "models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def utc_date_range(days: int, end_date: str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp(datetime.now(tz=UTC).date())
    start = end - pd.Timedelta(days=days)
    return start, end


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def download_prices(iso, start: pd.Timestamp, end: pd.Timestamp, node: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _call_get_lmp(**kwargs) -> pd.DataFrame:
        try:
            df = iso.get_lmp(**kwargs)
            return _normalize_cols(df)
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                raise ValueError(
                    f"No LMP data returned for market '{kwargs.get('market', 'unknown')}' "
                    f"between {start.date()} and {end.date()}. "
                    "This usually means the date range is in the future or data is unavailable. "
                    "Try a smaller --days window or set an explicit --end-date (YYYY-MM-DD)."
                ) from e
            raise

    def _fetch_market(market_name: str) -> pd.DataFrame:
        try:
            return _call_get_lmp(date=start, end=end, market=market_name)
        except TypeError as e:
            if "unexpected keyword argument 'market'" not in str(e):
                raise
            df_all = _call_get_lmp(date=start, end=end)
            if "market" not in df_all.columns:
                raise ValueError("LMP data missing 'market' column; cannot split DA/RT")

            def _norm_val(val: str) -> str:
                return str(val).lower().replace(" ", "").replace("_", "")

            mask = df_all["market"].apply(_norm_val) == _norm_val(market_name)
            if not mask.any():
                raise ValueError(
                    f"No rows for market '{market_name}' in LMP data; markets available: {df_all['market'].unique()}"
                )
            return df_all.loc[mask].copy()

    da = _fetch_market("day_ahead")
    rt = _fetch_market("real_time")

    node_col = "location" if "location" in da.columns else ("node" if "node" in da.columns else None)
    if node_col is None:
        raise ValueError("Could not find a node/location column in DA LMP data")

    da_n = da[da[node_col] == node].copy()
    rt_n = rt[rt[node_col] == node].copy()

    da_n = da_n.rename(columns={"lmp": "da_lmp", node_col: "node"})[["time", "node", "da_lmp"]]

    rt_n["time_hour"] = rt_n["time"].dt.floor("h")
    rt_h = (
        rt_n.groupby(["time_hour"], as_index=False)
        .agg(rt_lmp=("lmp", "mean"))
        .rename(columns={"time_hour": "time"})
    )
    rt_h["node"] = node
    rt_h = rt_h[["time", "node", "rt_lmp"]]

    return da_n, rt_h


def download_load_features(iso, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dfs = []
    try:
        load_act = iso.get_load(date=start, end=end)
        load_act = _normalize_cols(load_act)
        numeric_cols = [c for c in load_act.columns if c not in {"time", "zone"} and pd.api.types.is_numeric_dtype(load_act[c])]
        val = "load" if "load" in load_act.columns else (numeric_cols[0] if numeric_cols else None)
        if val:
            load_act = load_act[["time", val]].rename(columns={val: "load_actual_mw"})
            dfs.append(load_act)
    except Exception:
        pass

    try:
        load_fc = iso.get_load_forecast(date=start, end=end)
        load_fc = _normalize_cols(load_fc)
        numeric_cols = [c for c in load_fc.columns if c not in {"time", "zone"} and pd.api.types.is_numeric_dtype(load_fc[c])]
        val = "load" if "load" in load_fc.columns else (numeric_cols[0] if numeric_cols else None)
        if val:
            load_fc = load_fc[["time", val]].rename(columns={val: "load_forecast_mw"})
            dfs.append(load_fc)
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame(columns=["time", "load_actual_mw", "load_forecast_mw"])

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="time", how="outer")

    return out.sort_values("time").reset_index(drop=True)


def build_feature_table(da: pd.DataFrame, rt_h: pd.DataFrame, load_df: pd.DataFrame) -> pd.DataFrame:
    df = da.merge(rt_h, on=["time", "node"], how="inner")
    if not load_df.empty:
        df = df.merge(load_df, on="time", how="left")

    df["da_rt_spread"] = df["da_lmp"] - df["rt_lmp"]
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    if "load_actual_mw" in df.columns and "load_forecast_mw" in df.columns:
        df["load_forecast_error_mw"] = df["load_actual_mw"] - df["load_forecast_mw"]
        df["load_forecast_error_abs_mw"] = df["load_forecast_error_mw"].abs()
    else:
        df["load_forecast_error_mw"] = np.nan
        df["load_forecast_error_abs_mw"] = np.nan

    df = df.sort_values("time").reset_index(drop=True)
    df = df.dropna(subset=["da_lmp", "rt_lmp", "da_rt_spread"])
    return df


def train_model(df: pd.DataFrame, cfg: Config, paths: dict) -> dict:
    feature_cols = ["hour", "dow", "month", "da_lmp", "load_forecast_error_mw", "load_forecast_error_abs_mw"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0.0)
    y = df["da_rt_spread"].copy()

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=600,
        random_state=cfg.seed,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    model_path = os.path.join(paths["models"], f"rf_da_rt_spread_{cfg.node.lower().replace(' ', '_')}.joblib")
    joblib.dump({"model": model, "features": feature_cols}, model_path)

    return {"model_path": model_path, "mae": mae, "r2": r2, "n_train": int(len(X_train)), "n_test": int(len(X_test)), "features": feature_cols}


def plot_outputs(df: pd.DataFrame, cfg: Config, paths: dict) -> dict:
    fig_paths = {}

    plt.figure()
    df.set_index("time")["da_rt_spread"].rolling(24).mean().plot()
    plt.title(f"PJM {cfg.node} DA-RT Spread (24h rolling mean)")
    plt.xlabel("Time")
    plt.ylabel("DA - RT ($/MWh)")
    p1 = os.path.join(paths["fig"], "spread_timeseries.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    fig_paths["spread_timeseries"] = p1

    plt.figure()
    df["da_rt_spread"].clip(-200, 200).hist(bins=80)
    plt.title(f"PJM {cfg.node} DA-RT Spread Distribution (clipped)")
    plt.xlabel("DA - RT ($/MWh)")
    plt.ylabel("Count")
    p2 = os.path.join(paths["fig"], "spread_hist.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    fig_paths["spread_hist"] = p2

    if df["load_forecast_error_mw"].notna().any():
        plt.figure()
        sample = df.dropna(subset=["load_forecast_error_mw"]).copy()
        sample = sample[(sample["load_forecast_error_mw"].abs() < 8000) & (sample["da_rt_spread"].abs() < 300)]
        plt.scatter(sample["load_forecast_error_mw"], sample["da_rt_spread"], s=6, alpha=0.35)
        plt.title(f"Load Forecast Error vs DA-RT Spread (PJM {cfg.node})")
        plt.xlabel("Actual - Forecast (MW)")
        plt.ylabel("DA - RT ($/MWh)")
        p3 = os.path.join(paths["fig"], "error_vs_spread.png")
        plt.tight_layout()
        plt.savefig(p3, dpi=160)
        plt.close()
        fig_paths["error_vs_spread"] = p3

    return fig_paths


def write_report(cfg: Config, paths: dict, metrics: dict, df: pd.DataFrame, figs: dict) -> str:
    q = df["da_rt_spread"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    lines = [
        f"# PJM DA vs RT Spread Report ({cfg.node})",
        "",
        "## Data window",
        f"- Days: {cfg.days}",
        f"- Rows (hourly): {len(df):,}",
        "",
        "## Spread stats (DA - RT, $/MWh)",
        f"- p01: {q.get(0.01):.2f}",
        f"- p05: {q.get(0.05):.2f}",
        f"- p50: {q.get(0.50):.2f}",
        f"- p95: {q.get(0.95):.2f}",
        f"- p99: {q.get(0.99):.2f}",
        "",
        "## Baseline model (RandomForest)",
        f"- MAE: {metrics['mae']:.2f}",
        f"- R2: {metrics['r2']:.3f}",
        f"- Train rows: {metrics['n_train']:,}",
        f"- Test rows: {metrics['n_test']:,}",
        f"- Saved model: {os.path.relpath(metrics['model_path'], paths['root'])}",
        "",
        "## Figures",
    ]
    for k, p in figs.items():
        lines.append(f"- {k}: {os.path.relpath(p, paths['root'])}")
    lines += [
        "",
        "## Resume bullet template",
        "- Built a DA vs RT spread model using public market data, engineered load forecast error features, and trained a baseline model to identify when DA pricing underestimates RT risk.",
    ]

    report_path = os.path.join(paths["root"], "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def save_data(df: pd.DataFrame, cfg: Config, paths: dict) -> str:
    out = os.path.join(paths["data"], f"pjm_{cfg.node.lower().replace(' ', '_')}_hourly_features.parquet")
    df.to_parquet(out, index=False)
    return out


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=str, default="PJM RTO", help="PJM node or location name")
    ap.add_argument("--days", type=int, default=60, help="How many days back to pull")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (default: today UTC)")
    args = ap.parse_args()
    return Config(node=args.node, days=args.days, outdir=args.outdir, seed=args.seed, end_date=args.end_date)


def main() -> None:
    cfg = parse_args()
    paths = ensure_dirs(cfg.outdir)

    iso = PJM()
    start, end = utc_date_range(cfg.days, cfg.end_date)

    da, rt_h = download_prices(iso, start=start, end=end, node=cfg.node)
    load_df = download_load_features(iso, start=start, end=end)

    df = build_feature_table(da, rt_h, load_df)
    data_path = save_data(df, cfg, paths)
    figs = plot_outputs(df, cfg, paths)
    metrics = train_model(df, cfg, paths)
    report_path = write_report(cfg, paths, metrics, df, figs)

    print("Saved feature table:", data_path)
    print("Saved report:", report_path)
    print("Model metrics:", metrics)


if __name__ == "__main__":
    main()
