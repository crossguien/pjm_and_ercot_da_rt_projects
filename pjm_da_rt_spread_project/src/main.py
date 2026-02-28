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
import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

try:
    from gridstatus import PJM
except Exception:
    PJM = None

UTC = timezone.utc


@dataclass
class Config:
    node: str
    days: int
    outdir: str
    seed: int = 7
    end_date: str | None = None
    mode: str = "online"
    fallback_sample: bool = False
    demo: bool = False
    entry_z: float = 0.6
    base_mw: float = 10.0
    max_mw: float = 75.0
    trade_cost_per_mwh: float = 0.2
    risk_budget_per_trade: float = 250.0
    daily_loss_limit: float = 1200.0


def ensure_dirs(outdir: str) -> dict:
    paths = {
        "root": outdir,
        "data": os.path.join(outdir, "data"),
        "fig": os.path.join(outdir, "figures"),
        "models": os.path.join(outdir, "models"),
        "backtests": os.path.join(outdir, "backtests"),
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


def _load_sample_data(node: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data"))
    da_path = os.path.join(base, "da_prices.csv")
    rt_path = os.path.join(base, "rt_prices.csv")
    load_path = os.path.join(base, "load.csv")

    da = pd.read_csv(da_path, parse_dates=["time"])
    rt = pd.read_csv(rt_path, parse_dates=["time"])
    load_df = pd.read_csv(load_path, parse_dates=["time"])

    da = da[da["node"] == node].copy()
    rt = rt[rt["node"] == node].copy()
    return da, rt, load_df


def _supports_market_arg(func) -> bool:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return "market" in sig.parameters


def download_prices(iso, start: pd.Timestamp, end: pd.Timestamp, node: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_aliases = {
        "day_ahead": ["day_ahead", "DAY_AHEAD_HOURLY", "DAY_AHEAD_HOURLY_EX_ANTE", "DAY_AHEAD_HOURLY_EX_POST"],
        "real_time": ["real_time", "REAL_TIME_HOURLY", "REAL_TIME_5_MIN", "REAL_TIME_HOURLY_FINAL"],
    }

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
        if _supports_market_arg(iso.get_lmp):
            errs = []
            for market in market_aliases.get(market_name, [market_name]):
                try:
                    return _call_get_lmp(date=start, end=end, market=market)
                except ValueError as e:
                    errs.append(str(e))
            raise ValueError(
                f"Unable to fetch market '{market_name}' using aliases {market_aliases.get(market_name, [market_name])}. "
                f"Last error: {errs[-1] if errs else 'unknown'}"
            )

        df_all = _call_get_lmp(date=start, end=end)
        if "market" not in df_all.columns:
            raise ValueError("LMP data missing 'market' column; cannot split DA/RT")

        def _norm_val(val: str) -> str:
            return "".join(ch for ch in str(val).lower() if ch.isalnum())

        normalized_candidates = {_norm_val(v) for v in market_aliases.get(market_name, [market_name])}
        mask = df_all["market"].apply(_norm_val).isin(normalized_candidates)
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

    test_df = df.iloc[split:][["time", "node", "da_lmp", "rt_lmp", "da_rt_spread", "hour", "dow"]].copy()
    test_df["pred_spread"] = pred

    return {
        "model_path": model_path,
        "mae": mae,
        "r2": r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "test_frame": test_df,
    }


def _safe_sharpe(pnl: pd.Series) -> float:
    vol = float(pnl.std(ddof=0))
    if vol <= 1e-9:
        return float("nan")
    return float((pnl.mean() / vol) * np.sqrt(24 * 365))


def run_virtuals_backtest(test_df: pd.DataFrame, cfg: Config, paths: dict) -> dict:
    bt = test_df.sort_values("time").reset_index(drop=True).copy()
    if bt.empty:
        raise ValueError("Backtest input is empty.")

    bt["vol_lookback"] = bt["da_rt_spread"].rolling(24, min_periods=6).std().shift(1)
    fallback_vol = float(bt["da_rt_spread"].std(ddof=0))
    if not np.isfinite(fallback_vol) or fallback_vol <= 1e-9:
        fallback_vol = 10.0
    bt["vol_lookback"] = bt["vol_lookback"].fillna(fallback_vol).clip(lower=0.5)
    bt["signal_strength"] = bt["pred_spread"] / bt["vol_lookback"]
    bt["raw_direction"] = np.sign(bt["pred_spread"])
    bt["should_trade"] = bt["signal_strength"].abs() >= cfg.entry_z

    bt["raw_size_mw"] = cfg.base_mw * bt["signal_strength"].abs()
    bt["risk_cap_size_mw"] = cfg.risk_budget_per_trade / bt["vol_lookback"]
    bt["size_mw"] = bt[["raw_size_mw", "risk_cap_size_mw"]].min(axis=1).clip(upper=cfg.max_mw)
    bt["size_mw"] = np.where(bt["should_trade"], bt["size_mw"], 0.0)

    bt["trade_date"] = bt["time"].dt.floor("D")
    bt["direction"] = bt["raw_direction"].astype(float)
    bt["pnl_pre_risk"] = bt["direction"] * bt["size_mw"] * bt["da_rt_spread"] - bt["size_mw"].abs() * cfg.trade_cost_per_mwh
    bt["risk_blocked"] = False

    running_day = None
    day_pnl = 0.0
    for idx, row in bt.iterrows():
        if running_day != row["trade_date"]:
            running_day = row["trade_date"]
            day_pnl = 0.0

        if day_pnl <= -cfg.daily_loss_limit and row["size_mw"] > 0:
            bt.at[idx, "size_mw"] = 0.0
            bt.at[idx, "risk_blocked"] = True
            bt.at[idx, "pnl_pre_risk"] = 0.0
            continue

        day_pnl += float(bt.at[idx, "pnl_pre_risk"])

    bt["pnl"] = bt["direction"] * bt["size_mw"] * bt["da_rt_spread"] - bt["size_mw"].abs() * cfg.trade_cost_per_mwh
    bt["trade_flag"] = bt["size_mw"] > 0
    bt["trade_type"] = np.where(bt["direction"] > 0, "DEC", "INC")
    bt.loc[~bt["trade_flag"], "trade_type"] = "FLAT"
    bt["cum_pnl"] = bt["pnl"].cumsum()
    bt["running_peak"] = bt["cum_pnl"].cummax()
    bt["drawdown"] = bt["cum_pnl"] - bt["running_peak"]

    traded = bt[bt["trade_flag"]].copy()
    hit_rate = float((traded["pnl"] > 0).mean()) if not traded.empty else float("nan")
    max_dd = float(bt["drawdown"].min()) if not bt.empty else float("nan")
    sharpe = _safe_sharpe(bt["pnl"])

    bt["vol_regime"] = np.where(bt["vol_lookback"] >= bt["vol_lookback"].median(), "high_vol", "low_vol")
    bt["time_regime"] = np.where((bt["dow"] < 5) & (bt["hour"].between(7, 22)), "weekday_onpeak", "offpeak")
    bt["signal_regime"] = np.where(bt["pred_spread"] >= 0, "dec_signal", "inc_signal")

    regime_rows = []
    for regime_col in ["vol_regime", "time_regime", "signal_regime"]:
        for regime_value, g in bt.groupby(regime_col):
            g_traded = g[g["trade_flag"]]
            regime_rows.append(
                {
                    "regime_type": regime_col,
                    "regime": regime_value,
                    "hours": int(len(g)),
                    "trades": int(g["trade_flag"].sum()),
                    "pnl": float(g["pnl"].sum()),
                    "sharpe": _safe_sharpe(g["pnl"]),
                    "hit_rate": float((g_traded["pnl"] > 0).mean()) if not g_traded.empty else float("nan"),
                }
            )
    regime_df = pd.DataFrame(regime_rows)
    regime_df = regime_df.sort_values(["regime_type", "regime"]).reset_index(drop=True)

    trades_path = os.path.join(paths["backtests"], "virtuals_backtest_trades.csv")
    regime_path = os.path.join(paths["backtests"], "regime_performance.csv")
    bt.to_csv(trades_path, index=False)
    regime_df.to_csv(regime_path, index=False)

    return {
        "summary": {
            "hours": int(len(bt)),
            "trades": int(bt["trade_flag"].sum()),
            "total_pnl": float(bt["pnl"].sum()),
            "avg_pnl_per_hour": float(bt["pnl"].mean()),
            "avg_pnl_per_trade": float(traded["pnl"].mean()) if not traded.empty else float("nan"),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "risk_blocks": int(bt["risk_blocked"].sum()),
            "trades_path": trades_path,
            "regime_path": regime_path,
        },
        "backtest_frame": bt,
        "regime_df": regime_df,
    }


def plot_outputs(df: pd.DataFrame, cfg: Config, paths: dict, backtest_df: pd.DataFrame | None = None) -> dict:
    fig_paths = {}
    import matplotlib.pyplot as plt

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

    if backtest_df is not None and not backtest_df.empty:
        plt.figure()
        bt_plot = backtest_df.set_index("time")["cum_pnl"]
        bt_plot.plot()
        plt.title(f"Virtuals Backtest Equity Curve (PJM {cfg.node})")
        plt.xlabel("Time")
        plt.ylabel("Cumulative PnL ($)")
        p4 = os.path.join(paths["fig"], "virtuals_equity_curve.png")
        plt.tight_layout()
        plt.savefig(p4, dpi=160)
        plt.close()
        fig_paths["virtuals_equity_curve"] = p4

    return fig_paths


def write_report(cfg: Config, paths: dict, metrics: dict, strategy: dict, df: pd.DataFrame, figs: dict) -> str:
    q = df["da_rt_spread"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict()
    strat = strategy["summary"]
    regime_df = strategy["regime_df"]
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
        "## Virtuals strategy design",
        "- Trade instrument: hourly DA vs RT virtuals at selected node",
        "- Signal: model-predicted spread normalized by rolling realized vol",
        f"- Entry rule: trade when abs(pred_spread / vol_lookback) >= {cfg.entry_z:.2f}",
        "- Direction rule: predicted positive spread => DEC, predicted negative spread => INC",
        f"- Sizing: min(base_mw * |signal|, risk_budget_per_trade / vol_lookback, max_mw) with base_mw={cfg.base_mw:.1f}, max_mw={cfg.max_mw:.1f}",
        f"- Cost model: {cfg.trade_cost_per_mwh:.2f} $/MWh per traded MWh",
        f"- Risk cap: stop initiating new trades after daily PnL <= -{cfg.daily_loss_limit:.0f}",
        "",
        "## Backtest performance (out-of-sample)",
        f"- Hours tested: {strat['hours']:,}",
        f"- Trades: {strat['trades']:,}",
        f"- Total PnL: {strat['total_pnl']:.2f}",
        f"- Avg PnL per trade: {strat['avg_pnl_per_trade']:.2f}",
        f"- Sharpe (hourly annualized): {strat['sharpe']:.3f}",
        f"- Max drawdown: {strat['max_drawdown']:.2f}",
        f"- Hit rate: {strat['hit_rate']:.2%}",
        f"- Daily risk cap blocks: {strat['risk_blocks']}",
        f"- Trade log: {os.path.relpath(strat['trades_path'], paths['root'])}",
        f"- Regime table: {os.path.relpath(strat['regime_path'], paths['root'])}",
        "",
        "## Regime performance",
        "| Regime type | Regime | Hours | Trades | PnL | Sharpe | Hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regime_df.itertuples(index=False):
        hit_rate = f"{row.hit_rate:.2%}" if np.isfinite(row.hit_rate) else "nan"
        sharpe = f"{row.sharpe:.3f}" if np.isfinite(row.sharpe) else "nan"
        lines.append(
            f"| {row.regime_type} | {row.regime} | {row.hours} | {row.trades} | {row.pnl:.2f} | {sharpe} | {hit_rate} |"
        )
    lines += [
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
    ap.add_argument("--mode", choices=["online", "offline"], default="online", help="Use live data or bundled sample data")
    ap.add_argument("--fallback-sample", action="store_true", help="Use sample data if live fetch fails")
    ap.add_argument("--demo", action="store_true", help="Skip model training/plots for a fast demo run")
    ap.add_argument("--entry-z", type=float, default=0.6, help="Z-score style entry threshold on predicted spread")
    ap.add_argument("--base-mw", type=float, default=10.0, help="Base position size in MW")
    ap.add_argument("--max-mw", type=float, default=75.0, help="Maximum size per hour in MW")
    ap.add_argument("--trade-cost-per-mwh", type=float, default=0.2, help="Transaction + slippage cost in $/MWh")
    ap.add_argument("--risk-budget-per-trade", type=float, default=250.0, help="Per-trade risk budget in dollars")
    ap.add_argument("--daily-loss-limit", type=float, default=1200.0, help="Daily stop loss in dollars")
    args = ap.parse_args()
    return Config(
        node=args.node,
        days=args.days,
        outdir=args.outdir,
        seed=args.seed,
        end_date=args.end_date,
        mode=args.mode,
        fallback_sample=args.fallback_sample,
        demo=args.demo,
        entry_z=args.entry_z,
        base_mw=args.base_mw,
        max_mw=args.max_mw,
        trade_cost_per_mwh=args.trade_cost_per_mwh,
        risk_budget_per_trade=args.risk_budget_per_trade,
        daily_loss_limit=args.daily_loss_limit,
    )


def main() -> None:
    cfg = parse_args()
    paths = ensure_dirs(cfg.outdir)

    start, end = utc_date_range(cfg.days, cfg.end_date)

    if cfg.mode == "offline":
        da, rt_h, load_df = _load_sample_data(cfg.node)
    else:
        try:
            if PJM is None:
                raise SystemExit("Missing dependency gridstatus. Install with: pip install -r requirements.txt")
            iso = PJM()
            da, rt_h = download_prices(iso, start=start, end=end, node=cfg.node)
            load_df = download_load_features(iso, start=start, end=end)
        except Exception:
            if not cfg.fallback_sample:
                raise
            print("Live data fetch failed (or missing API key); falling back to sample data.")
            da, rt_h, load_df = _load_sample_data(cfg.node)

    df = build_feature_table(da, rt_h, load_df)
    if df.empty:
        raise ValueError("No overlapping DA/RT rows; try --mode offline or adjust --node/--days.")
    data_path = save_data(df, cfg, paths)
    if cfg.demo:
        metrics = {"model_path": "n/a", "mae": float("nan"), "r2": float("nan"), "n_train": 0, "n_test": 0, "features": []}
        strategy = {
            "summary": {
                "hours": 0,
                "trades": 0,
                "total_pnl": float("nan"),
                "avg_pnl_per_hour": float("nan"),
                "avg_pnl_per_trade": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
                "hit_rate": float("nan"),
                "risk_blocks": 0,
                "trades_path": "n/a",
                "regime_path": "n/a",
            },
            "backtest_frame": pd.DataFrame(),
            "regime_df": pd.DataFrame(columns=["regime_type", "regime", "hours", "trades", "pnl", "sharpe", "hit_rate"]),
        }
        figs = {}
        report_path = write_report(cfg, paths, metrics, strategy, df, figs)
        print("Demo mode: skipped model training and plots.")
    else:
        metrics = train_model(df, cfg, paths)
        strategy = run_virtuals_backtest(metrics["test_frame"], cfg, paths)
        figs = plot_outputs(df, cfg, paths, backtest_df=strategy["backtest_frame"])
        report_path = write_report(cfg, paths, metrics, strategy, df, figs)

    model_print = {k: v for k, v in metrics.items() if k != "test_frame"}
    print("Saved feature table:", data_path)
    print("Saved report:", report_path)
    print("Model metrics:", model_print)
    if not cfg.demo:
        print("Strategy summary:", strategy["summary"])


if __name__ == "__main__":
    main()
