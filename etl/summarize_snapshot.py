#!/usr/bin/env python3
# /home/coder/options/etl/summarize_snapshot.py
import os, glob, sys
from datetime import date
from dotenv import load_dotenv
import pandas as pd

ROOT = "/home/coder/options"
load_dotenv(os.path.join(ROOT, ".env"))

LABEL = os.getenv("SNAPSHOT_LABEL", "S1")
TRADE_DATE = os.getenv("TRADE_DATE", date.today().isoformat())
PARQ_QUOTES_ROOT = os.getenv("PARQ_ROOT", "/mnt/user/options/parquet/option_quotes")
PARQ_SUMM_ROOT = os.getenv("PARQ_SUMM_ROOT", "/mnt/user/options/parquet/summaries")

TARGET_ABS_DELTA = float(os.getenv("TARGET_ABS_DELTA", "0.20"))
TOP_N_PER_BUCKET = int(os.getenv("TOP_N_PER_BUCKET", "3"))
DTE_MIN = int(os.getenv("DTE_MIN", "3"))
DTE_MAX = int(os.getenv("DTE_MAX", "10"))

def load_snapshot_df(trade_date: str, label: str) -> pd.DataFrame:
    pattern = f"{PARQ_QUOTES_ROOT}/trade_date={trade_date}/snapshot={label}/symbol=*/part.parquet"
    files = glob.glob(pattern)
    print(f"[summary] looking for files: {pattern}")
    print(f"[summary] found {len(files)} parquet parts")

    if not files:
        return pd.DataFrame()

    frames = []
    errors = 0
    for f in files:
        try:
            # engine='pyarrow' handles zstd fine; if not present it will raise
            df = pd.read_parquet(f, engine="pyarrow")
            frames.append(df)
        except Exception as e:
            errors += 1
            # show the first few errors; don’t crash the whole run
            if errors <= 10:
                print(f"[summary] ERROR reading {f}: {e}", file=sys.stderr)
    if not frames:
        print(f"[summary] all reads failed ({errors} errors)")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def main():
    df = load_snapshot_df(TRADE_DATE, LABEL)
    if df.empty:
        print(f"No data for {TRADE_DATE} {LABEL}")
        return

    # required columns
    for col in ["delta", "dte", "right", "symbol", "expiration", "strike", "bid", "ask", "mid", "iv"]:
        if col not in df.columns:
            df[col] = pd.NA

    tmp = df.dropna(subset=["delta"]).copy()
    if "dte" in tmp.columns:
        tmp = tmp[tmp["dte"].between(DTE_MIN, DTE_MAX)]

    tmp["adelta"] = tmp["delta"].astype(float).abs()
    tmp["dist"] = (tmp["adelta"] - TARGET_ABS_DELTA).abs()

    picks = (
        tmp.sort_values(["symbol", "right", "dist"], kind="stable")
           .groupby(["symbol", "right"], as_index=False, sort=False)
           .head(TOP_N_PER_BUCKET)
           .reset_index(drop=True)
    )

    outdir = f"{PARQ_SUMM_ROOT}/trade_date={TRADE_DATE}/snapshot={LABEL}"
    os.makedirs(outdir, exist_ok=True)
    picks.to_parquet(f"{outdir}/picks.parquet", index=False)
    picks.to_csv(f"{outdir}/picks.csv", index=False)
    print(f"[summary] wrote picks: {len(picks)} rows to\n  {outdir}/picks.parquet\n  {outdir}/picks.csv")

if __name__ == "__main__":
    main()
