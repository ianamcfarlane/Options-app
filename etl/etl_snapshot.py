#!/usr/bin/env python3
# /home/coder/options/etl/etl_snapshot.py

import os, time, sqlite3
from datetime import datetime, timezone, date
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

# -------- config / env --------
ROOT = "/home/coder/options"  # code-server workspace
load_dotenv(os.path.join(ROOT, ".env"))

VERBOSE = os.getenv("SNAPSHOT_VERBOSE", "1") == "1"
TRADIER_TOKEN = (os.getenv("TRADIER_TOKEN") or "").strip()

# DB: allow DB_URL=sqlite:////mnt/user/options/db/options.db or absolute path
def _db_path_from_url(raw: str) -> str:
    if not raw:
        return ""
    if raw.startswith("sqlite:"):
        u = urlparse(raw)
        return u.path or ""
    return raw

raw_db_url = (os.getenv("DB_URL", f"sqlite:////{ROOT}/db/options.db") or "").strip()
DB_PATH = _db_path_from_url(raw_db_url)
if not DB_PATH.startswith("/"):
    DB_PATH = "/" + DB_PATH
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Parquet root (host share when provided via env)
PARQ_ROOT = os.getenv("PARQ_ROOT", f"{ROOT}/parquet/option_quotes")

SNAPSHOT_LABEL = os.getenv("SNAPSHOT_LABEL", "S1")
TRADE_DATE = date.today().isoformat()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

BASE = "https://api.tradier.com"  # switch to https://sandbox.tradier.com for sandbox token
HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json",
    "User-Agent": "options-snapshot/1.0",
}

def ts():
    """Short timestamp for logs."""
    return datetime.now().strftime("%H:%M:%S")

# -------- db helpers --------
def get_active_symbols():
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT symbol FROM symbols WHERE is_active=1 ORDER BY symbol"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()

def upsert_contracts(option_rows):
    if not option_rows:
        return
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        cur.execute("BEGIN")
        for r in option_rows:
            s = r.get("option_symbol")
            if not s:
                continue
            cur.execute(
                """INSERT OR IGNORE INTO option_contracts
                   (option_symbol, symbol, right, strike, expiration)
                   VALUES (?,?,?,?,?)""",
                (s, r["symbol"], r["right"], r["strike"], r["expiration"]),
            )
        con.commit()
    finally:
        con.close()

def record_snapshot_run(rows_ingested: int, expiries):
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO snapshot_runs
               (run_id, snapshot_ts, snapshot_label, symbols_count,
                rows_ingested, expiries_covered, status, message)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                RUN_ID,
                datetime.now(timezone.utc).isoformat(),
                SNAPSHOT_LABEL,
                len(get_active_symbols()),
                rows_ingested,
                ",".join(sorted(set(expiries))),
                "OK",
                "",
            ),
        )
        con.commit()
    finally:
        con.close()

# -------- Tradier calls --------
_exp_cache: dict[str, str] = {}

def nearest_expiration(symbol: str) -> str | None:
    """Get the next expiration (>= today) for the symbol."""
    if symbol in _exp_cache:
        return _exp_cache[symbol]
    url = f"{BASE}/v1/markets/options/expirations"
    r = requests.get(
        url,
        headers=HEADERS,
        params={"symbol": symbol, "includeAllRoots": "false", "strikes": "false"},
        timeout=20,
    )
    r.raise_for_status()
    dates = (r.json().get("expirations") or {}).get("date") or []
    today = TRADE_DATE
    for d in dates:
        if d >= today:
            _exp_cache[symbol] = d
            return d
    return dates[-1] if dates else None

def fetch_chain(symbol: str):
    exp = nearest_expiration(symbol)
    if not exp:
        return []
    url = f"{BASE}/v1/markets/options/chains"
    params = {"symbol": symbol, "expiration": exp, "greeks": "true"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=25)
    r.raise_for_status()
    data = (r.json().get("options") or {}).get("option", []) or []
    now = datetime.now(timezone.utc).isoformat()
    out = []
    for o in data:
        g = o.get("greeks") or {}
        out.append(
            {
                "option_symbol": o.get("symbol"),
                "symbol": o.get("underlying", symbol),
                "right": (o.get("option_type") or "").upper()[:1],
                "strike": o.get("strike"),
                "expiration": o.get("expiration_date"),
                "bid": o.get("bid"),
                "ask": o.get("ask"),
                "last": o.get("last"),
                "volume": o.get("volume"),
                "open_interest": o.get("open_interest"),
                "iv": g.get("mid_iv"),
                "delta": g.get("delta"),
                "gamma": g.get("gamma"),
                "theta": g.get("theta"),
                "vega": g.get("vega"),
                "rho": g.get("rho"),
                "snapshot_ts": now,
                "snapshot_label": SNAPSHOT_LABEL,
                "run_id": RUN_ID,
                "source": "tradier_4x",
            }
        )
    return out

# -------- IO --------
def write_symbol_parquet(df: pd.DataFrame, symbol: str):
    outdir = f"{PARQ_ROOT}/trade_date={TRADE_DATE}/snapshot={SNAPSHOT_LABEL}/symbol={symbol}"
    os.makedirs(outdir, exist_ok=True)
    df.to_parquet(f"{outdir}/part.parquet", index=False, compression="zstd")

# -------- main --------
def main():
    if not TRADIER_TOKEN:
        raise SystemExit("TRADIER_TOKEN missing in .env")

    symbols = get_active_symbols()
    total = len(symbols)
    total_rows = 0
    expiries: list[str] = []

    if VERBOSE:
        print(f"{ts()} Starting snapshot {SNAPSHOT_LABEL} with {total} symbols; RUN_ID={RUN_ID}", flush=True)

    for i, sym in enumerate(symbols, 1):
        try:
            if VERBOSE:
                print(f"{ts()} [{i}/{total}] {sym} …", flush=True)

            rows = fetch_chain(sym)
            if rows:
                df = pd.DataFrame(rows)
                df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
                df["dte"] = (
                    pd.to_datetime(df["expiration"]).dt.date - date.today()
                ).apply(lambda d: d.days)
                write_symbol_parquet(df, sym)
                upsert_contracts(rows)
                expiries += list(df["expiration"].dropna().unique())
                total_rows += len(df)

                if VERBOSE:
                    exps_here = df["expiration"].dropna().nunique()
                    print(
                        f"{ts()} [{i}/{total}] {sym}: {len(df)} rows, {exps_here} expiries — cumulative {total_rows}",
                        flush=True,
                    )

            time.sleep(0.35)  # be nice to the API

        except requests.HTTPError as e:
            try:
                msg = e.response.json()
            except Exception:
                msg = e.response.text
            print(f"{ts()} [{i}/{total}] {sym} HTTP {e.response.status_code}: {msg}", flush=True)
            time.sleep(0.8)
        except Exception as e:
            print(f"{ts()} [{i}/{total}] {sym} ERROR: {e}", flush=True)
            time.sleep(0.8)

    record_snapshot_run(total_rows, expiries)
    print(
        f"{ts()} RUN {RUN_ID} {SNAPSHOT_LABEL} finished: symbols={total} rows={total_rows} unique_expiries={len(set(expiries))}",
        flush=True,
    )

if __name__ == "__main__":
    main()
