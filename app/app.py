"""
app.py
===============================================================================
Options Snapshots – Strategy Helper + KaChing + Picks + Journal

- Wires modular tabs:
    1) Strategy helper (from strategy_helper.render)
    2) KaChing Put Calendar (from calendar_options.render)
    3) Filters (persisted to simple TOML)
    4) Picks (~0.20Δ)
    5) Snapshots Browser
    6) Trade Journal (ideas -> trades, close trades)
    7) Help / Definitions
    8) TradingView (placeholder)

Notes:
- Parquet + SQLite paths come from env with safe defaults.
- Trend helpers use yfinance with snapshot fallback.

@version 2025-09-03
"""

from __future__ import annotations

# ---- stdlib
import os, glob, sqlite3, math, json
from functools import lru_cache
from datetime import date, datetime

# ---- third-party
import pandas as pd
import numpy as np
import streamlit as st

# ---- local tabs
from calendar_options import render as render_calendar_options
from strategy_helper import render as render_strategy_helper

# ---- optional market data (trend/last)
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False


# =============================================================================
# Page config (once)
# =============================================================================
st.set_page_config(page_title="Options Snapshots", layout="wide")


# =============================================================================
# Simple prefs file (flat TOML-ish)
# =============================================================================
PREFS_PATH = "/app/.app_prefs.toml"
try:
    import tomllib  # Python 3.11+
    def _read_toml(p: str) -> dict:
        try:
            with open(p, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}
except Exception:
    def _read_toml(_p: str) -> dict:
        return {}

def _write_prefs(d: dict) -> None:
    """Tiny TOML-ish writer for flat keys (no nested tables)."""
    try:
        lines = []
        for k, v in d.items():
            if isinstance(v, str): lines.append(f'{k} = "{v}"')
            elif isinstance(v, (int, float)): lines.append(f"{k} = {v}")
            elif isinstance(v, bool): lines.append(f"{k} = {str(v).lower()}")
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


# =============================================================================
# Defaults
# =============================================================================
DEFAULTS = {
    "default_symbol": "AAPL",
    # Finder / Strategy helper guardrails
    "finder_min_vol_oi": 200,
    "finder_target_width": 5.0,
    "finder_allow_width_fuzz": 0.51,
    "liq_min_vol_oi": 50,
    # Delta windows (beginner-friendly)
    "bull_long_call_delta_min": 0.35,
    "bull_long_call_delta_max": 0.55,
    "bear_short_put_delta_min": 0.20,
    "bear_short_put_delta_max": 0.35,
    "neutral_short_delta": 0.18,
    # Distance from spot (guardrails)
    "max_call_short_pct_above": 0.10,
    "max_put_short_pct_below": 0.10,
    # Highlight targets
    "target_debit_pct": 0.40,
    "target_credit_pct": 0.30,
}
def _merge_prefs() -> dict:
    saved = _read_toml(PREFS_PATH)
    return {**DEFAULTS, **(saved or {})}
PREFS = _merge_prefs()


# =============================================================================
# Paths / Globals
# =============================================================================
PARQ_ROOT      = os.environ.get("PARQ_ROOT", "/home/coder/options/parquet/option_quotes")
PARQ_SUMM_ROOT = os.environ.get("PARQ_SUMM_ROOT", "/home/coder/options/parquet/summaries")
DB_PATH        = os.environ.get("DB_PATH", "/home/coder/options/db/options.db")

# DEBUG_MODE = st.sidebar.checkbox("🔍 Debug mode", value=False)
#def debug(msg):
#   if DEBUG_MODE:
#        st.sidebar.write("🛠", msg)


# =============================================================================
# DB helpers & schema
# =============================================================================
def db_connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def ensure_schema():
    con = db_connect()
    cur = con.cursor()
    # ideas staging
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trade_ideas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
        snapshot_date TEXT,
        snapshot_label TEXT,
        symbol TEXT,
        expiry TEXT,
        strategy TEXT,
        legs_json TEXT,
        est_price REAL,
        direction TEXT,
        notes TEXT
    );
    """)
    # live trades
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts DATETIME DEFAULT CURRENT_TIMESTAMP,
        trade_date TEXT,
        symbol TEXT,
        strategy TEXT,
        expiry TEXT,
        legs_json TEXT,
        entry_price REAL,
        exit_price REAL,
        qty INTEGER DEFAULT 1,
        status TEXT DEFAULT 'OPEN',
        pnl REAL,
        notes TEXT
    );
    """)
    con.commit()
    con.close()

def idea_insert(snapshot_date, snapshot_label, symbol, expiry, strategy, legs_json, est_price, direction, notes=""):
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trade_ideas
        (snapshot_date, snapshot_label, symbol, expiry, strategy, legs_json, est_price, direction, notes)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (snapshot_date, snapshot_label, symbol, expiry, strategy, json.dumps(legs_json or {}), float(est_price), direction, notes))
    con.commit()
    con.close()

def ideas_load_df() -> pd.DataFrame:
    con = db_connect()
    df = pd.read_sql_query("SELECT * FROM trade_ideas ORDER BY created_ts DESC", con)
    con.close()
    return df

def ideas_delete(ids: list[int]):
    if not ids: return
    con = db_connect(); cur = con.cursor()
    cur.executemany("DELETE FROM trade_ideas WHERE id = ?", [(int(i),) for i in ids])
    con.commit(); con.close()

def trade_insert(trade_date, symbol, strategy, expiry, legs_json, entry_price, qty, notes=""):
    con = db_connect(); cur = con.cursor()
    cur.execute("""
        INSERT INTO trades (trade_date, symbol, strategy, expiry, legs_json, entry_price, qty, status, notes)
        VALUES (?,?,?,?,?,?,?, 'OPEN', ?)
    """, (trade_date, symbol, strategy, expiry, json.dumps(legs_json or {}), float(entry_price), int(qty), notes))
    con.commit(); con.close()

def trade_close(trade_id: int, exit_price: float):
    con = db_connect(); cur = con.cursor()
    row = cur.execute("SELECT entry_price, qty, strategy FROM trades WHERE id = ?", (int(trade_id),)).fetchone()
    if not row:
        con.close(); return
    entry_price, qty, strategy = float(row[0]), int(row[1]), str(row[2]) if row[2] else ""
    is_credit = any(k in strategy.lower() for k in ["credit", "short", "condor", "strangle"])
    contracts = 100 * qty
    pnl = (entry_price - float(exit_price)) * contracts if is_credit else (float(exit_price) - entry_price) * contracts
    cur.execute("""
        UPDATE trades SET exit_price = ?, status = 'CLOSED', pnl = ? WHERE id = ?
    """, (float(exit_price), float(pnl), int(trade_id)))
    con.commit(); con.close()

def trades_load_df() -> pd.DataFrame:
    con = db_connect()
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY created_ts DESC", con)
    con.close()
    return df

ensure_schema()


# =============================================================================
# Snapshot parquet utilities
# =============================================================================
@lru_cache(maxsize=256)
def list_trade_dates() -> list[str]:
    dates = []
    for p in glob.glob(os.path.join(PARQ_ROOT, "trade_date=*")):
        d = p.split("trade_date=")[-1]
        if d and os.path.isdir(p):
            dates.append(d)
    return sorted(dates)

@lru_cache(maxsize=512)
def list_snapshot_labels(trade_date: str) -> list[str]:
    labels = []
    root = os.path.join(PARQ_ROOT, f"trade_date={trade_date}")
    for p in glob.glob(os.path.join(root, "snapshot=*")):
        lab = p.split("snapshot=")[-1]
        if lab and os.path.isdir(p):
            labels.append(lab)
    return sorted(labels)

@st.cache_data(show_spinner=False)
def load_snapshot_df(trade_date: str, label: str) -> pd.DataFrame:
    patt = os.path.join(PARQ_ROOT, f"trade_date={trade_date}", f"snapshot={label}", "symbol=*","part.parquet")
    files = glob.glob(patt)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in ["option_symbol","symbol","right","expiration","snapshot_label","run_id","source"]:
        if c in df:
            df[c] = df[c].astype("string")
    return df

@st.cache_data(show_spinner=False)
def load_picks_df(trade_date: str, label: str) -> pd.DataFrame:
    pq  = os.path.join(PARQ_SUMM_ROOT, f"trade_date={trade_date}", f"snapshot={label}", "picks.parquet")
    csv = os.path.join(PARQ_SUMM_ROOT, f"trade_date={trade_date}", f"snapshot={label}", "picks.csv")
    if os.path.exists(pq):  return pd.read_parquet(pq)
    if os.path.exists(csv): return pd.read_csv(csv)
    return pd.DataFrame()

@lru_cache(maxsize=256)
def symbols_in_snapshot(trade_date: str, label: str) -> list[str]:
    patt = os.path.join(PARQ_ROOT, f"trade_date={trade_date}", f"snapshot={label}", "symbol=*")
    syms = [p.split("symbol=")[-1] for p in glob.glob(patt)]
    return sorted(syms)


# =============================================================================
# Trend helpers (yfinance + snapshot fallback)
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_trend_from_yf(symbol: str, fallback_df: pd.DataFrame | None = None):
    """
    Returns:
        (trend: 'Bullish'|'Neutral'|'Bearish'|'Unknown', last_px, source, debug_msg, debug_log)
    """
    import numpy as _np
    debug_log = []
    dbg = ""
    debug_log.append(f"yfinance request: {symbol}")

    if YF_OK:
        try:
            t = yf.Ticker(symbol)
            last_px = _np.nan
            fi = getattr(t, "fast_info", None)
            if fi is not None:
                try:
                    last_px = float(getattr(fi, "last_price", fi.get("last_price", _np.nan)))
                    debug_log.append(f"fast_info.last_price: {last_px}")
                except Exception as e:
                    dbg = f"yf fast_info err: {e}"
                    debug_log.append(dbg)

            hist = t.history(period="4mo", interval="1d", auto_adjust=False, prepost=False)
            if hist is None or hist.empty:
                hist = yf.download(symbol, period="6mo", interval="1d", progress=False)

            if hist is not None and not hist.empty:
                close = hist["Close"].astype(float)
                if _np.isnan(last_px) and len(close) > 0:
                    last_px = float(close.iloc[-1])

                sma10 = close.rolling(10).mean()
                sma20 = close.rolling(20).mean()
                trend = "Unknown"
                if len(close) >= 25 and not _np.isnan(sma10.iloc[-1]) and not _np.isnan(sma20.iloc[-1]):
                    d10 = float(sma10.diff().iloc[-1]) if not _np.isnan(sma10.diff().iloc[-1]) else 0.0
                    if sma10.iloc[-1] > sma20.iloc[-1] and d10 > 0:
                        trend = "Bullish"
                    elif sma10.iloc[-1] < sma20.iloc[-1] and d10 < 0:
                        trend = "Bearish"
                    else:
                        trend = "Neutral"
                return trend, float(last_px), "yfinance", "history_ok", debug_log

            if not _np.isnan(last_px):
                return "Unknown", float(last_px), "yfinance", "fast_info_only", debug_log

        except Exception as e:
            dbg = f"yf error: {type(e).__name__}: {e}"
            debug_log.append(dbg)

    # Snapshot fallback for last price
    try:
        if fallback_df is not None and not fallback_df.empty:
            snap_last = (
                fallback_df.loc[fallback_df["symbol"] == symbol, "last"]
                .dropna()
                .astype(float)
            )
            if len(snap_last) > 0:
                return "Unknown", float(snap_last.iloc[-1]), "snapshot", "fallback_snapshot_last", debug_log
    except Exception as e:
        debug_log.append(f"snapshot fallback error: {e}")

    return "Unknown", float("nan"), "none", dbg or "no_data", debug_log


def get_weekly_trend(symbol: str):
    """Simple 5/10 SMA + RSI(5) view; returns (trend, earnings_date or None, debug_str)."""
    try:
        df = yf.download(symbol, period="6mo", interval="1d") if YF_OK else pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.empty or len(df) < 10:
            return "unknown", None, "not_enough_data"

        df["SMA5"] = df["Close"].rolling(5).mean()
        df["SMA10"] = df["Close"].rolling(10).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(5).mean()
        loss = -delta.where(delta < 0, 0).rolling(5).mean()
        rs = gain / loss
        df["RSI5"] = 100 - (100 / (1 + rs))

        valid = df.dropna(subset=["SMA5", "SMA10", "RSI5"])
        if valid.empty:
            return "unknown", None, "indicators_na"
        last = valid.iloc[-1]
        rsi = float(last["RSI5"])
        sma5 = float(last["SMA5"])
        sma10 = float(last["SMA10"])

        if sma5 > sma10 and rsi < 70:
            t = "Bullish"
        elif sma5 < sma10 and rsi > 30:
            t = "Bearish"
        else:
            t = "Neutral"

        earnings_date = None
        if YF_OK:
            try:
                cal = yf.Ticker(symbol).calendar
                if "Earnings Date" in cal.index:
                    earnings_date = cal.loc["Earnings Date"].values[0]
            except Exception:
                pass

        dbg = f"SMA5={sma5:.2f} SMA10={sma10:.2f} RSI5={rsi:.1f}"
        return t, earnings_date, dbg
    except Exception as e:
        return "unknown", None, str(e)


# =============================================================================
# Sidebar (commented out)
# =============================================================================
# with st.sidebar:
#     st.text_input("Summaries root", PARQ_SUMM_ROOT, key="paths_summ_root")
#     st.text_input("SQLite DB", DB_PATH, key="paths_db")
#     st.caption("Trade dates available")
#     try:
#         dates_str = ", ".join(sorted({p.split("trade_date=")[-1].split("/")[0]
#                                       for p in glob.glob(os.path.join(PARQ_ROOT, "trade_date=*"))}))
#         st.write(dates_str or "—")
#     except Exception as e:
#         st.write(f"(Path error: {e})")


# =============================================================================
# Global state
# =============================================================================
all_dates = list_trade_dates()
default_date = PREFS.get("last_date") or (all_dates[-1] if all_dates else date.today().isoformat())
st.session_state.setdefault("work_date", default_date)
work_date = st.session_state["work_date"]

labels_for_default = list_snapshot_labels(work_date)
default_label = PREFS.get("last_label") or (labels_for_default[-1] if labels_for_default else "S1")
st.session_state.setdefault("work_label", default_label)
work_label = st.session_state["work_label"]


# =============================================================================
# Tabs
# =============================================================================
TARGET_DELTA = 0.20
tabs = st.tabs([
    "Strategy helper",
    "KaChing (Put Calendar)",
    "Filters",
    f"Picks (~{TARGET_DELTA:.2f}Δ)",
    "Snapshots",
    "Trade Journal",
    "Help / Definitions",
    "TradingView 🔍",
])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs  # 8 tabs ⇢ 8 vars (fixed)

# -- Tab 1: Strategy helper (modular)
with tab1:
    st.subheader("Strategy helper")
    render_strategy_helper(
        all_dates, PREFS, list_snapshot_labels,
        load_snapshot_df, symbols_in_snapshot,
        fetch_trend_from_yf, get_weekly_trend, idea_insert
    )

# -- Tab 2: KaChing
with tab2:
    render_calendar_options()

# -- Tab 3: Filters (editable, saved)
with tab3:
    st.subheader("Finder & Strategy Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        finder_min_vol_oi  = st.number_input("Finder: Min Volume/OI (either)", 0, 100000, value=int(PREFS["finder_min_vol_oi"]), step=50)
        liq_min_vol_oi     = st.number_input("Strategy: Min Volume/OI (either)", 0, 100000, value=int(PREFS["liq_min_vol_oi"]), step=10)
    with c2:
        finder_target_width = st.number_input("Finder: Target spread width ($)", 0.5, 100.0, value=float(PREFS["finder_target_width"]), step=0.5)
        width_fuzz          = st.number_input("Finder: Width tolerance ($)", 0.0, 5.0, value=float(PREFS["finder_allow_width_fuzz"]), step=0.01)
    with c3:
        target_debit_pct  = st.slider("Highlight: Debit (% width)", 0.05, 0.95, float(PREFS["target_debit_pct"]), 0.05)
        target_credit_pct = st.slider("Highlight: Credit (% width)", 0.05, 0.95, float(PREFS["target_credit_pct"]), 0.05)

    st.markdown("##### Delta windows & guardrails")
    d1, d2, d3 = st.columns(3)
    with d1:
        bull_long_min = st.slider("Bullish long call Δ min", 0.05, 0.90, float(PREFS["bull_long_call_delta_min"]), 0.01)
        bull_long_max = st.slider("Bullish long call Δ max", 0.10, 0.95, float(PREFS["bull_long_call_delta_max"]), 0.01)
    with d2:
        bear_short_min = st.slider("Bearish short put Δ min", 0.05, 0.90, float(PREFS["bear_short_put_delta_min"]), 0.01)
        bear_short_max = st.slider("Bearish short put Δ max", 0.10, 0.95, float(PREFS["bear_short_put_delta_max"]), 0.01)
    with d3:
        neutral_short_delta = st.slider("Neutral short Δ (~)", 0.05, 0.50, float(PREFS["neutral_short_delta"]), 0.01)

    g1, g2 = st.columns(2)
    with g1:
        max_call_pct_above = st.slider("Max short CALL % above spot (weekly)", 0.00, 0.50, float(PREFS["max_call_short_pct_above"]), 0.01)
    with g2:
        max_put_pct_below  = st.slider("Max short PUT % below spot (weekly)", 0.00, 0.50, float(PREFS["max_put_short_pct_below"]), 0.01)

    if st.button("Save filters"):
        _write_prefs({
            **PREFS,
            "finder_min_vol_oi": int(finder_min_vol_oi),
            "finder_target_width": float(finder_target_width),
            "finder_allow_width_fuzz": float(width_fuzz),
            "liq_min_vol_oi": int(liq_min_vol_oi),
            "bull_long_call_delta_min": float(bull_long_min),
            "bull_long_call_delta_max": float(bull_long_max),
            "bear_short_put_delta_min": float(bear_short_min),
            "bear_short_put_delta_max": float(bear_short_max),
            "neutral_short_delta": float(neutral_short_delta),
            "max_call_short_pct_above": float(max_call_pct_above),
            "max_put_short_pct_below": float(max_put_pct_below),
            "target_debit_pct": float(target_debit_pct),
            "target_credit_pct": float(target_credit_pct),
        })
        st.success("Saved. Reload Strategy helper to apply.")

# -- Tab 4: Picks
with tab4:
    st.subheader("~0.20Δ Picks")
    r1, r2 = st.columns(2)
    p_date = r1.selectbox("Trade date", all_dates or [work_date],
                          index=max(0, (all_dates or [work_date]).index(work_date)))
    p_labels = list_snapshot_labels(p_date)
    p_label = r2.selectbox("Snapshot label", p_labels or ["S1"],
                           index=max(0, (p_labels or ["S1"]).index(work_label if p_labels else "S1")))
    df_picks = load_picks_df(p_date, p_label)
    if df_picks.empty:
        st.warning("No picks saved for this date/snapshot.")
    else:
        st.caption(f"{len(df_picks):,} picks")
        st.dataframe(df_picks, use_container_width=True, height=560)

# -- Tab 5: Snapshots
with tab5:
    st.subheader("Snapshot Browser")
    r1, r2 = st.columns(2)
    s_date = r1.selectbox("Trade date", all_dates or [work_date],
                          index=max(0, (all_dates or [work_date]).index(work_date)),
                          key="snap_trade_date")
    s_labels = list_snapshot_labels(s_date)
    s_label = r2.selectbox("Snapshot label", s_labels or ["S1"],
                           index=max(0, (s_labels or ["S1"]).index(work_label if s_labels else "S1")),
                           key="snap_snapshot_label")
    df_snap = load_snapshot_df(s_date, s_label)
    if df_snap.empty:
        st.warning("No rows found for this date/snapshot.")
    else:
        st.caption(f"{len(df_snap):,} rows from trade_date={s_date} / snapshot={s_label}")
        st.dataframe(df_snap.head(500), use_container_width=True, height=520)
        with st.expander("Columns / dtypes"):
            st.write(pd.DataFrame({"dtype": df_snap.dtypes.astype(str)}))

# -- Tab 6: Trade Journal
with tab6:
    st.subheader("Trade Journal")

    st.markdown("### Ideas (staging)")
    ideas = ideas_load_df()
    if ideas.empty:
        st.info("No ideas yet. Save from the Strategy Helper.")
    else:
        st.dataframe(ideas, use_container_width=True, height=300)
        sel_ids = st.multiselect("Select idea IDs", ideas["id"].tolist())

        with st.expander("Promote to live trade"):
            trade_date_val = st.date_input("Trade date", value=date.today())
            entry_price = st.number_input("Entry price (per contract; debit or credit)", 0.0, 1_000_000.0, 1.00, 0.05)
            qty = st.number_input("Qty (contracts)", 1, 10_000, 1, 1)
            notes = st.text_input("Notes", "")
            if st.button("Create trade(s) from selected ideas"):
                for _id in sel_ids:
                    row = ideas.loc[ideas["id"] == _id].iloc[0]
                    trade_insert(
                        trade_date=str(trade_date_val),
                        symbol=row["symbol"],
                        strategy=row["strategy"],
                        expiry=row["expiry"],
                        legs_json=json.loads(row["legs_json"]) if row.get("legs_json") else {},
                        entry_price=float(entry_price),
                        qty=int(qty),
                        notes=notes
                    )
                st.success(f"Created {len(sel_ids)} trade(s).")

        with st.expander("Delete ideas"):
            del_ids = st.multiselect("IDs to delete", ideas["id"].tolist(), key="ideas_del_ids")
            if st.button("Delete selected ideas"):
                ideas_delete(del_ids)
                st.success(f"Deleted {len(del_ids)} idea(s). Refresh the tab.")

    st.divider()
    st.markdown("### Trades")
    trades = trades_load_df()
    if trades.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(trades, use_container_width=True, height=300)
        close_id = st.selectbox("Trade ID to close", [None] + trades["id"].tolist())
        exit_px = st.number_input("Exit price (per contract)", 0.0, 1_000_000.0, 0.50, 0.05)
        if st.button("Close trade"):
            if close_id is None:
                st.warning("Pick a trade ID.")
            else:
                trade_close(int(close_id), float(exit_px))
                st.success(f"Closed trade {close_id} at {exit_px:.2f}.")

# -- Tab 7: Help / Definitions
with tab7:
    st.subheader("Help / Definitions")
    st.markdown("""
- **Credit spread:** Receive a net credit up front; max profit is the credit; max loss is width − credit.
- **Debit spread:** Pay a net debit; max loss is the debit; max profit is width − debit.
- **KaChing (put calendar):** Long-dated protective put + sell weekly puts.
- **Δ (delta):** Sensitivity of option price to $1 move in the underlying.
    """)

# -- Tab 8: TradingView (placeholder)
with tab8:
    st.subheader("TradingView 🔍")
    st.info("Embed your charts here (e.g., components.iframe with your Pine links).")
