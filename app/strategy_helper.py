"""
strategy_helper.py
==================
Streamlit Strategy Helper tab (modularized from app.py).

This tab screens options chains for:
- Call/Put Debit Spreads
- Call/Put Credit Spreads
- Long Straddles / Long Strangles
- Long Call/Put Calendars

It expects the host app (app.py) to supply snapshot loaders and simple trend helpers.

Public API:
    render(all_dates, PREFS, list_snapshot_labels, load_snapshot_df,
           symbols_in_snapshot, fetch_trend_from_yf, get_weekly_trend, idea_insert)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import math
import pandas as pd
import streamlit as st


# ---------- Small UI helpers ----------

def _fmt_money(x: float) -> str:
    """Format a number as $X.XX with sign."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _trend_icon(trend: str) -> str:
    """Return a small emoji hint for trend labels."""
    t = (trend or "").strip().lower()
    if t.startswith("bull"):
        return "🟢⬆️"
    if t.startswith("bear"):
        return "🔴⬇️"
    if t.startswith("neut"):
        return "🟡⏸️"
    return "⚪❓"


# ---------- Strategy finders ----------
# All finders return a DataFrame with columns:
# ["Symbol","Expiry","Strategy","Legs","Est. price","Direction","Max Profit","Max Loss","ROC%"]

def _ensure_mid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mid price and |delta| convenience columns."""
    df = df.copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2
    if "adelta" not in df.columns:
        df["adelta"] = df["delta"].abs()
    return df


def find_call_debit_spreads(df: pd.DataFrame, min_delta: float = 0.35, max_delta: float = 0.55) -> pd.DataFrame:
    """Call Debit: buy lower strike call, sell higher strike call (same expiry)."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    calls = df[df["right"] == "C"].sort_values(["expiration", "strike"])
    for expiry, exp_calls in calls.groupby("expiration"):
        exp_calls = exp_calls.reset_index(drop=True)
        for i in range(len(exp_calls) - 1):
            buy = exp_calls.iloc[i]
            sell = exp_calls.iloc[i + 1]
            d = float(buy["delta"])
            if not (min_delta <= d <= max_delta):
                continue
            debit = float(buy["mid"]) - float(sell["mid"])
            width = abs(float(sell["strike"]) - float(buy["strike"]))
            max_profit = width - debit
            max_loss = debit
            roc = (max_profit / max_loss * 100) if max_loss > 0 else None
            results.append({
                "Symbol": buy["symbol"],
                "Expiry": expiry,
                "Strategy": "Call Debit Spread",
                "Legs": [
                    {"action": "Buy", "type": "Call", "strike": float(buy["strike"]), "price": float(buy["mid"])},
                    {"action": "Sell", "type": "Call", "strike": float(sell["strike"]), "price": float(sell["mid"])},
                ],
                "Est. price": round(debit, 2),
                "Direction": "Debit",
                "Max Profit": round(max_profit, 2),
                "Max Loss": round(max_loss, 2),
                "ROC%": None if roc is None else round(roc, 1),
            })
    return pd.DataFrame(results)


def find_call_credit_spreads(df: pd.DataFrame, min_delta: float = 0.18, max_delta: float = 0.35) -> pd.DataFrame:
    """Call Credit: sell lower strike call, buy higher strike call (same expiry). Bearish/neutral."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    calls = df[df["right"] == "C"].sort_values(["expiration", "strike"])
    for expiry, exp_calls in calls.groupby("expiration"):
        exp_calls = exp_calls.reset_index(drop=True)
        for i in range(len(exp_calls) - 1):
            sell = exp_calls.iloc[i]
            buy = exp_calls.iloc[i + 1]
            d = float(sell["delta"])
            if not (min_delta <= d <= max_delta):
                continue
            credit = float(sell["mid"]) - float(buy["mid"])
            width = abs(float(buy["strike"]) - float(sell["strike"]))
            max_profit = credit
            max_loss = max(width - credit, 0.0)
            roc = (max_profit / max_loss * 100) if max_loss > 0 else None
            results.append({
                "Symbol": sell["symbol"],
                "Expiry": expiry,
                "Strategy": "Call Credit Spread",
                "Legs": [
                    {"action": "Sell", "type": "Call", "strike": float(sell["strike"]), "price": float(sell["mid"])},
                    {"action": "Buy",  "type": "Call", "strike": float(buy["strike"]),  "price": float(buy["mid"])},
                ],
                "Est. price": round(credit, 2),
                "Direction": "Credit",
                "Max Profit": round(max_profit, 2),
                "Max Loss": round(max_loss, 2),
                "ROC%": None if roc is None else round(roc, 1),
            })
    return pd.DataFrame(results)


def find_put_debit_spreads(df: pd.DataFrame, min_delta: float = -0.55, max_delta: float = -0.35) -> pd.DataFrame:
    """Put Debit: buy higher strike put, sell lower strike put (same expiry). Bearish."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    puts = df[df["right"] == "P"].sort_values(["expiration", "strike"])
    for expiry, exp_puts in puts.groupby("expiration"):
        exp_puts = exp_puts.reset_index(drop=True)
        for i in range(len(exp_puts) - 1):
            buy = exp_puts.iloc[i]
            sell = exp_puts.iloc[i + 1]
            d = float(buy["delta"])
            if not (min_delta <= d <= max_delta):
                continue
            debit = float(buy["mid"]) - float(sell["mid"])
            width = abs(float(buy["strike"]) - float(sell["strike"]))
            max_profit = width - debit
            max_loss = debit
            roc = (max_profit / max_loss * 100) if max_loss > 0 else None
            results.append({
                "Symbol": buy["symbol"],
                "Expiry": expiry,
                "Strategy": "Put Debit Spread",
                "Legs": [
                    {"action": "Buy", "type": "Put", "strike": float(buy["strike"]), "price": float(buy["mid"])},
                    {"action": "Sell","type": "Put", "strike": float(sell["strike"]), "price": float(sell["mid"])},
                ],
                "Est. price": round(debit, 2),
                "Direction": "Debit",
                "Max Profit": round(max_profit, 2),
                "Max Loss": round(max_loss, 2),
                "ROC%": None if roc is None else round(roc, 1),
            })
    return pd.DataFrame(results)


def find_put_credit_spreads(df: pd.DataFrame, min_delta: float = -0.35, max_delta: float = -0.18) -> pd.DataFrame:
    """Put Credit: sell higher strike put, buy lower strike put (same expiry). Bullish/neutral."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    puts = df[df["right"] == "P"].sort_values(["expiration", "strike"])
    for expiry, exp_puts in puts.groupby("expiration"):
        exp_puts = exp_puts.reset_index(drop=True)
        for i in range(len(exp_puts) - 1):
            sell = exp_puts.iloc[i]
            buy = exp_puts.iloc[i + 1]
            d = float(sell["delta"])
            if not (min_delta <= d <= max_delta):
                continue
            credit = float(sell["mid"]) - float(buy["mid"])
            width = abs(float(buy["strike"]) - float(sell["strike"]))
            max_profit = credit
            max_loss = max(width - credit, 0.0)
            roc = (max_profit / max_loss * 100) if max_loss > 0 else None
            results.append({
                "Symbol": sell["symbol"],
                "Expiry": expiry,
                "Strategy": "Put Credit Spread",
                "Legs": [
                    {"action": "Sell", "type": "Put", "strike": float(sell["strike"]), "price": float(sell["mid"])},
                    {"action": "Buy",  "type": "Put", "strike": float(buy["strike"]),  "price": float(buy["mid"])},
                ],
                "Est. price": round(credit, 2),
                "Direction": "Credit",
                "Max Profit": round(max_profit, 2),
                "Max Loss": round(max_loss, 2),
                "ROC%": None if roc is None else round(roc, 1),
            })
    return pd.DataFrame(results)


def find_long_straddles(df: pd.DataFrame) -> pd.DataFrame:
    """Long Straddle: buy ATM call and ATM put (same expiry)."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    for expiry, exp in df.groupby("expiration"):
        calls = exp[exp["right"] == "C"]
        puts = exp[exp["right"] == "P"]
        if calls.empty or puts.empty:
            continue
        call = calls.iloc[calls["adelta"].sub(0).abs().argsort().iloc[0]]
        put = puts.iloc[puts["adelta"].sub(0).abs().argsort().iloc[0]]
        est = float(call["mid"]) + float(put["mid"])
        results.append({
            "Symbol": call["symbol"],
            "Expiry": expiry,
            "Strategy": "Long Straddle",
            "Legs": [
                {"action": "Buy", "type": "Call", "strike": float(call["strike"]), "price": float(call["mid"])},
                {"action": "Buy", "type": "Put",  "strike": float(put["strike"]),  "price": float(put["mid"])},
            ],
            "Est. price": round(est, 2),
            "Direction": "Debit",
            "Max Profit": None,
            "Max Loss": round(est, 2),
            "ROC%": None,
        })
    return pd.DataFrame(results)


def find_long_strangles(df: pd.DataFrame, call_off: int = 1, put_off: int = 1) -> pd.DataFrame:
    """Long Strangle: buy slightly OTM call+put (same expiry)."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    for expiry, exp in df.groupby("expiration"):
        calls = exp[exp["right"] == "C"].sort_values("strike")
        puts = exp[exp["right"] == "P"].sort_values("strike")
        if calls.empty or puts.empty:
            continue
        call = calls.iloc[min(call_off, len(calls) - 1)]
        put = puts.iloc[max(0, len(puts) - 1 - put_off)]
        est = float(call["mid"]) + float(put["mid"])
        results.append({
            "Symbol": call["symbol"],
            "Expiry": expiry,
            "Strategy": "Long Strangle",
            "Legs": [
                {"action": "Buy", "type": "Call", "strike": float(call["strike"]), "price": float(call["mid"])},
                {"action": "Buy", "type": "Put",  "strike": float(put["strike"]),  "price": float(put["mid"])},
            ],
            "Est. price": round(est, 2),
            "Direction": "Debit",
            "Max Profit": None,
            "Max Loss": round(est, 2),
            "ROC%": None,
        })
    return pd.DataFrame(results)


def find_long_call_calendar_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Long Call Calendar: buy longer-dated call, sell nearer call, same strike."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    calls = df[df["right"] == "C"].sort_values(["strike", "expiration"])
    for strike, s in calls.groupby("strike"):
        expiries = s["expiration"].unique()
        if len(expiries) < 2:
            continue
        short_exp, long_exp = expiries[0], expiries[-1]
        sell = s[s["expiration"] == short_exp].iloc[0]
        buy = s[s["expiration"] == long_exp].iloc[0]
        debit = float(buy["mid"]) - float(sell["mid"])
        results.append({
            "Symbol": buy["symbol"],
            "Expiry": f"{short_exp}/{long_exp}",
            "Strategy": "Long Call Calendar Spread",
            "Legs": [
                {"action": "Buy", "type": "Call", "strike": float(strike), "price": float(buy["mid"]),  "expiry": long_exp},
                {"action": "Sell","type": "Call", "strike": float(strike), "price": float(sell["mid"]), "expiry": short_exp},
            ],
            "Est. price": round(debit, 2),
            "Direction": "Debit",
            "Max Profit": None,
            "Max Loss": round(debit, 2),
            "ROC%": None,
        })
    return pd.DataFrame(results)


def find_long_put_calendar_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Long Put Calendar: buy longer-dated put, sell nearer put, same strike."""
    df = _ensure_mid_cols(df)
    results: List[Dict[str, Any]] = []
    puts = df[df["right"] == "P"].sort_values(["strike", "expiration"])
    for strike, s in puts.groupby("strike"):
        expiries = s["expiration"].unique()
        if len(expiries) < 2:
            continue
        short_exp, long_exp = expiries[0], expiries[-1]
        sell = s[s["expiration"] == short_exp].iloc[0]
        buy = s[s["expiration"] == long_exp].iloc[0]
        debit = float(buy["mid"]) - float(sell["mid"])
        results.append({
            "Symbol": buy["symbol"],
            "Expiry": f"{short_exp}/{long_exp}",
            "Strategy": "Long Put Calendar Spread",
            "Legs": [
                {"action": "Buy", "type": "Put", "strike": float(strike), "price": float(buy["mid"]),  "expiry": long_exp},
                {"action": "Sell","type": "Put", "strike": float(strike), "price": float(sell["mid"]), "expiry": short_exp},
            ],
            "Est. price": round(debit, 2),
            "Direction": "Debit",
            "Max Profit": None,
            "Max Loss": round(debit, 2),
            "ROC%": None,
        })
    return pd.DataFrame(results)


# ---------- Suggestion rubric ----------

def _suggest_strategy_row(r: pd.Series, weekly_trend: str) -> str:
    """
    Heuristic suggestion:
      - Bullish: Calls (0.35–0.55Δ) → Call Debit; Puts (-0.35 to -0.18Δ) → Put Credit
      - Bearish: Puts (-0.55 to -0.35Δ) → Put Debit; Calls (0.18–0.35Δ) → Call Credit
      - Else: Calendar/Neutral
    """
    right = r.get("right", "")
    d = float(r.get("delta", 0))
    t = (weekly_trend or "").lower()
    if t.startswith("bull"):
        if right == "C" and 0.35 <= d <= 0.55:
            return "Call Debit"
        if right == "P" and -0.35 <= d <= -0.18:
            return "Put Credit"
    if t.startswith("bear"):
        if right == "P" and -0.55 <= d <= -0.35:
            return "Put Debit"
        if right == "C" and 0.18 <= d <= 0.35:
            return "Call Credit"
    return "Calendar/Neutral"


# ---------- Public render() ----------

def render(
    all_dates: list,
    PREFS: dict,
    list_snapshot_labels,
    load_snapshot_df,
    symbols_in_snapshot,
    fetch_trend_from_yf,
    get_weekly_trend,
    idea_insert,
):
    st.subheader("Strategy helper")

    # --- Hide selectors, auto-select latest trade date and snapshot label ---
    h_date = all_dates[-1] if all_dates else date.today().isoformat()
    h_labels = list_snapshot_labels(h_date)
    h_label = h_labels[-1] if h_labels else "S1"

    df_all = load_snapshot_df(h_date, h_label)
    if df_all.empty:
        st.warning("No snapshot data found for this date/label.")
        st.stop()

    # --- Symbol picker (single dropdown, active) ---
    syms = ["All"] + symbols_in_snapshot(h_date, h_label)
    pref_sym = PREFS.get("default_symbol", "AAPL")
    sym_default = "All"

    m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1])
    sym = m1.selectbox("Symbol", syms, index=syms.index(sym_default), key="strat_sym_k2")

    # Filter df_all by symbol if not "All"
    if sym != "All":
        df_filtered_base = df_all[df_all["symbol"] == sym].copy()
    else:
        df_filtered_base = df_all.copy()

    df_filtered_base["mid"] = (df_filtered_base["bid"].fillna(0) + df_filtered_base["ask"].fillna(0)) / 2
    df_filtered_base["adelta"] = df_filtered_base["delta"].abs()

    trend, last_px, _, dbg, debug_log = fetch_trend_from_yf(sym if sym != "All" else pref_sym, fallback_df=df_all)
    trend2, earnings, debug2 = get_weekly_trend(sym if sym != "All" else pref_sym)
    m2.metric("Spot (yf/snapshot)", f"{last_px:.2f}" if not math.isnan(last_px) else "—")
    m3.metric("Weekly Trend (SMA5/10, RSI)", _trend_icon(trend2))
    m4.metric("Trend (3mo SMA10/20)", _trend_icon(trend))
    user_view = m5.selectbox(
        "Your view", ["Bullish", "Neutral", "Bearish"],
        index=["Bullish", "Neutral", "Bearish"].index(trend if trend in ["Bullish", "Neutral", "Bearish"] else "Neutral"),
        key="bias_sel_k"
    )

    if earnings:
        st.caption(f"Earnings Date: {earnings}")
    st.caption(f"Trend debug: {debug2}")

    # --- Top 10 option plays (auto scan, no strategy picker) ---
    st.markdown("#### Top 10 Option Plays (auto-selected by liquidity/volume/OI)")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        min_vol_oi = st.slider("Min Volume/OI", 0, 1000, 200, 10)
    with filter_col2:
        min_price = st.slider("Min Option Price", 0.0, 10.0, 0.5, 0.1)
    with filter_col3:
        max_price = st.slider("Max Option Price", 0.0, 20.0, 5.0, 0.1)

    df_filtered = df_filtered_base[
        ((df_filtered_base["volume"].fillna(0) >= min_vol_oi) | (df_filtered_base["open_interest"].fillna(0) >= min_vol_oi)) &
        (df_filtered_base["mid"].fillna(0) >= min_price) &
        (df_filtered_base["mid"].fillna(0) <= max_price)
    ].copy()

    df_filtered["score"] = df_filtered["volume"].fillna(0) + df_filtered["open_interest"].fillna(0)

    # --- Strategy suggestion column ---
    def suggest_strategy(row):
        if row["right"] == "C":
            return "Call Debit Spread, Call Credit Spread, Long Call Calendar Spread"
        elif row["right"] == "P":
            return "Put Debit Spread, Put Credit Spread, Long Put Calendar Spread"
        else:
            return "Long Straddle, Long Strangle"

    df_filtered["Strategy Suggestion"] = df_filtered.apply(suggest_strategy, axis=1)

    top_plays = df_filtered.sort_values("score", ascending=False).head(10)

    if top_plays.empty:
        st.info("No option plays match the current filters.")
    else:
        st.dataframe(
            top_plays[["symbol", "expiration", "right", "strike", "mid", "volume", "open_interest", "score", "Strategy Suggestion"]],
            use_container_width=True
        )
        st.caption("Strategy suggestions are shown in the table.")

    st.markdown("#### Strategy Trade Ideas ")

    # --- Generate strategy ideas ---
    strategy_dfs = [
        find_call_debit_spreads(df_filtered_base),
        find_call_credit_spreads(df_filtered_base),
        find_put_debit_spreads(df_filtered_base),
        find_put_credit_spreads(df_filtered_base),
        find_long_straddles(df_filtered_base),
        find_long_strangles(df_filtered_base),
        find_long_call_calendar_spreads(df_filtered_base),
        find_long_put_calendar_spreads(df_filtered_base),
    ]
    trade_ideas = pd.concat(strategy_dfs, ignore_index=True)
    if not trade_ideas.empty:
        def format_legs(legs):
            return ", ".join(
                f"{leg['action']} {leg['type']} {leg['strike']} @ {leg['price']:.2f}" +
                (f" (exp {leg['expiry']})" if 'expiry' in leg else "")
                for leg in legs
            )
        trade_ideas["Legs"] = trade_ideas["Legs"].apply(format_legs)
        for col in ["Est. price", "Max Profit", "Max Loss"]:
            if col in trade_ideas:
                trade_ideas[col] = (trade_ideas[col].fillna(0) * 100).round(2)
                # available_strikes = sorted(set(df_filtered_base["strike"].unique()))
                # selected_strike = st.selectbox("Filter Strategy Ideas by strike", ["All"] + [str(s) for s in available_strikes], key="strike_filter_k")
                # if selected_strike != "All":
                #     trade_ideas_filtered = trade_ideas[trade_ideas["Legs"].str.contains(f" {selected_strike} ")]
                # else:
                #     trade_ideas_filtered = trade_ideas

                # st.dataframe(
                #     trade_ideas_filtered[
                #         ["Symbol", "Expiry", "Strategy", "Legs", "Est. price", "Direction", "Max Profit", "Max Loss", "ROC%"]
                #     ],
                #     use_container_width=True
                # )
                # else:
                #     st.info("No trade ideas found for current filters.")

                # st.divider()

                # ---- Output + Save selected to Ideas ----
                # out = []  # Initialize out as an empty list to avoid NameError
                # if out:
                #     df_out = pd.DataFrame(out)
                #     st.subheader("Suggestions")
                #     st.caption("“Est. price” is per 1 contract: debit for debit spreads; credit for credit spreads.")
                #     st.dataframe(df_out, use_container_width=True)

                #     choices = list(range(len(df_out)))
                #     pick_idx = st.multiselect("Select rows to save as Ideas", choices, key="sel_rows_k")
                #     notes_txt = st.text_input("Notes (applies to all selected ideas)", "", key="idea_notes_k")
                #     if st.button("Save selected to Ideas"):
                #         for i in pick_idx:
                #             r = df_out.iloc[i].to_dict()
                #             idea_insert(
                #                 snapshot_date=h_date,
                #                 snapshot_label=h_label,
                #                 symbol=sym,
                #                 expiry=r.get("Expiry",""),
                #                 strategy=r.get("Strategy",""),
                #                 legs_json=r.get("Legs", {}),
                #                 est_price=float(r.get("Est. price") or 0.0),
                #                 direction=r.get("Direction"),
                #                 notes=notes_txt or ""
                #             )
                #         st.success(f"Saved {len(pick_idx)} idea(s) to Ideas inbox.")
                # else:
                #     st.info("No suggestions for current filters.")

                # st.caption("All results per 1 option contract (×100 shares). Weekly (same-expiry) only. Guardrails keep strikes near spot.")

    if not top_plays.empty:
        # Build choices for selectbox
        top_choices = [
            f"{row['symbol']} {row['right']} {row['strike']} {row['expiration']}"
            for _, row in top_plays.iterrows()
        ]
        selected_play = st.selectbox(
            "Select an option play to filter strategies below",
            ["All"] + top_choices,
            key="top_play_select_k"
        )
    else:
        selected_play = "All"

    # --- Filter trade_ideas based on selection ---
    if selected_play != "All":
        parts = selected_play.split()
        sel_symbol, sel_right, sel_strike, sel_expiry = parts[0], parts[1], float(parts[2]), parts[3]
        trade_ideas_filtered = trade_ideas[
            (trade_ideas["Symbol"] == sel_symbol) &
            (trade_ideas["Legs"].str.contains(f"{sel_right} {sel_strike}")) &
            (trade_ideas["Expiry"].astype(str).str.contains(sel_expiry))
        ]
    else:
        trade_ideas_filtered = trade_ideas

    st.dataframe(
        trade_ideas_filtered[
            ["Symbol", "Expiry", "Strategy", "Legs", "Est. price", "Direction", "Max Profit", "Max Loss", "ROC%"]
        ],
        use_container_width=True
    )
