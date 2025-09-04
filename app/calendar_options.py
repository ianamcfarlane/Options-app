import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json, os

from tradier_utils import get_option_chain_with_greeks, get_tradier_expirations
from embed_tradingview import render_tradingview_chart

# Earnings + trend from yfinance
import yfinance as yf

def fetch_trend(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="6mo")
        df["SMA10"] = df["Close"].rolling(10).mean()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA5"] = df["Close"].rolling(5).mean()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(5).mean()
        loss = -delta.where(delta < 0, 0).rolling(5).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        last = df["Close"].iloc[-1]
        t3 = "Bullish" if df["SMA10"].iloc[-1] > df["SMA20"].iloc[-1] else "Bearish"
        t1 = "Bullish" if df["SMA5"].iloc[-1] > df["SMA10"].iloc[-1] else "Bearish"
        return round(last, 2), t3, t1, round(df["RSI"].iloc[-1], 1)
    except:
        return 0, "-", "-", 0

def get_next_fridays():
    fridays = []
    today = datetime.today()
    while len(fridays) < 2:
        today += timedelta(days=1)
        if today.weekday() == 4:
            fridays.append(today.strftime("%Y-%m-%d"))
    return fridays[0], fridays[1]

def find_put(df, delta_target=0.2, min_bid=0.2, min_theta=None, max_vega=None):
    df = df.copy()
    df["delta"] = df.get("delta", pd.Series([-0.2] * len(df))).abs()
    if min_theta is not None:
        df["theta"] = df.get("theta", pd.Series([0.0] * len(df)))
        df = df[df["theta"] >= min_theta]
    if max_vega is not None:
        df["vega"] = df.get("vega", pd.Series([0.0] * len(df)))
        df = df[df["vega"] <= max_vega]
    df = df[df["bid"] >= min_bid]
    df["delta_diff"] = (df["delta"] - delta_target).abs()
    if df.empty:
        return None
    return df.sort_values("delta_diff").iloc[0]

def save_trade_idea(symbol, long_put, short_put, cost, long_exp, short_exp):
    idea = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "strategy": "Put Calendar",
        "expiry": short_exp,
        "legs_json": [
            {"type": "long_put", "strike": float(long_put.strike), "expiry": long_exp},
            {"type": "short_put", "strike": float(short_put.strike), "expiry": short_exp}
        ],
        "est_price": round(cost, 2),
        "notes": f"Long {long_put.strike} / Short {short_put.strike}"
    }
    try:
        path = os.path.join("Options", "trade_ideas.json")
        existing = []
        if os.path.exists(path):
            with open(path, "r") as f:
                existing = json.load(f)
        existing.append(idea)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        return True
    except:
        return False

def render():
    st.subheader("📆 KaChing Calendar Strategy (with Tradier)")

    symbol = st.text_input("Symbol", value="SOFI").upper()
    if not symbol:
        return

    last, trend3, trend1, rsi = fetch_trend(symbol)
    try:
        earnings = yf.Ticker(symbol).calendar.loc["Earnings Date"].values[0].strftime("%Y-%m-%d")
    except:
        earnings = "N/A"

    st.markdown(f"**Price:** ${last} | 3M: {trend3} | Weekly: {trend1} (RSI: {rsi}) | Earnings: {earnings}")
    render_tradingview_chart(
        symbol=symbol,
        interval="D",
        range_="3M",
        theme="dark",
        studies=("Volume@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"),
        allow_symbol_change=True,
        hide_side_toolbar=False,
        height=620
    )
    st.divider()

    # Strategy config
    with st.expander("🔧 Configure Strategy"):
        col1, col2, col3 = st.columns(3)
        with col1:
            delta_target = st.slider("Target Δ", 0.1, 0.4, 0.2, 0.01)
            long_dte = st.slider("Min Long Put DTE", 100, 600, 120, 10)
        with col2:
            min_bid = st.number_input("Min Premium", 0.05, 5.0, 0.25, 0.05)
            min_theta = st.number_input("Min Theta", 0.0, 2.0, 0.10, 0.05)
        with col3:
            max_vega = st.number_input("Max Vega", 0.0, 2.0, 1.5, 0.1)

    # Expirations
    expirations = get_tradier_expirations(symbol)
    long_exp = next((d for d in expirations
                     if (datetime.strptime(d, "%Y-%m-%d") - datetime.today()).days >= long_dte), None)
    short1, short2 = get_next_fridays()

    df_long = get_option_chain_with_greeks(symbol, long_exp)
    df_short1 = get_option_chain_with_greeks(symbol, short1)
    df_short2 = get_option_chain_with_greeks(symbol, short2)

    long_put = find_put(df_long, delta_target, min_bid, min_theta, max_vega)
    short_put1 = find_put(df_short1, delta_target, min_bid, min_theta, max_vega)
    short_put2 = find_put(df_short2, delta_target, min_bid, min_theta, max_vega)

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"### {short1}")
        if isinstance(short_put1, pd.Series):
            st.write(f"PUT {short_put1.strike} @ ${short_put1.bid} (Δ {short_put1.delta:.2f})")
        else:
            st.warning("⚠️ No match for this week")

    with colB:
        st.markdown(f"### {short2}")
        if isinstance(short_put2, pd.Series):
            st.write(f"PUT {short_put2.strike} @ ${short_put2.bid} (Δ {short_put2.delta:.2f})")
        else:
            st.warning("⚠️ No match for next week")

    st.divider()
    if isinstance(long_put, pd.Series):
        st.success(f"Long PUT: {long_exp} @ {long_put.strike} (${long_put.ask:.2f}, Δ {long_put.delta:.2f})")

    if isinstance(long_put, pd.Series) and isinstance(short_put1, pd.Series):
        cost = float(long_put.ask) - float(short_put1.bid)
        st.info(f"Net Calendar Cost: ${cost:.2f} per share")
        if st.button("💾 Save Trade Idea"):
            if save_trade_idea(symbol, long_put, short_put1, cost, long_exp, short1):
                st.success("Saved!")
            else:
                st.error("Failed to save")