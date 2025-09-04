def get_weekly_trend(symbol):
    try:
        df = yf.download(symbol, period="2mo", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]  # flatten columns

        if len(df) < 10:
            df = yf.download(symbol, period="6mo", interval="1d")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            if len(df) < 10:
                return "unknown", None, "Not enough data"

        df["SMA5"] = df["Close"].rolling(5).mean()
        df["SMA10"] = df["Close"].rolling(10).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(5).mean()
        loss = -delta.where(delta < 0, 0).rolling(5).mean()
        rs = gain / loss
        df["RSI5"] = 100 - (100 / (1 + rs))

        print(df.tail(15))  # See the last 15 rows and indicator columns
        print("Columns:", df.columns)
        print("Total rows:", len(df))

        valid = df.dropna(subset=["SMA5", "SMA10", "RSI5"])
        if valid.empty:
            missing_cols = [col for col in ["SMA5", "SMA10", "RSI5"] if df[col].isna().all()]
            return "unknown", None, f"Missing indicator columns: {missing_cols}, total rows: {len(df)}"
        last = valid.iloc[-1]
        rsi = last["RSI5"]
        sma5 = last["SMA5"]
        sma10 = last["SMA10"]

        rsi = round(rsi, 1)
        sma5 = round(sma5, 2)
        sma10 = round(sma10, 2)

        if sma5 > sma10 and rsi < 70:
            trend = "Bullish"
        elif sma5 < sma10 and rsi > 30:
            trend = "Bearish"
        else:
            trend = "Neutral"

        # Earnings date (optional)
        earnings_date = None
        try:
            yf_ticker = yf.Ticker(symbol)
            cal = yf_ticker.calendar
            if "Earnings Date" in cal.index:
                earnings_date = cal.loc["Earnings Date"].values[0]
        except Exception:
            pass

        debug_info = f"SMA5: {sma5}, SMA10: {sma10}, RSI5: {rsi}"
        return trend, earnings_date, debug_info

    except Exception as e:
        return "unknown", None, str(e)