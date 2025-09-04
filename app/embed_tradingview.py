# embed_tradingview.py
# Generated: 2025-09-03
# =========================================================
# TradingView <iframe> embed for Streamlit (KaChing tab)
# - Dynamically updates by ticker
# - Supports daily/weekly toggle, range, theme
# - Preloads tv-basicstudies (Volume, RSI, MACD)
# - Uses st.components.v1.html to render responsive iframe
# =========================================================

from __future__ import annotations
from typing import Iterable, List, Tuple
from urllib.parse import quote_plus
import html

import streamlit as st
from streamlit.components.v1 import html as st_html


# ----------------------------- Public API -----------------------------

def render_tradingview_chart(
    symbol: str,
    *,
    interval: str = "D",
    range_: str = "3M",
    theme: str = "dark",
    style: int = 1,
    studies: Iterable[str] = ("Volume@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"),
    locale: str = "en",
    allow_symbol_change: bool = True,
    hide_side_toolbar: bool = False,
    height: int = 600,
    container_width: str = "100%",
    scrolling: bool = False,
) -> None:
    """
    Render a TradingView chart as an iframe in Streamlit.

    @param symbol: The symbol to display, e.g., "NASDAQ:AAPL", "NYSE:SOFI", "NVDA".
    @param interval: Chart interval. Examples: "1", "5", "15", "60", "D", "W", "M".
    @param range_: Visible range. Examples: "1D", "5D", "1M", "3M", "6M", "12M", "YTD".
    @param theme: "light" or "dark".
    @param style: Chart style. 1=candles, 3=bars, 4=hollow candles, 9=area, etc.
    @param studies: Iterable of indicator ids (tv-basicstudies supported in embed).
    @param locale: UI locale, e.g., "en".
    @param allow_symbol_change: Allow users to change ticker from the widget.
    @param hide_side_toolbar: Hide the side toolbar if True.
    @param height: Iframe height in pixels.
    @param container_width: CSS width for the iframe container (e.g. "100%", "900").
    @param scrolling: Whether the iframe should allow scrolling.
    """

    # --- sanitize inputs ---
    clean_symbol = _normalize_symbol(symbol)
    if not clean_symbol:
        st.warning("Enter a valid ticker (e.g., **NVDA** or **NASDAQ:NVDA**).")
        return

    # TradingView expects comma-separated, URL-encoded studies string
    studies_param = _encode_studies(studies)

    # Build the widget URL
    url = (
        "https://s.tradingview.com/widgetembed/?"
        f"symbol={quote_plus(clean_symbol)}"
        f"&interval={quote_plus(interval)}"
        f"&range={quote_plus(range_)}"
        f"&theme={quote_plus(theme)}"
        f"&style={quote_plus(str(style))}"
        f"&studies={studies_param}"
        f"&locale={quote_plus(locale)}"
        "&toolbarbg=f1f3f6"
        f"&allow_symbol_change={'true' if allow_symbol_change else 'false'}"
        f"&hide_side_toolbar={'true' if hide_side_toolbar else 'false'}"
    )

    # Assemble responsive iframe
    # Note: width is set by the container; height is fixed for stability inside Streamlit.
    iframe_html = f"""
    <div style="width:{html.escape(container_width)};max-width:100%;">
      <iframe
        src="{html.escape(url)}"
        width="100%"
        height="{int(height)}"
        frameborder="0"
        allowtransparency="true"
        scrolling="{ 'yes' if scrolling else 'no' }">
      </iframe>
    </div>
    """

    st_html(iframe_html, height=height, scrolling=scrolling)


# --------------------------- Helper Functions ---------------------------

def _normalize_symbol(symbol: str | None) -> str:
    """
    Normalize a user-entered symbol to a TradingView-friendly string.
    - Pass-through if already prefixed: "NASDAQ:AAPL".
    - Strip whitespace; uppercase non-prefixed tickers like "nvda" -> "NVDA".
    """
    if not symbol:
        return ""
    s = symbol.strip()
    if ":" in s:
        # Looks like EXCHANGE:SYMBOL already
        return s.upper()
    # Plain ticker: let TV infer the exchange
    return s.upper()


def _encode_studies(studies: Iterable[str]) -> str:
    """
    Convert an iterable of study ids to a comma-separated, URL-encoded string.
    Examples:
      ["Volume@tv-basicstudies", "RSI@tv-basicstudies"] ->
      "Volume%40tv-basicstudies%2CRSI%40tv-basicstudies"
    """
    studies_list: List[str] = [s for s in (studies or []) if isinstance(s, str) and s.strip()]
    if not studies_list:
        return ""
    # TradingView expects literal commas between encoded studies
    return quote_plus(",".join(studies_list))


# ------------------------- Example Standalone Page -------------------------

def _example_page() -> None:
    """
    Minimal, self-contained Streamlit page demonstrating dynamic ticker control.
    - Ticker input
    - Daily/Weekly toggle
    - Range selector (3M default)
    - Indicator chooser
    """
    st.set_page_config(page_title="KaChing – TradingView Embed", layout="wide")
    st.title("📈 TradingView Embed (KaChing)")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col1:
        ticker = st.text_input("Ticker (e.g., NVDA or NASDAQ:NVDA)", value="NASDAQ:NVDA").strip()

    with col2:
        tf = st.selectbox("Timeframe", options=["D (Daily)", "W (Weekly)"], index=0)
        interval = "W" if tf.startswith("W") else "D"

    with col3:
        range_ = st.selectbox("Range", options=["1M", "3M", "6M", "12M", "YTD"], index=1)

    with col4:
        theme = st.selectbox("Theme", options=["dark", "light"], index=0)

    # Indicators
    all_inds: List[Tuple[str, str]] = [
        ("Volume", "Volume@tv-basicstudies"),
        ("RSI", "RSI@tv-basicstudies"),
        ("MACD", "MACD@tv-basicstudies"),
        ("Stochastic", "StochasticRSI@tv-basicstudies"),
        ("EMA (50)", "Moving Average Exponential@tv-basicstudies"),
    ]
    pretty_to_id = {p: i for p, i in all_inds}
    chosen_pretty = st.multiselect(
        "Indicators",
        options=[p for p, _ in all_inds],
        default=["Volume", "RSI", "MACD"],
    )
    studies = [pretty_to_id[p] for p in chosen_pretty]

    # Render
    render_tradingview_chart(
        symbol=ticker,
        interval=interval,
        range_=range_,
        theme=theme,
        style=1,  # candles
        studies=studies,
        allow_symbol_change=True,
        hide_side_toolbar=False,
        height=620,
    )


# Allow running this file directly with:  streamlit run embed_tradingview.py
if __name__ == "__main__":
    _example_page()
