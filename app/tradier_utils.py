"""
tradier_utils.py
===================================================================
Utility to fetch Tradier options with Greeks and expiration dates
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_token():
    token = os.getenv("TRADIER_TOKEN")
    if not token:
        raise ValueError("TRADIER_TOKEN not found.")
    return token

def get_headers():
    return {
        "Authorization": f"Bearer {get_token()}",
        "Accept": "application/json"
    }

def get_option_chain_with_greeks(symbol, expiration, greeks=True, option_type="put"):
    """
    Fetch option chain (PUTs by default) from Tradier for given expiration date.
    Returns a pandas DataFrame or empty if error.
    """
    url = "https://api.tradier.com/v1/markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": str(greeks).lower(),
        "includeAllRoots": "true",
        "strikes": "",
        "option_type": option_type
    }

    try:
        response = requests.get(url, params=params, headers=get_headers())
        response.raise_for_status()
        data = response.json()
        options = data.get("options", {}).get("option", [])
        return pd.DataFrame(options)
    except Exception as e:
        print(f"[Tradier] ❌ Option chain error: {e}")
        return pd.DataFrame()

def get_tradier_expirations(symbol):
    """
    Return available expiration dates for a given symbol from Tradier.
    """
    url = "https://api.tradier.com/v1/markets/options/expirations"
    params = {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"}
    try:
        response = requests.get(url, headers=get_headers(), params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("expirations", {}).get("date", [])
    except Exception as e:
        print(f"[Tradier] ❌ Expiration fetch error: {e}")
        return []