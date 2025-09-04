"""
tradier_utils.py
===================================================================
Utility to fetch Tradier options with Greeks (PUTs only by default)
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
def get_token():
    load_dotenv()
    token = os.getenv("TRADIER_TOKEN")
    if not token:
        raise ValueError("TRADIER_TOKEN not found in environment.")
    return token

headers = {
    "Authorization": f"Bearer {get_token()}",
    "Accept": "application/json"
}


def get_option_chain_with_greeks(symbol, expiration, greeks=True, option_type="put"):
    """
    Fetch option chain (default: PUTs) from Tradier for given expiration date.
    Returns a pandas DataFrame or empty if error.
    """
    params = {
        "symbol": symbol,
        "expiration": expiration,
        "greeks": str(greeks).lower(),
        "includeAllRoots": "true",
        "strikes": "",
        "option_type": option_type
    }

    try:
        response = requests.get(TRADIER_ENDPOINT, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        options = data.get("options", {}).get("option", [])

        if not options:
            print(f"[Tradier] ❌ No options returned for {symbol} on {expiration}")
        else:
            print(f"[Tradier] ✅ Retrieved {len(options)} {option_type.upper()} options for {symbol} on {expiration}")

        return pd.DataFrame(options)
    except requests.exceptions.RequestException as e:
        print(f"[Tradier] ❌ Request failed: {e}")
    except Exception as e:
        print(f"[Tradier] ❌ Unexpected error: {e}")
    return pd.DataFrame()
