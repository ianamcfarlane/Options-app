# apply_schema.py
# Run:  python apply_schema.py --db "Y:\\db\\options.db" --sql "Y:\\etl\\init_schema.sql" [--seed]
import argparse, os, sqlite3, sys

def read_sql(path: str) -> str:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] SQL file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def apply_schema(db_path: str, sql_path: str):
    ddl = read_sql(sql_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(ddl)
        con.commit()
        print(f"[OK] Schema applied to {db_path}")
    finally:
        con.close()

def seed_symbols(db_path: str):
    seeds = [
        ("SPY","SPDR S&P 500","ARCA"), ("QQQ","Invesco QQQ","NASDAQ"),
        ("IWM","iShares Russell 2000","ARCA"), ("DIA","SPDR Dow Jones","ARCA"),
        ("XLF","Financial Select","ARCA"), ("XLK","Tech Select","ARCA"),
        ("XLE","Energy Select","ARCA"), ("XLI","Industrial Select","ARCA"),
        ("XLV","Health Care Select","ARCA"), ("XLY","Consumer Discretionary","ARCA"),
        ("AAPL","Apple","NASDAQ"), ("MSFT","Microsoft","NASDAQ"),
        ("NVDA","NVIDIA","NASDAQ"), ("AMD","AMD","NASDAQ"),
        ("META","Meta","NASDAQ"), ("AMZN","Amazon","NASDAQ"),
        ("GOOGL","Alphabet","NASDAQ"), ("TSLA","Tesla","NASDAQ"),
        ("NFLX","Netflix","NASDAQ"), ("AVGO","Broadcom","NASDAQ"),
        ("INTC","Intel","NASDAQ"), ("MU","Micron","NASDAQ"),
        ("QCOM","Qualcomm","NASDAQ"), ("ORCL","Oracle","NYSE"),
        ("CRM","Salesforce","NYSE"), ("COST","Costco","NASDAQ"),
        ("JPM","JPMorgan","NYSE"), ("BAC","Bank of America","NYSE"),
        ("WFC","Wells Fargo","NYSE"), ("GS","Goldman Sachs","NYSE"),
        ("MS","Morgan Stanley","NYSE"), ("UNH","UnitedHealth","NYSE"),
        ("MRK","Merck","NYSE"), ("PFE","Pfizer","NYSE"),
        ("LLY","Eli Lilly","NYSE"), ("XOM","Exxon","NYSE"),
        ("CVX","Chevron","NYSE"), ("PEP","PepsiCo","NASDAQ"),
        ("KO","Coca-Cola","NYSE"), ("NKE","Nike","NYSE"),
        ("DIS","Disney","NYSE"), ("BA","Boeing","NYSE"),
        ("CAT","Caterpillar","NYSE"), ("DE","Deere","NYSE"),
        ("UBER","Uber","NYSE"), ("ABNB","Airbnb","NASDAQ"),
        ("SHOP","Shopify","NYSE"), ("PLTR","Palantir","NYSE"),
        ("SOFI","SoFi","NASDAQ"), ("F","Ford","NYSE"),
        ("DKNG","DraftKings","NASDAQ")
    ]
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT OR IGNORE INTO symbols(symbol,name,exchange) VALUES (?,?,?)",
            seeds
        )
        con.commit()
        print(f"[OK] Seeded {len(seeds)} symbols.")
    finally:
        con.close()

def show_tables(db_path: str):
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        print("[INFO] Tables:", ", ".join(tables))
    finally:
        con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",  default=r"Y:\db\options.db", help="Path to SQLite DB file")
    parser.add_argument("--sql", default=r"Y:\etl\init_schema.sql", help="Path to init_schema.sql")
    parser.add_argument("--seed", action="store_true", help="Seed 50 starter symbols")
    args = parser.parse_args()

    apply_schema(args.db, args.sql)
    if args.seed:
        seed_symbols(args.db)
    show_tables(args.db)
