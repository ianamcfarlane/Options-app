-- Symbols you track
CREATE TABLE IF NOT EXISTS symbols (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  is_active INTEGER DEFAULT 1
);

-- Static option contract metadata (OCC symbols)
CREATE TABLE IF NOT EXISTS option_contracts (
  option_symbol TEXT PRIMARY KEY,
  symbol TEXT,
  right TEXT CHECK (right IN ('C','P')),
  strike REAL,
  expiration DATE,
  multiplier INTEGER DEFAULT 100
);

-- One row per snapshot run (audit)
CREATE TABLE IF NOT EXISTS snapshot_runs (
  run_id TEXT PRIMARY KEY,
  snapshot_ts TIMESTAMP,
  snapshot_label TEXT,          -- S1/S2/S3/S4
  symbols_count INTEGER,
  rows_ingested INTEGER,
  expiries_covered TEXT,
  status TEXT,
  message TEXT
);

-- Small, fast per‑symbol summary at each snapshot (for the UI)
CREATE TABLE IF NOT EXISTS snapshot_symbol_summary (
  run_id TEXT,
  symbol TEXT,
  dte_min INTEGER,
  dte_max INTEGER,
  iv_rank REAL,
  emove_1w REAL,
  short_put_020_delta REAL,
  short_put_020_credit REAL,
  PRIMARY KEY (run_id, symbol)
);

-- OPTIONAL: if you decide to store chains in SQLite instead of Parquet
-- (Otherwise you can skip this table and keep chains in Parquet only.)
CREATE TABLE IF NOT EXISTS option_quotes_intraday (
  option_symbol TEXT,
  symbol TEXT,
  right TEXT,
  strike REAL,
  expiration DATE,
  bid REAL, ask REAL, last REAL, mid REAL,
  volume INTEGER, open_interest INTEGER,
  iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
  dte INTEGER,
  snapshot_ts TIMESTAMP,
  snapshot_label TEXT,
  run_id TEXT,
  source TEXT,
  PRIMARY KEY (option_symbol, snapshot_ts)
);
CREATE INDEX IF NOT EXISTS idx_quotes_by_symbol_time
  ON option_quotes_intraday(symbol, snapshot_ts);
