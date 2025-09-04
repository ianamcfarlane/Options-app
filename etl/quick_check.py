import sqlite3
con = sqlite3.connect(r"Y:\db\options.db")
print(con.execute("SELECT COUNT(*) FROM symbols").fetchone())
print(con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
con.close()
