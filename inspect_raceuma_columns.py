# -*- coding: utf-8 -*-
"""Race.mdb の RaceUma テーブルカラム一覧を確認"""
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent
DATA = SCRIPT / "TukuAcc7" / "Data"
if not DATA.exists():
    DATA = SCRIPT / "Data"
RACE_MDB = DATA / "Race.mdb"
if not RACE_MDB.exists():
    RACE_MDB = DATA / "Kako.mdb"

if not RACE_MDB.exists():
    print(f"Race.mdb not found: {DATA}")
    sys.exit(1)

try:
    import pyodbc
except ImportError:
    print("pyodbc required")
    sys.exit(1)

conn = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={RACE_MDB};")
cur = conn.cursor()
cur.execute("SELECT TOP 1 * FROM [RaceUma]")
cols = [d[0] for d in cur.description] if cur.description else []
# 騎手コード・枠番のサンプル値を取得
cur.execute("SELECT TOP 20 [枠番], [騎手コード] FROM [RaceUma] WHERE [確定着順] IS NOT NULL")
sample_rows = cur.fetchall()
out_path = SCRIPT / "jv_data" / "raceuma_columns_utf8.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for i, c in enumerate(cols):
        if c:
            f.write(f"{i}: {c}\n")
    f.write("\n=== Sample 枠番, 騎手コード ===\n")
    for row in sample_rows:
        f.write(f"  waku={row[0]!r}, jockey={row[1]!r}\n")
cur.execute("""
    SELECT TOP 50 R.[開催日], R.[競馬場名], R.[レース番号], U.[枠番], U.[騎手コード], U.[確定着順]
    FROM [Race] R INNER JOIN [RaceUma] U ON R.[RaceID] = U.[RaceID]
    WHERE U.[確定着順] IS NOT NULL
    ORDER BY R.[開催日] DESC
""")
join_rows = cur.fetchall()
with open(out_path, "a", encoding="utf-8") as f:
    f.write("\n=== JOIN sample (開催日降順50件) ===\n")
    for row in join_rows:
        f.write(f"  date={row[0]!r} waku={row[3]!r} jockey={row[4]!r}\n")
print(f"Wrote {out_path}")
cur.close()
conn.close()

# 騎手・枠に関連しそうなカラムを抽出
keywords = ["騎手", "枠", "Kishu", "Kisyu", "Waku", "jockey", "waku"]
print("=== RaceUma full columns ===")
for i, c in enumerate(cols):
    if c:
        flag = " ***" if any(kw in str(c) for kw in keywords) else ""
        print(f"  {i}: {c!r}{flag}")

print("\n=== Sample row (first 20 values) ===")
conn = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={RACE_MDB};")
cur = conn.cursor()
cur.execute("SELECT TOP 1 * FROM [RaceUma]")
row = cur.fetchone()
if row:
    for i, (col, val) in enumerate(zip(cols[:20], row[:20] if row else [])):
        print(f"  {col!r}: {val!r}")
cur.close()
conn.close()
