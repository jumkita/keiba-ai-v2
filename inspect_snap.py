# -*- coding: utf-8 -*-
"""Snap フォルダと MDB の構造を確認するスクリプト"""
from pathlib import Path
import sys
import zipfile

SCRIPT = Path(__file__).resolve().parent
# UTF-8でコンソール出力（Windows対応）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
DATA = SCRIPT / "TukuAcc7" / "Data"
SNAP_DIR = DATA / "Snap"

def main():
    print("=== Snap フォルダ確認 ===")
    print(f"SNAP_DIR exists: {SNAP_DIR.exists()}")
    if SNAP_DIR.exists():
        zips = list(SNAP_DIR.glob("*.zip"))
        print(f"ZIP files: {len(zips)}")
        if zips:
            z = zips[0]
            print(f"Sample: {z.name}")
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist()[:15]:
                    print(f"  - {n}")
    
    print("\n=== MDB ファイル検索 ===")
    for name in ["Snap.mdb", "Race.mdb", "Master.mdb"]:
        p = DATA / name
        alt = SCRIPT / name
        print(f"  {name}: DATA={p.exists()}, SCRIPT={alt.exists()}")
    
    if (DATA / "Snap.mdb").exists():
        print("\n=== Snap.mdb テーブル一覧 ===")
        try:
            import pyodbc
            conn = pyodbc.connect(
                f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={DATA / 'Snap.mdb'};"
            )
            cur = conn.cursor()
            for row in cur.tables(tableType="TABLE"):
                tname = getattr(row, "table_name", None) or row[2]
                if tname and not str(tname).startswith("MSys"):
                    cur2 = conn.cursor()
                    cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
                    cols = [d[0] for d in cur2.description] if cur2.description else []
                    cur2.execute(f"SELECT COUNT(*) FROM [{tname}]")
                    cnt = (cur2.fetchone() or [0])[0]
                    cur2.close()
                    print(f"\n--- {tname} (count={cnt}) ---")
                    out_path = SCRIPT / "jv_data" / "snap_columns.txt"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    mode = "w" if not out_path.exists() else "a"
                    with open(out_path, mode, encoding="utf-8") as of:
                        of.write(f"\n--- {tname} (count={cnt}) ---\n")
                        for i, c in enumerate(cols):
                            line = f"  {i}: {c}\n"
                            of.write(line)
                            print(line.rstrip())
            conn.close()
        except Exception as e:
            print(f"  Error: {e}")

def test_extract():
    """extract_future_from_snap のテスト"""
    import predict_pipeline as pp
    snap = SCRIPT / "TukuAcc7" / "Data" / "Snap.mdb"
    if snap.exists():
        rows = pp.extract_future_from_snap(snap, SCRIPT / "jv_data" / "future_races.csv")
        print(f"抽出件数: {len(rows)}")
        if rows:
            print("先頭3件:")
            for r in rows[:3]:
                print(r)
    else:
        print("Snap.mdb not found")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        test_extract()
    else:
        main()
