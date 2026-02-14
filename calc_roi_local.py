# -*- coding: utf-8 -*-
"""
ローカル結果データ（Payoff.mdb）と AI 予測（weekly_prediction.json）を結合し、
3戦略の収支・回収率を計算する。Web スクレイピングは行わない。

Input:
  - docs/weekly_prediction.json
  - TukuAcc7/Data/Payoff.mdb

Strategy:
  - Plan A: ◎単勝一点
  - Plan B: ◎複勝一点
  - Plan C: 上位5頭ワイドBOX（10点）

Usage:
  python calc_roi_local.py              # 予測JSONの日付を自動使用
  python calc_roi_local.py 20260214     # 日付を指定
  python calc_roi_local.py --all        # 全日付で集計
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

SCRIPT_DIR = Path(__file__).resolve().parent
JSON_PATH = SCRIPT_DIR / "docs" / "weekly_prediction.json"

# Payoff ファイルの候補パス（先に存在するものを使用）
PAYOFF_CANDIDATES = [
    SCRIPT_DIR / "TukuAcc7" / "Data" / "Payoff.mbd",
    SCRIPT_DIR / "TukuAcc7" / "Data" / "Payoff.mdb",
    SCRIPT_DIR / "TukuAcc7" / "Data" / "PayOff.mbd",
    SCRIPT_DIR / "TukuAcc7" / "Payoff.mbd",
    SCRIPT_DIR / "TukuAcc7" / "Payoff.mdb",
    SCRIPT_DIR / "Payoff.mbd",
]


def _resolve_payoff_path() -> Path:
    """実際に存在する Payoff ファイルのパスを返す。無ければ候補の先頭を返す。"""
    for p in PAYOFF_CANDIDATES:
        if p.exists():
            return p
    return PAYOFF_CANDIDATES[0]

# 列名の候補（いずれかにマッチすればその列として扱う）
COL_DATE = ("日付", "年月日", "date", "日付日", "開催日")
COL_PLACE = ("場所", "場所コード", "場コード", "place", "競馬場")
COL_RACE = ("レース", "R", "レース番号", "race", "R番")
COL_RANK1 = ("1着", "1着馬番", "着順1", "rank1", "単勝馬番")
COL_RANK2 = ("2着", "2着馬番", "着順2", "rank2")
COL_RANK3 = ("3着", "3着馬番", "着順3", "rank3")
COL_TANSHO = ("単勝", "単勝配当", "単勝払戻", "tansho")
COL_FUKUSHO = ("複勝", "複勝配当", "複勝払戻", "fukusho")
COL_WIDE = ("ワイド", "ワイド配当", "wide")


def _to_int(val: Any, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


def _to_str(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    return s


# ---------------------------------------------------------------------------
# Step 1: ファイル形式の解析 (Analyze)
# ---------------------------------------------------------------------------
def is_likely_binary(path: Path) -> bool:
    """先頭数バイトでバイナリ（例: Access .mdb）かどうか推定。"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        head = f.read(32)
    # Access/Jet は "Standard Jet DB" 等のシグネチャを持つことがある
    if head[:4] in (b"\x00\x01\x00\x00", b"Stan"):
        return True
    # 印字可能 ASCII / 改行 / タブ / カンマ 以外が多ければバイナリとみなす
    printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
    return printable < len(head) * 0.7


def read_first_lines(path: Path, max_lines: int = 20) -> tuple[str | None, str | None]:
    """
    テキストとして先頭 max_lines 行を読み込む。Shift_JIS / UTF-8 を試行。
    戻り値: (encoding_used, text) または (None, None)
    """
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    for enc in ("shift_jis", "cp932", "utf-8", "utf-8-sig"):
        try:
            text = raw.decode(enc)
            lines = text.splitlines()[:max_lines]
            return enc, "\n".join(lines)
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None


def detect_delimiter(lines: list[str]) -> str:
    """行リストから区切り文字を推定（カンマ / タブ）。"""
    if not lines:
        return ","
    tab_counts = [line.count("\t") for line in lines if line.strip()]
    comma_counts = [line.count(",") for line in lines if line.strip()]
    if tab_counts and max(tab_counts) > max(comma_counts or [0]):
        return "\t"
    return ","


def find_column_index(header_cells: list[str], candidates: tuple[str, ...]) -> int:
    """ヘッダー行のセルリストから、候補名のいずれかと一致する列インデックスを返す。見つからなければ -1。"""
    for i, cell in enumerate(header_cells):
        c = _to_str(cell).strip()
        for cand in candidates:
            if cand in c or c in cand or c.lower() == cand.lower():
                return i
    return -1


def analyze_payoff_file(path: Path) -> dict[str, Any]:
    """
    Payoff.mbd の形式を解析する。
    戻り値: {
        "format": "csv_comma" | "csv_tab" | "binary" | "not_found" | "unknown",
        "encoding": str | None,
        "delimiter": str,
        "headers": list[str],
        "column_map": { "date": 0, "place": 1, ... },
        "sample_rows": list[list[str]],
        "message": str,
    }
    """
    result = {
        "format": "unknown",
        "encoding": None,
        "delimiter": ",",
        "headers": [],
        "column_map": {},
        "sample_rows": [],
        "message": "",
    }
    if not path.exists():
        result["format"] = "not_found"
        result["message"] = f"ファイルが存在しません: {path}"
        return result

    if is_likely_binary(path):
        result["format"] = "binary"
        result["message"] = (
            "Payoff.mbd はバイナリ形式の可能性があります（.mdb の誤記の場合は "
            "Microsoft Access で開き、CSV 等でエクスポートしてから Payoff.mbd として保存してください）。"
        )
        return result

    enc, text = read_first_lines(path, 20)
    if enc is None or text is None:
        result["format"] = "unknown"
        result["message"] = "テキストとしてデコードできませんでした（Shift_JIS / UTF-8 を試行済み）。"
        return result

    result["encoding"] = enc
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        result["message"] = "有効な行がありません。"
        return result

    delim = detect_delimiter(lines)
    result["delimiter"] = delim
    result["format"] = "csv_tab" if delim == "\t" else "csv_comma"

    # ヘッダー候補: 1行目
    header_line = lines[0]
    headers = [c.strip() for c in header_line.split(delim)]
    result["headers"] = headers

    # 列インデックスのマッピング
    col = result["column_map"]
    col["date"] = find_column_index(headers, COL_DATE)
    col["place"] = find_column_index(headers, COL_PLACE)
    col["race"] = find_column_index(headers, COL_RACE)
    col["rank1"] = find_column_index(headers, COL_RANK1)
    col["rank2"] = find_column_index(headers, COL_RANK2)
    col["rank3"] = find_column_index(headers, COL_RANK3)
    col["tansho"] = find_column_index(headers, COL_TANSHO)
    col["fukusho"] = find_column_index(headers, COL_FUKUSHO)
    col["wide"] = find_column_index(headers, COL_WIDE)

    # 複勝・ワイドは「複勝1」「複勝2」「複勝3」のように複数列の可能性
    for i, h in enumerate(headers):
        h = _to_str(h).strip()
        if "複勝" in h or "fukusho" in h.lower():
            if "1" in h or "１" in h or "1着" in h:
                col["fukusho1"] = i
            elif "2" in h or "２" in h or "2着" in h:
                col["fukusho2"] = i
            elif "3" in h or "３" in h or "3着" in h:
                col["fukusho3"] = i
            elif col.get("fukusho", -1) < 0:
                col["fukusho"] = i
        if ("ワイド" in h or "wide" in h.lower()) and col.get("wide", -1) < 0:
            col["wide"] = i

    result["sample_rows"] = []
    for ln in lines[1:11]:
        result["sample_rows"].append([c.strip() for c in ln.split(delim)])

    result["message"] = (
        f"カンマ区切りのCSVとして認識しました（エンコーディング: {enc}）。"
        if delim == ","
        else f"タブ区切りのTSVとして認識しました（エンコーディング: {enc}）。"
    )
    return result


def load_payoff_records(path: Path, analysis: dict[str, Any]) -> list[dict]:
    """
    解析結果に基づき Payoff.mbd を読み、レース単位の辞書リストを返す。
    キー: date(YYYYMMDD), place(2桁), race(2桁), rank1, rank2, rank3, tansho, fukusho_list[3], wide_list[3]
    """
    if analysis["format"] not in ("csv_comma", "csv_tab") or not path.exists():
        return []
    enc = analysis["encoding"] or "utf-8"
    delim = analysis["delimiter"]
    col = analysis["column_map"]
    records = []
    with open(path, "r", encoding=enc, newline="", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)
    if not rows:
        return records
    headers = rows[0]
    for row in rows[1:]:
        if len(row) <= max(col.get("date", 0), col.get("place", 0), col.get("race", 0)):
            continue
        date_val = _to_str(row[col["date"]]) if col.get("date", -1) >= 0 else ""
        place_val = _to_str(row[col["place"]]) if col.get("place", -1) >= 0 else ""
        race_val = _to_str(row[col["race"]]) if col.get("race", -1) >= 0 else ""
        # 日付を YYYYMMDD に正規化
        date_val = re.sub(r"[^0-9]", "", date_val)
        if len(date_val) == 8:
            pass
        elif len(date_val) == 6:
            date_val = "20" + date_val if int(date_val[:2]) < 50 else "19" + date_val
        else:
            continue
        place_val = re.sub(r"[^0-9]", "", place_val)
        if len(place_val) == 1:
            place_val = place_val.zfill(2)
        race_val = re.sub(r"[^0-9]", "", race_val)
        if len(race_val) == 1:
            race_val = race_val.zfill(2)
        if not place_val or not race_val:
            continue
        rank1 = _to_int(row[col["rank1"]]) if col.get("rank1", -1) >= 0 and col["rank1"] < len(row) else 0
        rank2 = _to_int(row[col["rank2"]]) if col.get("rank2", -1) >= 0 and col["rank2"] < len(row) else 0
        rank3 = _to_int(row[col["rank3"]]) if col.get("rank3", -1) >= 0 and col["rank3"] < len(row) else 0
        tansho = _to_int(row[col["tansho"]]) if col.get("tansho", -1) >= 0 and col["tansho"] < len(row) else 0
        fukusho_list = [0, 0, 0]
        if col.get("fukusho1", -1) >= 0 and col["fukusho1"] < len(row):
            fukusho_list[0] = _to_int(row[col["fukusho1"]])
        if col.get("fukusho2", -1) >= 0 and col["fukusho2"] < len(row):
            fukusho_list[1] = _to_int(row[col["fukusho2"]])
        if col.get("fukusho3", -1) >= 0 and col["fukusho3"] < len(row):
            fukusho_list[2] = _to_int(row[col["fukusho3"]])
        if (not any(fukusho_list)) and col.get("fukusho", -1) >= 0 and col["fukusho"] < len(row):
            raw = _to_str(row[col["fukusho"]])
            nums = [ _to_int(x) for x in re.findall(r"\d+", raw) ]
            if len(nums) >= 3:
                fukusho_list = nums[:3]
            else:
                fukusho_list[0] = _to_int(row[col["fukusho"]])
        wide_list = [0, 0, 0]
        # ワイドは 1-2着, 1-3着, 2-3着 の3列ある想定
        wide_idx = col.get("wide", -1)
        if wide_idx >= 0 and wide_idx + 2 < len(row):
            wide_list = [_to_int(row[wide_idx]), _to_int(row[wide_idx + 1]), _to_int(row[wide_idx + 2])]
        records.append({
            "date": date_val,
            "place": place_val,
            "race": race_val,
            "rank1": rank1,
            "rank2": rank2,
            "rank3": rank3,
            "tansho": tansho,
            "fukusho_list": fukusho_list,
            "wide_list": wide_list,
        })
    return records


# ---------------------------------------------------------------------------
# Step 1b: Access .mdb を pyodbc で直接読み込む
# ---------------------------------------------------------------------------
def inspect_mdb(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """
    .mdb のテーブル一覧とカラム一覧を返す。
    戻り値: (table_names, {table_name: [col_name, ...]})
    """
    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path};"
    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    tables = []
    schema: dict[str, list[str]] = {}
    for row in cur.tables(tableType="TABLE"):
        tname = row.table_name
        if tname.startswith("MSys"):
            continue
        tables.append(tname)
        try:
            cur2 = conn.cursor()
            cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
            schema[tname] = [d[0] for d in cur2.description] if cur2.description else []
            cur2.close()
        except Exception:
            schema[tname] = []
    cur.close()
    conn.close()
    return tables, schema


def _parse_jv_race_id(race_id: str) -> tuple[str, str, str, str]:
    """
    JV-Data 形式の RaceID (16桁) から日付・場コード・レース番号を抽出。

    JV-Data RaceID 構造 (16桁) - 実データから確認:
      [0:4]   YYYY  年
      [4:8]   MMDD  月日
      [8:10]  JJ    場コード (01=札幌, 02=函館, 03=福島, 04=新潟, 05=東京,
                               06=中山, 07=中京, 08=京都, 09=阪神, 10=小倉)
      [10:12] KK    回次
      [12:14] NN    日次 (or サブID)
      [14:16] RR    レース番号 (01-12)

    例: 2026021405010501 → date=20260214, 場=05(東京), R=01(1R)

    戻り値: (date_yyyymmdd, place, race, year)
    """
    rid = str(race_id).strip()
    if len(rid) < 16:
        return ("", "", "", "")
    year = rid[0:4]
    date = rid[0:8]     # YYYYMMDD
    place = rid[8:10]   # 場コード
    race = rid[14:16]   # レース番号（末尾2桁）
    return (date, place, race, year)


def load_payoff_from_mdb(path: Path, target_date: str = "") -> list[dict]:
    """
    Payoff.mdb (JV-Data 形式の「払戻」テーブル) を pyodbc で直接読み込む。

    実際のカラム:
      RaceID (16桁JV-Data形式), 単勝馬番1, 単勝払戻金1, 複勝馬番1-3, 複勝払戻金1-3,
      ワイド組番1-7, ワイド払戻金1-7, etc.
    """
    if not HAS_PYODBC:
        print("  [NG] pyodbc がインストールされていません。pip install pyodbc を実行してください。")
        return []

    print("  Access .mdb を pyodbc で読み込みます...")
    try:
        tables, schema = inspect_mdb(path)
    except Exception as e:
        print(f"  [NG] .mdb 接続エラー: {e}")
        return []

    print(f"  テーブル一覧: {tables}")
    for t in tables:
        cols = schema.get(t, [])
        print(f"    [{t}] カラム数: {len(cols)}")

    # 払戻テーブルを探す
    pay_table = None
    for t in tables:
        if "払戻" in t or "payoff" in t.lower() or "pay" in t.lower():
            pay_table = t
            break
    if not pay_table and tables:
        pay_table = tables[0]
    if not pay_table:
        print("  テーブルが見つかりません。")
        return []

    print(f"  使用テーブル: {pay_table}")

    # SQL で必要なカラムだけ取得（データ作成年月日も追加して日付フィルタに使用）
    needed_cols = [
        "RaceID",
        "データ作成年月日",
        "単勝馬番1", "単勝払戻金1",
        "複勝馬番1", "複勝払戻金1",
        "複勝馬番2", "複勝払戻金2",
        "複勝馬番3", "複勝払戻金3",
        "ワイド組番1", "ワイド払戻金1",
        "ワイド組番2", "ワイド払戻金2",
        "ワイド組番3", "ワイド払戻金3",
        "ワイド組番4", "ワイド払戻金4",
        "ワイド組番5", "ワイド払戻金5",
        "ワイド組番6", "ワイド払戻金6",
        "ワイド組番7", "ワイド払戻金7",
    ]
    available = set(schema.get(pay_table, []))
    select_cols = [c for c in needed_cols if c in available]
    if "RaceID" not in available:
        print("  [NG] RaceID カラムが見つかりません。")
        return []
    select_sql = ", ".join(f"[{c}]" for c in select_cols)

    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path};"
    conn = pyodbc.connect(conn_str)
    cur = conn.cursor()
    cur.execute(f"SELECT {select_sql} FROM [{pay_table}]")
    desc = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"  全行数: {len(rows)}")

    if not rows or not desc:
        return []

    # カラム名→インデックスのマップ
    ci = {c: i for i, c in enumerate(desc)}

    def _v(row, col_name: str, default=None):
        """カラム名からセル値を取得"""
        idx = ci.get(col_name, -1)
        if idx < 0 or idx >= len(row):
            return default
        return row[idx]

    # サンプル表示
    print(f"  取得カラム: {desc}")
    for si in range(min(3, len(rows))):
        vals = {c: str(rows[si][ci[c]]) for c in desc[:12]}
        print(f"  サンプル行{si}: {vals}")

    # ----- フィルタ戦略 -----
    # RaceID 構造: [0:4]=年, [4:8]=月日(MMDD), [8:10]=場コード, [10:12]=回次,
    #              [12:14]=日次, [14:16]=レース番号
    # → RaceID[0:8] = YYYYMMDD で日付フィルタが可能

    if target_date and len(target_date) == 8:
        # RaceID の先頭8桁 = YYYYMMDD で直接フィルタ
        use_rows = [row for row in rows
                    if _to_str(_v(row, "RaceID")).startswith(target_date)]
        print(f"  RaceID 日付フィルタ ({target_date}): {len(use_rows)}件 / {len(rows)}件")
    else:
        # target_date がない場合は全件
        use_rows = rows
        print(f"  日付フィルタなし: 全 {len(rows)} 件を使用")

    records = []
    skipped = 0
    for row in use_rows:
        race_id_raw = _to_str(_v(row, "RaceID"))
        if len(race_id_raw) < 16:
            skipped += 1
            continue
        date_val, place, race_num, year = _parse_jv_race_id(race_id_raw)

        # 単勝: 馬番と払戻金
        rank1 = _to_int(_v(row, "単勝馬番1"))
        tansho = _to_int(_v(row, "単勝払戻金1"))

        # 複勝: 馬番1-3 と払戻金1-3
        fuku_nums = [
            _to_int(_v(row, "複勝馬番1")),
            _to_int(_v(row, "複勝馬番2")),
            _to_int(_v(row, "複勝馬番3")),
        ]
        fuku_pays = [
            _to_int(_v(row, "複勝払戻金1")),
            _to_int(_v(row, "複勝払戻金2")),
            _to_int(_v(row, "複勝払戻金3")),
        ]

        # ワイド: 組番1-7 と払戻金1-7
        # 「ワイド組番」は "0102" のような4桁文字列 → 馬番ペア (01, 02) を表す
        wide_entries = []  # list of (uma_a, uma_b, haraikin)
        for wi in range(1, 8):
            kumi_col = f"ワイド組番{wi}"
            hari_col = f"ワイド払戻金{wi}"
            kumi_raw = _to_str(_v(row, kumi_col))
            hari = _to_int(_v(row, hari_col))
            if not kumi_raw or kumi_raw == "None" or len(kumi_raw) < 4:
                continue
            uma_a = _to_int(kumi_raw[:2])
            uma_b = _to_int(kumi_raw[2:4])
            if uma_a and uma_b and hari:
                wide_entries.append((uma_a, uma_b, hari))

        records.append({
            "race_id_jv": race_id_raw,
            "date": date_val,          # YYYYMMDD
            "year": year,
            "place": place,
            "race": race_num,
            "rank1": rank1,
            "tansho": tansho,
            "fuku_nums": fuku_nums,    # [馬番1, 馬番2, 馬番3]
            "fuku_pays": fuku_pays,    # [払戻1, 払戻2, 払戻3]
            "wide_entries": wide_entries,  # [(馬A, 馬B, 払戻), ...]
        })

    print(f"  フィルタ後レコード数: {len(records)} (スキップ: {skipped})")
    if records:
        r0 = records[0]
        print(f"  先頭レコード例: RaceID={r0['race_id_jv']}, place={r0['place']}, "
              f"race={r0['race']}, rank1={r0['rank1']}, tansho={r0['tansho']}, "
              f"fuku_nums={r0['fuku_nums']}, wide_entries={r0['wide_entries'][:2]}")

    # デバッグ: (place, race) ペアの分布を表示
    pr_set = set()
    for rec in records:
        pr_set.add((rec["place"], rec["race"]))
    places = sorted(set(r["place"] for r in records))
    print(f"  含まれる場コード: {places}")
    print(f"  (place, race) ユニーク数: {len(pr_set)}")
    # 各場コードのユニークレース番号
    for pl in places[:5]:
        races_unique = sorted(set(r["race"] for r in records if r["place"] == pl))
        count = sum(1 for r in records if r["place"] == pl)
        print(f"    場{pl}: R={races_unique} ({count}件)")

    # RaceID サンプル表示（各桁の意味を確認するため）
    sample_ids = sorted(set(r["race_id_jv"] for r in records))[:10]
    if sample_ids:
        print(f"  RaceID サンプル (先頭10件):")
        for sid in sample_ids:
            print(f"    {sid} => [{sid[0:4]}][{sid[4:6]}][{sid[6:8]}][{sid[8:10]}][{sid[10:12]}][{sid[12:14]}][{sid[14:16]}]")

    return records


# ---------------------------------------------------------------------------
# Step 2: データの結合 (Merge)
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> list[dict]:
    """weekly_prediction.json を読み、レースごとの予測リストを返す。"""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data.get("races") or []:
        race_id = (_to_str(r.get("race_id")) or "").strip()
        if len(race_id) < 12:
            continue
        date = race_id[:8]
        place = race_id[8:10]
        race = race_id[10:12]
        preds = r.get("predictions") or []
        horses = [
            {"horse_num": _to_int(p.get("horse_num")), "horse_name": _to_str(p.get("horse_name")), "mark": _to_str(p.get("mark")), "score": _to_int(p.get("score"))}
            for p in preds
        ]
        out.append({
            "date": date,
            "place": place,
            "race": race,
            "race_id": race_id,
            "race_name": _to_str(r.get("race_name")),
            "horses": horses,
        })
    return out


def merge_predictions_and_payoff(
    predictions: list[dict],
    payoff_records: list[dict],
) -> list[dict]:
    """
    (place, race) でマージし、予測＋結果を1レース1件のリストで返す。

    予測データの場コード集合で Payoff をフィルタしてから突合する。
    同一 (place, race) に複数件ある場合、RaceID プレフィックスの出現頻度で
    最も多い開催を選択する（同一開催日のデータが12R揃っているはず）。
    """
    # 予測側の場コード集合
    pred_places = {p["place"] for p in predictions}
    print(f"  予測データの場コード: {sorted(pred_places)}")

    # Payoff を予測の場コードだけにフィルタ
    filtered_payoff = [p for p in payoff_records if p["place"] in pred_places]
    print(f"  場コードフィルタ後の Payoff: {len(filtered_payoff)}件 (元: {len(payoff_records)}件)")

    # (place, race) ごとに候補をグルーピング
    key_candidates: dict[tuple[str, str], list[dict]] = {}
    for p in filtered_payoff:
        key = (p["place"], p["race"])
        key_candidates.setdefault(key, []).append(p)

    # 重複がある場合: RaceID プレフィックス [0:10] の出現頻度で最多開催を選ぶ
    # (同じ開催日なら12R全て同じプレフィックスを持つ)
    prefix_count: dict[str, int] = {}
    for p in filtered_payoff:
        jv_id = p.get("race_id_jv", "")
        if len(jv_id) >= 10:
            pfx = jv_id[:10]
            prefix_count[pfx] = prefix_count.get(pfx, 0) + 1

    # 各場コードで最頻のプレフィックスを特定 = 最新（直近）の開催
    best_prefix: dict[str, str] = {}  # place -> best prefix
    for pl in pred_places:
        pl_prefixes = {pfx: cnt for pfx, cnt in prefix_count.items()
                       if len(pfx) >= 8 and pfx[6:8] == pl}
        if pl_prefixes:
            # 最多かつ最新（プレフィックスが大きい＝より後の開催）を選ぶ
            best = max(pl_prefixes, key=lambda x: (pl_prefixes[x], x))
            best_prefix[pl] = best
            print(f"  場{pl}: 最新開催プレフィックス={best} ({pl_prefixes[best]}R)")

    # (place, race) → 最適な1件を選択
    key_to_payoff: dict[tuple[str, str], dict] = {}
    for key, candidates in key_candidates.items():
        pl = key[0]
        bp = best_prefix.get(pl, "")
        if bp:
            # best prefix に合致するものを優先
            match = [c for c in candidates
                     if c.get("race_id_jv", "").startswith(bp)]
            if match:
                key_to_payoff[key] = match[0]
                continue
        # フォールバック: 最後の1件
        key_to_payoff[key] = candidates[-1]

    merged = []
    for pred in predictions:
        key = (pred["place"], pred["race"])
        pay = key_to_payoff.get(key)
        if pay is None:
            continue
        merged.append({**pred, "payoff": pay})

    # デバッグ: マッチしなかったレースを表示
    if len(merged) < len(predictions):
        pred_keys = {(p["place"], p["race"]) for p in predictions}
        pay_keys = set(key_to_payoff.keys())
        missing = pred_keys - pay_keys
        if missing:
            print(f"  [INFO] マッチしなかった予測レース: {sorted(missing)[:10]}")
    return merged


# ---------------------------------------------------------------------------
# Step 3: 収支計算 (Calculate ROI)
# ---------------------------------------------------------------------------
def run_plans(merged: list[dict], log_hits: bool = True) -> dict[str, Any]:
    """
    Plan A / B / C の投資額・回収額・回収率を計算する。

    payoff 構造 (mdb 由来):
      rank1: int           単勝馬番 (1着)
      tansho: int          単勝払戻金 (100円あたり)
      fuku_nums: [int*3]   複勝馬番 (1着, 2着, 3着)
      fuku_pays: [int*3]   複勝払戻金 (各100円あたり)
      wide_entries: [(uma_a, uma_b, haraikin), ...]  ワイド的中ペアと払戻金
    """
    result = {
        "plan_a": {"stake": 0, "payout": 0, "recovery_pct": 0.0, "hits": []},
        "plan_b": {"stake": 0, "payout": 0, "recovery_pct": 0.0, "hits": []},
        "plan_c": {"stake": 0, "payout": 0, "recovery_pct": 0.0, "hits": []},
    }
    for m in merged:
        horses = m.get("horses") or []
        maru = next((h for h in horses if h.get("mark") == "◎"), None)
        if not maru:
            continue
        pay = m.get("payoff") or {}
        maru_num = _to_int(maru.get("horse_num"))
        top5 = horses[:5]
        top5_nums = {_to_int(h.get("horse_num")) for h in top5}
        race_label = m.get("race_name", f"place={m.get('place')} R{m.get('race')}")

        # ---- Plan A: ◎単勝一点（100円） ----
        result["plan_a"]["stake"] += 100
        rank1 = _to_int(pay.get("rank1"))
        if rank1 and rank1 == maru_num:
            payout_a = _to_int(pay.get("tansho"))
            result["plan_a"]["payout"] += payout_a
            if log_hits:
                result["plan_a"]["hits"].append({
                    "race": race_label, "maru": maru_num, "payout": payout_a,
                })

        # ---- Plan B: ◎複勝一点（100円） ----
        result["plan_b"]["stake"] += 100
        fuku_nums = pay.get("fuku_nums") or [0, 0, 0]
        fuku_pays = pay.get("fuku_pays") or [0, 0, 0]
        payout_b = 0
        for fi in range(3):
            if fuku_nums[fi] and fuku_nums[fi] == maru_num:
                payout_b = fuku_pays[fi]
                break
        result["plan_b"]["payout"] += payout_b
        if log_hits and payout_b:
            result["plan_b"]["hits"].append({
                "race": race_label, "maru": maru_num, "payout": payout_b,
            })

        # ---- Plan C: 上位5頭ワイドBOX（10点＝1000円） ----
        result["plan_c"]["stake"] += 1000
        # 選んだ5頭から作れる全ペア
        our_pairs: set[tuple[int, int]] = set()
        nums = list(top5_nums)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = min(nums[i], nums[j]), max(nums[i], nums[j])
                our_pairs.add((a, b))
        # ワイド的中ペアとの照合
        wide_entries = pay.get("wide_entries") or []
        payout_c = 0
        hit_details = []
        for uma_a, uma_b, haraikin in wide_entries:
            pair = (min(uma_a, uma_b), max(uma_a, uma_b))
            if pair in our_pairs:
                payout_c += haraikin
                hit_details.append(f"{uma_a}-{uma_b}:{haraikin}")
        result["plan_c"]["payout"] += payout_c
        if log_hits and payout_c:
            result["plan_c"]["hits"].append({
                "race": race_label, "payout": payout_c, "details": hit_details,
            })

    for key in ("plan_a", "plan_b", "plan_c"):
        st = result[key]["stake"]
        py = result[key]["payout"]
        result[key]["recovery_pct"] = (py / st * 100) if st else 0.0
    return result


def _print_plans(plans: dict[str, Any], merged_count: int, target_date: str) -> list[str]:
    """結果を表示しつつ、ログ用の行リストを返す。"""
    lines: list[str] = []
    header = f"[{target_date}] {merged_count}R マージ"
    print(f"\n--- 各戦略の結果 ({target_date}) ---")
    lines.append(f"=== {target_date} ({merged_count} races) ===")
    for name, key in [("Plan A (単勝一点)", "plan_a"),
                      ("Plan B (複勝一点)", "plan_b"),
                      ("Plan C (ワイドBOX 5頭)", "plan_c")]:
        p = plans[key]
        line = f"  {name}: 投資額 {p['stake']}円 / 回収額 {p['payout']}円 / 回収率 {p['recovery_pct']:.1f}%"
        print(line)
        lines.append(line)
    print("\n--- 的中ログ ---")
    for key in ("plan_a", "plan_b", "plan_c"):
        hits = plans[key].get("hits") or []
        if hits:
            hit_line = f"  [{key}] {len(hits)}件: {hits[:5]}{'...' if len(hits) > 5 else ''}"
            print(hit_line)
            lines.append(hit_line)
    return lines


def _save_log(lines: list[str], target_date: str) -> Path:
    """結果をログファイルに追記保存する。"""
    log_dir = SCRIPT_DIR / "jv_data" / "reports"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "roi_history.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- {timestamp} ---\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")
    return log_file


def _save_csv_record(plans: dict[str, Any], target_date: str, merged_count: int) -> Path:
    """収支結果を CSV に1行追加する（累積記録用）。"""
    csv_dir = SCRIPT_DIR / "jv_data" / "reports"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / "roi_history.csv"
    write_header = not csv_file.exists()
    with open(csv_file, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date", "races", "timestamp",
                        "planA_stake", "planA_payout", "planA_recovery",
                        "planB_stake", "planB_payout", "planB_recovery",
                        "planC_stake", "planC_payout", "planC_recovery"])
        pa, pb, pc = plans["plan_a"], plans["plan_b"], plans["plan_c"]
        w.writerow([
            target_date, merged_count, datetime.now().strftime("%Y-%m-%d %H:%M"),
            pa["stake"], pa["payout"], f"{pa['recovery_pct']:.1f}",
            pb["stake"], pb["payout"], f"{pb['recovery_pct']:.1f}",
            pc["stake"], pc["payout"], f"{pc['recovery_pct']:.1f}",
        ])
    return csv_file


def _run_roi(payoff_path: Path, target_date: str) -> tuple[int, dict[str, Any] | None]:
    """
    指定日の収支計算を実行する。
    戻り値: (マージ件数, plans辞書 or None)
    """
    # Step 1: ファイル形式の解析
    analysis = analyze_payoff_file(payoff_path)
    if analysis["format"] == "not_found":
        print(f"  [NG] ファイルが見つかりません: {payoff_path}")
        return 0, None

    payoff_records: list[dict] = []

    if analysis["format"] == "binary":
        if not HAS_PYODBC:
            print("  [NG] pyodbc が未インストールです。")
            return 0, None
        payoff_records = load_payoff_from_mdb(payoff_path, target_date=target_date)
        if not payoff_records:
            print(f"  [INFO] {target_date} のデータが Payoff.mdb に見つかりません。")
            return 0, None
    elif analysis["format"] in ("csv_comma", "csv_tab"):
        payoff_records = load_payoff_records(payoff_path, analysis)

    if not payoff_records:
        return 0, None

    # Step 2: 結合
    predictions = load_predictions(JSON_PATH)
    if not predictions:
        print("  [NG] 予測データがありません。")
        return 0, None
    merged = merge_predictions_and_payoff(predictions, payoff_records)
    if not merged:
        return 0, None

    # Step 3: 収支計算
    plans = run_plans(merged, log_hits=True)
    return len(merged), plans


def main() -> int:
    # --- 引数パーサー ---
    parser = argparse.ArgumentParser(description="AI予測の収支計算 (ローカル Payoff.mdb)")
    parser.add_argument("date", nargs="?", default=None,
                        help="対象日 YYYYMMDD（省略時: weekly_prediction.json の日付）")
    parser.add_argument("--all", action="store_true",
                        help="Payoff.mdb の全日付を集計")
    parser.add_argument("--quiet", action="store_true",
                        help="デバッグ出力を抑制")
    args = parser.parse_args()

    print("=== ローカル結果データによる収支計算（calc_roi_local.py） ===\n")

    payoff_path = _resolve_payoff_path()
    print(f"[Payoff] {payoff_path}")
    if not payoff_path.exists():
        data_dir = SCRIPT_DIR / "TukuAcc7" / "Data"
        if data_dir.exists():
            try:
                names = sorted(p.name for p in data_dir.iterdir())
                print(f"  TukuAcc7/Data 内のファイル: {names}")
            except OSError:
                pass
        print("  [NG] Payoff ファイルが見つかりません。")
        return 1

    # 対象日の決定
    if args.date:
        target_date = args.date.replace("-", "").replace("/", "")[:8]
    else:
        preds = load_predictions(JSON_PATH)
        target_date = preds[0].get("date", "") if preds else ""

    if not target_date or len(target_date) != 8:
        target_date = datetime.now().strftime("%Y%m%d")

    print(f"[対象日] {target_date}")
    print(f"[予測JSON] {JSON_PATH}\n")

    merged_count, plans = _run_roi(payoff_path, target_date)

    if not plans or merged_count == 0:
        print(f"\n{target_date} のデータがないため収支計算をスキップしました。")
        return 0

    # 結果表示
    log_lines = _print_plans(plans, merged_count, target_date)

    # ログ保存
    log_path = _save_log(log_lines, target_date)
    csv_path = _save_csv_record(plans, target_date, merged_count)
    print(f"\n[保存] ログ: {log_path}")
    print(f"[保存] CSV:  {csv_path}")
    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
