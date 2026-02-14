# -*- coding: utf-8 -*-
"""
Accessデータベース (Race.mdb, Master.mdb) から機械学習用の学習データと血統・騎手辞書を抽出する。

出力:
  - jv_data/learning_dataset.csv  学習用データ（race_key, horse_id, sire_id, broodmare_sire_id, jockey_id, wakuban, ...）
  - config/sire_id_map.csv        父ID辞書 (sire_id, sire_name)
  - config/bms_id_map.csv         母父ID辞書 (bms_id, bms_name)
  - config/jockey_id_map.csv      騎手ID辞書 (jockey_id, jockey_name)
  - config/trainer_id_map.csv     調教師ID辞書 (trainer_id, trainer_name)

使い方:
  python extract_from_mdb.py
  python extract_from_mdb.py "C:\path\to\TukuAcc7"
"""
from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

# パス設定
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "TukuAcc7" / "Data"
OUTPUT_LEARNING = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
OUTPUT_SIRE_MAP = SCRIPT_DIR / "config" / "sire_id_map.csv"
OUTPUT_BMS_MAP = SCRIPT_DIR / "config" / "bms_id_map.csv"
OUTPUT_JOCKEY_MAP = SCRIPT_DIR / "config" / "jockey_id_map.csv"
OUTPUT_TRAINER_MAP = SCRIPT_DIR / "config" / "trainer_id_map.csv"

# 競馬場コードマップ (01-10 ↔ 競馬場名)
PLACE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
for k, v in list(PLACE_CODE_MAP.items()):
    PLACE_CODE_MAP[v] = k.zfill(2)
    PLACE_CODE_MAP[str(k).zfill(2)] = str(k).zfill(2)

# コース区分・馬場状態・回りの変換（JRA-VAN仕様準拠）


def _normalize_str(val) -> str:
    """余計な空白・全角スペースを除去した文字列を返す（Fuzzy 判定用）。"""
    if val is None:
        return ""
    return str(val).replace("\u3000", " ").strip()


def map_course_type(val) -> str | None:
    """
    芝ダ別（日本語文字列）から course_type を判定。極めて緩い Fuzzy マッチング。
    戻り値: "1"(芝), "2"(ダート), None(障害・不明で除外)
    ※「障」を先に判定し、次に「芝」「ダ」の部分一致を行う。
    """
    s = _normalize_str(val)
    if not s:
        return None
    if "障" in s:
        return None  # 障芝, 障ダ など障害は除外
    if "芝" in s:
        return "1"  # 芝, 芝外, 芝内, 芝直 など
    if "ダ" in s:
        return "2"  # ダ
    # 数値コード（TrackCD）仕様: 10-22=芝, それ以外=ダート, 30+=障害で除外
    try:
        n = int(float(s))
        if n >= 30:
            return None  # 障害
        if 10 <= n <= 22:
            return "1"  # 芝
        return "2"  # 1-9, 23-29 等 = ダート
    except (ValueError, TypeError):
        pass
    return None


def map_state(val) -> int:
    """
    馬場状態（日本語 or BabaCD数値）から state を数値化。
    日本語: 良->1, 稍重->2, 重->3, 不良->4
    数値: 10-12=良, 20-22=稍重, 30-39=重, 40+=不良
    """
    s = _normalize_str(val)
    if not s:
        return 1
    if "稍" in s or s == "稍重":
        return 2
    if "不" in s or s == "不良":
        return 4
    if s == "重":
        return 3
    if s == "良":
        return 1
    try:
        n = int(float(s))
        if 10 <= n <= 19:
            return 1
        if 20 <= n <= 29:
            return 2
        if 30 <= n <= 39:
            return 3
        if n >= 40:
            return 4
    except (ValueError, TypeError):
        pass
    return 1


def map_rotation(val) -> int:
    """
    回り カラムから rotation を数値化。Fuzzy マッチング（含む判定）。
    """
    v = _normalize_str(val)
    if not v:
        return 0
    if "右" in v:
        return 1
    if "左" in v:
        return 2
    if "直" in v:
        return 3
    try:
        n = int(float(v))
        if 1 <= n <= 3:
            return n
    except (ValueError, TypeError):
        pass
    return 0


# 学習データ出力カラム（必須）
LEARNING_COLUMNS = [
    "race_key", "horse_id", "sire_id", "broodmare_sire_id", "jockey_id", "trainer_id", "wakuban", "weight_carry",
    "rank", "time", "distance", "course_type", "rotation", "weather", "state", "date", "racecourse",
    "time_diff",
]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _to_str(v, default: str = "") -> str:
    if v is None or (isinstance(v, float) and (v != v or v == float("nan"))):
        return default
    s = str(v).strip()
    return s if s and s.lower() != "nan" else default


def _to_int(v, default: int = 0) -> int:
    if v is None or v == "" or (isinstance(v, float) and (v != v or v == float("nan"))):
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _to_float(v, default: float = 0.0) -> float:
    if v is None or v == "" or (isinstance(v, float) and (v != v or v == float("nan"))):
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def list_tables_and_columns(conn) -> list[tuple[str, list[str]]]:
    """MDBの全テーブルとカラム一覧を取得する。"""
    cur = conn.cursor()
    tables = []
    for row in cur.tables(tableType="TABLE"):
        tname = getattr(row, "table_name", None) or row[2]
        if tname and tname.startswith("MSys"):
            continue
        try:
            cur2 = conn.cursor()
            cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
            cols = [d[0] for d in cur2.description] if cur2.description else []
            cur2.close()
            tables.append((tname, cols))
        except Exception as e:
            logger.warning("テーブル %s のカラム取得に失敗: %s", tname, e)
            tables.append((tname, []))
    cur.close()
    return tables


def find_col(cols: list[str], *candidates: str) -> str | None:
    """カラム一覧から候補に一致するカラム名を返す。"""
    col_set = {c.strip() for c in cols if c}
    col_lower = {(c or "").strip().lower(): c for c in cols if c}
    for c in candidates:
        for col in cols:
            if col and (col.strip() == c or (col or "").strip().lower() == c.lower()):
                return col
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None


def find_ra_table(tables: list[tuple[str, list[str]]]) -> tuple[str | None, dict]:
    """
    RA系テーブル（レース詳細）を探す。
    戻り値: (テーブル名, {学習用カラム: MDB列名})
    """
    ra_keywords = ("日付", "開催日", "競馬場", "R番", "レース番号", "回次", "距離", "芝ダ", "天候", "馬場", "出走数")
    for tname, cols in tables:
        col_str = " ".join(c or "" for c in cols).lower()
        score = sum(1 for kw in ra_keywords if kw in col_str)
        if score < 3:
            continue
        date_col = find_col(cols, "日付", "開催日", "race_date", "Date")
        place_col = find_col(cols, "競馬場", "競馬場名", "場所", "place_id", "場コード")
        round_col = find_col(cols, "R番", "レース番号", "round", "R")
        dist_col = find_col(cols, "距離", "distance")
        course_col = find_col(cols, "芝ダ別", "TrackCD", "トラックコード", "芝ダ", "コース", "course_type", "芝ダコード")
        weather_col = find_col(cols, "天候", "weather", "weather_code")
        state_col = find_col(cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場", "state", "track_condition")
        mawari_col = find_col(cols, "回り", "Mawari", "周り", "右左", "回りコード", "course_direction")
        if date_col and (place_col or round_col):
            racecourse_col = find_col(cols, "競馬場名", "競馬場")
            mapping = {
                "date": date_col,
                "place_id": place_col,
                "round": round_col,
                "distance": dist_col,
                "course_type": course_col,
                "rotation": mawari_col,
                "weather": weather_col,
                "state": state_col,
                "racecourse": racecourse_col or place_col,
            }
            return (tname, {k: v for k, v in mapping.items() if v})
    return (None, {})


def find_se_table(tables: list[tuple[str, list[str]]]) -> tuple[str | None, dict]:
    """
    SE系テーブル（馬毎成績）を探す。
    戻り値: (テーブル名, {学習用カラム: MDB列名})
    """
    se_keywords = ("着順", "確定着順", "競走馬コード", "UmaCode", "走破タイム", "馬番", "枠番")
    for tname, cols in tables:
        col_str = " ".join(c or "" for c in cols).lower()
        score = sum(1 for kw in se_keywords if kw in col_str)
        if score < 2:
            continue
        chakujun_col = find_col(cols, "着順", "確定着順", "chakujun", "Chakujun", "順位")
        horse_col = find_col(cols, "競走馬コード", "馬コード", "UmaCode", "horse_id")
        time_col = find_col(cols, "走破タイム", "race_time", "タイム", "走破")
        sire_col = find_col(cols, "父コード", "父", "sire_id", "FuchichiCode")
        bms_col = find_col(cols, "母父コード", "母父", "broodmare_sire_id", "HahachichiCode")
        wakuban_col = find_col(cols, "Wakuban", "枠番", "wakuban")
        jockey_col = find_col(cols, "KishuCode", "騎手コード", "jockey_id")
        trainer_col = find_col(cols, "ChokyosiCode", "調教師コード", "trainer_id")
        weight_col = find_col(cols, "Futan", "斤量", "weight_carry")
        date_col = find_col(cols, "日付", "開催日", "race_date", "日")
        place_col = find_col(cols, "競馬場", "競馬場名", "場所", "place_id", "場コード")
        round_col = find_col(cols, "R番", "レース番号", "round", "R")
        dist_col = find_col(cols, "距離", "distance")
        course_col = find_col(cols, "芝ダ別", "TrackCD", "トラックコード", "芝ダ", "course_type")
        mawari_col = find_col(cols, "回り", "Mawari", "周り", "右左", "回りコード", "course_direction")
        weather_col = find_col(cols, "天候", "weather")
        state_col = find_col(cols, "馬場状態", "馬場", "state")
        if chakujun_col and horse_col:
            racecourse_col = find_col(cols, "競馬場名", "競馬場")
            time_diff_col = find_col(cols, "タイム差", "着差", "time_diff")
            mapping = {
                "rank": chakujun_col,
                "horse_id": horse_col,
                "time": time_col,
                "sire_id": sire_col,
                "broodmare_sire_id": bms_col,
                "wakuban": wakuban_col,
                "jockey_id": jockey_col,
                "trainer_id": trainer_col,
                "weight_carry": weight_col,
                "date": date_col,
                "place_id": place_col,
                "round": round_col,
                "distance": dist_col,
                "course_type": course_col,
                "rotation": mawari_col,
                "weather": weather_col,
                "state": state_col,
                "racecourse": racecourse_col or place_col,
                "time_diff": time_diff_col,
            }
            return (tname, {k: v for k, v in mapping.items() if v})
    return (None, {})


def find_combined_table(tables: list[tuple[str, list[str]]]) -> tuple[str | None, dict]:
    """
    RA+SEが統合されたテーブルを探す（着順・日付・馬コードが同一テーブルにある）。
    """
    for tname, cols in tables:
        has_chakujun = find_col(cols, "着順", "確定着順", "chakujun", "順位")
        has_date = find_col(cols, "日付", "開催日", "race_date")
        has_horse = find_col(cols, "競走馬コード", "馬コード", "UmaCode", "horse_id")
        if not (has_chakujun and has_date and has_horse):
            continue
        place_or_racecourse = find_col(cols, "競馬場名", "競馬場", "場所", "place_id")
        mapping = {
            "rank": find_col(cols, "着順", "確定着順", "chakujun", "順位"),
            "horse_id": find_col(cols, "競走馬コード", "馬コード", "UmaCode", "horse_id"),
            "time": find_col(cols, "走破タイム", "race_time", "タイム"),
            "sire_id": find_col(cols, "父コード", "父", "父繁殖登録番号", "sire_id", "FuchichiCode"),
            "broodmare_sire_id": find_col(cols, "母父コード", "母父", "母父繁殖登録番号", "broodmare_sire_id", "HahachichiCode"),
            "wakuban": find_col(cols, "Wakuban", "枠番", "wakuban"),
            "jockey_id": find_col(cols, "KishuCode", "騎手コード", "jockey_id"),
            "trainer_id": find_col(cols, "ChokyosiCode", "調教師コード", "trainer_id"),
            "weight_carry": find_col(cols, "Futan", "斤量", "weight_carry"),
            "date": find_col(cols, "日付", "開催日", "race_date"),
            "place_id": place_or_racecourse,
            "round": find_col(cols, "R番", "レース番号", "round"),
            "distance": find_col(cols, "距離", "distance"),
            "course_type": find_col(cols, "芝ダ別", "TrackCD", "トラックコード", "芝ダ", "course_type"),
            "rotation": find_col(cols, "回り", "Mawari", "周り", "右左", "回りコード", "course_direction"),
            "weather": find_col(cols, "天候", "weather"),
            "state": find_col(cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場", "state"),
            "racecourse": place_or_racecourse,
            "time_diff": find_col(cols, "タイム差", "着差", "time_diff"),
        }
        return (tname, {k: v for k, v in mapping.items() if v})
    return (None, {})


def find_race_raceuma_pair(tables: list[tuple[str, list[str]]]) -> tuple[dict | None, dict | None]:
    """
    Race + RaceUma のペアを探す（TukuAcc7 形式）。
    戻り値: (ra_map, se_map) または (None, None)
    """
    race_tbl = None
    raceuma_tbl = None
    ra_map = {}
    se_map = {}
    for tname, cols in tables:
        if not cols:
            continue
        col_str = " ".join(c or "" for c in cols)
        if "RaceID" not in col_str:
            continue
        # Race と RaceUma をテーブル名で明確に区別（RaceUma で上書きされないように）
        if tname == "Race" and find_col(cols, "開催日") and (find_col(cols, "競馬場名") or find_col(cols, "競馬場")):
            race_tbl = (tname, cols)
        elif tname == "RaceUma" and find_col(cols, "UmaCode") and find_col(cols, "馬番"):
            raceuma_tbl = (tname, cols)
    if not race_tbl or not raceuma_tbl:
        return None, None
    _, ra_cols = race_tbl
    _, se_cols = raceuma_tbl
    chakujun_col = find_col(se_cols, "着順", "確定着順", "chakujun", "順位")
    if not chakujun_col:
        return None, None  # 着順がない RaceUma は学習に使えない
    place_racecourse_col = find_col(ra_cols, "競馬場名", "競馬場", "場コード")
    ra_map = {
        "date": find_col(ra_cols, "開催日", "日付"),
        "place_id": place_racecourse_col,
        "round": find_col(ra_cols, "レース番号", "R番"),
        "distance": find_col(ra_cols, "距離", "distance"),
        "course_type": find_col(ra_cols, "芝ダ", "芝ダ別", "TrackCD", "トラックコード", "コース", "course_type"),
        "rotation": find_col(ra_cols, "回り", "Mawari", "周り", "右左", "回りコード", "course_direction"),
        "weather": find_col(ra_cols, "天候", "天候コード", "weather", "weather_code"),
        "state": find_col(ra_cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場", "state", "track_condition"),
        "racecourse": place_racecourse_col,
    }
    se_racecourse_col = find_col(se_cols, "競馬場名", "競馬場")  # RaceUma に必ずある
    time_diff_col = find_col(se_cols, "タイム差", "着差", "time_diff", "Chakusa")
    se_map = {
        "rank": chakujun_col,
        "horse_id": find_col(se_cols, "UmaCode", "競走馬コード", "馬コード"),
        "time": find_col(se_cols, "走破タイム", "タイム"),
        "sire_id": find_col(se_cols, "父コード", "父", "FuchichiCode"),
        "broodmare_sire_id": find_col(se_cols, "母父コード", "母父", "HahachichiCode"),
        "wakuban": find_col(se_cols, "Wakuban", "枠番", "wakuban"),
        "jockey_id": find_col(se_cols, "KishuCode", "騎手コード", "jockey_id"),
        "trainer_id": find_col(se_cols, "ChokyosiCode", "調教師コード", "trainer_id"),
        "weight_carry": find_col(se_cols, "Futan", "斤量", "weight_carry"),
        "date": find_col(se_cols, "開催日", "日付"),
        "place_id": find_col(se_cols, "競馬場名", "競馬場"),
        "round": find_col(se_cols, "レース番号", "R番"),
        "racecourse": se_racecourse_col,
        "time_diff": time_diff_col,
    }
    ra_map = {k: v for k, v in ra_map.items() if v}
    se_map = {k: v for k, v in se_map.items() if v}
    if not (ra_map.get("date") or se_map.get("date")) or not se_map.get("horse_id"):
        return None, None
    return ra_map, se_map


def build_race_key(date_val: str, place_val: str, round_val: int) -> str:
    """race_key を生成 (YYYYMMDD_JJ_RR)。"""
    d = re.sub(r"[^0-9]", "", _to_str(date_val))[:8]
    if len(d) != 8:
        return ""
    place_code = _to_str(place_val)
    if place_code in PLACE_CODE_MAP:
        jj = PLACE_CODE_MAP[place_code] if len(place_code) <= 2 else place_code
    else:
        jj = place_code.zfill(2) if place_code.isdigit() else "00"
    rr = str(_to_int(round_val)).zfill(2)
    return f"{d}_{jj}_{rr}"


# 全角数字・記号を半角に変換（着差パース用）
_FULL_TO_HALF_TIME = str.maketrans(
    "０１２３４５６７８９＋－．　",
    "0123456789+-. ",
)

def _normalize_time_diff_str(s: str) -> str:
    """着差文字列をクリーニング（全角→半角、スペース除去）。"""
    if not s:
        return ""
    s = str(s).translate(_FULL_TO_HALF_TIME)
    s = s.replace(" ", "").replace("\u3000", "").strip()
    s = s.replace("+", "").strip()
    return s


def parse_time_diff(time_diff_val, rank_val, default: float = 2.0) -> float:
    """
    着差を float に変換。1着の場合は 0。
    文字列（"+0.2", "0．2", "　1.5　" 等）は float にパース。全角・スペースはクリーニング。変換できない場合は default。
    """
    rank_int = _to_int(rank_val, 99)
    if rank_int == 1:
        return 0.0
    if time_diff_val is None or time_diff_val == "":
        return default
    s = _normalize_time_diff_str(str(time_diff_val))
    if not s:
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def is_valid_rank(rank_val) -> bool:
    """確定着順かどうか（取消・除外はスキップ）。"""
    v = _to_str(rank_val)
    if not v:
        return False
    v = v.strip()
    if v in ("取消", "除外", "中止", "取", "除", "中"):
        return False
    try:
        n = int(float(v))
        return 1 <= n <= 30
    except (ValueError, TypeError):
        return False


def extract_learning_data(conn_race, data_path: Path) -> list[dict]:
    """Race.mdb から学習データを抽出する。"""
    tables = list_tables_and_columns(conn_race)
    logger.info("=== Race.mdb テーブル一覧 ===")
    for tname, cols in tables:
        logger.info("  %s: %s", tname, cols if cols else [])

    ra_table, ra_map = find_ra_table(tables)
    se_table, se_map = find_se_table(tables)
    combined_table, combined_map = find_combined_table(tables)
    race_ra_map, race_se_map = find_race_raceuma_pair(tables)

    if race_ra_map and race_se_map:
        if not race_ra_map.get("course_type"):
            for tname, cols in tables:
                if tname == "Race" and cols:
                    logger.warning("Race テーブルに芝ダ/TrackCD列がありません。カラム: %s", cols[:30])
                    break
        if not race_ra_map.get("state"):
            for tname, cols in tables:
                if tname == "Race" and cols:
                    logger.warning("Race テーブルに馬場状態/BabaCD列がありません。カラム: %s", cols[:30])
                    break
        logger.info("race_ra_map: course_type=%s, state=%s, racecourse(R)=%s", race_ra_map.get("course_type"), race_ra_map.get("state"), race_ra_map.get("racecourse"))
        logger.info("race_se_map: racecourse(U)=%s", race_se_map.get("racecourse"))

    rows_out: list[dict] = []

    if race_ra_map and race_se_map:
        race_tname = "Race"
        raceuma_tname = "RaceUma"
        for t, c in tables:
            if not c:
                continue
            cs = " ".join(c or [])
            if t == "Race" and "RaceID" in cs and find_col(c, "開催日"):
                race_tname = t
            elif t == "RaceUma" and "RaceID" in cs and find_col(c, "UmaCode"):
                raceuma_tname = t
        logger.info("Race + RaceUma を使用: %s JOIN %s", race_tname, raceuma_tname)
        sel_parts = []
        idx_map = {}
        pos = 0
        for key in ("date", "place_id", "round"):
            col = race_ra_map.get(key) or race_se_map.get(key)
            if col and key not in idx_map:
                sel_parts.append(f"R.[{race_ra_map.get(key)}]" if race_ra_map.get(key) else f"U.[{race_se_map.get(key)}]")
                idx_map[key] = pos
                pos += 1
        for key in ("rank", "horse_id", "time", "sire_id", "broodmare_sire_id", "jockey_id", "trainer_id", "wakuban", "weight_carry", "time_diff"):
            col = race_se_map.get(key)
            if col:
                sel_parts.append(f"U.[{col}]")
                idx_map[key] = pos
                pos += 1
        for key in ("racecourse",):
            col = race_se_map.get(key)
            if col and key not in idx_map:
                sel_parts.append(f"U.[{col}]")
                idx_map[key] = pos
                pos += 1
        for key in ("distance", "course_type", "rotation", "weather", "state"):
            col = race_ra_map.get(key)
            if col and key not in idx_map:
                sel_parts.append(f"R.[{col}]")
                idx_map[key] = pos
                pos += 1
        sql = f"SELECT {', '.join(sel_parts)} FROM [{race_tname}] R INNER JOIN [{raceuma_tname}] U ON R.[RaceID] = U.[RaceID]"
        cur = conn_race.cursor()
        try:
            cur.execute(sql)
            raw_count = 0
            debug_printed = 0
            for row in cur.fetchall():
                raw_count += 1
                r = _row_to_learning_dict_by_index(row, idx_map, race_ra_map, race_se_map)
                if r:
                    if debug_printed < 5:
                        i_td = idx_map.get("time_diff")
                        raw_td = row[i_td] if i_td is not None and i_td < len(row) else None
                        print(f"[DEBUG time_diff] 件数{debug_printed + 1} raw time_diff={raw_td!r} -> time_diff={r.get('time_diff')}")
                        debug_printed += 1
                    rows_out.append(r)
            if raw_count > 0 and len(rows_out) == 0:
                logger.warning("JOINで %d 件取得したが、course_type等のフィルタで0件になりました。", raw_count)
            elif rows_out:
                sample = rows_out[0]
                logger.info("先頭行サンプル: course_type=%s, state=%s, racecourse=%s", sample.get("course_type"), sample.get("state"), sample.get("racecourse"))
        except Exception as e:
            logger.error("Race+RaceUma JOIN エラー: %s", e)
        cur.close()
    elif combined_table and combined_map:
        logger.info("統合テーブルを使用: %s", combined_table)
        cur = conn_race.cursor()
        select_cols = [v for v in combined_map.values() if v]
        select_str = ", ".join(f"[{c}]" for c in select_cols)
        try:
            cur.execute(f"SELECT {select_str} FROM [{combined_table}]")
            for row in cur.fetchall():
                r = _row_to_learning_dict(row, select_cols, combined_map, is_combined=True)
                if r:
                    rows_out.append(r)
        except Exception as e:
            logger.error("SELECT エラー: %s", e)
        cur.close()
    elif ra_table and se_table and ra_map and se_map:
        logger.info("RAテーブル: %s, SEテーブル: %s を結合", ra_table, se_table)
        join_key_cols = ["date", "place_id", "round"]
        if not all(ra_map.get(k) and se_map.get(k) for k in join_key_cols):
            logger.warning("結合キー(date, place_id, round)が不足しています。")
            return []
        join_cond = " AND ".join(
            f"RA.[{ra_map[k]}] = SE.[{se_map[k]}]"
            for k in join_key_cols if ra_map.get(k) and se_map.get(k)
        )
        ra_select = [f"RA.[{c}]" for c in ra_map.values() if c]
        se_select = [f"SE.[{c}]" for c in se_map.values() if c]
        sql = f"SELECT {', '.join(ra_select)}, {', '.join(se_select)} FROM [{ra_table}] RA INNER JOIN [{se_table}] SE ON {join_cond}"
        cur = conn_race.cursor()
        try:
            cur.execute(sql)
            idx_map = {}
            pos = 0
            for key in ra_map.keys():
                if ra_map.get(key):
                    idx_map[key] = pos
                    pos += 1
            for key in se_map.keys():
                if se_map.get(key) and key not in idx_map:
                    idx_map[key] = pos
                    pos += 1
            for row in cur.fetchall():
                r = _row_to_learning_dict_by_index(row, idx_map, ra_map, se_map)
                if r:
                    rows_out.append(r)
        except Exception as e:
            logger.error("JOIN エラー: %s", e)
        cur.close()
    else:
        logger.warning("RA/SE または統合テーブルを検出できませんでした。")
        return []

    # デバッグ: 先頭5件の time_diff（パース後）
    if rows_out:
        print("[DEBUG] 先頭5件の time_diff (パース後):")
        for i, rec in enumerate(rows_out[:5]):
            print(f"  [{i+1}] time_diff={rec.get('time_diff')}")

    return rows_out


def _row_to_learning_dict(row, select_cols: list[str], col_map: dict, is_combined: bool = True) -> dict | None:
    """1行を学習用辞書に変換する。障害レース・不明は除外。"""
    def get_val(key: str):
        mdb_col = col_map.get(key)
        if not mdb_col or mdb_col not in select_cols:
            return None
        idx = select_cols.index(mdb_col)
        return row[idx] if idx < len(row) else None

    rank_val = get_val("rank")
    if not is_valid_rank(rank_val):
        return None

    date_val = get_val("date")
    place_val = get_val("place_id")
    round_val = get_val("round")

    race_key = build_race_key(date_val, place_val, round_val)
    if not race_key:
        return None

    date_str = _to_str(date_val)
    if len(date_str) >= 10:
        date_str = date_str[:10]
    elif re.match(r"^[0-9]{8}$", date_str):
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    horse_id = _to_str(get_val("horse_id"))
    if not horse_id:
        return None

    course_val = get_val("course_type")
    course_type = map_course_type(course_val)

    jockey_val = get_val("jockey_id")
    jockey_id = _to_str(jockey_val) if jockey_val is not None else ""  # str: 先頭ゼロ落ち防止
    trainer_val = get_val("trainer_id")
    trainer_id = _to_str(trainer_val) if trainer_val is not None else ""  # str: 先頭ゼロ落ち防止
    wakuban_val = get_val("wakuban")
    wakuban = _to_int(wakuban_val, 0) if wakuban_val is not None else 0
    if not (1 <= wakuban <= 8):
        wakuban = 0
    weight_val = get_val("weight_carry")
    weight_carry = _to_float(weight_val, 0.0) if weight_val is not None else 0.0

    time_diff_val = get_val("time_diff")
    return {
        "race_key": race_key,
        "horse_id": horse_id,
        "sire_id": _to_str(get_val("sire_id")),
        "broodmare_sire_id": _to_str(get_val("broodmare_sire_id")),
        "jockey_id": jockey_id,
        "trainer_id": trainer_id,
        "wakuban": wakuban,
        "weight_carry": weight_carry,
        "rank": _to_int(rank_val, 0),
        "time": _to_float(get_val("time"), 0.0),
        "distance": _to_int(get_val("distance"), 0),
        "course_type": course_type,
        "rotation": map_rotation(get_val("rotation")),
        "weather": _to_str(get_val("weather")),
        "state": map_state(get_val("state")),
        "date": date_str,
        "racecourse": _resolve_racecourse(get_val("racecourse") or get_val("place_id")),
        "time_diff": parse_time_diff(time_diff_val, rank_val, 2.0),
    }


def _resolve_racecourse(val) -> str:
    """競馬場コード(01-10)を競馬場名に変換。"""
    s = _to_str(val)
    if not s:
        return ""
    if s in PLACE_CODE_MAP and len(s) <= 2 and s.isdigit():
        return PLACE_CODE_MAP.get(s, s)
    return s


def _row_to_learning_dict_by_index(row, idx_map: dict, ra_map: dict, se_map: dict) -> dict | None:
    """JOIN結果の1行を学習用辞書に変換する（インデックスマップ使用）。"""
    def get_val(key: str):
        i = idx_map.get(key)
        if i is None or i >= len(row):
            return None
        return row[i]

    rank_val = get_val("rank")
    if not is_valid_rank(rank_val):
        return None

    date_val = get_val("date")
    place_val = get_val("place_id")
    round_val = get_val("round")

    race_key = build_race_key(date_val, place_val, round_val)
    if not race_key:
        return None

    date_str = _to_str(date_val)
    if re.match(r"^[0-9]{8}$", date_str):
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    elif len(date_str) >= 10:
        date_str = date_str[:10]

    horse_id = _to_str(get_val("horse_id"))
    if not horse_id:
        return None

    course_val = get_val("course_type")
    course_type = map_course_type(course_val)

    jockey_val = get_val("jockey_id")
    jockey_id = _to_str(jockey_val) if jockey_val is not None else ""  # str: 先頭ゼロ落ち防止
    trainer_val = get_val("trainer_id")
    trainer_id = _to_str(trainer_val) if trainer_val is not None else ""  # str: 先頭ゼロ落ち防止
    wakuban_val = get_val("wakuban")
    wakuban = _to_int(wakuban_val, 0) if wakuban_val is not None else 0
    if not (1 <= wakuban <= 8):
        wakuban = 0
    weight_val = get_val("weight_carry")
    weight_carry = _to_float(weight_val, 0.0) if weight_val is not None else 0.0

    time_diff_val = get_val("time_diff")
    return {
        "race_key": race_key,
        "horse_id": horse_id,
        "sire_id": _to_str(get_val("sire_id")),
        "broodmare_sire_id": _to_str(get_val("broodmare_sire_id")),
        "jockey_id": jockey_id,
        "trainer_id": trainer_id,
        "wakuban": wakuban,
        "weight_carry": weight_carry,
        "rank": _to_int(rank_val, 0),
        "time": _to_float(get_val("time"), 0.0),
        "distance": _to_int(get_val("distance"), 0),
        "course_type": course_type,
        "rotation": map_rotation(get_val("rotation")),
        "weather": _to_str(get_val("weather")),
        "state": map_state(get_val("state")),
        "date": date_str,
        "racecourse": _resolve_racecourse(get_val("racecourse") or get_val("place_id")),
        "time_diff": parse_time_diff(time_diff_val, rank_val, 2.0),
    }


def load_race_info_from_race_db(conn_race) -> dict[str, dict]:
    """Race.mdb の Race テーブルから (race_key) -> {distance, weather, state, rotation} を取得。"""
    result: dict[str, dict] = {}
    tables = list_tables_and_columns(conn_race)
    race_tbl = None
    date_col = place_col = round_col = dist_col = weather_col = state_col = mawari_col = course_col = None
    for tname, cols in tables:
        if not cols or tname != "Race":
            continue
        if find_col(cols, "開催日") and (find_col(cols, "競馬場名") or find_col(cols, "競馬場")):
            race_tbl = tname
            date_col = find_col(cols, "開催日")
            place_col = find_col(cols, "競馬場名", "競馬場", "場コード")
            round_col = find_col(cols, "レース番号", "R番")
            dist_col = find_col(cols, "距離", "distance")
            weather_col = find_col(cols, "天候", "天候コード", "weather")
            state_col = find_col(cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場", "state")
            mawari_col = find_col(cols, "回り", "Mawari", "周り", "右左", "回りコード", "course_direction")
            course_col = find_col(cols, "芝ダ", "芝ダ別", "TrackCD", "トラックコード")
            break
    if not race_tbl or not all([date_col, place_col, round_col]):
        return result
    select_cols = [date_col, place_col, round_col]
    if dist_col:
        select_cols.append(dist_col)
    if course_col:
        select_cols.append(course_col)
    if weather_col:
        select_cols.append(weather_col)
    if state_col:
        select_cols.append(state_col)
    if mawari_col:
        select_cols.append(mawari_col)
    cur = conn_race.cursor()
    try:
        cur.execute(
            f"SELECT {', '.join(f'[{c}]' for c in select_cols)} FROM [{race_tbl}]"
        )
        for row in cur.fetchall():
            d = _to_str(row[0])
            p = _to_str(row[1])
            r = _to_str(row[2])
            if not d or not (p or r):
                continue
            key = build_race_key(d, p, _to_int(r))
            if not key:
                continue
            info = {}
            idx = 3
            if dist_col and len(row) > idx:
                info["distance"] = _to_int(row[idx])
                idx += 1
            if course_col and len(row) > idx:
                ct = map_course_type(row[idx])
                if ct:
                    info["course_type"] = ct
                idx += 1
            if weather_col and len(row) > idx:
                info["weather"] = _to_str(row[idx])
                idx += 1
            if state_col and len(row) > idx:
                info["state"] = row[idx]  # 生値（enrich で map_state）
                idx += 1
            if mawari_col and len(row) > idx:
                info["rotation"] = map_rotation(row[idx])
            if info:
                result[key] = info
    except Exception as e:
        logger.warning("Race.mdb Raceテーブル読込エラー: %s", e)
    cur.close()
    return result


def load_race_info_from_master(conn_master) -> dict[str, dict]:
    """Master.mdb のレコードマスタから (date_place_round) -> {distance, weather, state} を取得。"""
    result: dict[str, dict] = {}
    tables = list_tables_and_columns(conn_master)
    rec_table = None
    date_col = place_col = round_col = dist_col = weather_col = state_col = None
    for tname, cols in tables:
        if not cols:
            continue
        if find_col(cols, "RaceID") and find_col(cols, "距離") and find_col(cols, "開催日"):
            rec_table = tname
            date_col = find_col(cols, "開催日")
            place_col = find_col(cols, "競馬場名", "競馬場", "場コード")
            round_col = find_col(cols, "レース番号", "R番")
            dist_col = find_col(cols, "距離")
            weather_col = find_col(cols, "天候", "天候コード", "weather")
            state_col = find_col(cols, "馬場状態", "馬場", "state")
            break
    if not rec_table or not all([date_col, place_col, round_col, dist_col]):
        return result
    select_cols = [date_col, place_col, round_col, dist_col]
    if weather_col:
        select_cols.append(weather_col)
    if state_col:
        select_cols.append(state_col)
    cur = conn_master.cursor()
    try:
        cur.execute(
            f"SELECT {', '.join(f'[{c}]' for c in select_cols)} FROM [{rec_table}]"
        )
        for row in cur.fetchall():
            d = _to_str(row[0])
            p = _to_str(row[1])
            r = _to_str(row[2])
            dist = _to_int(row[3])
            if d and (p or r):
                key = build_race_key(d, p, _to_int(r))
                if key:
                    info = {"distance": dist}
                    if weather_col and len(row) > 4:
                        info["weather"] = _to_str(row[4])
                    if state_col and len(row) > 5:
                        info["state"] = _to_str(row[5])
                    result[key] = info
    except Exception as e:
        logger.warning("レコードマスタ読込エラー: %s", e)
    cur.close()
    return result


def enrich_race_info(rows: list[dict], race_info_map: dict[str, dict]) -> None:
    """学習データの distance, course_type, weather, state, rotation を補完。"""
    for r in rows:
        rk = r.get("race_key", "")
        if not rk or rk not in race_info_map:
            continue
        info = race_info_map[rk]
        if r.get("distance", 0) == 0 and info.get("distance"):
            r["distance"] = info["distance"]
        if not r.get("course_type") and info.get("course_type"):
            r["course_type"] = info["course_type"]
        if not r.get("weather") and info.get("weather"):
            r["weather"] = info["weather"]
        if r.get("state") is None or r.get("state") == 0:
            if info.get("state") is not None:
                r["state"] = map_state(info["state"])
        if r.get("rotation") is None or r.get("rotation") == 0:
            if info.get("rotation") is not None:
                r["rotation"] = map_rotation(info["rotation"])


def load_pedigree_from_master(conn_master) -> tuple[dict[str, str], dict[str, str]]:
    """Master.mdb から UmaCode -> sire_id, UmaCode -> broodmare_sire_id のマップを取得。"""
    sire_map: dict[str, str] = {}
    bms_map: dict[str, str] = {}
    tables = list_tables_and_columns(conn_master)
    um_table = None
    um_id_col = None
    sire_col = None
    dam_col = None
    for tname, cols in tables:
        if not cols:
            continue
        id_c = find_col(cols, "UmaCode", "競走馬コード", "馬コード")
        s_c = find_col(cols, "父繁殖登録番号", "父コード", "FuchichiCode")
        d_c = find_col(cols, "母繁殖登録番号", "母コード", "BochichiCode")
        if id_c and s_c:
            um_table = tname
            um_id_col = id_c
            sire_col = s_c
            dam_col = d_c
            break
    if not um_table or not um_id_col or not sire_col:
        return sire_map, bms_map
    breed_table = None
    breed_id_col = None
    breed_sire_col = None
    for tname, cols in tables:
        if not cols:
            continue
        id_c = find_col(cols, "繁殖登録番号")
        s_c = find_col(cols, "父馬繁殖登録番号", "父繁殖登録番号")
        if id_c and s_c and "繁殖" in " ".join(cols):
            breed_table = tname
            breed_id_col = id_c
            breed_sire_col = s_c
            break
    cur = conn_master.cursor()
    dam_to_bms: dict[str, str] = {}
    if breed_table and breed_id_col and breed_sire_col:
        try:
            cur.execute(f"SELECT [{breed_id_col}], [{breed_sire_col}] FROM [{breed_table}]")
            for row in cur.fetchall():
                dam_id = _to_str(row[0])
                bms_id = _to_str(row[1])
                if dam_id and bms_id:
                    dam_to_bms[dam_id] = bms_id
        except Exception as e:
            logger.warning("繁殖馬マスタ読込エラー: %s", e)
    cols_needed = [um_id_col, sire_col]
    if dam_col:
        cols_needed.append(dam_col)
    try:
        cur.execute(f"SELECT {', '.join(f'[{c}]' for c in cols_needed)} FROM [{um_table}]")
        for row in cur.fetchall():
            uma = _to_str(row[0])
            sire = _to_str(row[1])
            if uma:
                if sire:
                    sire_map[uma] = sire
                if dam_col and len(row) > 2:
                    dam = _to_str(row[2])
                    if dam and dam in dam_to_bms:
                        bms_map[uma] = dam_to_bms[dam]
    except Exception as e:
        logger.warning("競走馬マスタ読込エラー: %s", e)
    cur.close()
    return sire_map, bms_map


def enrich_pedigree(rows: list[dict], sire_map: dict[str, str], bms_map: dict[str, str]) -> None:
    """学習データの sire_id, broodmare_sire_id を補完。"""
    for r in rows:
        hid = r.get("horse_id", "")
        if not hid:
            continue
        if not r.get("sire_id") and hid in sire_map:
            r["sire_id"] = sire_map[hid]
        if not r.get("broodmare_sire_id") and hid in bms_map:
            r["broodmare_sire_id"] = bms_map[hid]


def extract_pedigree_maps(conn_master) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Master.mdb の競走馬マスタから sire_id_map, bms_id_map を抽出する。"""
    tables = list_tables_and_columns(conn_master)
    logger.info("=== Master.mdb テーブル一覧 ===")
    for tname, cols in tables:
        logger.info("  %s: %s", tname, cols if cols else [])

    um_table = None
    sire_id_col = None
    sire_name_col = None
    bms_id_col = None
    bms_name_col = None

    for tname, cols in tables:
        s_id = find_col(cols, "父繁殖登録番号", "父コード", "FuchichiCode")
        s_name = find_col(cols, "父馬名", "父名")
        b_id = find_col(cols, "母父繁殖登録番号", "母父コード", "HahachichiCode")
        b_name = find_col(cols, "母父馬名", "母父名")
        if s_id and s_name and b_id and b_name:
            um_table = tname
            sire_id_col = s_id
            sire_name_col = s_name
            bms_id_col = b_id
            bms_name_col = b_name
            break

    if not um_table or not sire_id_col or not sire_name_col or not bms_id_col or not bms_name_col:
        logger.warning("競走馬マスタ（父・母父繁殖登録番号・馬名）を検出できませんでした。")
        return [], []

    cur = conn_master.cursor()
    sire_list: list[tuple[str, str]] = []
    sire_seen: set[str] = set()
    bms_list: list[tuple[str, str]] = []
    bms_seen: set[str] = set()
    try:
        cur.execute(
            f"SELECT [{sire_id_col}], [{sire_name_col}], [{bms_id_col}], [{bms_name_col}] FROM [{um_table}]"
        )
        for row in cur.fetchall():
            sid = _to_str(row[0])
            sname = _to_str(row[1])
            bid = _to_str(row[2])
            bname = _to_str(row[3])
            if sid and sname and sid not in sire_seen:
                sire_seen.add(sid)
                sire_list.append((sid, sname))
            if bid and bname and bid not in bms_seen:
                bms_seen.add(bid)
                bms_list.append((bid, bname))
    except Exception as e:
        logger.warning("血統辞書抽出エラー: %s", e)
    cur.close()

    return sire_list, bms_list


def extract_jockey_map_from_raceuma(conn_race) -> list[tuple[str, str]]:
    """
    Race.mdb の RaceUma から (騎手コード, 騎手名漢字) を重複排除して取得。
    Master に騎手マスタがない場合のフォールバック。
    """
    tables = list_tables_and_columns(conn_race)
    for tname, cols in tables:
        if not cols:
            continue
        col_str = " ".join(c or "" for c in cols)
        if "RaceUma" in tname or ("RaceID" in col_str and "UmaCode" in col_str):
            jockey_col = find_col(cols, "騎手コード", "KishuCode")
            name_col = find_col(cols, "騎手名漢字", "騎手名", "KishuMei")
            if jockey_col and name_col:
                cur = conn_race.cursor()
                result: list[tuple[str, str]] = []
                seen: set[str] = set()
                try:
                    cur.execute(f"SELECT [{jockey_col}], [{name_col}] FROM [{tname}] WHERE [{jockey_col}] IS NOT NULL")
                    for row in cur.fetchall():
                        jid = _to_str(row[0])
                        jname = _to_str(row[1]) if len(row) > 1 else "Unknown"
                        if jid and jid != "00000" and jid not in seen:
                            seen.add(jid)
                            result.append((jid, jname if jname else "Unknown"))
                except Exception as e:
                    logger.warning("RaceUma 騎手辞書抽出エラー: %s", e)
                cur.close()
                return result
    return []


def extract_trainer_map_from_raceuma(conn_race) -> list[tuple[str, str]]:
    """Race.mdb の RaceUma から調教師辞書を抽出（Master に CH がない場合のフォールバック）。"""
    tables = list_tables_and_columns(conn_race)
    for tname, cols in tables:
        if not cols:
            continue
        col_str = " ".join(c or "" for c in cols)
        if "RaceUma" in tname or ("RaceID" in col_str and "UmaCode" in col_str):
            trainer_col = find_col(cols, "ChokyosiCode", "調教師コード")
            name_col = find_col(cols, "調教師名漢字", "調教師名", "ChokyosiMei")
            if trainer_col and name_col:
                cur = conn_race.cursor()
                result: list[tuple[str, str]] = []
                seen: set[str] = set()
                try:
                    cur.execute(f"SELECT [{trainer_col}], [{name_col}] FROM [{tname}] WHERE [{trainer_col}] IS NOT NULL")
                    for row in cur.fetchall():
                        tid = _to_str(row[0])
                        tname_val = _to_str(row[1]) if len(row) > 1 else "Unknown"
                        if tid and tid != "00000" and tid not in seen:
                            seen.add(tid)
                            result.append((tid, tname_val if tname_val else "Unknown"))
                except Exception as e:
                    logger.warning("RaceUma 調教師辞書抽出エラー: %s", e)
                cur.close()
                return result
    return []


def extract_trainer_map(conn_master) -> list[tuple[str, str]]:
    """
    Master.mdb の調教師マスタ（CH 系テーブル）から trainer_id_map を抽出する。
    CH / CHOKYOSI 等で始まるテーブルを優先。
    戻り値: [(trainer_id, trainer_name), ...]
    """
    tables = list_tables_and_columns(conn_master)
    ch_tables = [(t, c) for t, c in tables if t and str(t).upper().startswith("CH")]
    cand_tables = ch_tables if ch_tables else tables
    trainer_table = None
    id_col = None
    name_col = None
    for tname, cols in cand_tables:
        if not cols:
            continue
        id_c = find_col(cols, "ChokyosiCode", "調教師コード", "trainer_id")
        name_c = find_col(cols, "ChokyosiName", "調教師名", "調教師名漢字", "RyakuBamei", "ChokyosiMei", "trainer_name")
        if id_c and name_c:
            trainer_table = tname
            id_col = id_c
            name_col = name_c
            break
    if not trainer_table or not id_col or not name_col:
        logger.warning("調教師マスタ（調教師コード・調教師名）を検出できませんでした。空のCSVを作成します。")
        return []
    cur = conn_master.cursor()
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        cur.execute(f"SELECT [{id_col}], [{name_col}] FROM [{trainer_table}]")
        for row in cur.fetchall():
            tid = _to_str(row[0]) if row[0] is not None else ""
            tname_val = _to_str(row[1]) if row[1] is not None else "Unknown"
            if tid and tid not in seen:
                seen.add(tid)
                result.append((tid, tname_val))
    except Exception as e:
        logger.warning("調教師辞書抽出エラー: %s", e)
    cur.close()
    return result


def extract_jockey_map(conn_master) -> list[tuple[str, str]]:
    """
    Master.mdb の騎手マスタ（KS 系テーブル）から jockey_id_map を抽出する。
    KS / KISHU 等で始まるテーブルを優先し、KishuCode, KishuName (騎手名/RyakuBamei) を取得。
    戻り値: [(jockey_id, jockey_name), ...]
    """
    tables = list_tables_and_columns(conn_master)
    # 1) KS で始まるテーブルを優先
    ks_tables = [(t, c) for t, c in tables if t and str(t).upper().startswith("KS")]
    cand_tables = ks_tables if ks_tables else tables
    jockey_table = None
    id_col = None
    name_col = None
    for tname, cols in cand_tables:
        if not cols:
            continue
        id_c = find_col(cols, "KishuCode", "騎手コード", "jockey_id")
        name_c = find_col(cols, "KishuName", "騎手名", "騎手名漢字", "RyakuBamei", "KishuMei", "jockey_name")
        if id_c and name_c:
            jockey_table = tname
            id_col = id_c
            name_col = name_c
            break
    if not jockey_table or not id_col or not name_col:
        logger.warning("騎手マスタ（騎手コード・騎手名）を検出できませんでした。")
        return []
    cur = conn_master.cursor()
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        cur.execute(f"SELECT [{id_col}], [{name_col}] FROM [{jockey_table}]")
        for row in cur.fetchall():
            jid = _to_str(row[0]) if row[0] is not None else ""  # str: 先頭ゼロ落ち防止
            jname = _to_str(row[1]) if row[1] is not None else "Unknown"
            if jid and jid not in seen:
                seen.add(jid)
                result.append((jid, jname))
    except Exception as e:
        logger.warning("騎手辞書抽出エラー: %s", e)
    cur.close()
    return result


def main() -> int:
    try:
        import pyodbc
    except ImportError:
        logger.error("pyodbc がインストールされていません。pip install pyodbc")
        return 1

    drivers = [x for x in pyodbc.drivers() if "Access" in x or "ACE" in x]
    if not drivers:
        logger.error("Microsoft Access 用 ODBC ドライバーが見つかりません。")
        return 1

    data_path = Path(sys.argv[1]) if len(sys.argv) >= 2 and Path(sys.argv[1]).exists() else DATA_PATH
    if not (data_path / "Race.mdb").exists() and (data_path / "Data").exists():
        data_path = data_path / "Data"
    race_mdb = data_path / "Race.mdb"
    master_mdb = data_path / "Master.mdb"
    if not race_mdb.exists():
        race_mdb = data_path / "Kako.mdb"
    if not race_mdb.exists():
        logger.error("Race.mdb または Kako.mdb が見つかりません: %s", data_path)
        return 1

    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={race_mdb};"
    try:
        conn_race = pyodbc.connect(conn_str)
    except Exception as e:
        logger.error("Race.mdb 接続エラー: %s", e)
        return 1

    rows = extract_learning_data(conn_race, data_path)
    conn_race.close()

    # Race.mdb に着順がない場合（出走表のみ）は Kako.mdb を試す
    if not rows and race_mdb.name == "Race.mdb":
        kako_mdb = data_path / "Kako.mdb"
        if kako_mdb.exists():
            logger.info("Race.mdb に着順データがありません。Kako.mdb を試します。")
            try:
                conn_kako = pyodbc.connect(
                    f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={kako_mdb};"
                )
                rows = extract_learning_data(conn_kako, data_path)
                conn_kako.close()
            except Exception as e:
                logger.error("Kako.mdb 接続エラー: %s", e)

    if not rows:
        logger.warning("学習データが 0 件でした。")
    else:
        race_info_map: dict[str, dict] = {}
        try:
            conn_r = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={race_mdb};")
            race_info_map = load_race_info_from_race_db(conn_r)
            conn_r.close()
        except Exception as e:
            logger.warning("Race.mdb レース情報読込スキップ: %s", e)
        if master_mdb.exists():
            try:
                conn_m = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={master_mdb};")
                sire_map, bms_map = load_pedigree_from_master(conn_m)
                master_race_info = load_race_info_from_master(conn_m)
                conn_m.close()
                if sire_map or bms_map:
                    enrich_pedigree(rows, sire_map, bms_map)
                    n_sire = sum(1 for r in rows if r.get("sire_id"))
                    n_bms = sum(1 for r in rows if r.get("broodmare_sire_id"))
                    logger.info("血統補完: sire_id %d 件, broodmare_sire_id %d 件", n_sire, n_bms)
                for k, v in master_race_info.items():
                    if k not in race_info_map:
                        race_info_map[k] = {}
                    for fk, fv in v.items():
                        if fv and not race_info_map[k].get(fk):
                            race_info_map[k][fk] = fv
            except Exception as e:
                logger.warning("Master補完スキップ: %s", e)
        if race_info_map:
            enrich_race_info(rows, race_info_map)
            n_dist = sum(1 for r in rows if r.get("distance") and str(r.get("distance")) != "0")
            n_weather = sum(1 for r in rows if r.get("weather"))
            n_state = sum(1 for r in rows if r.get("state"))
            logger.info("レース情報補完: distance %d 件, weather %d 件, state %d 件", n_dist, n_weather, n_state)
        # course_type が None の行を明示的に除外（dropna）
        rows = [r for r in rows if r.get("course_type") is not None and str(r.get("course_type", "")).strip()]
        OUTPUT_LEARNING.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_LEARNING, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEARNING_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        # === Final Verification ===
        from collections import Counter
        ct_counts = Counter(str(r.get("course_type", "")) for r in rows)
        st_counts = Counter(r.get("state", 0) for r in rows)
        rot_counts = Counter(r.get("rotation", 0) for r in rows)
        rc_counts = Counter(r.get("racecourse", "") for r in rows)
        logger.info("=== Final Verification ===")
        logger.info("course_type (1=Turf, 2=Dirt): %s", dict(ct_counts))
        logger.info("state (1=良,2=稍重,3=重,4=不良): %s", dict(st_counts))
        logger.info("rotation (0=不明,1=右,2=左,3=直): %s", dict(rot_counts))
        logger.info("racecourse (top 5): %s", dict(rc_counts.most_common(5)))
        logger.info("出力: %s (%d 件)", OUTPUT_LEARNING, len(rows))

    if master_mdb.exists():
        conn_str_m = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={master_mdb};"
        try:
            conn_master = pyodbc.connect(conn_str_m)
        except Exception as e:
            logger.error("Master.mdb 接続エラー: %s", e)
        else:
            sire_list, bms_list = extract_pedigree_maps(conn_master)
            jockey_list = extract_jockey_map(conn_master)
            if not jockey_list:
                try:
                    conn_race2 = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={race_mdb};")
                    jockey_list = extract_jockey_map_from_raceuma(conn_race2)
                    conn_race2.close()
                    if jockey_list:
                        logger.info("騎手辞書を RaceUma から抽出: %d 件", len(jockey_list))
                except Exception as e:
                    logger.warning("RaceUma 騎手辞書フォールバック失敗: %s", e)
            trainer_list = extract_trainer_map(conn_master)
            if not trainer_list:
                try:
                    conn_race3 = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={race_mdb};")
                    trainer_list = extract_trainer_map_from_raceuma(conn_race3)
                    conn_race3.close()
                    if trainer_list:
                        logger.info("調教師辞書を RaceUma から抽出: %d 件", len(trainer_list))
                except Exception as e:
                    logger.warning("RaceUma 調教師辞書フォールバック失敗: %s", e)
            conn_master.close()
            if sire_list:
                OUTPUT_SIRE_MAP.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_SIRE_MAP, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sire_id", "sire_name"])
                    w.writerows(sire_list)
                logger.info("出力: %s (%d 件)", OUTPUT_SIRE_MAP, len(sire_list))
            if bms_list:
                OUTPUT_BMS_MAP.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_BMS_MAP, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["bms_id", "bms_name"])
                    w.writerows(bms_list)
                logger.info("出力: %s (%d 件)", OUTPUT_BMS_MAP, len(bms_list))
            if jockey_list:
                OUTPUT_JOCKEY_MAP.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_JOCKEY_MAP, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["jockey_id", "jockey_name"])
                    w.writerows(jockey_list)
                logger.info("出力: %s (%d 件)", OUTPUT_JOCKEY_MAP, len(jockey_list))
            OUTPUT_TRAINER_MAP.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_TRAINER_MAP, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["trainer_id", "trainer_name"])
                w.writerows(trainer_list)
            if trainer_list:
                logger.info("出力: %s (%d 件)", OUTPUT_TRAINER_MAP, len(trainer_list))
            else:
                logger.warning("調教師辞書が 0 件のため、空のCSVを作成しました: %s", OUTPUT_TRAINER_MAP)
    else:
        logger.warning("Master.mdb が見つかりません。血統辞書は出力しません: %s", master_mdb)

    return 0


if __name__ == "__main__":
    sys.exit(main())
