# -*- coding: utf-8 -*-
"""
未来のレースを予測し、HTMLレポートを出力する統合スクリプト。

Inputs:
  - TukuAcc7/Data/Snap.mdb (出馬表)
  - config/sire_features_master.csv, config/bms_features_master.csv
  - config/global_stats.json
  - config/sire_name_to_int.pkl, config/bms_name_to_int.pkl, config/jockey_name_to_int.pkl, config/trainer_name_to_int.pkl
  - config/racecourse_name_to_int.pkl
  - config/sire_id_map.csv, config/bms_id_map.csv, config/jockey_id_map.csv, config/trainer_id_map.csv (任意)
  - jv_data/models/lgb_target_rank_1to3.txt (LightGBM)

Outputs:
  - jv_data/future_races.csv
  - jv_data/reports/report_YYYYMMDD_HHMM.html
  - jv_data/history/prediction_log.csv（予測ログ・後日精度検証用）
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# パス設定
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "TukuAcc7" / "Data"
SNAP_MDB = DATA_PATH / "Snap.mdb"
OUTPUT_FUTURE = SCRIPT_DIR / "jv_data" / "future_races.csv"
CONFIG = SCRIPT_DIR / "config"
MODEL_PATH = SCRIPT_DIR / "jv_data" / "models" / "lgb_target_rank_1to3.txt"
PREDICTION_LOG_PATH = SCRIPT_DIR / "jv_data" / "history" / "prediction_log.csv"
MODEL_VERSION = "ver_2.0"
PREDICTION_LOG_COLUMNS = [
    "race_id", "date", "race_name", "horse_num", "horse_name", "horse_id",
    "ai_score", "logic_score", "final_score", "rank_predict", "mark", "model_version",
]

# 競馬場コード
PLACE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}
for k, v in list(PLACE_CODE_MAP.items()):
    PLACE_CODE_MAP[v] = k.zfill(2)

# JyoCD -> 競馬場名 変換（Snap.mdb 用。int の場合は str に変換してマッチング）
JYO_NAME_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

# 距離・コース区分
DISTANCES = ["short", "mile", "mid", "long"]
COURSES = ["turf", "dirt"]

# スコア重み（マジックナンバー排除）
WEIGHT_AI_SCORE = 0.7
WEIGHT_LOGIC_SCORE = 0.3
WEIGHT_SIRE = 0.7
WEIGHT_BMS = 0.3
DEFAULT_GLOBAL_MEAN = 0.05  # データが全くない場合の最低保証値
ODDS_DUMMY_VALUE = 0.0      # オッズを使わない（未来予測では未確定のため）
WEIGHT_CARRY_DEFAULT = 55.0  # 斤量未設定時のフォールバック値

# 学習時 (train_model.py) と完全一致する特徴量リスト（Logic + クロス + 近5走含む）
FEATURE_COLS = [
    "sire_id_int",
    "bms_id_int",
    "jockey_id_int",
    "trainer_id_int",
    "racecourse_id_int",
    "course_type",
    "state",
    "rotation",
    "wakuban",
    "weight_carry",
    "distance",
    "sire_state_int",
    "course_wakuban_int",
    "sire_dist_int",
    "sire_type_int",
    "bms_dist_int",
    "bms_type_int",
    "bms_state_int",
    "jockey_course_int",
    "logic_score",
    "prev_rank_1", "prev_rank_2", "prev_rank_3", "prev_rank_4", "prev_rank_5",
    "prev_time_diff_1", "prev_time_diff_2", "prev_time_diff_3", "prev_time_diff_4", "prev_time_diff_5",
    "avg_rank_5",
    "avg_time_diff_5",
    "interval",
    "recency",  # 改善点5: 前走の鮮度（1/(1+interval/60)）
]


def _to_str(v, default: str = "") -> str:
    if v is None or v == "":
        return default
    s = str(v).strip()
    return s if s else default


def _to_int(v, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _to_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def classify_distance(meters: int) -> str:
    if meters <= 0:
        return "mile"
    if meters < 1400:
        return "short"
    if meters <= 1800:
        return "mile"
    if meters <= 2400:
        return "mid"
    return "long"


def classify_course(ct: str) -> str:
    c = _to_str(ct)
    return "turf" if c == "1" else "dirt"


def get_dist_cat(meters: int) -> str:
    """距離を4区分に分類（クロス特徴量用。train_model と一致）。"""
    d = _to_int(meters, 1600)
    if d < 1400:
        return "Short"
    if d < 1800:
        return "Mile"
    if d < 2400:
        return "Middle"
    return "Long"


def build_race_key(date_val: str, place_val: str, round_val: int) -> str:
    d = re.sub(r"[^0-9]", "", _to_str(date_val))[:8]
    if len(d) != 8:
        return ""
    p = _to_str(place_val)
    jj = PLACE_CODE_MAP.get(p, p.zfill(2) if p.isdigit() else "00")
    rr = str(_to_int(round_val)).zfill(2)
    return f"{d}_{jj}_{rr}"


def build_race_key_from_parts(year: int, month: int, day: int, jyo_cd: str, race_num: str) -> str:
    """YYYYMMDD_JJ_RR 形式の race_key を生成。各パーツはゼロ埋め。"""
    yy = str(_to_int(year)).zfill(4)
    mm = str(_to_int(month)).zfill(2)
    dd = str(_to_int(day)).zfill(2)
    jj = _to_str(jyo_cd)
    jj = PLACE_CODE_MAP.get(jj, jj.zfill(2) if jj.isdigit() else "00")
    rr = str(_to_int(race_num)).zfill(2)
    return f"{yy}{mm}{dd}_{jj}_{rr}"


def _course_type_from_track(track_cd: str | int | None) -> str:
    """TrackCD: 10〜22 は芝(1)、それ以外はダート(2)。判別不能時はダート。"""
    try:
        v = int(float(_to_str(track_cd, "0") or 0))
        return "1" if 10 <= v <= 22 else "2"
    except (ValueError, TypeError):
        return "2"


def _jyo_to_name(jyo_cd) -> str:
    """JyoCD (01-10) を競馬場名に変換。"""
    s = _to_str(jyo_cd)
    if not s:
        return ""
    if s.isdigit() and len(s) <= 2:
        s = s.zfill(2)
    return JYO_NAME_MAP.get(s, s)


def _state_from_baba(baba_cd) -> int:
    """BabaCD: 10-19=良(1), 20-29=稍重(2), 30-39=重(3), 40+=不良(4)。"""
    try:
        n = int(float(_to_str(baba_cd, "0") or 0))
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


def _rotation_from_mawari(mawari_val) -> int:
    """回り: 右/1->1, 左/2->2, 直/3->3。"""
    v = _to_str(mawari_val)
    if not v:
        return 0
    if "右" in v or v in ("1", "右", "右回"):
        return 1
    if "左" in v or v in ("2", "左", "左回"):
        return 2
    if "直" in v or v in ("3", "直", "直線"):
        return 3
    try:
        n = int(float(v))
        if 1 <= n <= 3:
            return n
    except (ValueError, TypeError):
        pass
    return 0


# ========== 1. 未来データ抽出 ==========
def _find_col(cols: list[str], *candidates: str) -> str | None:
    """カラム名から候補のいずれかと一致するものを返す。"""
    col_lower = {(c or "").strip().lower(): c for c in cols if c}
    for c in candidates:
        for col in cols:
            if col and (col.strip() == c or (col or "").strip().lower() == c.lower()):
                return col
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None


def _list_snap_tables(conn) -> list[tuple[str, list[str]]]:
    """Snap.mdb のテーブル一覧（MSys除外）を返す。"""
    tables = []
    cur = conn.cursor()
    for row in cur.tables(tableType="TABLE"):
        tname = getattr(row, "table_name", None) or row[2]
        if not tname or str(tname).startswith("MSys"):
            continue
        try:
            cur2 = conn.cursor()
            cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
            cols = [d[0] for d in cur2.description] if cur2.description else []
            cur2.close()
            if cols:
                cur2 = conn.cursor()
                cur2.execute(f"SELECT COUNT(*) FROM [{tname}]")
                cnt = (cur2.fetchone() or [0])[0]
                cur2.close()
                if cnt > 0:
                    tables.append((tname, cols))
        except Exception:
            pass
    cur.close()
    return tables


def _load_race_info_map(data_path: Path) -> dict[str, dict]:
    """Race.mdb から RaceID -> {date, place, round, distance, course_type} を取得。"""
    result: dict[str, dict] = {}
    race_mdb = data_path / "Race.mdb"
    if not race_mdb.exists():
        return result
    try:
        import pyodbc
        conn = pyodbc.connect(
            f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={race_mdb};"
        )
        cur = conn.cursor()
        for row in cur.tables(tableType="TABLE"):
            tname = getattr(row, "table_name", None) or row[2]
            if tname != "Race":
                continue
            cur2 = conn.cursor()
            cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
            cols = [d[0] for d in cur2.description] if cur2.description else []
            cur2.close()
            rid = _find_col(cols, "RaceID")
            date_c = _find_col(cols, "開催日", "日付")
            place_c = _find_col(cols, "競馬場名", "競馬場", "場コード")
            round_c = _find_col(cols, "レース番号", "R番")
            dist_c = _find_col(cols, "距離", "distance")
            course_c = _find_col(cols, "芝ダ", "芝ダ別", "トラックコード")
            state_c = _find_col(cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場")
            mawari_c = _find_col(cols, "回り", "Mawari", "周り", "右左")
            if not all([rid, date_c, place_c, round_c]):
                break
            sel = f"[{rid}],[{date_c}],[{place_c}],[{round_c}]"
            if dist_c:
                sel += f",[{dist_c}]"
            if course_c:
                sel += f",[{course_c}]"
            if state_c:
                sel += f",[{state_c}]"
            if mawari_c:
                sel += f",[{mawari_c}]"
            cur2 = conn.cursor()
            cur2.execute(f"SELECT {sel} FROM [{tname}]")
            di = 4
            ci = 4 + (1 if dist_c else 0)
            si = ci + (1 if course_c else 0)
            mi = si + (1 if state_c else 0)
            for r in cur2.fetchall():
                race_id = _to_str(r[0])
                dist_val = _to_int(r[di], 1600) if dist_c and len(r) > di else 1600
                course_val = _to_str(r[ci]) if course_c and len(r) > ci else "2"
                state_val = r[si] if state_c and len(r) > si else None
                mawari_val = r[mi] if mawari_c and len(r) > mi else None
                info = {
                    "date": _to_str(r[1]),
                    "place": _to_str(r[2]),
                    "round": _to_str(r[3]),
                    "distance": dist_val,
                    "course": course_val,
                    "state": _state_from_baba(state_val) if state_val is not None else 1,
                    "rotation": _rotation_from_mawari(mawari_val) if mawari_val is not None else 0,
                }
                rk = build_race_key(info["date"], info["place"], _to_int(info["round"]))
                if rk:
                    info["race_key"] = rk
                result[race_id] = info
            cur2.close()
            conn.close()
            break
    except Exception:
        pass
    return result


def _load_master_pedigree(data_path: Path) -> dict[str, dict]:
    """Master.mdb から UmaCode -> {horse_name, sire_id, broodmare_sire_id} を取得。"""
    result: dict[str, dict] = {}
    master_mdb = data_path / "Master.mdb"
    if not master_mdb.exists():
        return result
    try:
        import pyodbc
        conn = pyodbc.connect(
            f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={master_mdb};"
        )
        cur = conn.cursor()
        for row in cur.tables(tableType="TABLE"):
            tname = getattr(row, "table_name", None) or row[2]
            cur2 = conn.cursor()
            cur2.execute(f"SELECT TOP 1 * FROM [{tname}]")
            cols = [d[0] for d in cur2.description] if cur2.description else []
            cur2.close()
            uma_c = _find_col(cols, "UmaCode", "競走馬コード", "馬コード")
            name_c = _find_col(cols, "馬名", "Bamei", "競走馬名")
            sire_c = _find_col(cols, "父繁殖登録番号", "父コード", "FuchichiCode")
            bms_c = _find_col(cols, "母父繁殖登録番号", "母父コード", "HahachichiCode")
            if not uma_c:
                continue
            sel = f"[{uma_c}]"
            idx = 1
            if name_c:
                sel += f",[{name_c}]"
                name_idx = idx
                idx += 1
            else:
                name_idx = None
            if sire_c:
                sel += f",[{sire_c}]"
                sire_idx = idx
                idx += 1
            else:
                sire_idx = None
            if bms_c:
                sel += f",[{bms_c}]"
                bms_idx = idx
            else:
                bms_idx = None
            cur2 = conn.cursor()
            cur2.execute(f"SELECT {sel} FROM [{tname}]")
            for r in cur2.fetchall():
                uma = _to_str(r[0])
                rec = {"horse_name": "", "sire_id": "0", "broodmare_sire_id": "0"}
                if name_idx is not None and len(r) > name_idx:
                    rec["horse_name"] = _to_str(r[name_idx])
                if sire_idx is not None and len(r) > sire_idx:
                    rec["sire_id"] = _to_str(r[sire_idx])
                if bms_idx is not None and len(r) > bms_idx:
                    rec["broodmare_sire_id"] = _to_str(r[bms_idx])
                result[uma] = rec
            cur2.close()
        conn.close()
    except Exception:
        pass
    return result


def extract_future_from_snap(snap_path: Path, output_csv: Path | None = None) -> list[dict]:
    """
    Snap.mdb から未来レース情報を抽出し、CSV に保存する。
    出走表馬群（RaceID+UmaCode）を Race.mdb / Master.mdb と結合して整形する。
    """
    if not snap_path.exists():
        print(f"警告: ファイルが見つかりません: {snap_path}", file=sys.stderr)
        return []
    try:
        import pyodbc
    except ImportError:
        raise RuntimeError("pyodbc がインストールされていません。pip install pyodbc")
    drivers = [x for x in pyodbc.drivers() if "Access" in x or "ACE" in x]
    if not drivers:
        raise RuntimeError("Microsoft Access 用 ODBC ドライバーが見つかりません")

    data_path = snap_path.parent
    conn = pyodbc.connect(f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={snap_path};")
    tables = _list_snap_tables(conn)
    target_table = None
    target_cols: list[str] = []
    for tname, cols in tables:
        if _find_col(cols, "RaceID") and _find_col(cols, "UmaCode"):
            target_table = tname
            target_cols = cols
            break
    if not target_table or not target_cols:
        conn.close()
        raise RuntimeError("Snap.mdb に RaceID と UmaCode を持つテーブルが見つかりません")

    race_id_c = _find_col(target_cols, "RaceID")
    uma_c = _find_col(target_cols, "UmaCode")
    date_c = _find_col(target_cols, "開催日", "日付")
    place_c = _find_col(target_cols, "競馬場名", "場名", "場", "競馬場", "JyoCD", "場コード", "競馬場コード")
    jyo_c = _find_col(target_cols, "JyoCD", "場コード", "競馬場コード")
    round_c = _find_col(target_cols, "レース番号", "登録レース番号", "R番")
    dist_c = _find_col(target_cols, "距離")
    course_c = _find_col(target_cols, "芝ダ", "芝ダ別", "TrackCD", "トラックコード")
    baba_c = _find_col(target_cols, "BabaCD", "馬場状態コード", "馬場状態", "馬場")
    mawari_c = _find_col(target_cols, "回り", "Mawari", "周り", "右左", "回りコード")
    horse_c = _find_col(target_cols, "馬名")
    umaban_c = _find_col(target_cols, "馬番")
    sire_c = _find_col(target_cols, "父繁殖登録番号", "父コード", "FuchichiCode")
    bms_c = _find_col(target_cols, "母父繁殖登録番号", "母父コード", "HahachichiCode")
    # オッズは未来データ（Snap.mdb）に含まれないため取得しない
    jockey_c = _find_col(target_cols, "騎手コード", "KishuCode", "jockey_id")
    waku_c = _find_col(target_cols, "枠番", "Wakuban", "waku")
    trainer_c = _find_col(target_cols, "ChokyosiCode", "調教師コード", "調教師Code")
    futan_c = _find_col(target_cols, "Futan", "斤量", "負担重量")

    race_info_map = _load_race_info_map(data_path)
    master_map = _load_master_pedigree(data_path)

    sel_cols = [race_id_c, uma_c]
    for c in [date_c, place_c, jyo_c, round_c, dist_c, course_c, baba_c, mawari_c, horse_c, umaban_c, sire_c, bms_c, jockey_c, waku_c, trainer_c, futan_c]:
        if c and c not in sel_cols:
            sel_cols.append(c)
    sel_str = ", ".join(f"[{c}]" for c in sel_cols)
    cur = conn.cursor()
    cur.execute(f"SELECT {sel_str} FROM [{target_table}]")
    raw_rows = cur.fetchall()
    conn.close()

    col_idx = {c: i for i, c in enumerate(sel_cols)}

    def _v(row, col: str | None, default=None):
        if not col or col not in col_idx:
            return default
        i = col_idx[col]
        return row[i] if i < len(row) else default

    rows = []
    for r in raw_rows:
        race_id = _to_str(_v(r, race_id_c))
        uma_code = _to_str(_v(r, uma_c))
        if not race_id or not uma_code:
            continue

        info = race_info_map.get(race_id, {})
        date_val = _v(r, date_c) or info.get("date")
        place_val = _v(r, place_c) if _v(r, place_c) is not None else info.get("place")
        jyo_val = _v(r, jyo_c) if jyo_c else place_val
        round_val = _v(r, round_c) if _v(r, round_c) is not None else info.get("round")
        distance = _to_int(_v(r, dist_c) or info.get("distance"), 1600)
        course_raw = _v(r, course_c) or info.get("course")
        baba_val = _v(r, baba_c)
        mawari_val = _v(r, mawari_c)
        state_val = _state_from_baba(baba_val) if (baba_val is not None and baba_val != "") else info.get("state", 1)
        rotation_val = _rotation_from_mawari(mawari_val) if (mawari_val is not None and mawari_val != "") else info.get("rotation", 0)
        horse_name = _v(r, horse_c)
        sire_id = _v(r, sire_c)
        bms_id = _v(r, bms_c)
        umaban = _v(r, umaban_c)
        jockey_id = _to_str(_v(r, jockey_c))
        waku_val = _v(r, waku_c)
        waku = _to_int(waku_val, 0) if waku_val is not None else 0
        if not (1 <= waku <= 8):
            waku = 0
        trainer_id = _to_str(_v(r, trainer_c)) if trainer_c else ""
        futan_val = _v(r, futan_c)
        weight_carry = _to_float(futan_val, WEIGHT_CARRY_DEFAULT) if futan_val is not None and futan_val != "" else WEIGHT_CARRY_DEFAULT

        master_rec = master_map.get(uma_code, {})
        if not horse_name:
            horse_name = master_rec.get("horse_name", "")
        if not sire_id:
            sire_id = master_rec.get("sire_id", "0")
        if not bms_id:
            bms_id = master_rec.get("broodmare_sire_id", "0")

        if info.get("race_key"):
            key = info["race_key"]
        elif date_val and place_val is not None and round_val is not None:
            key = build_race_key(str(date_val), str(place_val), _to_int(round_val))
        elif race_id and "_" in str(race_id) and len(str(race_id)) >= 10:
            key = str(race_id)
        else:
            continue
        if not key or key == "00000000_00_00":
            continue

        place_str = _to_str(place_val)
        jyo_str = _to_str(jyo_val or place_val)
        if place_str in JYO_NAME_MAP.values():
            racecourse_name = place_str
        elif jyo_str and (jyo_str in JYO_NAME_MAP or (jyo_str.isdigit() and len(jyo_str) <= 2)):
            racecourse_name = _jyo_to_name(jyo_str)
        else:
            racecourse_name = _jyo_to_name(jyo_str) if jyo_str else ""

        date_str = _to_str(date_val)
        race_num = _to_int(round_val, 0)
        rows.append({
            "race_key": key,
            "race_name": "未定",
            "date": date_str,
            "race_number": race_num,
            "horse_id": _to_str(uma_code),
            "horse_name": _to_str(horse_name),
            "umaban": _to_str(umaban),
            "sire_id": _to_str(sire_id),
            "broodmare_sire_id": _to_str(bms_id),
            "bms_id": _to_str(bms_id),
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "waku": waku,
            "wakuban": waku,
            "weight_carry": weight_carry,
            "distance": distance,
            "course_type": _course_type_from_track(course_raw),
            "racecourse": racecourse_name,
            "state": state_val,
            "rotation": rotation_val,
        })

    out_path = output_csv or OUTPUT_FUTURE
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["race_key", "race_name", "date", "race_number", "horse_id", "horse_name", "umaban", "sire_id", "broodmare_sire_id", "bms_id", "jockey_id", "trainer_id", "waku", "wakuban", "weight_carry", "distance", "course_type", "racecourse", "state", "rotation"],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(rows)
        print(f"Snap.mdb から {len(rows)} 件の出走データを抽出しました")

    return rows


# ========== 前走情報（履歴データ）==========
# 前走・動的特征の欠損時デフォルト（学習時と揃える）
DEFAULT_PREV_RANK = 99
DEFAULT_PREV_TIME_DIFF = 2.0
DEFAULT_INTERVAL = 999
N_PREV_RACES = 5


def load_prev_race_map(learning_csv: Path) -> dict[str, list[dict]]:
    """
    learning_dataset.csv を読み込み、各馬(horse_id)の「直近最大5走」をリストで保持する。
    戻り値: horse_id -> [ {rank, time_diff, date}, ... ] (日付降順で最大5件)
    """
    result: dict[str, list[dict]] = {}
    if not learning_csv.exists():
        return result
    try:
        with open(learning_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            has_time_diff = "time_diff" in cols
            rows_by_horse: dict[str, list[dict]] = {}
            for row in reader:
                hid = _to_str(row.get("horse_id", ""))
                if not hid:
                    continue
                date_str = _to_str(row.get("date", ""))
                rank = _to_int(row.get("rank"), DEFAULT_PREV_RANK)
                time_val = _to_float(row.get("time"), 0.0)
                if hid not in rows_by_horse:
                    rows_by_horse[hid] = []
                rows_by_horse[hid].append({
                    "date": date_str,
                    "rank": rank,
                    "time": time_val,
                    "time_diff": _to_float(row.get("time_diff"), 0.0) if has_time_diff else DEFAULT_PREV_TIME_DIFF,
                })
            for hid, races in rows_by_horse.items():
                races_sorted = sorted(races, key=lambda x: (x["date"], x.get("time", 0)), reverse=True)
                result[hid] = races_sorted[:N_PREV_RACES]
    except Exception:
        pass
    return result


def _parse_race_date(race_key: str) -> str:
    """race_key (YYYYMMDD_JJ_RR) から日付部分 YYYY-MM-DD を返す。"""
    parts = _to_str(race_key).split("_")
    if len(parts) < 1 or len(parts[0]) != 8:
        return ""
    d = parts[0]
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _days_between(date_str: str, race_date_str: str) -> int:
    """2つの日付文字列 (YYYY-MM-DD または YYYYMMDD) の間の日数を返す。"""
    def norm(s: str) -> str:
        s = re.sub(r"[^0-9]", "", _to_str(s))[:8]
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else ""
    a, b = norm(date_str), norm(race_date_str)
    if not a or not b:
        return DEFAULT_INTERVAL
    try:
        from datetime import datetime as dt
        d1 = dt.strptime(a, "%Y-%m-%d")
        d2 = dt.strptime(b, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except Exception:
        return DEFAULT_INTERVAL


def attach_prev_race_to_future(future: list[dict], prev_map: dict[str, list[dict]]) -> None:
    """future の各行に近5走情報 (prev_rank_1〜5, prev_time_diff_1〜5, avg_rank_5, avg_time_diff_5, interval) を付与。"""
    for r in future:
        hid = _to_str(r.get("horse_id", ""))
        race_date = _parse_race_date(r.get("race_key", ""))
        races = prev_map.get(hid, []) if hid else []
        # 1〜5走前を横展開。不足分はデフォルトで埋める
        for i in range(1, N_PREV_RACES + 1):
            idx = i - 1
            if idx < len(races):
                r[f"prev_rank_{i}"] = races[idx]["rank"]
                r[f"prev_time_diff_{i}"] = races[idx].get("time_diff", DEFAULT_PREV_TIME_DIFF)
            else:
                r[f"prev_rank_{i}"] = DEFAULT_PREV_RANK
                r[f"prev_time_diff_{i}"] = DEFAULT_PREV_TIME_DIFF
        # 前走からの間隔（最新1走の日付との差）
        if races:
            r["interval"] = _days_between(races[0]["date"], race_date)
        else:
            r["interval"] = DEFAULT_INTERVAL
        # 集約: 過去5走の平均（欠損99/2.0は除外。学習時と同様）
        rank_vals = [r.get(f"prev_rank_{i}", DEFAULT_PREV_RANK) for i in range(1, N_PREV_RACES + 1)]
        td_vals = [r.get(f"prev_time_diff_{i}", DEFAULT_PREV_TIME_DIFF) for i in range(1, N_PREV_RACES + 1)]
        valid_ranks = [x for x in rank_vals if x != DEFAULT_PREV_RANK]
        r["avg_rank_5"] = sum(valid_ranks) / len(valid_ranks) if valid_ranks else DEFAULT_PREV_RANK
        valid_td = [x for x in td_vals if x < DEFAULT_PREV_TIME_DIFF]
        r["avg_time_diff_5"] = sum(valid_td) / len(valid_td) if valid_td else DEFAULT_PREV_TIME_DIFF


# ========== 2. 前処理・マッピング ==========
def load_fallback_maps() -> tuple[dict[str, str], dict[str, str]]:
    """血統遡及用マッピングを読み込む。ファイルがなければ空辞書を返す。"""
    sire_gf_map: dict[str, str] = {}
    bms_father_map: dict[str, str] = {}
    p1 = CONFIG / "sire_to_grandfather.csv"
    if p1.exists():
        with open(p1, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = _to_str(row.get("sire_id"))
                v = _to_str(row.get("grandfather_sire_id"))
                if k and v:
                    sire_gf_map[k] = v
    p2 = CONFIG / "bms_to_father.csv"
    if p2.exists():
        with open(p2, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = _to_str(row.get("bms_id"))
                v = _to_str(row.get("bms_father_id"))
                if k and v:
                    bms_father_map[k] = v
    return sire_gf_map, bms_father_map


def load_id_map(path: Path, id_col: str, name_col: str) -> dict[str, str]:
    m = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = _to_str(row.get(id_col))
            v = _to_str(row.get(name_col))
            if k:
                m[k] = v if v else "Unknown"
    return m


# ========== 3. 血統スコア ==========
def load_master(path: Path) -> dict[str, dict]:
    master = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        name_col = "sire_name" if "sire_name" in (reader.fieldnames or []) else "bms_name"
        for row in reader:
            n = _to_str(row.get(name_col))
            if n:
                master[n] = {k: _to_float(v) for k, v in row.items() if k != name_col}
    return master


def get_stats_value(global_stats: dict, course: str, dist: str) -> float:
    """global_stats から place_rate を安全に取得。キーなし・0 の場合は DEFAULT_GLOBAL_MEAN を返す。"""
    v = global_stats.get(course, {}).get(dist, {}).get("place_rate")
    if v is None or (isinstance(v, (int, float)) and v <= 0):
        return DEFAULT_GLOBAL_MEAN
    return _to_float(v, DEFAULT_GLOBAL_MEAN)


def _get_score_from_master(name: str, course: str, dist: str, master: dict) -> float | None:
    """マスタからスコアを取得。なければ None。"""
    if not name or name not in master:
        return None
    place_col = f"{course}_{dist}_place"
    v = master[name].get(place_col)
    if v is not None and v > 0:
        return v
    return None


def get_sire_score(
    sire_name: str, sire_id: str, course: str, dist: str,
    sire_master: dict, sire_id_map: dict, sire_gf_map: dict, global_stats: dict,
) -> tuple[float, bool]:
    """父スコアを取得。Fallback対応。戻り値: (score, used_fallback)"""
    sire_id_str = _to_str(sire_id)
    sc = _get_score_from_master(sire_name, course, dist, sire_master)
    if sc is not None:
        return (sc, False)
    gf_id = sire_gf_map.get(sire_id_str) if sire_id_str else None
    if gf_id:
        gf_name = sire_id_map.get(gf_id, "Unknown")
        if gf_name and gf_name != "Unknown":
            sc = _get_score_from_master(gf_name, course, dist, sire_master)
            if sc is not None:
                print(f"[Fallback] Sire {sire_name} -> GrandSire {gf_name}")
                return (sc, True)
    return (get_stats_value(global_stats, course, dist), False)


def get_bms_score(
    bms_name: str, bms_id: str, course: str, dist: str,
    bms_master: dict, bms_id_map: dict, bms_father_map: dict, global_stats: dict,
) -> tuple[float, bool]:
    """母父スコアを取得。Fallback対応。戻り値: (score, used_fallback)"""
    bms_id_str = _to_str(bms_id)
    sc = _get_score_from_master(bms_name, course, dist, bms_master)
    if sc is not None:
        return (sc, False)
    father_id = bms_father_map.get(bms_id_str) if bms_id_str else None
    if father_id:
        father_name = bms_id_map.get(father_id, "Unknown")
        if father_name and father_name != "Unknown":
            sc = _get_score_from_master(father_name, course, dist, bms_master)
            if sc is not None:
                print(f"[Fallback] BMS {bms_name} -> BMS Father {father_name}")
                return (sc, True)
    return (get_stats_value(global_stats, course, dist), False)


def get_pedigree_score(name: str, course: str, dist: str, master: dict, global_stats: dict, fallback_name: str | None = None) -> float:
    """後方互換用。単純なスコア取得。"""
    sc = _get_score_from_master(name, course, dist, master)
    if sc is not None:
        return sc
    if fallback_name:
        sc = _get_score_from_master(fallback_name, course, dist, master)
        if sc is not None:
            return sc
    return get_stats_value(global_stats, course, dist)


# ========== 4. Logic Score ==========
def minmax_normalize(values: list[float], fallback_rank: list[int] | None = None) -> list[float]:
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        if fallback_rank:
            n = len(values)
            return [1.0 - (r / max(n - 1, 1)) for r in fallback_rank]
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ========== 5. AI予測 ==========
def _model_path_for_lgb(model_path: Path) -> str:
    """LightGBM は日本語パスで失敗することがあるため、一時パスにコピーして返す。"""
    p = str(model_path)
    if all(ord(c) < 128 for c in p):
        return p
    import shutil
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tmp = tf.name
    try:
        shutil.copy(model_path, tmp)
        return tmp
    except Exception:
        return p


def predict_lgb(features_list: list[dict], model_path: Path) -> list[float]:
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError as e:
        raise RuntimeError("lightgbm がインストールされていません。pip install lightgbm") from e
    path_str = _model_path_for_lgb(model_path)
    booster = lgb.Booster(model_file=path_str)
    feature_names = booster.feature_name() or []
    if "odds_tansho" in feature_names:
        print(
            "警告: モデルはオッズ(odds_tansho)で学習されています。"
            "未来予測ではオッズは未確定のため ODDS_DUMMY_VALUE を使用しています。"
            "オッズなしで再学習することを推奨します。（python train_model.py）",
            file=sys.stderr,
        )
    X = []
    for f in features_list:
        row = [float(f.get(n, 0)) for n in feature_names]
        X.append(row)
    pred = booster.predict(X)
    return pred.tolist() if hasattr(pred, "tolist") else list(pred)


# ========== 6. レポート出力 ==========
MARKS = ["◎", "○", "▲", "△", "⭐"]


def _race_key_to_label(race_key: str) -> str:
    """race_key (YYYYMMDD_JJ_RR) を 日付_競馬場名_レースNo 形式に変換。"""
    parts = _to_str(race_key).split("_")
    if len(parts) != 3:
        return race_key
    date_part, jj, rr = parts
    if len(date_part) == 8:
        date_fmt = f"{date_part[:4]}/{date_part[4:6]}/{date_part[6:8]}"
    else:
        date_fmt = date_part
    place_name = PLACE_CODE_MAP.get(jj, jj)
    race_no = str(_to_int(rr))
    return f"{date_fmt}_{place_name}_{race_no}R"


def _race_key_to_netkeiba_race_id(race_key: str) -> str:
    """race_key (YYYYMMDD_JJ_RR) を netkeiba の race_id 形式に変換。例: 20260215_05_01 → 202602150501"""
    parts = _to_str(race_key).split("_")
    if len(parts) != 3:
        return ""
    date_part, jj, rr = parts[0], parts[1], parts[2]
    if len(date_part) != 8 or not date_part.isdigit():
        return ""
    place = jj.zfill(2) if len(jj) <= 2 and (jj.isdigit() or jj in PLACE_CODE_MAP) else "00"
    if not place.isdigit():
        place = PLACE_CODE_MAP.get(jj, "00")
    r = str(_to_int(rr)).zfill(2)
    return f"{date_part}{place}{r}"


# 配信用 JSON / モバイルHTML の固定保存先（Headless 配信モデル）
DIST_DIR = SCRIPT_DIR / "docs"
DIST_WEEKLY_JSON = DIST_DIR / "weekly_prediction.json"
DIST_INDEX_HTML = DIST_DIR / "index.html"
MOBILE_HTML_TITLE_VERSION = "Ver 2.0"


def _write_predictions_json(report_rows: list[dict], _output_base_path: Path | None = None) -> dict:
    """予測結果を配信用 JSON 出力（docs/weekly_prediction.json）。オッズなし・AIスコアのみ。payload を返す。"""
    by_race = defaultdict(list)
    for r in report_rows:
        by_race[r["race_key"]].append(r)
    target_date = ""
    races_out = []
    for rk in sorted(by_race.keys()):
        items = sorted(by_race[rk], key=lambda x: -_to_float(x.get("final_score", 0)))
        if not target_date and len(rk.split("_")) >= 1:
            target_date = rk.split("_")[0]
        race_id_12 = _race_key_to_netkeiba_race_id(rk)
        label = _race_key_to_label(rk)
        race_name = f"{label.split('_')[1]}{_to_int(rk.split('_')[-1])}R" if len(rk.split("_")) >= 3 else rk
        predictions = []
        for r in items[:5]:
            predictions.append({
                "horse_num": _to_int(r.get("umaban", 0)),
                "horse_name": _to_str(r.get("horse_name", "")),
                "mark": _to_str(r.get("mark", "")),
                "score": round(_to_float(r.get("final_score", 0)) * 100),
            })
        races_out.append({
            "race_id": race_id_12,
            "race_name": race_name,
            "predictions": predictions,
        })
    netkeiba_url = ""
    if target_date:
        netkeiba_url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date}"
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target_date": target_date,
        "netkeiba_url": netkeiba_url,
        "races": races_out,
    }
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIST_WEEKLY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"JSON出力（配信用）: {DIST_WEEKLY_JSON}")
    return payload


def _write_mobile_html(report_rows: list[dict]) -> None:
    """インタラクティブ分析ダッシュボードを docs/index.html に出力。全頭データをJSON埋め込み。"""
    by_race = defaultdict(list)
    for r in report_rows:
        by_race[r["race_key"]].append(r)

    racing_data = []
    for rk in sorted(by_race.keys()):
        parts = rk.split("_")
        date_part = parts[0] if len(parts) >= 1 else ""
        date_label = f"{date_part[:4]}/{date_part[4:6]}/{date_part[6:8]}" if len(date_part) == 8 else date_part
        course_code = parts[1] if len(parts) >= 2 else ""
        course_name = PLACE_CODE_MAP.get(course_code, course_code)
        race_no = _to_int(parts[2], 0) if len(parts) >= 3 else 0
        race_name = f"{course_name}{race_no}R" if course_name else rk
        items = sorted(by_race[rk], key=lambda x: -_to_float(x.get("final_score", 0)))
        horses = []
        for r in items:
            horses.append({
                "mark": _to_str(r.get("mark", "")),
                "umaban": _to_int(r.get("umaban", 0)),
                "horse_name": _to_str(r.get("horse_name", "")),
                "score": round(_to_float(r.get("final_score", 0)) * 100),
            })
        racing_data.append({
            "date": date_part,
            "dateLabel": date_label,
            "course": course_name,
            "race_no": race_no,
            "race_name": race_name,
            "horses": horses,
        })

    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    target_date = racing_data[0]["date"] if racing_data else ""

    # </script> をエスケープしてJSON埋め込み
    json_str = json.dumps(racing_data, ensure_ascii=False)
    if "</script>" in json_str:
        json_str = json_str.replace("</script>", "<\\/script>")

    html = f"""<!DOCTYPE html>
<html lang="ja" data-bs-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI競馬予測 ({MOBILE_HTML_TITLE_VERSION}) - 分析ダッシュボード</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    :root {{ --accent-blue: #0d6efd; }}
    body {{ background: #0a0a0a; color: #e0e0e0; min-height: 100vh; }}
    .panel {{ background: #111; border: 1px solid #333; }}
    .text-primary {{ color: var(--accent-blue) !important; }}
    #filter-header {{ position: sticky; top: 0; z-index: 100; }}
    .table-horses {{ font-size: 0.95rem; }}
    .table-horses th {{ border-color: #333; white-space: nowrap; }}
    .table-horses td {{ border-color: #333; vertical-align: middle; }}
    .row-score-high {{ background: rgba(220, 53, 69, 0.2); }}
    .row-score-mid {{ background: rgba(253, 126, 20, 0.15); }}
    .row-score-low {{ background: rgba(13, 110, 253, 0.15); }}
    .sort-toggle {{ cursor: pointer; user-select: none; }}
    footer {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body class="d-flex flex-column min-vh-100">
  <header id="filter-header" class="panel border-bottom border-secondary py-3 mb-0">
    <div class="container">
      <h1 class="h5 mb-3 text-primary">AI競馬予測 ({MOBILE_HTML_TITLE_VERSION}) - 分析ダッシュボード</h1>
      <div class="row g-2">
        <div class="col-12 col-md-4">
          <label class="form-label small mb-0">開催日</label>
          <select id="sel-date" class="form-select form-select-sm bg-dark text-light border-secondary"></select>
        </div>
        <div class="col-12 col-md-4">
          <label class="form-label small mb-0">競馬場</label>
          <select id="sel-course" class="form-select form-select-sm bg-dark text-light border-secondary"></select>
        </div>
        <div class="col-12 col-md-4">
          <label class="form-label small mb-0">レース番号</label>
          <select id="sel-race" class="form-select form-select-sm bg-dark text-light border-secondary"></select>
        </div>
      </div>
      <div class="mt-2 d-flex align-items-center gap-2">
        <span class="small text-secondary">並び:</span>
        <span id="sort-score" class="sort-toggle badge bg-danger">スコア順</span>
        <span id="sort-umaban" class="sort-toggle badge bg-secondary">馬番順</span>
      </div>
    </div>
  </header>
  <main class="container flex-grow-1 py-3">
    <div id="race-title" class="h5 text-primary mb-2"></div>
    <div class="table-responsive">
      <table class="table table-dark table-horses table-striped">
        <thead>
          <tr>
            <th>印</th>
            <th>馬番</th>
            <th>馬名</th>
            <th>AIスコア</th>
          </tr>
        </thead>
        <tbody id="tbody-horses"></tbody>
      </table>
    </div>
  </main>
  <footer class="container py-3 border-top border-secondary">
    Updated at: {_escape_html(updated_at)}
  </footer>
  <script>
    const racingData = {json_str};
    let sortByScore = true;

    function getUniqueDates() {{
      const set = new Set(racingData.map(r => r.date));
      return Array.from(set).sort();
    }}
    function getCoursesByDate(date) {{
      const set = new Set(racingData.filter(r => r.date === date).map(r => r.course));
      return Array.from(set).sort((a, b) => (a || '').localeCompare(b || ''));
    }}
    function getRacesByDateCourse(date, course) {{
      return racingData.filter(r => r.date === date && r.course === course).map(r => ({{
        race_no: r.race_no,
        race_name: r.race_name,
        horses: r.horses
      }})).sort((a, b) => a.race_no - b.race_no);
    }}
    function getCurrentRace() {{
      const date = document.getElementById('sel-date').value;
      const course = document.getElementById('sel-course').value;
      const raceNo = document.getElementById('sel-race').value;
      const list = getRacesByDateCourse(date, course);
      return list.find(r => String(r.race_no) === raceNo) || list[0] || null;
    }}
    function rowClass(score) {{
      if (score >= 80) return 'row-score-high';
      if (score >= 60) return 'row-score-mid';
      return 'row-score-low';
    }}
    function renderTable() {{
      const race = getCurrentRace();
      const tbody = document.getElementById('tbody-horses');
      const titleEl = document.getElementById('race-title');
      if (!race) {{
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">レースを選択してください</td></tr>';
        titleEl.textContent = '';
        return;
      }}
      titleEl.textContent = race.race_name + ' — 全頭';
      let horses = race.horses.slice();
      if (!sortByScore) horses.sort((a, b) => (a.umaban || 0) - (b.umaban || 0));
      tbody.innerHTML = horses.map(h => {{
        const cls = rowClass(h.score);
        return `<tr class="${{cls}}"><td>${{escapeHtml(h.mark)}}</td><td>${{h.umaban}}</td><td>${{escapeHtml(h.horse_name)}}</td><td>${{h.score}}</td></tr>`;
      }}).join('');
    }}
    function escapeHtml(s) {{
      if (s == null) return '';
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }}
    function fillDates() {{
      const sel = document.getElementById('sel-date');
      const dates = getUniqueDates();
      sel.innerHTML = dates.map(d => {{
        const r = racingData.find(x => x.date === d);
        return `<option value="${{d}}">${{r ? r.dateLabel : d}}</option>`;
      }}).join('');
      if (dates.length) sel.value = dates[0];
    }}
    function fillCourses() {{
      const date = document.getElementById('sel-date').value;
      const sel = document.getElementById('sel-course');
      const courses = getCoursesByDate(date);
      sel.innerHTML = courses.map(c => `<option value="${{escapeHtml(c)}}">${{escapeHtml(c)}}</option>`).join('');
      if (courses.length) sel.value = courses[0];
    }}
    function fillRaces() {{
      const date = document.getElementById('sel-date').value;
      const course = document.getElementById('sel-course').value;
      const sel = document.getElementById('sel-race');
      const races = getRacesByDateCourse(date, course);
      sel.innerHTML = races.map(r => `<option value="${{r.race_no}}">${{r.race_no}}R</option>`).join('');
      if (races.length) sel.value = String(races[0].race_no);
    }}
    function onFilterChange() {{
      fillCourses();
      fillRaces();
      renderTable();
    }}
    document.getElementById('sel-date').addEventListener('change', onFilterChange);
    document.getElementById('sel-course').addEventListener('change', function() {{ fillRaces(); renderTable(); }});
    document.getElementById('sel-race').addEventListener('change', renderTable);
    document.getElementById('sort-score').addEventListener('click', function() {{ sortByScore = true; document.getElementById('sort-umaban').classList.remove('bg-primary'); document.getElementById('sort-umaban').classList.add('bg-secondary'); this.classList.add('bg-danger'); this.classList.remove('bg-secondary'); renderTable(); }});
    document.getElementById('sort-umaban').addEventListener('click', function() {{ sortByScore = false; document.getElementById('sort-score').classList.remove('bg-danger'); document.getElementById('sort-score').classList.add('bg-secondary'); this.classList.add('bg-primary'); this.classList.remove('bg-secondary'); renderTable(); }});
    fillDates();
    fillCourses();
    fillRaces();
    renderTable();
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIST_INDEX_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"モバイルHTML出力: {DIST_INDEX_HTML}")


def _escape_html(s: str) -> str:
    """HTML属性・本文用のエスケープ。"""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _append_prediction_log(report_rows: list[dict]) -> None:
    """予測ログを jv_data/history/prediction_log.csv に追記（MLOps・後日精度検証用）。"""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = PREDICTION_LOG_PATH.exists()
    rows = []
    for r in report_rows:
        date_val = r.get("date") or ""
        if hasattr(date_val, "strftime"):
            date_val = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
        else:
            date_val = str(date_val)[:10] if date_val else ""
        rows.append({
            "race_id": _to_str(r.get("race_key", "")),
            "date": date_val,
            "race_name": _to_str(r.get("race_name", "未定")),
            "horse_num": _to_str(r.get("umaban", "")),
            "horse_name": _to_str(r.get("horse_name", "")),
            "horse_id": _to_str(r.get("horse_id", "")),
            "ai_score": _to_float(r.get("ai_score", 0)),
            "logic_score": _to_float(r.get("logic_score", 0)),
            "final_score": _to_float(r.get("final_score", 0)),
            "rank_predict": _to_int(r.get("rank_predict", 0)),
            "mark": _to_str(r.get("mark", "")),
            "model_version": MODEL_VERSION,
        })
    with open(PREDICTION_LOG_PATH, "a" if file_exists else "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTION_LOG_COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerows(rows)
    print(f"予測ログ追記: {PREDICTION_LOG_PATH} ({len(rows)} 件)")


def build_html_report(report_rows: list[dict], output_path: Path) -> None:
    by_race = defaultdict(list)
    for r in report_rows:
        by_race[r["race_key"]].append(r)

    dates_set: set[str] = set()
    places_set: set[str] = set()
    races_set: set[str] = set()
    for rk in by_race:
        label = _race_key_to_label(rk)
        parts = label.split("_")
        if len(parts) >= 1:
            dates_set.add(parts[0])
        if len(parts) >= 2:
            places_set.add(parts[1])
        if len(parts) >= 3:
            races_set.add(parts[2])
    dates_sorted = sorted(dates_set)
    place_order = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    places_sorted = sorted(places_set, key=lambda x: (place_order.index(x) if x in place_order else 99, x))
    races_sorted = sorted(races_set, key=lambda x: _to_int(x.replace("R", "")))

    filter_html = '<div class="filters">'
    filter_html += '<label>競馬場: <select id="filter-place"><option value="" selected>全競馬場</option>'
    for p in places_sorted:
        filter_html += f'<option value="{p}">{p}</option>'
    filter_html += '</select></label> '
    filter_html += '<label>日付: <select id="filter-date"><option value="" selected>全日程</option>'
    for d in dates_sorted:
        filter_html += f'<option value="{d}">{d}</option>'
    filter_html += '</select></label> '
    filter_html += '<label>レースNo: <select id="filter-race"><option value="" selected>全レース</option>'
    for r in races_sorted:
        filter_html += f'<option value="{r}">{r}</option>'
    filter_html += '</select></label> '
    filter_html += '</div>'

    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>競馬予測レポート</title>
<style>
body{font-family:sans-serif;margin:8px;background:#f5f5f5}
.wrap{background:#fff;padding:12px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.filters{margin-bottom:12px;padding:8px;background:#fafafa;border-radius:4px}
.filters label{margin-right:16px}
.filters select{padding:4px 8px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:8px;text-align:left;border-bottom:1px solid #eee}
th{background:#fafafa;font-weight:600}
.mark{font-size:1.2em}
tr.hidden{display:none}
</style>
</head>
<body>
<h1>競馬予測レポート</h1>
<p>生成日時: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
<div class="wrap">
""" + filter_html + """
<table id="report">
<thead><tr>
<th>日付</th><th>競馬場</th><th>レースNo</th>
<th>印</th><th>馬番</th><th>馬名</th><th>父名</th><th>AI</th><th>Logic</th><th>Final</th>
</tr></thead>
<tbody>
"""
    for rk in sorted(by_race.keys()):
        items = by_race[rk]
        label = _race_key_to_label(rk)
        parts = label.split("_")
        date_display = parts[0] if len(parts) >= 1 else ""
        place_display = parts[1] if len(parts) >= 2 else ""
        race_display = parts[2] if len(parts) >= 3 else ""
        for i, r in enumerate(items):
            mark = r.get("mark", "")
            date_cell = date_display if i == 0 else ""
            place_cell = place_display if i == 0 else ""
            race_cell = race_display
            html += f"<tr data-date='{date_display}' data-place='{place_display}' data-race='{race_display}'><td>{date_cell}</td><td>{place_cell}</td><td>{race_cell}</td>"
            html += f"<td class='mark'>{mark}</td><td>{r.get('umaban','')}</td><td>{r.get('horse_name','')}</td><td>{r.get('sire_name','')}</td>"
            html += f"<td>{r.get('ai_score',0):.3f}</td><td>{r.get('logic_score',0):.3f}</td><td><b>{r.get('final_score',0):.3f}</b></td></tr>"
    html += """</tbody></table>
</div>
<script>
(function(){
  var body=document.querySelector('#report tbody');
  var rows=Array.from(body.querySelectorAll('tr'));
  var selPlace=document.getElementById('filter-place');
  var selDate=document.getElementById('filter-date');
  var selRace=document.getElementById('filter-race');
  function apply(){
    var place=(selPlace&&selPlace.value)||'';
    var date=(selDate&&selDate.value)||'';
    var race=(selRace&&selRace.value)||'';
    rows.forEach(function(r){
      var okPlace=!place||(r.dataset.place===place);
      var okDate=!date||(r.dataset.date===date);
      var okRace=!race||(r.dataset.race===race);
      r.classList.toggle('hidden',!(okPlace&&okDate&&okRace));
    });
  }
  if(selPlace)selPlace.addEventListener('change',apply);
  if(selDate)selDate.addEventListener('change',apply);
  if(selRace)selRace.addEventListener('change',apply);
  apply();
})();
</script>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    print("=== predict_pipeline.py 開始 ===")
    # 入力チェック
    if not SNAP_MDB.exists():
        alt = SCRIPT_DIR / "Snap.mdb"
        snap_path = alt if alt.exists() else SNAP_MDB
    else:
        snap_path = SNAP_MDB
    if not snap_path.exists():
        print(f"エラー: Snap.mdb が見つかりません: {snap_path}", file=sys.stderr)
        return 1
    for p, name in [
        (CONFIG / "sire_features_master.csv", "sire_features_master.csv"),
        (CONFIG / "bms_features_master.csv", "bms_features_master.csv"),
        (CONFIG / "global_stats.json", "global_stats.json"),
        (CONFIG / "sire_name_to_int.pkl", "sire_name_to_int.pkl"),
        (CONFIG / "bms_name_to_int.pkl", "bms_name_to_int.pkl"),
        (CONFIG / "jockey_name_to_int.pkl", "jockey_name_to_int.pkl"),
        (CONFIG / "trainer_name_to_int.pkl", "trainer_name_to_int.pkl"),
        (CONFIG / "racecourse_name_to_int.pkl", "racecourse_name_to_int.pkl"),
        (CONFIG / "interaction_dict.pkl", "interaction_dict.pkl"),
        (CONFIG / "sire_id_map.csv", "sire_id_map.csv"),
        (CONFIG / "bms_id_map.csv", "bms_id_map.csv"),
        (CONFIG / "jockey_id_map.csv", "jockey_id_map.csv"),
    ]:
        if not p.exists():
            print(f"エラー: {name} が見つかりません: {p}", file=sys.stderr)
            return 1
    model_exists = MODEL_PATH.exists()
    if not model_exists:
        print("警告: モデルファイルが見つかりません。Logic Score のみでレポートを出力します。")
    else:
        print(f"使用モデル: {MODEL_PATH}")
    print(f"血統ロジック重み: AI={WEIGHT_AI_SCORE}, Logic={WEIGHT_LOGIC_SCORE}")

    # 1. 未来データ抽出
    print("未来データ抽出中...")
    try:
        future = extract_future_from_snap(snap_path)
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    print(f"未来データ抽出完了: {len(future)} 件")
    # 更新日時より未来のレースのみを対象とする
    today_str = datetime.now().strftime("%Y%m%d")
    if len(future) > 0:
        def _is_future_race(r: dict) -> bool:
            rk = _to_str(r.get("race_key", ""))
            parts = rk.split("_")
            if len(parts) < 1:
                return False
            date_part = parts[0]
            return len(date_part) == 8 and date_part >= today_str
        future = [r for r in future if _is_future_race(r)]
        race_count = len(set(r["race_key"] for r in future))
        print(f"予測対象: 未来レース {race_count} レース {len(future)} 件")

    # 前走情報のマッピング（履歴データから）
    learning_csv = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
    prev_map = load_prev_race_map(learning_csv)
    attach_prev_race_to_future(future, prev_map)
    print(f"前走辞書: {len(prev_map)} 頭分の直近成績を読み込みました")

    OUTPUT_FUTURE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FUTURE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["race_key", "race_name", "date", "race_number", "horse_id", "horse_name", "umaban", "sire_id", "broodmare_sire_id", "bms_id", "jockey_id", "trainer_id", "waku", "wakuban", "weight_carry", "distance", "course_type", "racecourse", "state", "rotation"], extrasaction="ignore")
        w.writeheader()
        w.writerows(future)

    # 2. 前処理
    print("前処理・マッピング中...")
    sire_id_map = load_id_map(CONFIG / "sire_id_map.csv", "sire_id", "sire_name")
    bms_id_map = load_id_map(CONFIG / "bms_id_map.csv", "bms_id", "bms_name")
    jockey_id_map = load_id_map(CONFIG / "jockey_id_map.csv", "jockey_id", "jockey_name")
    trainer_id_map_path = CONFIG / "trainer_id_map.csv"
    trainer_map = load_id_map(trainer_id_map_path, "trainer_id", "trainer_name") if trainer_id_map_path.exists() else {}
    with open(CONFIG / "sire_name_to_int.pkl", "rb") as f:
        sire_name_to_int = pickle.load(f)
    with open(CONFIG / "bms_name_to_int.pkl", "rb") as f:
        bms_name_to_int = pickle.load(f)
    with open(CONFIG / "jockey_name_to_int.pkl", "rb") as f:
        jockey_name_to_int = pickle.load(f)
    with open(CONFIG / "trainer_name_to_int.pkl", "rb") as f:
        trainer_n2i = pickle.load(f)
    with open(CONFIG / "racecourse_name_to_int.pkl", "rb") as f:
        racecourse_n2i = pickle.load(f)
    with open(CONFIG / "interaction_dict.pkl", "rb") as f:
        interaction_dict = pickle.load(f)
    sire_state_dict = interaction_dict.get("sire_state", {})
    course_wakuban_dict = interaction_dict.get("course_wakuban", {})
    sire_dist_dict = interaction_dict.get("sire_dist", {})
    sire_type_dict = interaction_dict.get("sire_type", {})
    bms_dist_dict = interaction_dict.get("bms_dist", {})
    bms_type_dict = interaction_dict.get("bms_type", {})
    bms_state_dict = interaction_dict.get("bms_state", {})
    jockey_course_dict = interaction_dict.get("jockey_course", {})
    sire_master = load_master(CONFIG / "sire_features_master.csv")
    bms_master = load_master(CONFIG / "bms_features_master.csv")
    with open(CONFIG / "global_stats.json", encoding="utf-8") as f:
        global_stats = json.load(f)
    sire_gf_map, bms_father_map = load_fallback_maps()

    fallback_count = 0
    for r in future:
        r["sire_name"] = sire_id_map.get(_to_str(r["sire_id"]), "Unknown")
        r["bms_name"] = bms_id_map.get(_to_str(r["broodmare_sire_id"]), "Unknown")
        jockey_id = _to_str(r.get("jockey_id", ""))
        trainer_id = _to_str(r.get("trainer_id", ""))
        jockey_name = jockey_id_map.get(jockey_id, "Unknown")
        trainer_name = trainer_map.get(trainer_id, trainer_id) if trainer_id else "Unknown"
        if not trainer_name:
            trainer_name = "Unknown"
        r["sire_id_int"] = sire_name_to_int.get(r["sire_name"], 0)
        r["bms_id_int"] = bms_name_to_int.get(r["bms_name"], 0)
        r["jockey_id_int"] = jockey_name_to_int.get(jockey_name, 0)
        r["trainer_id_int"] = trainer_n2i.get(trainer_name, 0)
        r["wakuban"] = _to_int(r.get("waku", 0), 0)
        if not (1 <= r["wakuban"] <= 8):
            r["wakuban"] = 0
        r["weight_carry"] = _to_float(r.get("weight_carry", WEIGHT_CARRY_DEFAULT), WEIGHT_CARRY_DEFAULT)
        dist_cat = classify_distance(r["distance"])
        course_cat = classify_course(r["course_type"])
        sire_score, sire_fb = get_sire_score(
            r["sire_name"], _to_str(r["sire_id"]), course_cat, dist_cat,
            sire_master, sire_id_map, sire_gf_map, global_stats,
        )
        bms_score, bms_fb = get_bms_score(
            r["bms_name"], _to_str(r["broodmare_sire_id"]), course_cat, dist_cat,
            bms_master, bms_id_map, bms_father_map, global_stats,
        )
        if sire_fb or bms_fb:
            fallback_count += 1
        r["pedigree_score"] = WEIGHT_SIRE * sire_score + WEIGHT_BMS * bms_score
        r["_dist_cat"] = dist_cat
        r["_course_cat"] = course_cat

    print(f"祖父/父フォールバック適用数: {fallback_count} 件")

    # Logic Score (レース内正規化)
    by_race = defaultdict(list)
    for r in future:
        by_race[r["race_key"]].append(r)
    for rk, items in by_race.items():
        vals = [x["pedigree_score"] for x in items]
        normed = minmax_normalize(vals)
        for i, x in enumerate(items):
            x["logic_score"] = normed[i] if i < len(normed) else 0.5

    # 5. AI予測（学習時 FEATURE_COLS と完全一致。Logic + 近5走特徴含む）
    print("AI予測実行中...")
    # 予測直前: 過去履歴が横展開されていることを確認
    if future:
        try:
            import pandas as pd
            future_races = pd.DataFrame(future)
            print("=== future_races check (過去履歴・logic_score) ===")
            display_cols = ["horse_name", "prev_rank_1", "prev_rank_2", "prev_rank_3", "interval", "logic_score"]
            available = [c for c in display_cols if c in future_races.columns]
            if available:
                print(future_races[available].head())
            else:
                print(future_races.head())
        except Exception as ex:
            print(f"future_races 検証表示スキップ: {ex}")
    features_list = []
    for r in future:
        racecourse_name = _to_str(r.get("racecourse", ""))
        racecourse_id_int = racecourse_n2i.get(racecourse_name, 0)
        state_val = _to_int(r.get("state"), 1)
        waku_val = r["wakuban"]
        dist_cat = get_dist_cat(_to_int(r.get("distance"), 1600))
        course_type_val = 1 if r.get("course_type") == "1" else 2
        sire_state_int = sire_state_dict.get(f"{r['sire_id_int']}_{state_val}", 0)
        course_wakuban_int = course_wakuban_dict.get(f"{racecourse_id_int}_{waku_val}", 0)
        sire_dist_int = sire_dist_dict.get(f"{r['sire_id_int']}_{dist_cat}", 0)
        sire_type_int = sire_type_dict.get(f"{r['sire_id_int']}_{course_type_val}", 0)
        bms_dist_int = bms_dist_dict.get(f"{r['bms_id_int']}_{dist_cat}", 0)
        bms_type_int = bms_type_dict.get(f"{r['bms_id_int']}_{course_type_val}", 0)
        bms_state_int = bms_state_dict.get(f"{r['bms_id_int']}_{state_val}", 0)
        jockey_course_int = jockey_course_dict.get(f"{r['jockey_id_int']}_{racecourse_id_int}", 0)
        feat = {
            "sire_id_int": r["sire_id_int"],
            "bms_id_int": r["bms_id_int"],
            "jockey_id_int": r["jockey_id_int"],
            "trainer_id_int": r["trainer_id_int"],
            "racecourse_id_int": racecourse_id_int,
            "course_type": course_type_val,
            "state": state_val,
            "rotation": _to_int(r.get("rotation"), 0),
            "wakuban": waku_val,
            "weight_carry": r["weight_carry"],
            "distance": r["distance"],
            "sire_state_int": sire_state_int,
            "course_wakuban_int": course_wakuban_int,
            "sire_dist_int": sire_dist_int,
            "sire_type_int": sire_type_int,
            "bms_dist_int": bms_dist_int,
            "bms_type_int": bms_type_int,
            "bms_state_int": bms_state_int,
            "jockey_course_int": jockey_course_int,
            "logic_score": _to_float(r.get("logic_score"), 0.5),
        }
        for i in range(1, N_PREV_RACES + 1):
            feat[f"prev_rank_{i}"] = _to_int(r.get(f"prev_rank_{i}"), DEFAULT_PREV_RANK)
            feat[f"prev_time_diff_{i}"] = _to_float(r.get(f"prev_time_diff_{i}"), DEFAULT_PREV_TIME_DIFF)
        feat["avg_rank_5"] = _to_float(r.get("avg_rank_5"), DEFAULT_PREV_RANK)
        feat["avg_time_diff_5"] = _to_float(r.get("avg_time_diff_5"), DEFAULT_PREV_TIME_DIFF)
        interval_val = _to_int(r.get("interval"), DEFAULT_INTERVAL)
        feat["interval"] = interval_val
        feat["recency"] = 1.0 / (1.0 + interval_val / 60.0)  # 改善点5: 前走の鮮度
        features_list.append(feat)

    if features_list and model_exists:
        try:
            import pandas as pd
            X_debug = pd.DataFrame(features_list)[FEATURE_COLS]
            print("予測入力特徴量 (先頭5件):")
            print(X_debug.head())
        except Exception as ex:
            print(f"デバッグ表示スキップ: {ex}")

    if model_exists:
        try:
            raw_preds = predict_lgb(features_list, MODEL_PATH)
        except Exception as e:
            print(f"警告: モデル予測失敗 - {e}。Logic Score のみで出力します。", file=sys.stderr)
            raw_preds = [r["logic_score"] for r in future]
    else:
        raw_preds = [r["logic_score"] for r in future]

    for i, r in enumerate(future):
        r["raw_pred"] = raw_preds[i] if i < len(raw_preds) else r["logic_score"]

    for rk, items in by_race.items():
        vals = [x["raw_pred"] for x in items]
        ranks = sorted(range(len(items)), key=lambda i: -items[i]["logic_score"])
        normed = minmax_normalize(vals, ranks)
        for i, x in enumerate(items):
            x["ai_score"] = normed[i] if i < len(normed) else 0.5
        # 改善点2: 3着内的のため AI と Logic のバランスで final_score を算出
        logic_vals = [x["logic_score"] for x in items]
        logic_ranks = sorted(range(len(items)), key=lambda i: -items[i]["logic_score"])
        logic_normed = minmax_normalize(logic_vals, logic_ranks)
        for i, x in enumerate(items):
            x["_logic_normed"] = logic_normed[i] if i < len(logic_normed) else 0.5
            x["final_score"] = WEIGHT_AI_SCORE * x["ai_score"] + WEIGHT_LOGIC_SCORE * x["_logic_normed"]
            del x["_logic_normed"]

    # 6. 最終判定（改善点1・2: 単勝精度・3着内的のため AI と Logic を WEIGHT_* でブレンド）
    for r in future:
        if "final_score" not in r:
            r["final_score"] = WEIGHT_AI_SCORE * r.get("ai_score", r.get("raw_pred", 0.5)) + WEIGHT_LOGIC_SCORE * r.get("logic_score", 0.5)

    report_rows = []
    for rk in sorted(by_race.keys()):
        items = sorted(by_race[rk], key=lambda x: -x["final_score"])
        for rank, r in enumerate(items):
            r["mark"] = MARKS[rank] if rank < len(MARKS) else ""
            r["rank_predict"] = rank + 1
            report_rows.append(r)

    out_path = SCRIPT_DIR / "jv_data" / "reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    build_html_report(report_rows, out_path)
    # JSON 出力（配信用・docs/weekly_prediction.json）+ モバイル用HTML（docs/index.html）
    payload = _write_predictions_json(report_rows)
    _write_mobile_html(report_rows)
    # MLOps: 予測ログを追記保存（後日の精度検証用）
    _append_prediction_log(report_rows)
    print(f"予測完了")
    print(f"レポート出力: {out_path}")
    print("=== predict_pipeline.py 正常終了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
