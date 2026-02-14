# -*- coding: utf-8 -*-
"""
学習データ (learning_dataset.csv) を基に、予測モデル用の特徴量マスタ・ID辞書・血統遡及マップを一括生成する。

Inputs:
  - jv_data/learning_dataset.csv (sire_id, broodmare_sire_id, jockey_id, trainer_id, wakuban, weight_carry, rank, distance, course_type)
  - config/sire_id_map.csv, config/bms_id_map.csv, config/jockey_id_map.csv, config/trainer_id_map.csv

Outputs:
  - config/sire_features_master.csv, config/bms_features_master.csv
  - config/global_stats.json
  - config/sire_name_to_int.pkl, config/bms_name_to_int.pkl, config/jockey_name_to_int.pkl, config/trainer_name_to_int.pkl
  - config/racecourse_name_to_int.pkl
  - config/sire_to_grandfather.csv, config/bms_to_father.csv

使い方:
  python preprocess_master.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# パス設定
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_LEARNING = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
INPUT_SIRE_MAP = SCRIPT_DIR / "config" / "sire_id_map.csv"
INPUT_BMS_MAP = SCRIPT_DIR / "config" / "bms_id_map.csv"
INPUT_JOCKEY_MAP = SCRIPT_DIR / "config" / "jockey_id_map.csv"
INPUT_TRAINER_MAP = SCRIPT_DIR / "config" / "trainer_id_map.csv"
CONFIG_DIR = SCRIPT_DIR / "config"
OUTPUT_SIRE_FEATURES = CONFIG_DIR / "sire_features_master.csv"
OUTPUT_BMS_FEATURES = CONFIG_DIR / "bms_features_master.csv"
OUTPUT_GLOBAL_STATS = CONFIG_DIR / "global_stats.json"
OUTPUT_SIRE_DICT = CONFIG_DIR / "sire_name_to_int.pkl"
OUTPUT_BMS_DICT = CONFIG_DIR / "bms_name_to_int.pkl"
OUTPUT_JOCKEY_DICT = CONFIG_DIR / "jockey_name_to_int.pkl"
OUTPUT_TRAINER_DICT = CONFIG_DIR / "trainer_name_to_int.pkl"
OUTPUT_RACECOURSE_DICT = CONFIG_DIR / "racecourse_name_to_int.pkl"
OUTPUT_SIRE_GF = CONFIG_DIR / "sire_to_grandfather.csv"
OUTPUT_BMS_FATHER = CONFIG_DIR / "bms_to_father.csv"

# カテゴリ定義
COURSE_TURF = "1"
COURSE_DIRT = "2"
COURSES = ["turf", "dirt"]

# 距離区分 (m): Short (<1400), Mile (1400-1800), Mid (1800-2400), Long (>2400)
DIST_SHORT = "short"
DIST_MILE = "mile"
DIST_MID = "mid"
DIST_LONG = "long"
DISTANCES = [DIST_SHORT, DIST_MILE, DIST_MID, DIST_LONG]

# 出走回数しきい値（10回未満は全体平均で補完）
MIN_SAMPLES = 10


def _to_int(v, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def _to_str(v, default: str = "") -> str:
    if v is None or v == "":
        return default
    s = str(v).strip()
    return s if s else default


def load_id_map(path: Path, id_col: str, name_col: str) -> dict[str, str]:
    """ID→名前マップを読み込む。"""
    result = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = _to_str(row.get(id_col))
            v = _to_str(row.get(name_col))
            if k:
                result[k] = v if v else "Unknown"
    return result


def classify_distance(meters: int) -> str | None:
    """距離を区分に分類する。"""
    if meters <= 0:
        return None
    if meters < 1400:
        return DIST_SHORT
    if meters <= 1800:
        return DIST_MILE
    if meters <= 2400:
        return DIST_MID
    return DIST_LONG


def classify_course(course_type: str) -> str | None:
    """コース区分。1=芝(turf), 2=ダート(dirt)。"""
    c = _to_str(course_type)
    if c == "1":
        return "turf"
    if c == "2":
        return "dirt"
    return None


def main() -> int:
    # --- 1. 入力ファイル存在確認 ---
    required = [
        (INPUT_LEARNING, "learning_dataset.csv"),
        (INPUT_SIRE_MAP, "sire_id_map.csv"),
        (INPUT_BMS_MAP, "bms_id_map.csv"),
        (INPUT_JOCKEY_MAP, "jockey_id_map.csv"),
    ]
    for p, name in required:
        if not p.exists():
            print(f"エラー: {name} が見つかりません: {p}", file=sys.stderr)
            return 1

    print("=== preprocess_master.py 開始 ===")
    print("データ準備: 入力ファイル読み込み中...")
    sire_map = load_id_map(INPUT_SIRE_MAP, "sire_id", "sire_name")
    bms_map = load_id_map(INPUT_BMS_MAP, "bms_id", "bms_name")
    jockey_map = load_id_map(INPUT_JOCKEY_MAP, "jockey_id", "jockey_name")
    if INPUT_TRAINER_MAP.exists():
        trainer_map = load_id_map(INPUT_TRAINER_MAP, "trainer_id", "trainer_name")
    else:
        trainer_map = {}
        print(f"   警告: trainer_id_map.csv が見つかりません。調教師はすべて Unknown として扱います。", file=sys.stderr)
    print(f"   sire_id_map:   {len(sire_map)} 件")
    print(f"   bms_id_map:    {len(bms_map)} 件")
    print(f"   jockey_id_map: {len(jockey_map)} 件")
    print(f"   trainer_id_map: {len(trainer_map)} 件")

    # --- 2. データ準備・クリーニング ---
    # データ検証: trainer_id / weight_carry の存在確認（後方互換）
    with open(INPUT_LEARNING, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
    has_trainer_id = "trainer_id" in header
    has_weight_carry = "weight_carry" in header
    if not has_trainer_id:
        print("   警告: learning_dataset.csv に trainer_id カラムがありません。調教師は Unknown として扱います。", file=sys.stderr)
    if not has_weight_carry:
        print("   警告: learning_dataset.csv に weight_carry カラムがありません。", file=sys.stderr)

    rows = []
    racecourse_names: set[str] = set()
    child_to_father: dict[str, str] = {}  # horse_id -> sire_id (父)
    has_racecourse = "racecourse" in (header or [])
    with open(INPUT_LEARNING, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sire_id = _to_str(row.get("sire_id"))
            bms_id = _to_str(row.get("broodmare_sire_id"))
            jockey_id = _to_str(row.get("jockey_id"))
            trainer_id = _to_str(row.get("trainer_id")) if has_trainer_id else ""
            horse_id = _to_str(row.get("horse_id"))
            rc = _to_str(row.get("racecourse")) if has_racecourse else ""
            if rc:
                racecourse_names.add(rc)
            course = classify_course(row.get("course_type"))
            dist_m = _to_int(row.get("distance"), 0)
            dist_cat = classify_distance(dist_m)
            rank = _to_int(row.get("rank"), 0)
            if horse_id and sire_id and horse_id != "0" and sire_id != "0":
                child_to_father[horse_id] = sire_id
            if course is None or dist_cat is None:
                continue
            trainer_name = trainer_map.get(trainer_id, "Unknown") if trainer_id else "Unknown"
            rows.append({
                "sire_name": sire_map.get(sire_id, "Unknown"),
                "bms_name": bms_map.get(bms_id, "Unknown"),
                "jockey_name": jockey_map.get(jockey_id, "Unknown"),
                "trainer_name": trainer_name,
                "course": course,
                "distance": dist_cat,
                "rank": rank,
                "is_win": 1 if rank == 1 else 0,
                "is_place": 1 if 1 <= rank <= 3 else 0,
            })
    print(f"   有効レコード: {len(rows)} 件（Course/Distance 除外後）")

    # --- 3. 全体平均の算出 (Global Mean) ---
    global_stats = {}
    for c in COURSES:
        global_stats[c] = {}
        for d in DISTANCES:
            sub = [r for r in rows if r["course"] == c and r["distance"] == d]
            n = len(sub)
            if n == 0:
                global_stats[c][d] = {"win_rate": 0.0, "place_rate": 0.0, "count": 0}
            else:
                win = sum(r["is_win"] for r in sub)
                place = sum(r["is_place"] for r in sub)
                global_stats[c][d] = {
                    "win_rate": win / n,
                    "place_rate": place / n,
                    "count": n,
                }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_GLOBAL_STATS, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)
    print("全体平均算出完了: config/global_stats.json に保存")

    # --- 4. 成績マスタの集計 (Aggregation) ---
    def aggregate(rows_list: list, name_key: str) -> dict[tuple[str, str, str], dict]:
        """Name x Course x Distance ごとに Win Rate, Place Rate を集計。"""
        counts = defaultdict(lambda: {"runs": 0, "wins": 0, "places": 0})
        for r in rows_list:
            key = (r[name_key], r["course"], r["distance"])
            counts[key]["runs"] += 1
            counts[key]["wins"] += r["is_win"]
            counts[key]["places"] += r["is_place"]
        return dict(counts)

    sire_agg = aggregate(rows, "sire_name")
    bms_agg = aggregate(rows, "bms_name")
    print("マスタ集計完了: sire / bms ごとに Course x Distance で集計")

    # --- 5. マスタCSV作成（10回未満は全体平均で補完）---
    feature_cols = []
    for c in COURSES:
        for d in DISTANCES:
            feature_cols.append(f"{c}_{d}_win")
            feature_cols.append(f"{c}_{d}_place")

    def build_master(agg: dict, name_key: str, output_path: Path) -> None:
        names = sorted(set(k[0] for k in agg))
        header = [name_key] + feature_cols
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for name in names:
                row = {name_key: name}
                for c in COURSES:
                    for d in DISTANCES:
                        key = (name, c, d)
                        rec = agg.get(key, {"runs": 0, "wins": 0, "places": 0})
                        g = global_stats[c][d]
                        if rec["runs"] >= MIN_SAMPLES:
                            wr = rec["wins"] / rec["runs"]
                            pr = rec["places"] / rec["runs"]
                        else:
                            wr = g["win_rate"]
                            pr = g["place_rate"]
                        row[f"{c}_{d}_win"] = round(wr, 4)
                        row[f"{c}_{d}_place"] = round(pr, 4)
                w.writerow(row)

    build_master(sire_agg, "sire_name", OUTPUT_SIRE_FEATURES)
    build_master(bms_agg, "bms_name", OUTPUT_BMS_FEATURES)
    print("マスタCSV保存完了: sire_features_master.csv, bms_features_master.csv")

    # --- 6. 遡及マップの生成 (Grandfather Logic) ---
    # 馬(horse_id) -> 父(sire_id) のペアを抽出。
    # sire_to_grandfather: 父として使われる horse_id -> その父(grandfather_sire_id)。ルックアップ時 sire_id で検索。
    # bms_to_father: 母父として使われる horse_id -> その父(bms_father_id)。ルックアップ時 bms_id で検索。
    pairs = sorted(child_to_father.items())

    with open(OUTPUT_SIRE_GF, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sire_id", "grandfather_sire_id"])
        w.writerows(pairs)

    with open(OUTPUT_BMS_FATHER, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bms_id", "bms_father_id"])
        w.writerows(pairs)

    print(f"家系図生成完了: sire_to_grandfather.csv ({len(pairs)} 件)")
    print(f"家系図生成完了: bms_to_father.csv ({len(pairs)} 件)")

    # --- 7. ID辞書の作成 (Unknown=0) ---
    def build_name_to_int(names: set[str]) -> dict[str, int]:
        sorted_names = sorted(names)
        if "Unknown" not in sorted_names:
            sorted_names = ["Unknown"] + sorted_names
        else:
            sorted_names = ["Unknown"] + [n for n in sorted_names if n != "Unknown"]
        return {n: i for i, n in enumerate(sorted_names)}

    sire_names = set(r["sire_name"] for r in rows)
    bms_names = set(r["bms_name"] for r in rows)
    jockey_names = set(r["jockey_name"] for r in rows)
    trainer_names = set(r["trainer_name"] for r in rows)

    sire_name_to_int = build_name_to_int(sire_names)
    bms_name_to_int = build_name_to_int(bms_names)
    jockey_name_to_int = build_name_to_int(jockey_names)
    trainer_name_to_int = build_name_to_int(trainer_names)

    with open(OUTPUT_SIRE_DICT, "wb") as f:
        pickle.dump(sire_name_to_int, f)
    with open(OUTPUT_BMS_DICT, "wb") as f:
        pickle.dump(bms_name_to_int, f)
    with open(OUTPUT_JOCKEY_DICT, "wb") as f:
        pickle.dump(jockey_name_to_int, f)
    with open(OUTPUT_TRAINER_DICT, "wb") as f:
        pickle.dump(trainer_name_to_int, f)

    # --- 競馬場辞書の作成 ---
    sorted_racecourses = sorted(racecourse_names)
    racecourse_name_to_int = {name: i + 1 for i, name in enumerate(sorted_racecourses)}
    racecourse_name_to_int[""] = 0
    if "Unknown" not in racecourse_name_to_int:
        racecourse_name_to_int["Unknown"] = 0
    with open(OUTPUT_RACECOURSE_DICT, "wb") as f:
        pickle.dump(racecourse_name_to_int, f)
    n_rc = len(racecourse_names)
    print(f"競馬場辞書: {n_rc}件 保存完了 (config/racecourse_name_to_int.pkl)")

    print(f"辞書保存完了: sire_name_to_int.pkl ({len(sire_name_to_int)} 件)")
    print(f"辞書保存完了: bms_name_to_int.pkl ({len(bms_name_to_int)} 件)")
    print(f"辞書保存完了: jockey_name_to_int.pkl ({len(jockey_name_to_int)} 件)")
    print(f"調教師辞書: {len(trainer_name_to_int)} 件 保存完了")
    print("preprocess_master.py 正常終了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
