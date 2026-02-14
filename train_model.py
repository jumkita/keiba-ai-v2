# -*- coding: utf-8 -*-
"""
学習データから LightGBM による競馬予測モデルを学習し、lgb_target_rank_1to3.txt を生成する。

Target: rank <= 3 (複勝) を 1、それ以外を 0 の二値分類。

Inputs:
  - jv_data/learning_dataset.csv
  - config/sire_name_to_int.pkl, config/bms_name_to_int.pkl, config/jockey_name_to_int.pkl, config/trainer_name_to_int.pkl
  - config/racecourse_name_to_int.pkl
  - config/sire_id_map.csv, config/bms_id_map.csv, config/jockey_id_map.csv, config/trainer_id_map.csv (ID→名前変換用)

Outputs:
  - jv_data/models/lgb_target_rank_1to3.txt
  - config/interaction_dict.pkl (sire_state, course_wakuban の辞書)

使い方:
  python train_model.py
"""
from __future__ import annotations

import csv
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_LEARNING = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
CONFIG = SCRIPT_DIR / "config"
OUTPUT_MODEL = SCRIPT_DIR / "jv_data" / "models" / "lgb_target_rank_1to3.txt"

# 特徴量リスト（odds_tansho, rank, time, date は絶対に含めない）
# Stacking: Logic Score + 近5走動的特征を入力に含める
# 改善点3: 複勝安定＝weight_carry(斤量), state(馬場), prev_rank/avg_rank_5 を活用
# 改善点6: 父・母父適性＝sire_dist_int, bms_dist_int 等の交互項を学習で確認
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
    "sire_dist_int",     # 改善点6: sire × 距離カテゴリ
    "sire_type_int",     # sire × 芝ダート
    "bms_dist_int",      # 改善点6: bms × 距離カテゴリ
    "bms_type_int",     # bms × 芝ダート
    "bms_state_int",    # bms × 馬場状態
    "jockey_course_int", # 騎手 × 競馬場
    "logic_score",       # 血統ロジック（レース内正規化）
    "prev_rank_1", "prev_rank_2", "prev_rank_3", "prev_rank_4", "prev_rank_5",
    "prev_time_diff_1", "prev_time_diff_2", "prev_time_diff_3", "prev_time_diff_4", "prev_time_diff_5",
    "avg_rank_5",
    "avg_time_diff_5",
    "interval",
    "recency",  # 改善点5: 前走の鮮度（1/(1+interval/60)、直近ほど高く）。追加後は再学習必須。
]

CATEGORICAL_FEATURES = [
    "sire_id_int",
    "bms_id_int",
    "jockey_id_int",
    "trainer_id_int",
    "racecourse_id_int",
    "course_type",
    "state",
    "rotation",
    "wakuban",
    "sire_state_int",
    "course_wakuban_int",
    "sire_dist_int",
    "sire_type_int",
    "bms_dist_int",
    "bms_type_int",
    "bms_state_int",
    "jockey_course_int",
]

# 前走・動的特征の欠損デフォルト（predict_pipeline と一致）
DEFAULT_PREV_RANK = 99
DEFAULT_PREV_TIME_DIFF = 2.0
DEFAULT_INTERVAL = 999
DEFAULT_GLOBAL_MEAN = 0.05
WEIGHT_SIRE = 0.7
WEIGHT_BMS = 0.3


def _to_str(v, default: str = "") -> str:
    if v is None or v == "" or (isinstance(v, float) and (v != v or v == float("nan"))):
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


def load_id_map(path: Path, id_col: str, name_col: str) -> dict[str, str]:
    m = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = _to_str(row.get(id_col))
            v = _to_str(row.get(name_col))
            if k:
                m[k] = v if v else "Unknown"
    return m


def _classify_distance(meters: int) -> str:
    if meters <= 0:
        return "mile"
    if meters < 1400:
        return "short"
    if meters <= 1800:
        return "mile"
    if meters <= 2400:
        return "mid"
    return "long"


def get_dist_cat(meters: int) -> str:
    """距離を4区分に分類（クロス特徴量用）。"""
    d = _to_int(meters, 1600)
    if d < 1400:
        return "Short"
    if d < 1800:
        return "Mile"
    if d < 2400:
        return "Middle"
    return "Long"


def _classify_course(ct: str) -> str:
    c = _to_str(ct)
    return "turf" if c == "1" else "dirt"


def _load_master(path: Path) -> dict[str, dict]:
    master = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        name_col = "sire_name" if "sire_name" in (reader.fieldnames or []) else "bms_name"
        for row in reader:
            n = _to_str(row.get(name_col))
            if n:
                master[n] = {k: _to_float(v) for k, v in row.items() if k != name_col}
    return master


def _get_place_rate(name: str, course: str, dist: str, master: dict, global_stats: dict) -> float:
    if not name or name not in master:
        return global_stats.get(course, {}).get(dist, {}).get("place_rate", DEFAULT_GLOBAL_MEAN) or DEFAULT_GLOBAL_MEAN
    key = f"{course}_{dist}_place"
    v = master[name].get(key)
    if v is not None and v > 0:
        return v
    return global_stats.get(course, {}).get(dist, {}).get("place_rate", DEFAULT_GLOBAL_MEAN) or DEFAULT_GLOBAL_MEAN


def main() -> int:
    print("=== train_model.py 開始 ===")

    required = [
        (INPUT_LEARNING, "learning_dataset.csv"),
        (CONFIG / "sire_name_to_int.pkl", "sire_name_to_int.pkl"),
        (CONFIG / "bms_name_to_int.pkl", "bms_name_to_int.pkl"),
        (CONFIG / "jockey_name_to_int.pkl", "jockey_name_to_int.pkl"),
        (CONFIG / "trainer_name_to_int.pkl", "trainer_name_to_int.pkl"),
        (CONFIG / "racecourse_name_to_int.pkl", "racecourse_name_to_int.pkl"),
        (CONFIG / "sire_id_map.csv", "sire_id_map.csv"),
        (CONFIG / "bms_id_map.csv", "bms_id_map.csv"),
        (CONFIG / "jockey_id_map.csv", "jockey_id_map.csv"),
    ]
    for p, name in required:
        if not p.exists():
            print(f"エラー: {name} が見つかりません: {p}", file=sys.stderr)
            return 1

    sire_id_map = load_id_map(CONFIG / "sire_id_map.csv", "sire_id", "sire_name")
    bms_id_map = load_id_map(CONFIG / "bms_id_map.csv", "bms_id", "bms_name")
    jockey_id_map = load_id_map(CONFIG / "jockey_id_map.csv", "jockey_id", "jockey_name")
    trainer_id_map_path = CONFIG / "trainer_id_map.csv"
    trainer_id_map = load_id_map(trainer_id_map_path, "trainer_id", "trainer_name") if trainer_id_map_path.exists() else {}

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

    try:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print("エラー: lightgbm がインストールされていません。pip install lightgbm", file=sys.stderr)
        return 1

    # --- 1. データ読み込み ---
    rows = []
    with open(INPUT_LEARNING, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sire_id = _to_str(r.get("sire_id"))
            bms_id = _to_str(r.get("broodmare_sire_id"))
            jockey_id = _to_str(r.get("jockey_id"))
            trainer_id = _to_str(r.get("trainer_id"))
            sire_name = sire_id_map.get(sire_id, "Unknown")
            bms_name = bms_id_map.get(bms_id, "Unknown")
            jockey_name = jockey_id_map.get(jockey_id, "Unknown")
            trainer_name = trainer_id_map.get(trainer_id, trainer_id) if trainer_id else "Unknown"
            if not trainer_name:
                trainer_name = "Unknown"

            sire_id_int = sire_name_to_int.get(sire_name, 0)
            bms_id_int = bms_name_to_int.get(bms_name, 0)
            jockey_id_int = jockey_name_to_int.get(jockey_name, 0)
            trainer_id_int = trainer_n2i.get(trainer_name, 0)
            racecourse = _to_str(r.get("racecourse"), "")
            racecourse_id_int = racecourse_n2i.get(racecourse, 0)

            wakuban = _to_int(r.get("wakuban"), 0)
            if not (1 <= wakuban <= 8):
                wakuban = 0
            distance = _to_int(r.get("distance"), 0)
            weight_carry = _to_float(r.get("weight_carry"), 0.0)
            course_type_raw = _to_str(r.get("course_type"))
            course_type = 1 if course_type_raw == "1" else (2 if course_type_raw == "2" else 0)
            state = _to_int(r.get("state"), 1)
            rotation = _to_int(r.get("rotation"), 0)
            rank = _to_int(r.get("rank"), 99)
            date = _to_str(r.get("date"), "")
            horse_id = _to_str(r.get("horse_id"), "")
            race_key = _to_str(r.get("race_key"), "")
            time_diff = _to_float(r.get("time_diff"), 2.0)

            target = 1 if 1 <= rank <= 3 else 0
            rows.append({
                "horse_id": horse_id,
                "race_key": race_key,
                "sire_name": sire_name,
                "bms_name": bms_name,
                "sire_id_int": sire_id_int,
                "bms_id_int": bms_id_int,
                "jockey_id_int": jockey_id_int,
                "trainer_id_int": trainer_id_int,
                "racecourse_id_int": racecourse_id_int,
                "course_type": course_type,
                "state": state,
                "rotation": rotation,
                "wakuban": wakuban,
                "weight_carry": weight_carry,
                "distance": distance,
                "target": target,
                "date": date,
                "rank": rank,
                "time_diff": time_diff,
            })

    df = pd.DataFrame(rows)

    # --- 近5走のラグ特徴量（馬×日付ソート → groupby shift(i)）---
    df = df.sort_values(["horse_id", "date"], na_position="first").reset_index(drop=True)
    for i in range(1, 6):
        df[f"prev_rank_{i}"] = df.groupby("horse_id")["rank"].shift(i).fillna(DEFAULT_PREV_RANK)
        df[f"prev_time_diff_{i}"] = df.groupby("horse_id")["time_diff"].shift(i).fillna(DEFAULT_PREV_TIME_DIFF)
    # 前走からの間隔（1走前のみ）
    df["prev_date"] = df.groupby("horse_id")["date"].shift(1)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["prev_date_dt"] = pd.to_datetime(df["prev_date"], errors="coerce")
    df["interval"] = (df["date_dt"] - df["prev_date_dt"]).dt.days.fillna(DEFAULT_INTERVAL)
    # 改善点5: 前走の鮮度（interval が短いほど 1 に近い。欠損は約0.06）
    df["recency"] = 1.0 / (1.0 + df["interval"].astype(float) / 60.0)
    df = df.drop(columns=["prev_date", "date_dt", "prev_date_dt"], errors="ignore")

    # --- 集約特徴量（過去5走の平均。欠損99/2.0は除外して計算）---
    rank_cols = [f"prev_rank_{i}" for i in range(1, 6)]
    td_cols = [f"prev_time_diff_{i}" for i in range(1, 6)]
    # 99 は欠損のため NaN に置換してから平均 → fillna(99)
    df["avg_rank_5"] = df[rank_cols].replace(DEFAULT_PREV_RANK, float("nan")).mean(axis=1).fillna(DEFAULT_PREV_RANK)
    # 2.0 以上は欠損扱いのため NaN に置換してから平均 → fillna(2.0)
    df["avg_time_diff_5"] = df[td_cols].where(df[td_cols] < DEFAULT_PREV_TIME_DIFF).mean(axis=1).fillna(DEFAULT_PREV_TIME_DIFF)

    # --- Lag Feature Audit ---
    print("=== Lag Feature Audit ===")
    total = len(df)
    print(f"prev_rank_1=99 rate: {len(df[df['prev_rank_1']==99]) / total:.2%}")
    print(f"prev_time_diff_1>=2.0 rate: {len(df[df['prev_time_diff_1']>=2.0]) / total:.2%}")
    print("\n--- Data Sample (Shifted, 1走前・5走前・平均) ---")
    sample_cols = ["horse_id", "date", "rank", "prev_rank_1", "prev_rank_5", "prev_time_diff_1", "avg_rank_5", "avg_time_diff_5"]
    print(df[[c for c in sample_cols if c in df.columns]].head(10))

    # --- Logic Score（血統スコアをレース内で min-max 正規化）---
    sire_master_path = CONFIG / "sire_features_master.csv"
    bms_master_path = CONFIG / "bms_features_master.csv"
    global_stats_path = CONFIG / "global_stats.json"
    if sire_master_path.exists() and bms_master_path.exists() and global_stats_path.exists():
        import json
        sire_master = _load_master(sire_master_path)
        bms_master = _load_master(bms_master_path)
        with open(global_stats_path, encoding="utf-8") as f:
            global_stats = json.load(f)
        pedigree_list = []
        for _, row in df.iterrows():
            course_cat = _classify_course(str(row.get("course_type", 2)))
            dist_cat = _classify_distance(_to_int(row.get("distance"), 1600))
            sire_place = _get_place_rate(row.get("sire_name", ""), course_cat, dist_cat, sire_master, global_stats)
            bms_place = _get_place_rate(row.get("bms_name", ""), course_cat, dist_cat, bms_master, global_stats)
            pedigree_list.append(WEIGHT_SIRE * sire_place + WEIGHT_BMS * bms_place)
        df["_pedigree"] = pedigree_list
        df["logic_score"] = 0.5
        for rk, grp in df.groupby("race_key"):
            vals = grp["_pedigree"].tolist()
            if not vals:
                continue
            mn, mx = min(vals), max(vals)
            if mx - mn >= 1e-9:
                normed = [(v - mn) / (mx - mn) for v in vals]
            else:
                normed = [0.5] * len(vals)
            df.loc[grp.index, "logic_score"] = normed
        df = df.drop(columns=["_pedigree"], errors="ignore")
        print("Logic Score を血統マスタから算出しました")
    else:
        df["logic_score"] = 0.5
        print("警告: sire/bms_features_master または global_stats が無いため logic_score=0.5 で統一")

    # 時系列分割用に日付で再ソート
    df = df.sort_values("date", na_position="first").reset_index(drop=True)

    # --- クロス特徴量（全網羅: sire/bms/jockey × 環境）---
    sire_state_combos = set()
    course_wakuban_combos = set()
    sire_dist_combos = set()
    sire_type_combos = set()
    bms_dist_combos = set()
    bms_type_combos = set()
    bms_state_combos = set()
    jockey_course_combos = set()
    for _, r in df.iterrows():
        dist_cat = get_dist_cat(_to_int(r.get("distance"), 1600))
        ct = r.get("course_type", 0)
        sire_state_combos.add(f"{r['sire_id_int']}_{r['state']}")
        course_wakuban_combos.add(f"{r['racecourse_id_int']}_{r['wakuban']}")
        sire_dist_combos.add(f"{r['sire_id_int']}_{dist_cat}")
        sire_type_combos.add(f"{r['sire_id_int']}_{ct}")
        bms_dist_combos.add(f"{r['bms_id_int']}_{dist_cat}")
        bms_type_combos.add(f"{r['bms_id_int']}_{ct}")
        bms_state_combos.add(f"{r['bms_id_int']}_{r['state']}")
        jockey_course_combos.add(f"{r['jockey_id_int']}_{r['racecourse_id_int']}")
    sire_state_dict = {c: i + 1 for i, c in enumerate(sorted(sire_state_combos))}
    course_wakuban_dict = {c: i + 1 for i, c in enumerate(sorted(course_wakuban_combos))}
    sire_dist_dict = {c: i + 1 for i, c in enumerate(sorted(sire_dist_combos))}
    sire_type_dict = {c: i + 1 for i, c in enumerate(sorted(sire_type_combos))}
    bms_dist_dict = {c: i + 1 for i, c in enumerate(sorted(bms_dist_combos))}
    bms_type_dict = {c: i + 1 for i, c in enumerate(sorted(bms_type_combos))}
    bms_state_dict = {c: i + 1 for i, c in enumerate(sorted(bms_state_combos))}
    jockey_course_dict = {c: i + 1 for i, c in enumerate(sorted(jockey_course_combos))}
    df["sire_state_int"] = df.apply(lambda r: sire_state_dict.get(f"{r['sire_id_int']}_{r['state']}", 0), axis=1)
    df["course_wakuban_int"] = df.apply(lambda r: course_wakuban_dict.get(f"{r['racecourse_id_int']}_{r['wakuban']}", 0), axis=1)
    df["sire_dist_int"] = df.apply(lambda r: sire_dist_dict.get(f"{r['sire_id_int']}_{get_dist_cat(_to_int(r.get('distance'), 1600))}", 0), axis=1)
    df["sire_type_int"] = df.apply(lambda r: sire_type_dict.get(f"{r['sire_id_int']}_{r.get('course_type', 0)}", 0), axis=1)
    df["bms_dist_int"] = df.apply(lambda r: bms_dist_dict.get(f"{r['bms_id_int']}_{get_dist_cat(_to_int(r.get('distance'), 1600))}", 0), axis=1)
    df["bms_type_int"] = df.apply(lambda r: bms_type_dict.get(f"{r['bms_id_int']}_{r.get('course_type', 0)}", 0), axis=1)
    df["bms_state_int"] = df.apply(lambda r: bms_state_dict.get(f"{r['bms_id_int']}_{r['state']}", 0), axis=1)
    df["jockey_course_int"] = df.apply(lambda r: jockey_course_dict.get(f"{r['jockey_id_int']}_{r['racecourse_id_int']}", 0), axis=1)
    interaction_dict = {
        "sire_state": sire_state_dict,
        "course_wakuban": course_wakuban_dict,
        "sire_dist": sire_dist_dict,
        "sire_type": sire_type_dict,
        "bms_dist": bms_dist_dict,
        "bms_type": bms_type_dict,
        "bms_state": bms_state_dict,
        "jockey_course": jockey_course_dict,
    }
    CONFIG.mkdir(parents=True, exist_ok=True)
    with open(CONFIG / "interaction_dict.pkl", "wb") as f:
        pickle.dump(interaction_dict, f)
    print(f"クロス特徴量辞書: sire_state={len(sire_state_dict)}, course_wakuban={len(course_wakuban_dict)}, sire_dist={len(sire_dist_dict)}, sire_type={len(sire_type_dict)}, bms_dist={len(bms_dist_dict)}, bms_type={len(bms_type_dict)}, bms_state={len(bms_state_dict)}, jockey_course={len(jockey_course_dict)}")

    # 欠損値処理（NaN を -1 または 0 で埋める）
    for col in FEATURE_COLS:
        if col in df.columns:
            if df[col].dtype in ("object", "string"):
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(-1 if col in CATEGORICAL_FEATURES else 0)

    # 日付でソート（時系列）
    df = df.sort_values("date", na_position="first").reset_index(drop=True)

    X = df[FEATURE_COLS].astype(float)
    y = df["target"]

    # --- 2. 時系列分割 (過去 80% Train, 直近 20% Valid) ---
    n = len(df)
    split_idx = int(n * 0.8)
    X_train, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"学習データ: {len(X_train)} 件 (複勝率: {y_train.mean():.2%})")
    print(f"検証データ: {len(X_valid)} 件 (複勝率: {y_valid.mean():.2%})")
    print(f"使用特徴量: {FEATURE_COLS}")

    # --- 3. LightGBM 学習 ---
    train_data = lgb.Dataset(
        X_train, label=y_train, feature_name=FEATURE_COLS,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    valid_data = lgb.Dataset(
        X_valid, label=y_valid, reference=train_data, feature_name=FEATURE_COLS,
        categorical_feature=CATEGORICAL_FEATURES,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": 1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True)],
    )

    # --- 4. モデル保存 ---
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tmp_path = tf.name
    try:
        model.save_model(tmp_path)
        shutil.copy(tmp_path, OUTPUT_MODEL)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    print(f"モデルを保存しました: {OUTPUT_MODEL}")

    # --- 5. 特徴量重要度・検証スコア ---
    feature_names = model.feature_name() or FEATURE_COLS
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
    print(f"\n特徴量重要度 (全{len(feat_imp)}件):")
    for i, (name, imp) in enumerate(feat_imp, 1):
        print(f"  {i:2d}. {name}: {imp:.1f}")
    logic_imp = next((imp for n, imp in feat_imp if n == "logic_score"), 0)
    prev_rank_1_imp = next((imp for n, imp in feat_imp if n == "prev_rank_1"), 0)
    avg_rank_5_imp = next((imp for n, imp in feat_imp if n == "avg_rank_5"), 0)
    interval_imp = next((imp for n, imp in feat_imp if n == "interval"), 0)
    print(f"\n[Stacking確認] logic_score={logic_imp:.1f}, prev_rank_1={prev_rank_1_imp:.1f}, avg_rank_5={avg_rank_5_imp:.1f}, interval={interval_imp:.1f}")

    valid_pred = model.predict(X_valid)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_valid, valid_pred)
        print(f"\n検証 AUC: {auc:.4f}")
    except Exception:
        pass

    print("train_model.py 正常終了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
