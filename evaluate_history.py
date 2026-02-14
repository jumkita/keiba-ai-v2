# -*- coding: utf-8 -*-
"""
過去の予測ログと実際のレース結果を照合し、モデルのパフォーマンスを評価するスクリプト。

Inputs:
  - jv_data/history/prediction_log.csv（Step1 で predict_pipeline が蓄積した予測ログ）
  - jv_data/learning_dataset.csv（確定したレース結果。extract_from_mdb で更新）

Outputs:
  - コンソール: 印別成績・スコア帯別成績・条件別成績
  - jv_data/reports/evaluation_summary.csv（照合結果の詳細）

使い方:
  python evaluate_history.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTION_LOG = SCRIPT_DIR / "jv_data" / "history" / "prediction_log.csv"
LEARNING_CSV = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
OUTPUT_SUMMARY = SCRIPT_DIR / "jv_data" / "reports" / "evaluation_summary.csv"


def _to_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _to_int(v, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def load_prediction_log() -> list[dict]:
    """予測ログを読み込む。"""
    if not PREDICTION_LOG.exists():
        return []
    rows = []
    with open(PREDICTION_LOG, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: row.get(k, "") for k in row.keys()})
    return rows


def load_results() -> dict[tuple[str, str], dict]:
    """learning_dataset から (race_key, horse_id) -> {rank, distance, state, racecourse, date} を返す。"""
    out = {}
    if not LEARNING_CSV.exists():
        return out
    with open(LEARNING_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            hid = (row.get("horse_id") or "").strip()
            rank_val = row.get("rank", "")
            try:
                rank = int(float(rank_val))
            except (ValueError, TypeError):
                continue
            if rk and hid and rank >= 1:
                out[(rk, hid)] = {
                    "rank": rank,
                    "distance": _to_int(row.get("distance"), 0),
                    "state": (row.get("state") or "").strip(),
                    "racecourse": (row.get("racecourse") or "").strip(),
                    "date": (row.get("date") or "").strip()[:10],
                }
    return out


def main() -> int:
    print("=== 予測ログ × 結果 照合（evaluate_history.py） ===\n")

    preds = load_prediction_log()
    if not preds:
        print(f"予測ログがありません: {PREDICTION_LOG}")
        print("先に predict_pipeline.py を実行して予測ログを蓄積してください。")
        return 1

    results = load_results()
    if not results:
        print(f"結果データがありません: {LEARNING_CSV}")
        print("extract_from_mdb.py で学習データを更新してください。")
        return 1

    # 結合: race_id = race_key, horse_id でマッチ。未開催は除外
    merged = []
    for p in preds:
        race_id = (p.get("race_id") or "").strip()
        horse_id = (p.get("horse_id") or "").strip()
        key = (race_id, horse_id)
        if key not in results:
            continue
        res = results[key]
        merged.append({
            **p,
            "actual_rank": res["rank"],
            "distance": res["distance"],
            "state": res["state"],
            "racecourse": res["racecourse"],
            "race_date": res["date"],
            "win": 1 if res["rank"] == 1 else 0,
            "place_2": 1 if res["rank"] <= 2 else 0,
            "place_3": 1 if res["rank"] <= 3 else 0,
        })
    if not merged:
        print("照合できたレコードが 0 件です（予測ログの race_id/horse_id と結果が一致しません）。")
        return 1
    print(f"照合件数: {len(merged)} 件（予測ログ {len(preds)} 件のうち結果と一致したもの）\n")

    # --- 印別成績 ---
    by_mark = defaultdict(lambda: {"win": 0, "place_2": 0, "place_3": 0, "n": 0})
    for m in merged:
        mark = (m.get("mark") or "無印").strip() or "無印"
        by_mark[mark]["n"] += 1
        by_mark[mark]["win"] += m["win"]
        by_mark[mark]["place_2"] += m["place_2"]
        by_mark[mark]["place_3"] += m["place_3"]

    mark_order = ["◎", "○", "▲", "△", "⭐", "無印"]
    print("【印別成績】")
    print(f"  {'印':<4} {'頭数':>6} {'勝率':>8} {'連対率':>8} {'複勝率':>8}  (回収率はオッズ未取得のため省略)")
    print("  " + "-" * 50)
    for mark in [x for x in mark_order if x in by_mark] + [x for x in sorted(by_mark) if x not in mark_order]:
        b = by_mark[mark]
        n = b["n"]
        if n == 0:
            continue
        win_rate = b["win"] / n * 100
        place2_rate = b["place_2"] / n * 100
        place3_rate = b["place_3"] / n * 100
        print(f"  {mark:<4} {n:>6} {win_rate:>6.1f}% {place2_rate:>6.1f}% {place3_rate:>6.1f}%")

    # --- スコア帯別（final_score を 0.1 刻みでビン）---
    by_bin = defaultdict(lambda: {"win": 0, "place_3": 0, "n": 0})
    for m in merged:
        fs = _to_float(m.get("final_score"), 0)
        bin_key = round(fs * 10) / 10
        by_bin[bin_key]["n"] += 1
        by_bin[bin_key]["win"] += m["win"]
        by_bin[bin_key]["place_3"] += m["place_3"]

    print("\n【スコア帯別成績（final_score 0.1刻み）】")
    print(f"  {'final_score':<12} {'頭数':>6} {'勝率':>8} {'複勝率':>8}")
    print("  " + "-" * 40)
    for bin_key in sorted(by_bin.keys(), reverse=True):
        b = by_bin[bin_key]
        n = b["n"]
        if n == 0:
            continue
        win_rate = b["win"] / n * 100
        place3_rate = b["place_3"] / n * 100
        print(f"  {bin_key:.1f}         {n:>6} {win_rate:>6.1f}% {place3_rate:>6.1f}%")

    # --- 条件別（競馬場・距離・馬場）---
    by_course = defaultdict(lambda: {"win": 0, "place_3": 0, "n": 0})
    by_dist = defaultdict(lambda: {"win": 0, "place_3": 0, "n": 0})
    by_state = defaultdict(lambda: {"win": 0, "place_3": 0, "n": 0})
    for m in merged:
        course = (m.get("racecourse") or "不明").strip() or "不明"
        dist = m.get("distance", 0)
        if dist <= 0:
            dist_key = "不明"
        elif dist < 1400:
            dist_key = "短距離"
        elif dist <= 1800:
            dist_key = "マイル"
        elif dist <= 2400:
            dist_key = "中距離"
        else:
            dist_key = "長距離"
        st = (m.get("state") or "不明").strip() or "不明"
        state_label = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}.get(st, st)
        by_course[course]["n"] += 1
        by_course[course]["win"] += m["win"]
        by_course[course]["place_3"] += m["place_3"]
        by_dist[dist_key]["n"] += 1
        by_dist[dist_key]["win"] += m["win"]
        by_dist[dist_key]["place_3"] += m["place_3"]
        by_state[state_label]["n"] += 1
        by_state[state_label]["win"] += m["win"]
        by_state[state_label]["place_3"] += m["place_3"]

    print("\n【競馬場別（複勝率）】")
    print(f"  {'競馬場':<8} {'頭数':>6} {'勝率':>8} {'複勝率':>8}")
    print("  " + "-" * 35)
    for course in sorted(by_course.keys()):
        b = by_course[course]
        n = b["n"]
        if n == 0:
            continue
        win_rate = b["win"] / n * 100
        place3_rate = b["place_3"] / n * 100
        print(f"  {course:<8} {n:>6} {win_rate:>6.1f}% {place3_rate:>6.1f}%")

    print("\n【距離別（複勝率）】")
    print(f"  {'距離':<8} {'頭数':>6} {'勝率':>8} {'複勝率':>8}")
    print("  " + "-" * 35)
    dist_order = ["短距離", "マイル", "中距離", "長距離", "不明"]
    for dist_key in [x for x in dist_order if x in by_dist] + [x for x in sorted(by_dist) if x not in dist_order]:
        b = by_dist[dist_key]
        n = b["n"]
        if n == 0:
            continue
        win_rate = b["win"] / n * 100
        place3_rate = b["place_3"] / n * 100
        print(f"  {dist_key:<8} {n:>6} {win_rate:>6.1f}% {place3_rate:>6.1f}%")

    print("\n【馬場状態別（複勝率）】")
    print(f"  {'馬場':<6} {'頭数':>6} {'勝率':>8} {'複勝率':>8}")
    print("  " + "-" * 35)
    state_order = ["良", "稍重", "重", "不良", "不明"]
    for st in [x for x in state_order if x in by_state] + [x for x in sorted(by_state) if x not in state_order]:
        b = by_state[st]
        n = b["n"]
        if n == 0:
            continue
        win_rate = b["win"] / n * 100
        place3_rate = b["place_3"] / n * 100
        print(f"  {st:<6} {n:>6} {win_rate:>6.1f}% {place3_rate:>6.1f}%")

    # --- 詳細を CSV に保存 ---
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_cols = [
        "race_id", "race_date", "race_name", "racecourse", "distance", "state",
        "horse_id", "horse_num", "horse_name", "mark", "rank_predict", "final_score",
        "actual_rank", "win", "place_2", "place_3", "model_version",
    ]
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    print(f"\n照合結果の詳細を保存しました: {OUTPUT_SUMMARY}")
    print("=== evaluate_history.py 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
