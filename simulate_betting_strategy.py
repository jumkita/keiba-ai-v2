# -*- coding: utf-8 -*-
"""
買い目フィルターシミュレーション: 過去のAI予測と確定結果を突き合わせ、
スコア閾値・単勝オッズ帯・券種ごとに「的中率」と「回収率」を計算し、
Winning Formula（勝利の方程式）を探索する。

Data Sources:
  - 予測: jv_data/history/prediction_log.csv（predict_pipeline が蓄積）
         または jv_data/reports/evaluation_summary.csv（evaluate_history.py 出力）
  - 結果: jv_data/learning_dataset.csv（確定成績・extract_from_mdb で更新）
  - オッズ（任意）: jv_data/result_odds.csv があると回収率を算出
    形式: race_id,horse_id,odds_tansho,odds_fukusho

Parameters:
  - AIスコア閾値: 60, 70, 80, 90 以上（final_score×100 で比較）
  - 単勝オッズ帯: 1.0〜4.9倍, 5.0〜9.9倍, 10.0〜19.9倍, 20倍以上
  - 券種: 単勝 / 複勝 / ワイド◎流し（ワイドは的中率のみ。払戻は別途ワイドオッズが必要）

Usage:
  python simulate_betting_strategy.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTION_LOG = SCRIPT_DIR / "jv_data" / "history" / "prediction_log.csv"
EVALUATION_SUMMARY = SCRIPT_DIR / "jv_data" / "reports" / "evaluation_summary.csv"
LEARNING_CSV = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
RESULT_ODDS_CSV = SCRIPT_DIR / "jv_data" / "result_odds.csv"
OUTPUT_SIMULATION_CSV = SCRIPT_DIR / "jv_data" / "reports" / "simulation_betting_strategy.csv"

# シミュレーション用パラメータ
SCORE_THRESHOLDS = [60, 70, 80, 90]  # 以上
ODDS_BANDS = [
    (1.0, 4.9, "1.0〜4.9倍"),
    (5.0, 9.9, "5.0〜9.9倍"),
    (10.0, 19.9, "10.0〜19.9倍"),
    (20.0, 9999.0, "20倍以上"),
]
BET_TYPES = ("単勝", "複勝", "ワイド◎流し")


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _to_int(v: Any, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def load_merged_data() -> list[dict]:
    """
    予測と結果をマージしたデータを返す。
    evaluation_summary.csv があればそれを使用、なければ prediction_log + learning_dataset をマージ。
    各レコードの final_score は 0〜1。スコア閾値用に score100 = final_score * 100 を付与する。
    """
    merged: list[dict] = []

    if EVALUATION_SUMMARY.exists():
        with open(EVALUATION_SUMMARY, encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                row = {k: row.get(k, "") for k in row.keys()}
                fs = _to_float(row.get("final_score"), 0)
                row["score100"] = fs * 100
                row["race_id"] = (row.get("race_id") or "").strip()
                row["horse_id"] = (row.get("horse_id") or "").strip()
                merged.append(row)
        return merged

    # prediction_log + learning_dataset でマージ（evaluate_history と同様）
    if not PREDICTION_LOG.exists():
        return []
    preds: list[dict] = []
    with open(PREDICTION_LOG, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            preds.append({k: row.get(k, "") for k in row.keys()})

    results: dict[tuple[str, str], dict] = {}
    if LEARNING_CSV.exists():
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
                    results[(rk, hid)] = {
                        "rank": rank,
                        "date": (row.get("date") or "").strip()[:10],
                    }

    for p in preds:
        race_id = (p.get("race_id") or "").strip()
        horse_id = (p.get("horse_id") or "").strip()
        key = (race_id, horse_id)
        if key not in results:
            continue
        res = results[key]
        fs = _to_float(p.get("final_score"), 0)
        merged.append({
            "race_id": race_id,
            "horse_id": horse_id,
            "horse_num": p.get("horse_num", ""),
            "horse_name": p.get("horse_name", ""),
            "mark": p.get("mark", ""),
            "final_score": fs,
            "score100": fs * 100,
            "actual_rank": res["rank"],
            "race_date": res["date"],
            "win": 1 if res["rank"] == 1 else 0,
            "place_2": 1 if res["rank"] <= 2 else 0,
            "place_3": 1 if res["rank"] <= 3 else 0,
        })
    return merged


def load_odds() -> dict[tuple[str, str], dict]:
    """(race_id, horse_id) -> {odds_tansho, odds_fukusho}"""
    out: dict[tuple[str, str], dict] = {}
    if not RESULT_ODDS_CSV.exists():
        return out
    with open(RESULT_ODDS_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_id") or "").strip()
            hid = (row.get("horse_id") or "").strip()
            if rk and hid:
                out[(rk, hid)] = {
                    "odds_tansho": _to_float(row.get("odds_tansho"), 0),
                    "odds_fukusho": _to_float(row.get("odds_fukusho"), 0),
                }
    return out


def in_odds_band(odds: float, low: float, high: float) -> bool:
    return low <= odds <= high


def run_simulation(
    merged: list[dict],
    odds_map: dict[tuple[str, str], dict],
) -> list[dict]:
    """
    全組み合わせでシミュレーションし、行のリストを返す。
    各行: score_threshold, odds_band_label, bet_type, n_bets, n_hits, hit_rate_pct, recovery_pct
    """
    has_odds = bool(odds_map)
    rows: list[dict] = []

    for score_th in SCORE_THRESHOLDS:
        # スコアフィルタ: score100 >= score_th（final_score は 0-1 なので score100 = final_score*100）
        subset = [m for m in merged if _to_float(m.get("score100"), 0) >= score_th]
        if not subset:
            continue

        for low, high, band_label in ODDS_BANDS:
            if has_odds:
                sub = [
                    s for s in subset
                    if in_odds_band(
                        odds_map.get((s["race_id"], s["horse_id"]), {}).get("odds_tansho", 0) or 0,
                        low, high,
                    )
                ]
            else:
                sub = subset  # オッズなし時は全件で「全オッズ」として 1 パターンのみ
                if (low, high) != (1.0, 4.9):
                    continue
                band_label = "全オッズ（オッズデータなし）"

            if not sub:
                continue

            for bet_type in BET_TYPES:
                n_bets = len(sub)
                if bet_type == "単勝":
                    n_hits = sum(1 for s in sub if s.get("win") == 1)
                    if has_odds:
                        payout = sum(
                            odds_map.get((s["race_id"], s["horse_id"]), {}).get("odds_tansho", 0)
                            for s in sub if s.get("win") == 1
                        )
                    else:
                        payout = 0.0
                elif bet_type == "複勝":
                    n_hits = sum(1 for s in sub if s.get("place_3") == 1)
                    if has_odds:
                        payout = sum(
                            odds_map.get((s["race_id"], s["horse_id"]), {}).get("odds_fukusho", 0)
                            for s in sub if s.get("place_3") == 1
                        )
                    else:
                        payout = 0.0
                else:
                    # ワイド◎流し: ◎（1番人気想定）が 1〜2着に入れば「流し」のどれかが的中
                    # ここでは「◎馬が3着以内」を複勝的扱いでカウント（ワイドは1-2着の2頭なので厳密には2着以内）
                    n_hits = sum(1 for s in sub if s.get("place_2") == 1)
                    payout = 0.0  # ワイドオッズは別形式のため未対応時は0

                hit_rate = (n_hits / n_bets * 100) if n_bets else 0
                # 回収率 = 払戻合計 / 投資（1点100円想定）。オッズは100円あたりの払戻倍率
                investment = n_bets * 100
                return_yen = payout * 100 if payout else 0
                recovery = (return_yen / investment * 100) if investment else 0.0

                rows.append({
                    "score_threshold": score_th,
                    "odds_band": band_label,
                    "bet_type": bet_type,
                    "n_bets": n_bets,
                    "n_hits": n_hits,
                    "hit_rate_pct": round(hit_rate, 1),
                    "recovery_pct": round(recovery, 1) if has_odds else None,
                })

            if not has_odds:
                break  # オッズなし時は「全オッズ」のみ

    return rows


def print_results(rows: list[dict], has_odds: bool) -> None:
    """表形式で出力し、Winning Formula を要約する。"""
    if not rows:
        print("シミュレーション結果が0件です。")
        return

    headers = ["スコア閾値", "単勝オッズ帯", "券種", "購入数", "的中数", "的中率(%)", "回収率(%)"]
    col_widths = [10, 18, 12, 8, 8, 10, 10]
    sep = " | "
    print("\n【シミュレーション結果】")
    print(sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(col_widths) + len(sep) * (len(headers) - 1)))

    for r in rows:
        rec = r["recovery_pct"]
        rec_str = f"{rec:.1f}%" if rec is not None else "N/A"
        print(sep.join([
            str(r["score_threshold"]).ljust(col_widths[0]),
            r["odds_band"].ljust(col_widths[1]),
            r["bet_type"].ljust(col_widths[2]),
            str(r["n_bets"]).rjust(col_widths[3]),
            str(r["n_hits"]).rjust(col_widths[4]),
            f"{r['hit_rate_pct']:.1f}%".rjust(col_widths[5]),
            rec_str.rjust(col_widths[6]),
        ]))

    # Winning Formula: 回収率100%超で、的中率もそこそこある条件を抽出
    if has_odds:
        candidates = [x for x in rows if x["recovery_pct"] is not None and x["recovery_pct"] >= 100 and x["n_bets"] >= 5]
        candidates.sort(key=lambda x: (-(x["recovery_pct"] or 0), -x["hit_rate_pct"]))
        print("\n【Winning Formula 候補】回収率100%以上 & 購入数5件以上")
        if candidates:
            for c in candidates[:15]:
                print(
                    f"  スコア{c['score_threshold']}以上 & 単勝オッズ「{c['odds_band']}」で {c['bet_type']} → "
                    f"的中率{c['hit_rate_pct']:.1f}% / 回収率{c['recovery_pct']:.1f}% (n={c['n_bets']})"
                )
        else:
            print("  該当する条件はありませんでした。")
    else:
        print("\n※ 回収率を出すには jv_data/result_odds.csv に race_id, horse_id, odds_tansho, odds_fukusho を用意してください。")

    # 結果をCSVに保存
    if rows:
        OUTPUT_SIMULATION_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_SIMULATION_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["score_threshold", "odds_band", "bet_type", "n_bets", "n_hits", "hit_rate_pct", "recovery_pct"],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\n詳細: {OUTPUT_SIMULATION_CSV}")


def main() -> int:
    print("=== 買い目フィルターシミュレーション（simulate_betting_strategy.py） ===\n")

    merged = load_merged_data()
    if not merged:
        print("予測×結果のマージデータが0件です。")
        print("  - jv_data/history/prediction_log.csv と jv_data/learning_dataset.csv を用意するか、")
        print("  - evaluate_history.py を実行して jv_data/reports/evaluation_summary.csv を生成してください。")
        return 1

    print(f"照合件数: {len(merged)} 件\n")

    odds_map = load_odds()
    has_odds = bool(odds_map)
    if has_odds:
        print(f"オッズデータ: {len(odds_map)} 件読み込み（回収率を計算します）")
    else:
        print("オッズデータがありません（jv_data/result_odds.csv）。的中率のみ算出します。")

    rows = run_simulation(merged, odds_map)
    print_results(rows, has_odds)
    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
