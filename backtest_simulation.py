# -*- coding: utf-8 -*-
"""
バックテストシミュレーション: AI予測データとWeb取得のレース結果（オッズ）を結合し、
複数購入パターンで回収率を計算する。

Input:
  - docs/weekly_prediction.json（races[].race_id, races[].predictions[].horse_name, score, mark）

Process:
  1. netkeiba レース結果ページをスクレイピング（着順・馬名・単勝・複勝）
  2. 予測と結果を馬名でマージ
  3. Case A〜D のシミュレーション → 表形式で出力

Usage:
  python backtest_simulation.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTION_JSON = SCRIPT_DIR / "docs" / "weekly_prediction.json"
BASE_URL = "https://db.netkeiba.com/race/{race_id}/"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def load_predictions(path: Path | None = None) -> list[dict]:
    """
    weekly_prediction.json を読み、レースごとの予測リストを返す。
    各要素: { race_id, race_name, target_date, horses: [ { horse_name, horse_num, score, mark }, ... ] }
    """
    path = path or PREDICTION_JSON
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    target_date = data.get("target_date") or ""
    out = []
    for r in data.get("races") or []:
        predictions = r.get("predictions") or []
        horses = [
            {
                "horse_name": p.get("horse_name", "").strip(),
                "horse_num": p.get("horse_num"),
                "score": int(p.get("score", 0)),
                "mark": (p.get("mark") or "").strip(),
            }
            for p in predictions
        ]
        out.append({
            "race_id": (r.get("race_id") or "").strip(),
            "race_name": (r.get("race_name") or "").strip(),
            "target_date": target_date,
            "horses": horses,
        })
    return out


def _normalize_name(s: str) -> str:
    return (s or "").strip().replace("　", " ").replace("\u3000", " ")


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else default


def fetch_race_result(race_id: str) -> dict[str, Any] | None:
    """
    netkeiba レース結果ページを取得し、着順・馬名・単勝オッズ・複勝オッズを返す。
    返却: { rows: [ { rank, horse_name, umaban, odds_tansho, odds_fukusho } ], wide_payouts: { (a,b): yen } }
    """
    url = BASE_URL.format(race_id=race_id)
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return {"error": str(e), "rows": [], "wide_payouts": {}}
    soup = BeautifulSoup(resp.text, "html.parser")

    # 結果テーブル: 着順・枠番・馬番・馬名・...・単勝・人気...
    table = soup.find("table", class_="race_table_01") or soup.find("table", class_="race_table_01 nk_tb_common")
    if not table:
        return {"rows": [], "wide_payouts": {}}

    thead = table.find("thead")
    headers = []
    if thead:
        for th in thead.find_all("tr")[-1].find_all("th"):
            headers.append((th.get_text() or "").strip())
    tbody = table.find("tbody")
    if not tbody:
        return {"rows": [], "wide_payouts": {}}

    # ヘッダーで「単勝」列インデックスを取得
    try:
        idx_tansho = headers.index("単勝") if "単勝" in headers else 12
    except ValueError:
        idx_tansho = 12
    idx_chakujun = 0
    idx_umaban = 2
    idx_bamei = 3

    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) <= max(idx_bamei, idx_tansho):
            continue
        rank_text = (cells[idx_chakujun].get_text() or "").strip()
        if not rank_text or not rank_text.isdigit():
            continue
        rank = int(rank_text)
        umaban_text = (cells[idx_umaban].get_text() or "").strip()
        umaban = int(umaban_text) if umaban_text.isdigit() else 0
        # 馬名は a タグ内のこともある
        name_cell = cells[idx_bamei]
        name_el = name_cell.find("a") or name_cell
        horse_name = _normalize_name((name_el.get_text() or "").strip())
        odds_tansho = _to_float(cells[idx_tansho].get_text() if idx_tansho < len(cells) else "")

        rows.append({
            "rank": rank,
            "horse_name": horse_name,
            "umaban": umaban,
            "odds_tansho": odds_tansho,
            "odds_fukusho": None,  # あとで払戻から割り当て
        })

    # 複勝払戻: 1着・2着・3着に付与（払戻表から 50〜3000 円の数値を3つ取得）
    pay_tables = soup.find_all("table", class_="pay_table_01") or soup.find_all("table")
    fukusho_yen = [0.0, 0.0, 0.0]
    for tbl in pay_tables:
        text = tbl.get_text() or ""
        if "複勝" not in text:
            continue
        for row in tbl.find_all("tr"):
            cells = row.find_all(["td", "th"])
            cell_texts = [(c.get_text() or "").strip() for c in cells]
            if "複勝" in cell_texts:
                nums = []
                for c in cells[1:]:
                    raw = (c.get_text() or "").replace(",", "").replace("，", "")
                    for m in re.finditer(r"\d+", raw):
                        n = int(m.group())
                        if 50 <= n <= 3000:  # 複勝払戻は通常この範囲
                            nums.append(n)
                if len(nums) >= 3:
                    fukusho_yen = [float(nums[0]), float(nums[1]), float(nums[2])]
                elif len(nums) >= 1:
                    fukusho_yen = [float(nums[0]), 0.0, 0.0]
                break
        break

    for i, row in enumerate(rows):
        if row["rank"] <= 3 and i < len(fukusho_yen):
            row["odds_fukusho"] = fukusho_yen[i] / 100.0 if fukusho_yen[i] else 0.0  # 100円あたり→倍率

    # ワイド払戻: 表は "6-11" "2-6" "2-11" と "380" "1,050" "790" のように並ぶ
    wide_payouts = {}
    for tbl in pay_tables:
        text = tbl.get_text() or ""
        if "ワイド" not in text:
            continue
        for row in tbl.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if any("ワイド" in ((c.get_text() or "")) for c in cells):
                nums = []
                for c in cells[1:]:
                    raw = (c.get_text() or "").replace(",", "").replace("，", "")
                    for m in re.finditer(r"\d+", raw):
                        nums.append(int(m.group()))
                # 馬番2つ+払戻 が3組: (6,11,380), (2,6,1050), (2,11,790)
                if len(nums) >= 9:
                    wide_payouts[(nums[0], nums[1])] = nums[2]
                    wide_payouts[(nums[3], nums[4])] = nums[5]
                    wide_payouts[(nums[6], nums[7])] = nums[8]
                elif len(nums) >= 6:
                    wide_payouts[(nums[0], nums[1])] = nums[2]
                    wide_payouts[(nums[3], nums[4])] = nums[5]
                elif len(nums) >= 3:
                    wide_payouts[(nums[0], nums[1])] = nums[2]
                break
        break

    return {"rows": rows, "wide_payouts": wide_payouts}


def fetch_all_results(race_ids: list[str]) -> dict[str, dict]:
    """各 race_id で結果を取得。1リクエストごとに time.sleep(1)。"""
    out = {}
    for i, rid in enumerate(race_ids):
        if i > 0:
            time.sleep(1)
        out[rid] = fetch_race_result(rid) or {}
    return out


def merge_predictions_and_results(
    predictions: list[dict],
    results: dict[str, dict],
) -> list[dict]:
    """
    レースごとに予測と結果を馬名でマージ。
    返却: 1行 = 1馬（レース×馬）のリスト。キー: race_id, horse_name, score, mark, rank, odds_tansho, odds_fukusho, umaban
    """
    merged = []
    for pred in predictions:
        race_id = pred["race_id"]
        res = results.get(race_id) or {}
        rows = res.get("rows") or []
        if not rows:
            continue
        name_to_result = {_normalize_name(r["horse_name"]): r for r in rows}
        for h in pred.get("horses") or []:
            name = _normalize_name(h.get("horse_name") or "")
            if not name:
                continue
            r = name_to_result.get(name)
            if r is None:
                # 微妙な表記差でマッチしない場合
                for k, v in name_to_result.items():
                    if name in k or k in name:
                        r = v
                        break
            if r is None:
                continue
            merged.append({
                "race_id": race_id,
                "race_name": pred.get("race_name", ""),
                "horse_name": name,
                "horse_num": h.get("horse_num"),
                "score": h.get("score", 0),
                "mark": h.get("mark", ""),
                "rank": r.get("rank"),
                "umaban": r.get("umaban"),
                "odds_tansho": r.get("odds_tansho") or 0.0,
                "odds_fukusho": r.get("odds_fukusho") or 0.0,
            })
    return merged


def simulate_case_a(merged: list[dict]) -> dict:
    """Case A: スコア1位（◎）を無条件で単勝購入。"""
    # レースごとに◎を1頭
    by_race = {}
    for m in merged:
        if m.get("mark") != "◎":
            continue
        rid = m["race_id"]
        if rid not in by_race:
            by_race[rid] = m
    bets = list(by_race.values())
    n = len(bets)
    stake = n * 100
    payout = sum(100 * (b["odds_tansho"] or 0) for b in bets if b.get("rank") == 1)
    return {"case": "A", "label": "◎単勝（無条件）", "n_bets": n, "stake": stake, "payout": payout}


def simulate_case_b(merged: list[dict]) -> dict:
    """Case B: スコア1位かつ単勝オッズ3.0倍以上のみ購入。"""
    by_race = {}
    for m in merged:
        if m.get("mark") != "◎":
            continue
        if (m.get("odds_tansho") or 0) < 3.0:
            continue
        rid = m["race_id"]
        if rid not in by_race:
            by_race[rid] = m
    bets = list(by_race.values())
    n = len(bets)
    stake = n * 100
    payout = sum(100 * (b["odds_tansho"] or 0) for b in bets if b.get("rank") == 1)
    return {"case": "B", "label": "◎単勝（3.0倍以上）", "n_bets": n, "stake": stake, "payout": payout}


def simulate_case_c(
    predictions: list[dict],
    results: dict[str, dict],
    merged: list[dict],
) -> dict:
    """Case C: スコア上位5頭のBOXワイド購入。1レースあたり10組み合わせ×100円。"""
    total_stake = 0
    total_payout = 0
    n_races = 0
    for pred in predictions:
        race_id = pred["race_id"]
        res = results.get(race_id) or {}
        rows = res.get("rows", [])
        wide_payouts = res.get("wide_payouts") or {}
        if not rows or not wide_payouts:
            continue
        # スコア上位5頭（予測の先頭5頭）
        top5 = (pred.get("horses") or [])[:5]
        if len(top5) < 2:
            continue
        name_to_result = {_normalize_name(r["horse_name"]): r for r in rows}
        umaban_set = set()
        for h in top5:
            name = _normalize_name(h.get("horse_name") or "")
            r = name_to_result.get(name)
            if r is not None:
                umaban_set.add(r.get("umaban"))
        if len(umaban_set) < 2:
            continue
        # BOX: 5C2 = 10通り、100円ずつ
        combos = [(a, b) if a < b else (b, a) for a in umaban_set for b in umaban_set if a != b]
        combos = list(dict.fromkeys(combos))
        stake_race = len(combos) * 100
        total_stake += stake_race
        n_races += 1
        # 本番の1-2, 1-3, 2-3 の馬番
        rank1 = next((r for r in rows if r.get("rank") == 1), None)
        rank2 = next((r for r in rows if r.get("rank") == 2), None)
        rank3 = next((r for r in rows if r.get("rank") == 3), None)
        if not rank1 or not rank2 or not rank3:
            continue
        u1, u2, u3 = rank1.get("umaban"), rank2.get("umaban"), rank3.get("umaban")
        winning_pairs = [
            (min(u1, u2), max(u1, u2)) if u1 and u2 else None,
            (min(u1, u3), max(u1, u3)) if u1 and u3 else None,
            (min(u2, u3), max(u2, u3)) if u2 and u3 else None,
        ]
        for pair in winning_pairs:
            if not pair:
                continue
            small, large = (min(pair), max(pair))
            if (small, large) in wide_payouts:
                total_payout += wide_payouts[(small, large)]
            else:
                for (a, b), yen in wide_payouts.items():
                    if (min(a, b), max(a, b)) == (small, large):
                        total_payout += yen
                        break
    return {
        "case": "C",
        "label": "上位5頭BOXワイド",
        "n_bets": n_races,
        "stake": total_stake,
        "payout": total_payout,
    }


def simulate_case_d(merged: list[dict]) -> dict:
    """Case D: スコア80以上かつ単勝オッズ10倍以上の複勝購入。"""
    bets = [m for m in merged if (m.get("score") or 0) >= 80 and (m.get("odds_tansho") or 0) >= 10.0]
    n = len(bets)
    stake = n * 100
    payout = sum(100 * (b["odds_fukusho"] or 0) for b in bets if (b.get("rank") or 99) <= 3)
    return {"case": "D", "label": "スコア80+＆単勝10倍+の複勝", "n_bets": n, "stake": stake, "payout": payout}


def run_simulations(
    predictions: list[dict],
    results: dict[str, dict],
    merged: list[dict],
) -> list[dict]:
    out = []
    out.append(simulate_case_a(merged))
    out.append(simulate_case_b(merged))
    out.append(simulate_case_c(predictions, results, merged))
    out.append(simulate_case_d(merged))
    return out


def main() -> int:
    print("=== バックテストシミュレーション（backtest_simulation.py） ===\n")

    predictions = load_predictions()
    if not predictions:
        print(f"予測データがありません: {PREDICTION_JSON}")
        return 1

    race_ids = [p["race_id"] for p in predictions]
    print(f"対象レース数: {len(race_ids)}（race_id 先頭3件: {race_ids[:3]}）")
    print("結果を取得中（1リクエスト/秒）...\n")

    results = fetch_all_results(race_ids)
    merged = merge_predictions_and_results(predictions, results)
    print(f"マージ件数: {len(merged)} 件\n")

    if not merged:
        print("マージできたデータが0件です。馬名の表記差や未開催の可能性があります。")
        return 1

    sims = run_simulations(predictions, results, merged)
    for s in sims:
        s["recovery_pct"] = (s["payout"] / s["stake"] * 100) if s["stake"] else 0.0

    df = pd.DataFrame(sims)
    df = df[["case", "label", "n_bets", "stake", "payout", "recovery_pct"]]
    df.columns = ["Case", "購入条件", "購入数", "投資(円)", "払戻(円)", "回収率(%)"]
    print("【シミュレーション結果】")
    print(df.to_string(index=False))
    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
