# -*- coding: utf-8 -*-
"""
2/10 予想とレース結果を照合し、的中率を算出・改善点を洗い出すスクリプト。

使い方:
  python compare_predictions_20260210.py

入力:
  - jv_data/reports/ 内の最新レポート HTML（2/10 予想を含むもの）
  - jv_data/learning_dataset.csv の 20260210 レース結果
     または jv_data/result_20260210.csv（オプション。形式: race_key,horse_id,rank）
  - jv_data/future_races.csv（馬番↔horse_id の対応用）

出力:
  - コンソールに的中率と改善点
  - jv_data/reports/accuracy_20260210.txt（オプションで保存）
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SCRIPT_DIR / "jv_data" / "reports"
LEARNING_CSV = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
FUTURE_CSV = SCRIPT_DIR / "jv_data" / "future_races.csv"
RESULT_CSV_OPT = SCRIPT_DIR / "jv_data" / "result_20260210.csv"
TARGET_DATE = "20260210"
PLACE_NAME_TO_CODE = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10",
}
# race_key の競馬場コード（例: 20260210_05_01 の 05）→ 競馬場名
CODE_TO_PLACE = {v: k for k, v in PLACE_NAME_TO_CODE.items()}


def _latest_report_for_date(date_display: str) -> Path | None:
    """指定日付（例: 2026/02/10）を含むレポートのうち最新を返す。"""
    date_part = date_display.replace("/", "")
    if len(date_part) == 8:
        target = date_part  # 20260210
    else:
        return None
    candidates = []
    for f in REPORTS_DIR.glob("report_*.html"):
        try:
            text = f.read_text(encoding="utf-8")
            if target in text or date_display in text:
                candidates.append((f.stat().st_mtime, f))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def parse_predictions_from_html(report_path: Path, date_display: str) -> dict[str, list[tuple[str, str, str]]]:
    """
    レポートHTMLから指定日の予想を抽出。
    戻り値: race_key -> [(mark, umaban, horse_name), ...]  (◎○▲の順)
    """
    text = report_path.read_text(encoding="utf-8")
    # <tr data-date='2026/02/10' data-place='東京' data-race='1R'> ... <td class='mark'>◎</td><td>5</td><td>馬名</td>
    tr_pattern = r"<tr data-date='" + re.escape(date_display) + r"' data-place='([^']+)' data-race='(\d+)R'>(.*?)</tr>"
    races = defaultdict(list)
    for m in re.finditer(tr_pattern, text, re.DOTALL):
        place_name, race_no, rest = m.group(1), m.group(2), m.group(3)
        code = PLACE_NAME_TO_CODE.get(place_name, "00")
        race_key = f"{TARGET_DATE}_{code}_{int(race_no):02d}"
        # 印・馬番・馬名は 4,5,6 番目の td
        td_pattern = r"<td[^>]*>([^<]*)</td>"
        tds = re.findall(td_pattern, rest)
        if len(tds) >= 6:
            mark = (tds[3] or "").strip()
            umaban = (tds[4] or "").strip()
            horse_name = (tds[5] or "").strip()
            if mark and umaban and horse_name:
                races[race_key].append((mark, umaban, horse_name))
    # 各レースはスコア順で並んでいるので ◎○▲ を先頭3頭として採用
    result = {}
    for rk, rows in races.items():
        top3 = [r for r in rows if r[0] in ("◎", "○", "▲")][:3]
        if top3:
            result[rk] = top3
    return result


def load_future_umaban_to_horse_id() -> dict[str, dict[str, str]]:
    """future_races.csv から race_key ごとに (umaban -> horse_id) を返す。"""
    umaban_map = defaultdict(dict)
    if not FUTURE_CSV.exists():
        return dict(umaban_map)
    with open(FUTURE_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            if not rk or not rk.startswith(TARGET_DATE):
                continue
            u = (row.get("umaban") or "").strip()
            hid = (row.get("horse_id") or "").strip()
            if u and hid:
                umaban_map[rk][u] = hid
    return dict(umaban_map)


def load_results_from_learning() -> dict[str, list[tuple[str, int]]]:
    """learning_dataset.csv から 20260210 の (horse_id, rank) を race_key ごとに返す。"""
    out = defaultdict(list)
    if not LEARNING_CSV.exists():
        return dict(out)
    with open(LEARNING_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            if not rk.startswith(TARGET_DATE):
                continue
            hid = (row.get("horse_id") or "").strip()
            rank_val = row.get("rank", "")
            try:
                rank = int(float(rank_val))
            except (ValueError, TypeError):
                continue
            if hid and rank >= 1:
                out[rk].append((hid, rank))
    for rk in out:
        out[rk].sort(key=lambda x: x[1])
    return dict(out)


def load_results_from_learning_with_wakuban() -> dict[str, list[tuple[str, int, int]]]:
    """learning_dataset.csv から 20260210 の (horse_id, rank, wakuban) を race_key ごとに返す。馬番が無い場合の枠番照合用。"""
    out = defaultdict(list)
    if not LEARNING_CSV.exists():
        return dict(out)
    with open(LEARNING_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            if not rk.startswith(TARGET_DATE):
                continue
            hid = (row.get("horse_id") or "").strip()
            rank_val = row.get("rank", "")
            waku_val = row.get("wakuban", "0")
            try:
                rank = int(float(rank_val))
                wakuban = int(float(waku_val)) if waku_val else 0
            except (ValueError, TypeError):
                continue
            if hid and rank >= 1:
                out[rk].append((hid, rank, wakuban))
    for rk in out:
        out[rk].sort(key=lambda x: x[1])
    return dict(out)


def _create_result_template() -> None:
    """future_races から 2/10 の result テンプレート（rank 未記入）を作成。"""
    rows = []
    with open(FUTURE_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("race_key") or "").strip().startswith(TARGET_DATE):
                rows.append({
                    "race_key": row.get("race_key", ""),
                    "horse_id": row.get("horse_id", ""),
                    "horse_name": row.get("horse_name", ""),
                    "rank": "",
                })
    if not rows:
        return
    RESULT_CSV_OPT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV_OPT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["race_key", "horse_id", "horse_name", "rank"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"結果入力用テンプレート作成: {RESULT_CSV_OPT}  （rank に 1,2,3... の着順を記入後、再実行）")


def _write_improvement_only_report(predictions: dict, total_races: int) -> None:
    """結果がない場合でも改善点レポートを書き出す。"""
    lines = [
        "=" * 60,
        "2/10 予想 vs 結果 照合レポート（結果未登録のため改善点のみ）",
        "=" * 60,
        f"予想対象レース数: {total_races}",
        "",
        "【結果の反映方法】",
        "  1. extract_from_mdb.py を実行し、Race.mdb に 2/10 結果を取り込んだ上で学習データを再出力",
        "  2. または jv_data/result_20260210.csv に race_key,horse_id,rank を用意",
        "",
        "【改善点の洗い出し（一般的な推奨）】",
    ]
    for i, s in enumerate([
        "単勝的中率向上: 1着予想の精度（血統・ロジックスコアの重み、特徴量の見直し）",
        "3着内的中率: 本命・対抗・単穴の選定（AIスコアとLogicスコアのバランス）",
        "複勝安定: ◎馬の安定性（前走成績・馬場適性・斤量）の特徴量強化",
        "過半数で1着を外す場合: 競馬場・距離・馬場状態別のモデル分岐や当日オッズの利用を検討",
        "前走データの鮮度: interval, prev_rank の重み付けを検証",
        "父・母父の競馬場/距離適性: sire_dist_int, bms_dist_int 等の交互項を学習で確認",
    ], 1):
        lines.append(f"  {i}. {s}")
    lines.extend(["", "=" * 60])
    report_text = "\n".join(lines)
    out_txt = SCRIPT_DIR / "jv_data" / "reports" / "accuracy_20260210.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"\n改善点レポート保存: {out_txt}")


def load_results_from_optional_csv() -> dict[str, list[tuple[str, int]]] | None:
    """result_20260210.csv があれば読み、race_key -> [(horse_id, rank)] を返す。"""
    if not RESULT_CSV_OPT.exists():
        return None
    out = defaultdict(list)
    with open(RESULT_CSV_OPT, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            hid = (row.get("horse_id") or "").strip()
            rank_val = row.get("rank", row.get("着順", ""))
            try:
                rank = int(float(rank_val))
            except (ValueError, TypeError):
                continue
            if rk and hid and rank >= 1:
                out[rk].append((hid, rank))
    for rk in out:
        out[rk].sort(key=lambda x: x[1])
    return dict(out) if out else None


def _load_race_meta_for_20260210() -> dict[str, dict[str, str]] | None:
    """改善点7用: learning_dataset から 20260210 の race_key → {racecourse, state} を返す。"""
    if not LEARNING_CSV.exists():
        return None
    out: dict[str, dict[str, str]] = {}
    with open(LEARNING_CSV, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rk = (row.get("race_key") or "").strip()
            if not rk.startswith(TARGET_DATE):
                continue
            if rk not in out:
                out[rk] = {
                    "racecourse": (row.get("racecourse") or "").strip() or "不明",
                    "state": (row.get("state") or "").strip() or "不明",
                }
    return out if out else None


def main() -> int:
    date_display = "2026/02/10"
    report_path = _latest_report_for_date(date_display)
    if not report_path:
        print(f"2/10 予想を含むレポートが見つかりません: {REPORTS_DIR}")
        return 1
    print(f"使用レポート: {report_path.name}")

    predictions = parse_predictions_from_html(report_path, date_display)
    if not predictions:
        print("2/10 の予想がレポート内にありません。")
        return 1
    print(f"予想対象レース数: {len(predictions)}")

    umaban_to_hid = load_future_umaban_to_horse_id()
    results = load_results_from_optional_csv()
    if results is None:
        results = load_results_from_learning()
    # 競馬場別算出用: future_races が無くても枠番で照合するため学習データから (horse_id, rank, wakuban) を取得
    results_with_wakuban = load_results_from_learning_with_wakuban()
    if not results:
        print("2/10 のレース結果がありません。")
        print("  - extract_from_mdb.py で Race.mdb から学習データを再抽出するか、")
        print(f"  - {RESULT_CSV_OPT.name} に race_key,horse_id,rank 形式で結果を用意してください。")
        _write_improvement_only_report(predictions, len(predictions))
        if not RESULT_CSV_OPT.exists():
            _create_result_template()
        return 1
    print(f"結果があるレース数: {len(results)}")

    # 両方にあるレースのみ集計
    common_races = sorted(set(predictions.keys()) & set(results.keys()))
    if not common_races:
        print("予想と結果の両方にあるレースがありません。")
        return 1

    hit_1 = 0
    hit_3 = 0
    hit_fukusho = 0
    total = len(common_races)
    details = []

    use_wakuban_fallback = False
    for rk in common_races:
        pred_list = predictions[rk]
        res_list = results[rk]
        res_rank1_hid = next((hid for hid, r in res_list if r == 1), None)
        res_top3_hids = {hid for hid, r in res_list if 1 <= r <= 3}
        pred_1st_umaban = (pred_list[0][1] if len(pred_list) >= 1 else "").strip()
        pred_hids = []
        for mark, u, name in pred_list[:3]:
            hid = umaban_to_hid.get(rk, {}).get((u or "").strip())
            if hid:
                pred_hids.append(hid)
        pred_1st_hid = pred_hids[0] if len(pred_hids) >= 1 else None
        pred_top3_hids = set(pred_hids[:3])
        pred_umabans = [pred_list[i][1].strip() for i in range(min(3, len(pred_list))) if pred_list[i][1]]

        _1 = 0
        _3 = 0
        _fuku = 0
        if pred_1st_hid is not None:
            _1 = 1 if (pred_1st_hid == res_rank1_hid) else 0
            _3 = 1 if (res_rank1_hid and res_rank1_hid in pred_top3_hids) else 0
            _fuku = 1 if (pred_1st_hid in res_top3_hids) else 0
        elif results_with_wakuban.get(rk) and pred_1st_umaban:
            # future_races に馬番→horse_id が無い場合: 学習データの枠番で照合（馬番≒枠番の近似、9頭以上では誤差あり）
            use_wakuban_fallback = True
            rw = results_with_wakuban[rk]
            res_rank1_wakuban = next((w for _, r, w in rw if r == 1), None)
            res_top3_wakubans = {str(w) for _, r, w in rw if 1 <= r <= 3}
            if res_rank1_wakuban is not None:
                _1 = 1 if pred_1st_umaban == str(res_rank1_wakuban) else 0
                _3 = 1 if str(res_rank1_wakuban) in pred_umabans else 0
            _fuku = 1 if pred_1st_umaban in res_top3_wakubans else 0

        hit_1 += _1
        hit_3 += _3
        hit_fukusho += _fuku
        details.append({
            "race_key": rk,
            "pred_1st": pred_list[0][2] if pred_list else "-",
            "actual_1st_hid": res_rank1_hid,
            "hit_1": _1,
            "hit_3": _3,
            "hit_fukusho": _fuku,
        })

    # 的中率
    rate_1 = (hit_1 / total * 100) if total else 0
    rate_3 = (hit_3 / total * 100) if total else 0
    rate_fukusho = (hit_fukusho / total * 100) if total else 0

    lines = []
    lines.append("=" * 60)
    lines.append("2/10 予想 vs 結果 照合レポート（report_20260209_2202 の予想ベース）")
    lines.append("=" * 60)
    lines.append(f"対象レース数: {total}")
    if use_wakuban_fallback:
        lines.append("（※future_races に2/10が無いため、学習データの枠番で予想馬番と照合。馬番≠枠番のレースは近似です）")
    lines.append("")
    lines.append("【的中率】")
    lines.append(f"  単勝（1着）的中: {hit_1}/{total} = {rate_1:.1f}%")
    lines.append(f"  3着内（1着が予想◎○▲のどれか）: {hit_3}/{total} = {rate_3:.1f}%")
    lines.append(f"  複勝（◎が3着以内）: {hit_fukusho}/{total} = {rate_fukusho:.1f}%")
    lines.append("")
    lines.append("【レース別】")
    for d in details:
        h = "◎" if d["hit_1"] else ("○▲" if d["hit_3"] else "×")
        lines.append(f"  {d['race_key']}: 予想1着={d['pred_1st']} → 単勝={d['hit_1']} 3着内={d['hit_3']} 複勝={d['hit_fukusho']} ({h})")
    # 競馬場別・馬場別の的中率（report_20260209_2202 の予想データベース）
    race_meta = _load_race_meta_for_20260210()
    by_place = defaultdict(lambda: {"hit_1": 0, "hit_3": 0, "hit_fukusho": 0, "total": 0})
    by_state = defaultdict(lambda: {"hit_1": 0, "hit_3": 0, "hit_fukusho": 0, "total": 0})
    for d in details:
        rk = d["race_key"]
        parts = rk.split("_")
        if race_meta and rk in race_meta:
            place = race_meta[rk].get("racecourse", "不明")
            state = race_meta[rk].get("state", "不明")
        else:
            place = CODE_TO_PLACE.get(parts[1], parts[1]) if len(parts) >= 2 else "不明"
            state = "不明"
        for key in ("hit_1", "hit_3", "hit_fukusho"):
            by_place[place][key] += d[key]
            by_state[str(state)][key] += d[key]
        by_place[place]["total"] += 1
        by_state[str(state)]["total"] += 1
    lines.append("")
    lines.append("【競馬場別的中率】")
    place_order = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    for place in sorted(by_place.keys(), key=lambda x: (place_order.index(x) if x in place_order else 99, x)):
        b = by_place[place]
        t = b["total"]
        if t:
            r1 = (b["hit_1"] / t * 100) if t else 0
            r3 = (b["hit_3"] / t * 100) if t else 0
            rf = (b["hit_fukusho"] / t * 100) if t else 0
            lines.append(f"  {place}: 単勝={b['hit_1']}/{t} ({r1:.0f}%)  3着内={b['hit_3']}/{t} ({r3:.0f}%)  複勝={b['hit_fukusho']}/{t} ({rf:.0f}%)")
    lines.append("【馬場状態別】")
    state_name = {"1": "良", "2": "稍重", "3": "重", "4": "不良"}
    for st in sorted(by_state.keys(), key=lambda x: (state_name.get(x, x), x)):
        b = by_state[st]
        t = b["total"]
        if t and st != "不明":
            r1 = (b["hit_1"] / t * 100) if t else 0
            r3 = (b["hit_3"] / t * 100) if t else 0
            rf = (b["hit_fukusho"] / t * 100) if t else 0
            lines.append(f"  {state_name.get(st, st)}: 単勝={b['hit_1']}/{t} ({r1:.0f}%)  3着内={b['hit_3']}/{t} ({r3:.0f}%)  複勝={b['hit_fukusho']}/{t} ({rf:.0f}%)")
    lines.append("")
    lines.append("【改善点の洗い出し】")
    improvements = []
    if rate_1 < 15:
        improvements.append("単勝的中率が低い: 1着予想の精度向上（血統・ロジックスコアの重み、特徴量の見直し）")
    if rate_3 < 35:
        improvements.append("3着内的中率が低い: 本命・対抗・単穴の選定ロジック（AIスコアとLogicスコアのバランス）の見直し")
    if rate_fukusho < 40:
        improvements.append("複勝的中率を上げる: ◎馬の安定性（前走成績・馬場適性・斤量）の特徴量強化")
    # 外したレースの傾向
    missed_races = [d for d in details if not d["hit_1"]]
    if len(missed_races) > total / 2:
        improvements.append("過半数で1着を外している: 競馬場・距離・馬場状態別のモデル分岐や、当日オッズ情報の利用を検討")
    improvements.append("前走データの鮮度: 直前の出走履歴（interval, prev_rank）の重み付けを検証")
    improvements.append("父・母父の競馬場/距離適性: sire_dist_int, bms_dist_int 等の交互項を学習で確認")
    for i, s in enumerate(improvements, 1):
        lines.append(f"  {i}. {s}")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    out_txt = SCRIPT_DIR / "jv_data" / "reports" / "accuracy_20260210.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report_text, encoding="utf-8")
    print(f"\nレポート保存: {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
