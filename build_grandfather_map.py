# -*- coding: utf-8 -*-
"""
学習データから「馬（子）→ 父」の関係を抽出し、
父用・母父用の遡及マッピングを保存する。

Outputs:
  - config/sire_to_grandfather.csv (sire_id, grandfather_sire_id)
  - config/bms_to_father.csv (bms_id, bms_father_id)

使い方:
  python build_grandfather_map.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"
OUTPUT_SIRE = SCRIPT_DIR / "config" / "sire_to_grandfather.csv"
OUTPUT_BMS = SCRIPT_DIR / "config" / "bms_to_father.csv"


def _to_str(v, default: str = "") -> str:
    if v is None or v == "":
        return default
    s = str(v).strip()
    return s if s else default


def main() -> int:
    if not INPUT.exists():
        print(f"エラー: {INPUT} が見つかりません。", file=sys.stderr)
        return 1

    pairs = set()
    with open(INPUT, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "horse_id" not in (reader.fieldnames or []) or "sire_id" not in (reader.fieldnames or []):
            print("エラー: learning_dataset.csv に horse_id または sire_id 列がありません。", file=sys.stderr)
            return 1
        for row in reader:
            horse = _to_str(row.get("horse_id"))
            sire = _to_str(row.get("sire_id"))
            if not horse or not sire or horse == "0" or sire == "0":
                continue
            pairs.add((horse, sire))

    OUTPUT_SIRE.parent.mkdir(parents=True, exist_ok=True)
    sorted_pairs = sorted(pairs)
    n = len(sorted_pairs)

    with open(OUTPUT_SIRE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sire_id", "grandfather_sire_id"])
        for horse_id, father_id in sorted_pairs:
            w.writerow([horse_id, father_id])

    with open(OUTPUT_BMS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bms_id", "bms_father_id"])
        for horse_id, father_id in sorted_pairs:
            w.writerow([horse_id, father_id])

    print(f"父マップ: {n} 件, 母父マップ: {n} 件 を保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
