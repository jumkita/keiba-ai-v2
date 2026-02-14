# -*- coding: utf-8 -*-
"""
週次運用自動化スクリプト（MLOps / PDCA 自動化）

毎週、JRA-VANデータを更新した後に実行し、
1. モデルの自己進化（最新結果での再学習）
2. レポートの自動更新（出馬表から予測HTML生成）
3. 予測ログの蓄積（predict_pipeline 内で prediction_log.csv に追記）
4. 精度検証（過去予測ログ × 結果の照合・印別・条件別成績）

をワンクリックで行います。

使い方:
  python run_weekly_automation.py

運用フロー:
  1. 金曜日: JV-Link 等で JRA-VAN データを更新 → 本バッチ実行（予測＋ログ蓄積）
  2. 月曜日: 結果反映後、再度本バッチ or evaluate_history.py のみ実行 → 精度検証
  3. 火曜日: evaluation_summary / コンソール結果を元に特徴量・重みを調整
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

# プロジェクトルート（このスクリプトの親ディレクトリ）
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / "weekly_operation.log"

# 実行するスクリプト（順序厳守）
# (script_name, description, optional): optional=True は失敗しても次へ進む
STEPS = [
    ("extract_from_mdb.py", "Extraction - レース結果・出馬表をMDBから抽出", False),
    ("preprocess_master.py", "Master Update - ID辞書・マスタ更新", False),
    ("train_model.py", "Model Retraining - 最新データで再学習", False),
    ("predict_pipeline.py", "Prediction - 予測レポート生成・予測ログ追記", False),
    ("evaluate_history.py", "Evaluation - 予測ログ×結果の照合・印別・条件別成績", True),
]


def log(msg: str, also_print: bool = True) -> None:
    """ログファイルと標準出力に書き込む。"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run_script(script_name: str, description: str, optional: bool = False) -> bool:
    """
    サブスクリプトを実行し、出力をログに記録する。
    戻り値: 成功 True / 失敗 False
    optional=True の場合は失敗しても次に進む。
    """
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        log(f"エラー: スクリプトが見つかりません: {script_path}", also_print=True)
        return False if not optional else True

    log(f"===== 開始: {description} ({script_name}) =====")
    start = datetime.datetime.now()

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # 子プロセスの stdout/stderr を UTF-8 に統一
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 10分タイムアウト
            env=env,
        )
        elapsed = (datetime.datetime.now() - start).total_seconds()
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if stdout:
            for line in stdout.splitlines():
                log(f"  [stdout] {line}", also_print=False)
        if stderr:
            for line in stderr.splitlines():
                log(f"  [stderr] {line}", also_print=True)

        if result.returncode != 0:
            log(f"===== 失敗: {script_name} (exit={result.returncode}, {elapsed:.1f}s) =====")
            if optional:
                log(f"  (optional のため続行)")
                return True
            return False

        log(f"===== 完了: {script_name} ({elapsed:.1f}s) =====")
        return True

    except subprocess.TimeoutExpired:
        log(f"エラー: {script_name} がタイムアウト（10分）しました。")
        return False
    except Exception as e:
        log(f"エラー: {script_name} 実行中に例外: {e}")
        return False


def main() -> int:
    log("=" * 60)
    log("週次運用自動化 開始")
    log("=" * 60)

    for step in STEPS:
        script_name = step[0]
        description = step[1]
        optional = step[2] if len(step) >= 3 else False
        log(f"Executing {script_name} ({description})...")
        if not run_script(script_name, description, optional=optional):
            log(f"Error in {script_name}. Aborting.")
            return 1

    log("=" * 60)
    log("All tasks completed. 週次運用自動化 完了")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
