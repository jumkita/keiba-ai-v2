# -*- coding: utf-8 -*-
"""
learning_dataset.csv の state と wakuban を検査し、
Feature Importance 0.0 になる原因を特定する。
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "jv_data" / "learning_dataset.csv"


def audit_features():
    print("=== Feature Audit Start ===")
    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
        print(f"Total rows: {len(df)}")

        # 1. Wakuban Check
        print("\n[Wakuban Analysis]")
        print(f"Unique Values: {sorted(df['wakuban'].unique())}")
        print("Count by Wakuban:")
        print(df['wakuban'].value_counts().sort_index())
        print("Mean Rank by Wakuban:")
        print(df.groupby('wakuban')['rank'].mean())
        print("Variance of mean rank:", df.groupby('wakuban')['rank'].mean().var())

        # 2. State Check
        print("\n[State Analysis]")
        print(f"Unique Values: {sorted(df['state'].unique())}")
        print("Count by State:")
        print(df['state'].value_counts().sort_index())
        print("Mean Rank by State:")
        print(df.groupby('state')['rank'].mean())
        print("Variance of mean rank:", df.groupby('state')['rank'].mean().var())

        # 3. 複勝率 (is_place) との相関（モデルの目的変数）
        df['is_place'] = (df['rank'] <= 3).astype(int)
        print("\n[Place Rate by State] (target=rank<=3)")
        print(df.groupby('state')['is_place'].mean())
        print("\n[Place Rate by Wakuban]")
        print(df.groupby('wakuban')['is_place'].mean())

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    audit_features()
