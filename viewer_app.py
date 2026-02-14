# -*- coding: utf-8 -*-
"""
競馬予測ビューア（Headless配信用・データ分析ツール風UI）

起動時に docs/weekly_prediction.json を読み込み、
AIスコア（0-100）をプログレスバーで可視化します。オッズは表示しません。

Usage:
  python viewer_app.py
  python viewer_app.py --url path/to/weekly_prediction.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import flet as ft

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "docs" / "weekly_prediction.json"

# スコア帯の色・ラベル（仕様）
SCORE_HOT = (80, "激熱", ft.Colors.RED_400)
SCORE_HOPE = (70, "有望", ft.Colors.ORANGE_400)
SCORE_CAUTION = (0, "注意", ft.Colors.BLUE_400)


def load_predictions(path_or_url: str) -> dict:
    """JSON ファイルまたは URL パスから予測データを読み込む。"""
    path = Path(path_or_url)
    if not path.is_absolute():
        path = SCRIPT_DIR / path_or_url
    if not path.exists():
        raise FileNotFoundError(f"予測ファイルが見つかりません: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_color(score: int) -> str:
    if score >= SCORE_HOT[0]:
        return SCORE_HOT[2]
    if score >= SCORE_HOPE[0]:
        return SCORE_HOPE[2]
    return SCORE_CAUTION[2]


def score_label(score: int) -> str:
    if score >= SCORE_HOT[0]:
        return SCORE_HOT[1]
    if score >= SCORE_HOPE[0]:
        return SCORE_HOPE[1]
    return SCORE_CAUTION[1]


def build_race_card(race: dict, netkeiba_url: str, page: ft.Page) -> ft.Card:
    """1レース分のカード（馬リスト + スコアプログレスバー + netkeiba ボタン）。"""
    rows = []
    for p in race.get("predictions", []):
        score = int(p.get("score", 0))
        color = score_color(score)
        label = score_label(score)
        rows.append(
            ft.Row(
                [
                    ft.Text(f"{p.get('horse_num', '')}", width=28, text_align=ft.TextAlign.CENTER),
                    ft.Text(p.get("mark", ""), width=24, text_align=ft.TextAlign.CENTER),
                    ft.Text(p.get("horse_name", ""), expand=True, overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Container(
                        content=ft.ProgressBar(value=score / 100, color=color, bgcolor=ft.Colors.SURFACE_VARIANT),
                        expand=True,
                        height=20,
                        border_radius=4,
                    ),
                    ft.Text(f"{score}", width=32, text_align=ft.TextAlign.RIGHT),
                    ft.Text(label, width=48, color=color),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        )
    content = ft.Column(rows, spacing=8)

    def open_netkeiba(_):
        if netkeiba_url:
            page.launch_url(netkeiba_url)

    open_btn = ft.IconButton(
        icon=ft.icons.OPEN_IN_NEW,
        tooltip="WEBでオッズを確認 (netkeiba)",
        on_click=open_netkeiba,
        icon_color=ft.Colors.ON_SURFACE_VARIANT,
    )
    header = ft.Row(
        [
            ft.Text(race.get("race_name", ""), size=18, weight=ft.FontWeight.BOLD),
            ft.Container(expand=True),
            ft.Text("WEBでオッズを確認 (netkeiba)", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            open_btn,
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )
    return ft.Card(
        content=ft.Container(
            content=ft.Column([header, ft.Divider(), content], spacing=12),
            padding=20,
        ),
        elevation=2,
    )


# 起動時オプション（Flet に渡す前にパースして sys.argv から除去）
PREDICTION_JSON_PATH: str = str(DEFAULT_JSON)


def main(page: ft.Page) -> None:
    try:
        data = load_predictions(PREDICTION_JSON_PATH)
    except FileNotFoundError as e:
        page.add(ft.Text(str(e), color=ft.Colors.ERROR))
        return
    except json.JSONDecodeError as e:
        page.add(ft.Text(f"JSON の読み込みに失敗しました: {e}", color=ft.Colors.ERROR))
        return

    # Dark theme・金融分析ツール風
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme_seed=ft.Colors.INDIGO,
        brightness=ft.Brightness.DARK,
    )
    page.dark_theme = page.theme
    page.title = "競馬予測ビューア（AIスコア）"
    page.padding = 0
    page.spacing = 0

    races = data.get("races", [])
    netkeiba_url = data.get("netkeiba_url", "")
    target_date = data.get("target_date", "")
    updated_at = data.get("updated_at", "")

    # 左サイドバー: レース一覧
    race_list_items = []
    selected_race_index = [0]

    def on_race_select(idx: int):
        selected_race_index[0] = idx
        detail.content = ft.Container(
            content=ft.Column(
                [build_race_card(races[idx], netkeiba_url, page)],
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
            padding=16,
        )
        detail.update()

    for i, r in enumerate(races):
        idx = i
        race_list_items.append(
            ft.ListTile(
                title=ft.Text(r.get("race_name", f"Race {i+1}")),
                data=idx,
                on_click=lambda e, i=idx: on_race_select(i),
            )
        )

    sidebar = ft.Container(
        content=ft.Column(
            [
                ft.Text("レース一覧", size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.ListView(race_list_items, expand=True, spacing=2),
            ],
            expand=True,
        ),
        width=220,
        padding=12,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        border=ft.border.only(right=ft.BorderSide(1, ft.Colors.OUTLINE)),
    )

    # 右メイン: 詳細カード（ヘッダー + 選択中レースのカード）
    detail = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(f"対象日: {target_date}", size=14, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Text(f"更新: {updated_at}", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                    ],
                    spacing=16,
                ),
                ft.Divider(),
                build_race_card(races[0], netkeiba_url, page) if races else ft.Text("レースがありません"),
            ],
            scroll=ft.ScrollMode.AUTO,
        ),
        expand=True,
        padding=16,
    )

    layout = ft.Row([sidebar, detail], expand=True)
    page.add(layout)


if __name__ == "__main__":
    import argparse as _argparse
    _p = _argparse.ArgumentParser()
    _p.add_argument("--url", default=str(DEFAULT_JSON), help="weekly_prediction.json のパス")
    _args, _rest = _p.parse_known_args()
    globals()["PREDICTION_JSON_PATH"] = _args.url
    # Flet 用に残りの引数のみ渡す（--url は除去済み）
    sys.argv = [sys.argv[0]] + _rest
    ft.app(target=main)
