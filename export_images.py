# -*- coding: utf-8 -*-
"""
X（Twitter）投稿用に、docs/index.html の全レースを PNG 画像化し、
煽り文を dist/tweets.txt に一括出力するスクリプト。

- Playwright（非同期）でヘッドレスブラウザを起動し、全レースを順に表示してスクリーンショット。
- 画像: dist/images/ に 20260215_Tokyo_01R.png 形式で保存。
- 投稿文: dist/tweets.txt に全レース分を書き出し。

Usage:
  pip install playwright
  playwright install chromium
  python export_images.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HTML_PATH = SCRIPT_DIR / "docs" / "index.html"
OUTPUT_DIR = SCRIPT_DIR / "dist" / "images"
TWEETS_PATH = SCRIPT_DIR / "dist" / "tweets.txt"

# 競馬場名 → ファイル名用 PascalCase（例: Tokyo, Kyoto）
COURSE_TO_PASCAL = {
    "札幌": "Sapporo",
    "函館": "Hakodate",
    "福島": "Fukushima",
    "新潟": "Niigata",
    "東京": "Tokyo",
    "中山": "Nakayama",
    "中京": "Chukyo",
    "京都": "Kyoto",
    "阪神": "Hanshin",
    "小倉": "Kokura",
}


def _course_pascal(course: str) -> str:
    """競馬場名を PascalCase に（20260215_Tokyo_01R.png 用）。"""
    s = COURSE_TO_PASCAL.get(course, course)
    return re.sub(r"[^a-zA-Z0-9]", "", s) or "Race"


def _image_filename(date: str, course: str, race_no: int) -> str:
    """例: 20260215_Tokyo_01R.png"""
    r = str(race_no).zfill(2)
    return f"{date}_{_course_pascal(course)}_{r}R.png"


def _honmei_taikou(race: dict) -> tuple[dict | None, dict | None]:
    """本命（◎）と対抗（○）の馬情報を返す。"""
    horses = race.get("horses") or []
    honmei = next((h for h in horses if h.get("mark") == "◎"), None)
    taikou = next((h for h in horses if h.get("mark") == "○"), None)
    return honmei, taikou


def _confidence_emoji(score: int) -> str:
    """AIスコアに応じた自信度絵文字。"""
    if score >= 90:
        return "🔥🔥🔥"
    if score >= 80:
        return "🔥🔥"
    if score >= 70:
        return "🔥"
    return "✨"


def _tweet_block(race: dict, image_name: str) -> str:
    """1レース分の投稿用テキストブロックを生成。"""
    race_name = race.get("race_name", "")
    honmei, taikou = _honmei_taikou(race)
    score = honmei.get("score", 0) if honmei else 0
    honmei_line = f"本命: ◎ {honmei['horse_name']} (Score: {score})" if honmei else "本命: —"
    taikou_line = f"対抗: ○ {taikou['horse_name']}" if taikou else "対抗: —"
    return (
        f"---\n"
        f"【{race_name} AI予想】\n"
        f"{honmei_line}\n"
        f"{taikou_line}\n"
        f"AI自信度: {_confidence_emoji(score)}\n"
        f"#競馬予想 #AI予想 #{race_name.replace(' ', '')}\n"
        f"(画像: {image_name})\n"
        f"---\n"
    )


async def _run_playwright() -> list[tuple[Path, dict]]:
    """全レースをループし、main 要素をスクリーンショット。"""
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise SystemExit(
            "playwright がインストールされていません。pip install playwright のあと playwright install chromium"
        ) from e

    if not HTML_PATH.exists():
        raise SystemExit(f"HTML が見つかりません: {HTML_PATH}")

    file_url = HTML_PATH.as_uri()
    saved: list[tuple[Path, dict]] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 800, "height": 900})
            await page.goto(file_url, wait_until="networkidle")
            # フォント・アイコン等のレンダリング待ち
            await page.wait_for_timeout(3000)
            await page.wait_for_selector("#sel-date", state="attached", timeout=5000)

            racing_data = await page.evaluate(
                "() => (typeof racingData !== 'undefined' ? racingData : [])"
            )
            if not racing_data:
                print("racingData が取得できませんでした。")
                return saved

            # 全レースを日付・競馬場・R順でソート
            all_races = sorted(
                racing_data,
                key=lambda r: (r.get("date", ""), r.get("course", ""), r.get("race_no", 0)),
            )
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            for race in all_races:
                date_val = race.get("date", "")
                course = race.get("course", "")
                race_no = race.get("race_no", 0)
                race_name = race.get("race_name", "")

                await page.select_option("#sel-date", value=date_val)
                await page.wait_for_timeout(200)
                await page.select_option("#sel-course", value=course)
                await page.wait_for_timeout(200)
                await page.select_option("#sel-race", value=str(race_no))
                await page.wait_for_timeout(400)

                image_name = _image_filename(date_val, course, race_no)
                out_path = OUTPUT_DIR / image_name
                main_el = page.locator("main").first
                await main_el.screenshot(path=str(out_path))
                meta = {
                    "date": date_val,
                    "date_label": race.get("dateLabel", ""),
                    "race_name": race_name,
                    "course": course,
                    "race_no": race_no,
                    "race": race,
                }
                saved.append((out_path, meta))
                print(f"保存: {out_path}")
        finally:
            await browser.close()

    return saved


def _write_tweets_txt(saved: list[tuple[Path, dict]]) -> None:
    """全レース分の煽り文を dist/tweets.txt に書き出す。"""
    TWEETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for out_path, meta in saved:
        race = meta.get("race", {})
        image_name = out_path.name
        blocks.append(_tweet_block(race, image_name))
    with open(TWEETS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print(f"投稿文: {TWEETS_PATH} ({len(blocks)} 件)")


def main() -> int:
    print(f"入力: {HTML_PATH}")
    print(f"画像出力: {OUTPUT_DIR}")
    print(f"投稿文出力: {TWEETS_PATH}")
    saved = asyncio.run(_run_playwright())
    if not saved:
        print("保存した画像はありません。")
        return 1
    _write_tweets_txt(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
