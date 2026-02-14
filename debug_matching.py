import json
import requests
from bs4 import BeautifulSoup
import re

# ==========================================
# 設定
# ==========================================
# テスト対象：東京 (05)
TARGET_CODE = "05"
TARGET_URL = "https://race.netkeiba.com/top/payback_list.html?kaisai_id=2026050105&kaisai_date=20260214"
JSON_PATH = "docs/weekly_prediction.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def main():
    print("=== データ照合診断を開始します ===")

    # -------------------------------------------------
    # STEP 1: JSON側の鍵チェック
    # -------------------------------------------------
    print("\n[STEP 1] JSONデータの解析チェック")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            races = data.get('races', [])
            print(f"   [OK] JSON読み込み成功: 全{len(races)}レース")

            # 東京(05)のデータがあるかテスト
            tokyo_count = 0
            sample_id = ""
            for r in races:
                rid = str(r.get('race_id', ''))
                # IDの8-9文字目が場所コード
                if len(rid) >= 10:
                    place_code = rid[8:10]
                    if place_code == TARGET_CODE:
                        tokyo_count += 1
                        if not sample_id:
                            sample_id = rid

            if tokyo_count > 0:
                print(f"   [OK] 東京(05)のデータ発見: {tokyo_count}件")
                print(f"   [i] サンプルID: {sample_id} -> 場所コード抽出: '{sample_id[8:10]}'")
            else:
                print("   [NG] 東京(05)のデータが見つかりません！")
                print(f"   [!] 先頭データのIDサンプル: {races[0].get('race_id') if races else 'None'}")
                return  # ここで終了

    except Exception as e:
        print(f"   [NG] JSONエラー: {e}")
        return

    # -------------------------------------------------
    # STEP 2: Web側のドアチェック
    # -------------------------------------------------
    print("\n[STEP 2] Webサイトへのアクセスチェック")
    try:
        resp = requests.get(TARGET_URL, headers=HEADERS)
        resp.encoding = 'EUC-JP'
        print(f"   HTTPステータス: {resp.status_code}")

        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string if soup.title else "No Title"
        print(f"   ページタイトル: {title.strip()}")

        if "403" in str(resp.status_code) or "アクセス" in title:
            print("   [NG] アクセスブロックされています！User-Agentが効いていません。")
            return

        # -------------------------------------------------
        # STEP 3: HTML構造チェック
        # -------------------------------------------------
        print("\n[STEP 3] HTMLテーブルの構造チェック")

        # 可能性のあるクラス名を列挙して探す
        class_candidates = ['RaceList_Box', 'Payback_Table_01', 'Race_Num']
        found_something = False

        for cls in class_candidates:
            items = soup.find_all(class_=cls)
            print(f"   ClassName '{cls}' の検索結果: {len(items)}個")
            if len(items) > 0:
                found_something = True

        if not found_something:
            print("   [NG] レース結果のテーブルが見つかりません。HTML構造が予想と違います。")
            print("   取得したHTMLの先頭500文字:")
            print(soup.prettify()[:500])
        else:
            print("   [OK] テーブル構造を発見しました。照合可能です。")

    except Exception as e:
        print(f"   [NG] Webアクセスエラー: {e}")


if __name__ == "__main__":
    main()
