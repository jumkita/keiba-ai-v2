import json
import requests
from bs4 import BeautifulSoup
import re

# ==========================================
# 1. 設定エリア
# ==========================================
TARGETS = {
    "05": { "name": "Tokyo",  "url": "https://race.netkeiba.com/top/payback_list.html?kaisai_id=2026050105&kaisai_date=20260214" },
    "08": { "name": "Kyoto",  "url": "https://race.netkeiba.com/top/payback_list.html?kaisai_id=2026080205&kaisai_date=20260214" },
    "10": { "name": "Kokura", "url": "https://race.netkeiba.com/top/payback_list.html?kaisai_id=2026100107&kaisai_date=20260214" }
}

JSON_PATH = "docs/weekly_prediction.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# ==========================================
# 2. ユーティリティ
# ==========================================
def clean_money(text):
    if not text: return 0
    clean = re.sub(r'[^\d]', '', text)
    return int(clean) if clean else 0

def clean_text_list(html_td):
    """ <br> 区切りをリスト化 """
    text = str(html_td).replace('<br>', ',').replace('<br/>', ',')
    soup = BeautifulSoup(text, 'html.parser')
    plain = soup.get_text()
    return [x.strip() for x in plain.replace('\n', ',').split(',') if x.strip()]

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("=== 2026/02/14 Settlement Report (Anchor Mode) ===")

    # JSON読み込み
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            json_races = raw_data.get('races', []) if isinstance(raw_data, dict) else raw_data
            print(f"JSON Loaded: {len(json_races)} races found.")
    except Exception as e:
        print(f"JSON Error: {e}")
        return

    balance_sheet = {
        "A_Win": {"invest": 0, "return": 0},
        "B_Place": {"invest": 0, "return": 0},
        "C_Wide": {"invest": 0, "return": 0}
    }

    for place_code, info in TARGETS.items():
        print(f"\nProcessing {info['name']} ({place_code})...")
        
        target_races = {}
        for r in json_races:
            rid = str(r.get('race_id', ''))
            if len(rid) >= 10 and rid[8:10] == place_code:
                r_no = int(rid[10:12])
                target_races[r_no] = r

        if not target_races:
            print("   -> No prediction data.")
            continue

        try:
            resp = requests.get(info['url'], headers=HEADERS)
            resp.encoding = 'EUC-JP'
            soup = BeautifulSoup(resp.text, 'html.parser')
            print(f"   -> Page Loaded. Searching for race tables...")

            # === アンカー探索ロジック ===
            # 1Rから12Rまで順番に探す
            found_races = 0
            
            for r_no in range(1, 13):
                if r_no not in target_races:
                    continue

                # 1. "1R", "2R" という文字（アンカー）を探す
                #    厳密には "1R " や "1R\n" なども含むため、正規表現で検索
                #    dtタグ, hタグ, divタグなどが候補
                
                anchor_text = f"{r_no}R"
                
                # テキストを含む要素をすべて探す
                candidates = soup.find_all(string=re.compile(rf"^\s*{r_no}R"))
                
                target_table = None
                
                # 候補の中から「直後にテーブルがあるもの」を探す
                for cand in candidates:
                    # 親要素を辿って、その近くに table があるか確認
                    parent = cand.parent
                    
                    # パターン1: 親の次の要素が table (div.Race_Num -> table)
                    # パターン2: 親の親の次の要素が table
                    # パターン3: 親のコンテナ内に table がある
                    
                    # 最も確実なのは「親要素の近くにある table」を見つけること
                    # 少し範囲を広げて親の親(grandparent)から table を探す
                    container = parent.find_parent('div', class_=re.compile("RaceList_Box"))
                    
                    if container:
                        # コンテナが見つかれば、その中の table を取得
                        t = container.find('table')
                        if t and '単勝' in t.get_text():
                            target_table = t
                            break
                    else:
                        # コンテナが見つからない場合、親の兄弟要素(siblings)から探す
                        # (古いレイアウト対応)
                        curr = parent
                        for _ in range(5): # 5回くらい親を遡るか兄弟を見る
                            if not curr: break
                            if curr.name == 'table' and '単勝' in curr.get_text():
                                target_table = curr
                                break
                            # 次の兄弟を見る
                            sib = curr.find_next_sibling('table')
                            if sib and '単勝' in sib.get_text():
                                target_table = sib
                                break
                            curr = curr.parent
                        if target_table: break

                if not target_table:
                    # 見つからなかった場合、単純に全てのテーブルから r_no 番目を推測するのは危険なのでスキップ
                    # ただし、debug用にログを出す
                    # print(f"   ⚠️ R{r_no} table not found via anchor '{anchor_text}'")
                    continue

                found_races += 1
                
                # === ここから集計ロジック（前回と同じ） ===
                ai_data = target_races[r_no]
                preds = ai_data.get('predictions', [])
                if not preds: continue

                honme = preds[0]
                honme_num = str(honme.get('horse_num'))
                box_5_nums = [str(p.get('horse_num')) for p in preds[:5]]

                win_amt = 0
                place_amt = 0
                wide_amt = 0

                rows = target_table.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    if not th: continue
                    header = th.get_text()
                    tds = row.find_all('td')
                    if len(tds) < 2: continue

                    r_nums = clean_text_list(tds[0])
                    r_pays = clean_text_list(tds[1])

                    # 単勝
                    if '単勝' in header:
                        if honme_num in r_nums:
                            try:
                                idx = r_nums.index(honme_num)
                                win_amt = clean_money(r_pays[idx])
                            except: pass
                    # 複勝
                    elif '複勝' in header:
                        if honme_num in r_nums:
                            try:
                                idx = r_nums.index(honme_num)
                                place_amt = clean_money(r_pays[idx])
                            except: pass
                    # ワイド
                    elif 'ワイド' in header:
                        for k, pair_str in enumerate(r_nums):
                            nums = re.findall(r'\d+', pair_str)
                            if len(nums) == 2:
                                p1, p2 = nums
                                if (p1 in box_5_nums) and (p2 in box_5_nums):
                                    if k < len(r_pays):
                                        wide_amt += clean_money(r_pays[k])

                balance_sheet["A_Win"]["invest"] += 100
                balance_sheet["A_Win"]["return"] += win_amt
                balance_sheet["B_Place"]["invest"] += 100
                balance_sheet["B_Place"]["return"] += place_amt
                balance_sheet["C_Wide"]["invest"] += 1000
                balance_sheet["C_Wide"]["return"] += wide_amt

                hit_log = []
                if win_amt: hit_log.append(f"Win:{win_amt}")
                if place_amt: hit_log.append(f"Place:{place_amt}")
                if wide_amt: hit_log.append(f"Wide:{wide_amt}")
                
                if hit_log:
                    print(f"   R{r_no} Hit! (No.{honme_num}): {' '.join(hit_log)}")

        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "="*40)
    print("   Total Balance Sheet")
    print("="*40)
    for key, name in [("A_Win", "Win"), ("B_Place", "Place"), ("C_Wide", "Wide BOX")]:
        d = balance_sheet[key]
        inv = d["invest"]
        ret = d["return"]
        profit = ret - inv
        roi = (ret / inv * 100) if inv > 0 else 0
        icon = "🔥" if roi > 100 else "💀"
        print(f"[{icon}] {name}")
        print(f"   Invest: {inv:,} yen  Return: {ret:,} yen")
        print(f"   Profit: {profit:+,} yen  (ROI: {roi:.1f}%)")
        print("-" * 30)

if __name__ == "__main__":
    main()