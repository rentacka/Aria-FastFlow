#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 ⚡ 👑 Studio 34: Clean Japanese Programming & Coding Dataset Builder 👑 ⚡ 🌸
=============================================================================
Google AI Principles & Gemma利用規約 100%完全準拠 (金融ワードゼロ)
日常的なPythonコーディング対話・基本関数・アルゴリズム・データ構造の対話ペアを生成し、
既存の日本語クリーンデータセットに合体させます。
=============================================================================
"""

import json
import os
import shutil
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_cache")
ORIG_DATA_PATH = os.path.join(DATA_DIR, "aria_home_training_set_ja_clean.jsonl")
BACKUP_PATH = os.path.join(DATA_DIR, "aria_home_training_set_ja_clean.orig.jsonl")

# 代表的なPythonコーディング対話テンプレート (金融ワード完全ゼロ)
CODE_TEMPLATES = [
    # 1. 2つの数値を足し算する関数
    (
        ["Pythonで2つの数値を足し算する関数を書いて", "2つの数を足すPythonのコードを教えて", "Pythonで加算する関数を作って", "足し算の関数をPythonで実装してください"],
        "Pythonで2つの数値を足し算する関数の実装例です：\n\n```python\ndef add(a: float, b: float) -> float:\n    \"\"\"2つの数値を加算して返します。\"\"\"\n    return a + b\n```\n\n引数 `a` と `b` を受け取り、その合計値を `return` で返します。"
    ),
    # 2. リストの重複を削除
    (
        ["Pythonでリストの重複を削除するコードを教えて", "リストから重複を取り除く関数を書いて", "Pythonでリストのユニークな要素だけを残すには？", "重複要素を削除するPythonコード"],
        "リストの重複を削除する最も簡潔な方法は `set` を利用することです：\n\n```python\ndef remove_duplicates(lst: list) -> list:\n    \"\"\"順序を保持しながら重複を削除します。\"\"\"\n    return list(dict.fromkeys(lst))\n```\n\n`dict.fromkeys()` を使うことで、要素の登場順序を保持したまま重複を高速に削除できます。"
    ),
    # 3. 偶数判定
    (
        ["Pythonで偶数かどうか判定する関数を書いて", "偶数をチェックするPythonコードを教えて", "数値が偶数か調べる関数"],
        "数値が偶数かどうかを判定する関数です：\n\n```python\ndef is_even(n: int) -> bool:\n    \"\"\"数値が偶数であればTrue、奇数であればFalseを返します。\"\"\"\n    return n % 2 == 0\n```\n\n剰余演算子 `%` を使い、2で割った余りが0かどうかを判定しています。"
    ),
    # 4. 階乗の計算
    (
        ["Pythonで階乗を計算する関数を書いて", "階乗を求めるPythonコードを教えて", "再帰またはループで階乗を計算する関数"],
        "階乗を計算する関数です：\n\n```python\ndef factorial(n: int) -> int:\n    \"\"\"nの階乗 (n!) を計算します。\"\"\"\n    if n < 0:\n        raise ValueError(\"負の数の階乗は定義されていません\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```\n\n`math.factorial(n)` を利用することでも同様に高速に計算できます。"
    ),
    # 5. 文字列の反転
    (
        ["Pythonで文字列を逆順にするコードを教えて", "文字列を反転させる関数を書いて", "Pythonで文字を逆から並べる方法"],
        "文字列を反転させるPythonコードです：\n\n```python\ndef reverse_string(s: str) -> str:\n    \"\"\"文字列を逆順にして返します。\"\"\"\n    return s[::-1]\n```\n\nスライス `[::-1]` を使うことで、シンプルかつ高速に逆順の文字列を取得できます。"
    ),
    # 6. リストの平均値を計算
    (
        ["Pythonでリストの平均値を計算する関数を書いて", "数値リストの平均を求めるコードを教えて", "平均値を計算するPython関数"],
        "リストの平均値を計算する関数の例です：\n\n```python\ndef calculate_average(numbers: list) -> float:\n    \"\"\"数値リストの平均値を返します。\"\"\"\n    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)\n```\n\n空リストの場合はゼロ除算を避けるガード節を設けています。"
    ),
    # 7. 最大値の探索
    (
        ["Pythonでリスト内の最大値を見つける関数を書いて", "リストから最大の要素を取得するコードを教えて", "最大値を求める関数"],
        "リスト内の最大値を求める関数です：\n\n```python\ndef find_max(numbers: list):\n    \"\"\"リスト内の最大値を返します。\"\"\"\n    if not numbers:\n        return None\n    max_val = numbers[0]\n    for num in numbers[1:]:\n        if num > max_val:\n            max_val = num\n    return max_val\n```\n\n組み込み関数 `max(numbers)` を使うと1行で取得することも可能です。"
    ),
    # 8. 回文判定
    (
        ["Pythonで回文かどうか判定する関数を書いて", "回文をチェックするコードを教えて", "文字列が回文か判定する関数"],
        "文字列が回文（前から読んでも後ろから読んでも同じ）かを判定する関数です：\n\n```python\ndef is_palindrome(text: str) -> bool:\n    \"\"\"回文であればTrueを返します。\"\"\"\n    cleaned = \"\".join(c.lower() for c in text if c.isalnum())\n    return cleaned == cleaned[::-1]\n```\n\n空白や記号を除去し、小文字に揃えてから比較しています。"
    ),
    # 9. 素数判定
    (
        ["Pythonで素数判定を行う関数を書いて", "素数かどうかをチェックするPythonコードを教えて", "素数を判定する関数"],
        "素数を判定する効率的な関数の実装例です：\n\n```python\ndef is_prime(n: int) -> bool:\n    \"\"\"与えられた整数が素数かどうかを判定します。\"\"\"\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n```\n\n$\\sqrt{n}$ までの試し割りを行うことで高速に判定できます。"
    ),
    # 10. ファイルの読み込み
    (
        ["Pythonでテキストファイルを安全に読み込むコードを教えて", "ファイルを開いて中身を読むPythonの書き方", "テキストファイルを読み込む関数を書いて"],
        "ファイルを安全に読み込むには `with` 構文を使用します：\n\n```python\ndef read_text_file(file_path: str) -> str:\n    \"\"\"テキストファイルをUTF-8で安全に読み込みます。\"\"\"\n    with open(file_path, \"r\", encoding=\"utf-8\") as f:\n        return f.read()\n```\n\n`with` 文を使うことで、処理完了時やエラー時にも自動的にファイルがクローズされます。"
    ),
    # 11. JSONファイルの読み書き
    (
        ["PythonでJSONファイルを読み書きするコードを教えて", "JSONを保存して読み込むPythonの書き方", "JSONの処理コードを書いて"],
        "JSONデータの保存と読み込みを行う関数です：\n\n```python\nimport json\n\ndef save_json(data: dict, file_path: str) -> None:\n    \"\"\"辞書データをJSONファイルに書き込みます。\"\"\"\n    with open(file_path, \"w\", encoding=\"utf-8\") as f:\n        json.dump(data, f, ensure_ascii=False, indent=2)\n\ndef load_json(file_path: str) -> dict:\n    \"\"\"JSONファイルを辞書として読み込みます。\"\"\"\n    with open(file_path, \"r\", encoding=\"utf-8\") as f:\n        return json.load(f)\n```"
    ),
    # 12. 辞書の値によるソート
    (
        ["Pythonで辞書を値でソートするコードを教えて", "辞書を値の大きさ順に並び替える方法", "辞書の値でソートする関数を書いて"],
        "辞書を値（バリュー）でソートするコード例です：\n\n```python\ndef sort_dict_by_value(d: dict, reverse: bool = False) -> dict:\n    \"\"\"辞書を値の昇順または降順でソートして返します。\"\"\"\n    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))\n```\n\n`sorted()` に `key=lambda item: item[1]` を指定することで値基準のソートが可能です。"
    ),
    # 13. フィボナッチ数列
    (
        ["Pythonでフィボナッチ数列を生成する関数を書いて", "フィボナッチ数列を求めるPythonコードを教えて", "フィボナッチ数列のジェネレータ"],
        "指定した要素数までのフィボナッチ数列を生成する関数です：\n\n```python\ndef fibonacci(n: int) -> list:\n    \"\"\"n個のフィボナッチ数をリストで返します。\"\"\"\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq\n```"
    ),
    # 14. リスト内包表記
    (
        ["Pythonのリスト内包表記の使い方を教えて", "内包表記で偶数の二乗リストを作るコード", "リスト内包表記のサンプルコードを書いて"],
        "リスト内包表記を使うと、簡潔に新しいリストを生成できます：\n\n```python\n# 0から9までの偶数の二乗リストを生成する例\nsquares = [x ** 2 for x in range(10) if x % 2 == 0]\nprint(squares)  # [0, 4, 16, 36, 64]\n```\n\n`for` ループを回すよりも高速で、可読性高く記述できます。"
    ),
    # 15. スタッククラスの実装
    (
        ["Pythonでシンプルなスタック（Stack）クラスを作って", "スタックのデータ構造を実装するコードを教えて", "PythonでStackクラスを書いて"],
        "リストを活用したシンプルなスタッククラスの実装例です：\n\n```python\nclass Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item) -> None:\n        self._items.append(item)\n\n    def pop(self):\n        if not self.is_empty():\n            return self._items.pop()\n        raise IndexError(\"空のスタックからpopできません\")\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n```"
    ),
    # 16. キュークラスの実装
    (
        ["Pythonでキュー（Queue）を実装するコードを教えて", "キューのクラスをPythonで書いて", "collections.dequeを使ったキューの実装"],
        "`collections.deque` を使用した効率的なキューの実装例です：\n\n```python\nfrom collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._items = deque()\n\n    def enqueue(self, item) -> None:\n        self._items.append(item)\n\n    def dequeue(self):\n        if not self.is_empty():\n            return self._items.popleft()\n        raise IndexError(\"空のキューからdequeueできません\")\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n```"
    ),
    # 17. 二分探索
    (
        ["Pythonで二分探索（バイナリサーチ）のコードを書いて", "ソート済みリストを二分探索する関数を教えて", "二分探索アルゴリズムのPython実装"],
        "ソート済みリストに対する二分探索の実装例です：\n\n```python\ndef binary_search(arr: list, target: int) -> int:\n    \"\"\"targetが見つかった場合はそのインデックス、なければ-1を返します。\"\"\"\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n```"
    ),
    # 18. クイックソート
    (
        ["Pythonでクイックソートを実装するコードを書いて", "クイックソートのアルゴリズムを教えて", "Pythonによるソートアルゴリズムの実装"],
        "Pythonで直感的に書けるクイックソートの実装例です：\n\n```python\ndef quick_sort(arr: list) -> list:\n    \"\"\"クイックソートにより昇順に並び替えます。\"\"\"\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```"
    ),
    # 19. エラー処理（try-except）
    (
        ["Pythonで例外処理（try-except）を行うコードの書き方を教えて", "ゼロ除算エラーを防ぐ例外処理コードを書いて", "try exceptの書き方例"],
        "ゼロ除算を防ぐ安全な割り算関数の例です：\n\n```python\ndef safe_divide(a: float, b: float) -> float:\n    \"\"\"ゼロ除算を安全にキャッチして処理します。\"\"\"\n    try:\n        return a / b\n    except ZeroDivisionError:\n        print(\"警告: ゼロ除算が発生しました。0.0を返します。\")\n        return 0.0\n```"
    ),
    # 20. 単語出現頻度のカウント
    (
        ["Pythonで文章中の単語の出現回数を数えるコードを教えて", "テキストの単語頻度をカウントする関数を書いて", "Counterを使った単語集計"],
        "`collections.Counter` を使って単語の出現頻度を集計するコード例です：\n\n```python\nfrom collections import Counter\n\ndef count_words(text: str) -> dict:\n    \"\"\"文章内の単語出現頻度を辞書形式で返します。\"\"\"\n    words = text.lower().split()\n    return dict(Counter(words))\n```"
    )
]

def main():
    print("🌸 ⚡ Building Clean Japanese Code Dataset (Zero Financial Risk)...")
    
    # 既存データのバックアップ
    if not os.path.exists(BACKUP_PATH):
        print(f"📦 Backing up original dataset to {BACKUP_PATH}...")
        shutil.copyfile(ORIG_DATA_PATH, BACKUP_PATH)
    
    # 既存データの読み込み
    with open(BACKUP_PATH, "r", encoding="utf-8") as f:
        orig_entries = [json.loads(line) for line in f if line.strip()]
    
    print(f"📄 Original entries count: {len(orig_entries)}")
    
    # コーディング対話ペアの生成（バリエーション展開で約200〜300件作成）
    code_entries = []
    
    # 丁寧な前置き・挨拶のバリエーション
    prefixes = ["", "アリアちゃん、", "ねえアリアちゃん、", "質問です！", "教えてほしいな。"]
    suffixes = ["", " お願いします！", " よろしくね(๑•̀ㅂ•́)و✧"]
    
    for prompts, resp in CODE_TEMPLATES:
        for p in prompts:
            # 基本形
            code_entries.append({"instruction": p, "response": resp})
            # バリエーション
            for pre in ["アリアちゃん、", "ねえ、"]:
                code_entries.append({"instruction": pre + p, "response": resp})
    
    print(f"💡 Generated new code entries: {len(code_entries)}")
    
    # 合体
    merged = orig_entries + code_entries
    print(f"🚀 Merged total entries: {len(merged)}")
    
    # 保存
    with open(ORIG_DATA_PATH, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Successfully written to {ORIG_DATA_PATH}!")
    
    # 古いN-gramキャッシュを削除して次回自動再生成を促す
    ngram_cache = os.path.join(DATA_DIR, "aria_ngram_cache_ja_clean.pkl")
    if os.path.exists(ngram_cache):
        os.remove(ngram_cache)
        print(f"🗑️ Cleaned old N-gram cache: {ngram_cache} (will regenerate automatically)")

if __name__ == "__main__":
    main()
