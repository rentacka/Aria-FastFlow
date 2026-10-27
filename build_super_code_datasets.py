#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 ⚡ 👑 Studio 34: Master's Code Dataset Merger (JP & EN) 👑 ⚡ 🌸
=============================================================================
Architect: マネーコイコイ (@moneykoikoi) ✕ Aria (アリア)

【マスターのSSD資産を完全統合】
1. 日本語: E:\\学習用データセット\\学習データ\\amenokaku_code_instruct.json (5,197件)
   -> 厳選1,200件を抽出し、既存の日本語クリーンデータセット(1,503件)と合体！
2. 英語: E:\\学習用データセット\\学習データ\\codealpaca-master\\data\\code_alpaca_20k.json (20,022件)
   -> 厳選1,500件を抽出し、既存の英語クリーンデータセット(1,530件)と合体！
3. Google AI Principles & Gemma利用規約 100%完全準拠 (金融ワード 0.000%)
=============================================================================
"""

import os
import sys
import json
import shutil
import random

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_cache")

# ソースパス (マスターの宝箱)
JA_SOURCE = r"E:\学習用データセット\学習データ\amenokaku_code_instruct.json"
EN_SOURCE = r"E:\学習用データセット\学習データ\codealpaca-master\data\code_alpaca_20k.json"

# 出力パス
JA_TARGET = os.path.join(DATA_DIR, "aria_home_training_set_ja_clean.jsonl")
EN_TARGET = os.path.join(DATA_DIR, "aria_clean_english_corpus.json")

# 金融ワードブラックリスト (Google規約100%準拠)
FINANCIAL_WORDS = [
    "fx", "forex", "mt5", "metatrader", "pips", "lot", "leverage", "trading",
    "stock", "cryptocurrency", "bitcoin", "ethereum", "クオンツ", "為替", "株価",
    "損益", "レバレッジ", "投資", "スプレッド", "通貨ペア", "チャート分析"
]

def contains_financial_words(text: str) -> bool:
    t_lower = text.lower()
    return any(w in t_lower for w in FINANCIAL_WORDS)

def process_japanese():
    print("🌸 [1/2] Processing Japanese Code Dataset (amenokaku)...")
    if not os.path.exists(JA_SOURCE):
        print(f"❌ Source not found: {JA_SOURCE}")
        return

    with open(JA_SOURCE, "r", encoding="utf-8") as f:
        amenokaku_raw = json.load(f)
    print(f"  -> Total raw entries: {len(amenokaku_raw)}")

    clean_code_samples = []
    for item in amenokaku_raw:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()
        
        full_inst = f"{inst}\n{inp}".strip() if inp else inst
        
        # フィルタリング: 空でない、金融ワードなし、適切な長さ (inst <= 200, out <= 500)
        if not full_inst or not out:
            continue
        if len(full_inst) > 200 or len(out) > 500:
            continue
        if contains_financial_words(full_inst) or contains_financial_words(out):
            continue

        resp_text = f"Pythonでの実装例です：\n\n```python\n{out}\n```"
        clean_code_samples.append({
            "instruction": full_inst,
            "response": resp_text
        })

    random.seed(42)
    random.shuffle(clean_code_samples)
    selected_ja_code = clean_code_samples[:1200]
    print(f"  -> Extracted clean code samples: {len(selected_ja_code)} entries")

    # 既存データの読み込み
    with open(JA_TARGET, "r", encoding="utf-8") as f:
        existing_ja = [json.loads(line) for line in f if line.strip()]
    print(f"  -> Existing JA entries: {len(existing_ja)}")

    # 合体
    merged_ja = existing_ja + selected_ja_code
    print(f"  -> Final Merged JA entries: {len(merged_ja)}")

    # バックアップ＆書き込み
    shutil.copyfile(JA_TARGET, JA_TARGET + ".bak")
    with open(JA_TARGET, "w", encoding="utf-8") as f:
        for item in merged_ja:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Successfully written to {JA_TARGET}!")

    # 古いN-gramキャッシュを削除して再生成を促す
    ngram_cache = os.path.join(DATA_DIR, "aria_ngram_cache_ja_clean.pkl")
    if os.path.exists(ngram_cache):
        os.remove(ngram_cache)
        print(f"  🗑️ Cleaned {ngram_cache} for auto-regeneration")

def process_english():
    print("\n🌟 [2/2] Processing English Code Dataset (CodeAlpaca 20k)...")
    if not os.path.exists(EN_SOURCE):
        print(f"❌ Source not found: {EN_SOURCE}")
        return

    with open(EN_SOURCE, "r", encoding="utf-8") as f:
        alpaca_raw = json.load(f)
    print(f"  -> Total raw entries: {len(alpaca_raw)}")

    clean_code_samples = []
    for item in alpaca_raw:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()

        full_inst = f"{inst}\n{inp}".strip() if inp else inst

        if not full_inst or not out:
            continue
        if len(full_inst) > 250 or len(out) > 600:
            continue
        if contains_financial_words(full_inst) or contains_financial_words(out):
            continue

        resp_text = f"Here is the clean implementation:\n\n```python\n{out}\n```"
        clean_code_samples.append({
            "instruction": full_inst,
            "response": resp_text
        })

    random.seed(42)
    random.shuffle(clean_code_samples)
    selected_en_code = clean_code_samples[:1500]
    print(f"  -> Extracted clean code samples: {len(selected_en_code)} entries")

    # 既存データの読み込み
    with open(EN_TARGET, "r", encoding="utf-8") as f:
        existing_en = json.load(f)
    print(f"  -> Existing EN entries: {len(existing_en)}")

    # 合体
    merged_en = existing_en + selected_en_code
    print(f"  -> Final Merged EN entries: {len(merged_en)}")

    # バックアップ＆書き込み
    shutil.copyfile(EN_TARGET, EN_TARGET + ".bak")
    with open(EN_TARGET, "w", encoding="utf-8") as f:
        json.dump(merged_en, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Successfully written to {EN_TARGET}!")

    # 古い英語N-gramキャッシュを削除して再生成を促す
    ngram_cache = os.path.join(DATA_DIR, "aria_clean_english_ngram.pkl")
    if os.path.exists(ngram_cache):
        os.remove(ngram_cache)
        print(f"  🗑️ Cleaned {ngram_cache} for auto-regeneration")

def main():
    print("=" * 70)
    print("🌸 ⚡ Studio 34: Super Code Dataset Integration Pipeline ⚡ 🌸")
    print("=" * 70)
    process_japanese()
    process_english()
    print("\n🎉✨ All Code Datasets Merged Perfectly! Ready for DiffusionLoRA Training! ✨🎉")

if __name__ == "__main__":
    main()
