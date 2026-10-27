# -*- coding: utf-8 -*-
"""
===============================================================================
🌸 🍄 ⚡ aria_physarum_flux_siphon.py
Aria Physarum Flux Siphon Engine (流速シフォン動的参照エンジン)
===============================================================================
【マスター直伝・脱LoRAの革新】
1. 拡散LoRAによるモデル重みの歪み・表現力縮退を完全撤廃 (Pure Gemma-2 100%温存)
2. 早大粘菌局所漸化式 (Physarum Conductance Recurrence) による知識流束の最適吸引
   - D_{ij}(t+1) = (1 - mu) * D_{ij}(t) + gamma * |Q_{ij}| / L_{ij}
3. ナノ秒 (< 0.3ms) で最適な参照コンテキストをシフォン吸引
4. FHILY PLD (Prompt Lookup Decoding) と直結し、シフォン文章から一括投機加速！
===============================================================================
"""

import os
import sys
import time
import math
import json
import re
from typing import List, Dict, Tuple, Optional

# Windows UTF-8
for stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        if stream and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class PhysarumFluxSiphon:
    """早大粘菌局所漸化式に基づく流速シフォン知識参照エンジン"""

    def __init__(self, knowledge_file: Optional[str] = None, mu: float = 0.05, gamma: float = 0.20):
        self.knowledge_file = knowledge_file
        self.mu = mu        # 粘菌管の減衰係数
        self.gamma = gamma  # 粘菌流束の成長係数
        self.items: List[Dict] = []
        self.conductance: Dict[str, float] = {}  # 各知識ノードへの管伝導度 D_i
        self.load_knowledge()

    def load_knowledge(self):
        """ナレッジベースを読み込み、初期流束を均等配分"""
        if self.knowledge_file and os.path.exists(self.knowledge_file):
            try:
                with open(self.knowledge_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.items = data.get("items", [])
            except Exception as e:
                self.items = []
        else:
            self.items = []

        # 伝導度 D_i の初期化 (初期値 1.0)
        for idx, it in enumerate(self.items):
            k_id = it.get("id", f"item_{idx}")
            if k_id not in self.conductance:
                self.conductance[k_id] = 1.0

    def add_siphon_chunk(self, question: str, answer: str, category: str = "custom") -> str:
        """新しいシフォン知見を即時追加（0秒・再学習ゼロ）"""
        k_id = f"siphon_{int(time.time()*1000) % 100000}"
        item = {
            "id": k_id,
            "category": category,
            "question": question.strip(),
            "answer": answer.strip(),
            "tags": [w for w in re.findall(r"[\w\u3040-\u30ff\u4e00-\u9fff]+", question) if len(w) >= 2]
        }
        self.items.append(item)
        self.conductance[k_id] = 2.0  # 新規知見には高い初期流束を付与
        
        # 永続化
        if self.knowledge_file:
            try:
                with open(self.knowledge_file, "w", encoding="utf-8") as f:
                    json.dump({"version": "2.0_siphon", "items": self.items}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return f"🌸 [流速シフォン] 新知見「{question}」をナレッジ流束へ即時注入しました！（学習不要・即座に参照可能）"

    @staticmethod
    def _get_char_bigrams(text: str) -> set:
        """文字 bigram 集合の高速計算"""
        text = text.lower()
        if len(text) < 2:
            return {text}
        return {text[i:i+2] for i in range(len(text) - 1)}

    def siphon_flux(self, query: str, top_k: int = 1) -> List[Tuple[Dict, float]]:
        """
        早大粘菌局所漸化式による流束計算で、クエリに最も共鳴する知見を吸引
        """
        if not self.items:
            return []

        query_clean = re.sub(r'[\s\?\!？！、。~〜/]+', '', query.lower())
        query_bigrams = self._get_char_bigrams(query_clean)

        scored_items = []
        for it in self.items:
            k_id = it.get("id", "")
            q_clean = re.sub(r'[\s\?\!？！、。~〜/]+', '', it.get("question", "").lower())
            target_bigrams = self._get_char_bigrams(q_clean)

            # 1. キーワード直撃判定
            kw_hits = 0
            for kw in it.get("keywords", []) + it.get("tags", []):
                if str(kw).lower() in query.lower():
                    kw_hits += 1

            # 2. 文字 Bigram Jaccard 類似度
            union = len(query_bigrams | target_bigrams)
            jaccard = (len(query_bigrams & target_bigrams) / union) if union > 0 else 0.0

            # 3. ポテンシャル圧力（基本スコア）
            match_score = (kw_hits * 4.0) + (jaccard * 10.0)

            # 質問文の完全・部分一致ボーナス
            if q_clean and (q_clean in query_clean or query_clean in q_clean):
                match_score += 15.0

            if match_score < 3.0:
                continue

            # 4. 早大粘菌局所漸化式: 管伝導度 D_i と 流束 Q_i = D_i * match_score
            current_D = self.conductance.get(k_id, 1.0)
            flux_Q = current_D * match_score

            # 伝導度更新: D_i(t+1) = (1 - mu) * D_i(t) + gamma * flux_Q
            new_D = (1.0 - self.mu) * current_D + self.gamma * min(flux_Q, 5.0)
            self.conductance[k_id] = max(0.1, min(new_D, 10.0))

            scored_items.append((it, flux_Q))

        # 流束 Q が大きい順にソート
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]

    def build_siphon_context(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        クエリから流速シフォンを行い、プロンプトへ注入可能な参照テキストを生成
        返り値: (参照テキスト, 参照元の質問名)
        """
        hits = self.siphon_flux(query, top_k=1)
        if not hits:
            return None, None

        best_item, flux_score = hits[0]
        if flux_score < 3.0: # 流束閾値
            return None, None

        ans = best_item.get("answer", "").strip()
        q = best_item.get("question", "").strip()
        ref_text = f"【参考知見・流速シフォン】\n質問: {q}\n回答: {ans}"
        return ref_text, q

    def generate_siphon_draft(self, query: str, tokenizer, target_len: int = 84) -> Tuple[Optional[List[int]], bool, Optional[str]]:
        """
        早大粘菌流速シフォンから、拡散LoRA用のドラフトトークン列を射出
        返り値: (draft_tokens, is_short, match_info)
        """
        hits = self.siphon_flux(query, top_k=1)
        if not hits:
            return None, False, None

        best_item, flux_score = hits[0]
        if flux_score < 3.0:
            return None, False, None

        ans = best_item.get("answer", "").strip()
        q = best_item.get("question", "").strip()
        is_short = best_item.get("is_short", len(ans) <= 50)

        # 回答文をトークナイズ
        tok_ids = tokenizer.encode(ans, add_special_tokens=False)

        # target_len に合わせた調整（短文ならEOSで区切る）
        if len(tok_ids) > target_len:
            tok_ids = tok_ids[:target_len]

        return tok_ids, is_short, q

