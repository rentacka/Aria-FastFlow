"""
Aria N-gram Step Compressor (Wavefront & Entropy-Gated Prior)
=============================================================
マスター直伝「人間の言葉のパイプライン（小脳連想記憶 ✕ 大脳拡散整音）」を完全数理化した
直交型 N-gram ステップ圧縮エンジン。

【3大直交化安全保証】
1. 局所波面伝播 (Locality): 確定トークンに直接隣接する [MASK] スロットにのみ波面伝播。孤立マスク同士の共鳴を物理遮断。
2. エントロピー・ゲート (Entropy Gating): モデル確信度が低い (H(p)大) 時のみ介入。自信がある時は介入度0。
3. 有界バイアス (Bounded Prior): ロジット加算幅を厳格にクリップし、Jon Barron ゼロサム多様体を完全保護。
"""

import math
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn.functional as F


class AriaNgramStepCompressor:
    def __init__(
        self,
        max_bias: float = 2.0,
        entropy_threshold: float = 0.85,
        min_count: int = 2,
    ):
        self.max_bias = max_bias
        self.entropy_threshold = entropy_threshold
        self.min_count = min_count
        
        # 連想記憶テーブル:
        # bigram: (left_tok,) -> {target_tok: count}
        # rev_bigram: (right_tok,) -> {target_tok: count}
        # trigram_cloze: (left_tok, right_tok) -> {target_tok: count}
        self.bigram_table: Dict[int, Dict[int, int]] = {}
        self.rev_bigram_table: Dict[int, Dict[int, int]] = {}
        self.cloze_table: Dict[Tuple[int, int], Dict[int, int]] = {}

    def build_from_texts(self, texts: List[str], tokenizer):
        """テキストコーパスから N-gram 連想辞書を決定論的に抽出"""
        for text in texts:
            if not text.strip():
                continue
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            L = len(token_ids)
            if L < 2:
                continue
            
            for i in range(L):
                tgt = token_ids[i]
                # Forward Bigram: left -> tgt
                if i > 0:
                    left = token_ids[i - 1]
                    if left not in self.bigram_table:
                        self.bigram_table[left] = {}
                    self.bigram_table[left][tgt] = self.bigram_table[left].get(tgt, 0) + 1
                
                # Reverse Bigram: right -> tgt
                if i < L - 1:
                    right = token_ids[i + 1]
                    if right not in self.rev_bigram_table:
                        self.rev_bigram_table[right] = {}
                    self.rev_bigram_table[right][tgt] = self.rev_bigram_table[right].get(tgt, 0) + 1
                
                # Cloze Trigram: (left, right) -> tgt
                if i > 0 and i < L - 1:
                    left = token_ids[i - 1]
                    right = token_ids[i + 1]
                    pair = (left, right)
                    if pair not in self.cloze_table:
                        self.cloze_table[pair] = {}
                    self.cloze_table[pair][tgt] = self.cloze_table[pair].get(tgt, 0) + 1

    def save_cache(self, filepath: str):
        """連想テーブルをキャッシュファイルとして保存"""
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "bigram": self.bigram_table,
                "rev_bigram": self.rev_bigram_table,
                "cloze": self.cloze_table,
            }, f)

    def load_cache(self, filepath: str) -> bool:
        """連想テーブルをキャッシュから高速ロード"""
        import os
        import pickle
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.bigram_table = data.get("bigram", {})
                self.rev_bigram_table = data.get("rev_bigram", {})
                self.cloze_table = data.get("cloze", {})
            return True
        except Exception:
            return False

    def apply_prior(
        self,
        shift_logits: torch.Tensor,       # [num_masked, vocab_size]
        pos_indices: torch.Tensor,        # [num_masked] (全系列中での絶対インデックス)
        full_tokens: torch.Tensor,        # [seq_len] (現在の生成中トークン列)
        mask_id: int,
        step_progress: float = 0.0,       # 0.0 (開始) -> 1.0 (完了)
    ) -> torch.Tensor:
        """
        局所波面伝播 ＆ エントロピー・ゲートによる安全な Logit Prior 注入。
        """
        if shift_logits.size(0) == 0:
            return shift_logits
        
        # 進行度アニーリング (初期ステップほど効き、終盤は拡散モデル自身に任せる)
        anneal_factor = max(0.0, 1.0 - step_progress)
        if anneal_factor <= 0.05:
            return shift_logits

        seq_len = full_tokens.size(0)
        
        # モデルの現在の予測確率と最大確信度
        with torch.no_grad():
            probs = F.softmax(shift_logits, dim=-1)
            max_conf, _ = probs.max(dim=-1) # [num_masked]

        for idx, pos in enumerate(pos_indices.tolist()):
            # 1. エントロピー・ゲート: すでにモデルが自信満々の場合は介入ゼロ (非干渉保証)
            if max_conf[idx].item() >= self.entropy_threshold:
                continue

            left_tok = full_tokens[pos - 1].item() if pos > 0 else mask_id
            right_tok = full_tokens[pos + 1].item() if pos < seq_len - 1 else mask_id
            
            left_valid = (left_tok != mask_id)
            right_valid = (right_tok != mask_id)

            # 2. 局所波面伝播: 両隣がマスクの孤立スロットには一切触れない (定在波完全根絶)
            if not left_valid and not right_valid:
                continue

            candidates: Dict[int, float] = {}

            # Cloze (両隣確定): 最も高精度なアンカー
            if left_valid and right_valid:
                pair = (left_tok, right_tok)
                if pair in self.cloze_table:
                    counts = self.cloze_table[pair]
                    total = sum(counts.values())
                    for tok, cnt in counts.items():
                        if cnt >= self.min_count:
                            candidates[tok] = candidates.get(tok, 0.0) + 1.5 * (cnt / total)

            # Forward Bigram
            if left_valid and left_tok in self.bigram_table:
                counts = self.bigram_table[left_tok]
                total = sum(counts.values())
                for tok, cnt in counts.items():
                    if cnt >= self.min_count:
                        candidates[tok] = candidates.get(tok, 0.0) + 1.0 * (cnt / total)

            # Reverse Bigram
            if right_valid and right_tok in self.rev_bigram_table:
                counts = self.rev_bigram_table[right_tok]
                total = sum(counts.values())
                for tok, cnt in counts.items():
                    if cnt >= self.min_count:
                        candidates[tok] = candidates.get(tok, 0.0) + 0.8 * (cnt / total)

            if not candidates:
                continue

            # 3. 有界バイアス (Bounded Prior): 上限キャップ付きで安全にロジット加算
            for tok_id, score in candidates.items():
                bias = min(self.max_bias, score * self.max_bias) * anneal_factor
                shift_logits[idx, tok_id] += bias

        return shift_logits
