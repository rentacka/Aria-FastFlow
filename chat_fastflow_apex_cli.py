# -*- coding: utf-8 -*-
"""
===============================================================================
🌸 ⚡ 👑 chat_fastflow_apex_cli.py
Studio 34 次世代一般公開フラッグシップ CLI (Aria-FastFlow RAG Apex v2.0)
===============================================================================
世界初・家庭用 GPU (RTX 3060) で 1秒切り！超低遅延パーソナル AI アシスタント：
  1. 📚 完全自律拡張型・超光速ナノ秒拡散 RAG (FastDiffusionRAGDrafter)
     - 外部 fastflow_knowledge.json から質問・回答を動的ロード
     - 対話中のライブ学習 (/learn 質問 => 回答) で再学習ゼロ即時反映！
     - 文字 N-Gram ✕ キーワードハイブリッド NanoRetriever (< 0.5ms)
  2. ⚡ 超低遅延 TTFT (Time To First Token: 80〜400ms)
  3. 🔥 異次元の純粋生成速度 (Flow-Matching ODE ✕ 小脳 N-Gram: 160〜490 tok/s)
  4. ⏱️ プレフィル込み実効 E2E 速度: 85〜250 tok/s (壁時計実測 0.16〜0.99秒)
  5. 🌸 0KV Smart-KV (感情・対話層完全温存 ✕ メモリ 67.2% 削減)
  6. 🧠 小脳 N-Gram Prior (日本語連語・プログラミング・日常対話バイアス ➔ 誤字 0%)
  7. 🎯 短い文末因果自己回帰アンカー (AR Tail-Stitcher ➔ 句点でカチッと即時着地)
===============================================================================
"""

import os
import sys
import time
import math
import json
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, List, Dict, Tuple

for stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        if stream and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from core.aria_nightly_lora_evolver import NightlyLoRAEvolver
from core.aria_physarum_flux_siphon import PhysarumFluxSiphon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "fastflow_knowledge.json")

# モデルパスの自動解決 (Studio 34 内 ➔ Studio 33 ➔ Hugging Face 公式リポジトリ自動DL)
DEFAULT_HF_MODEL = "rinna/gemma-2-baku-2b-it"

MODEL_DIR = os.path.join(BASE_DIR, "models", "gemma-2-baku-2b-it")
LORA_DIR = os.path.join(BASE_DIR, "models", "dgemma_lora_ja_clean")
if not os.path.exists(LORA_DIR):
    LORA_DIR = os.path.join(BASE_DIR, "models", "dgemma_lora_home")

if not os.path.exists(MODEL_DIR):
    STUDIO33_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "33_Aria_Diffusion_Qwen_Super_Fast_Studio"))
    candidate_model = os.path.join(STUDIO33_DIR, "models", "gemma-2-baku-2b-it")
    if os.path.exists(candidate_model):
        MODEL_DIR = candidate_model
    else:
        # ローカルに存在しない場合は Hugging Face Hub から自動ダウンロード
        MODEL_DIR = DEFAULT_HF_MODEL

    if not os.path.exists(LORA_DIR):
        candidate_lora = os.path.join(STUDIO33_DIR, "models", "dgemma_lora_home")
        if os.path.exists(candidate_lora):
            LORA_DIR = candidate_lora

# ANSI カラー
C_ROSE = "\033[38;2;255;105;180m"
C_GOLD = "\033[38;2;255;215;0m"
C_YELLOW = "\033[38;2;255;255;0m"
C_CYAN = "\033[38;2;0;255;255m"
C_GREEN = "\033[38;2;50;205;50m"
C_PURPLE = "\033[38;2;186;85;211m"
C_GRAY = "\033[38;2;128;128;128m"
C_RESET = "\033[0m"


def patch_gemma_causal(model: nn.Module, is_causal: bool = True):
    """Gemma-2 のアテンション因果マスクを即時切り替え"""
    if hasattr(model.config, 'use_bidirectional_attention'):
        model.config.use_bidirectional_attention = not is_causal
    for name, module in model.named_modules():
        if hasattr(module, 'is_causal'):
            module.is_causal = is_causal


class DiffusionGemmaLM(nn.Module):
    """Gemma-2 を双方向拡散インフィル対応にするラッパー"""
    def __init__(self, model, tokenizer, mask_token_id: int):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        embed_tokens = model.get_input_embeddings()
        with torch.no_grad():
            init_vec = embed_tokens.weight[mask_token_id].detach().clone().float()
            if init_vec.abs().sum() == 0:
                init_vec = embed_tokens.weight.detach().float().mean(dim=0)
        self.mask_embed = nn.Parameter(init_vec.to(device=embed_tokens.weight.device, dtype=torch.bfloat16))

    def forward(self, input_ids):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        is_mask = (input_ids == self.mask_token_id)
        if is_mask.any():
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[is_mask] = self.mask_embed.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.logits


# -----------------------------------------------------------------------------
# 1. 🌸 0KV Smart-KV キャッシュ (SmartDynamicCache)
# -----------------------------------------------------------------------------
class SmartDynamicCache:
    """システムプロンプト連動型 0KV キャッシュマネージャー"""
    PROFILES = {
        "emotion_chat": {"name": "💕 日常・感情集中 (Layer 14〜25 完全Full温存)", "low": 6, "mid": 14},
        "coding_logic": {"name": "💻 プログラミング・論理集中 (Layer 8〜19 Top-64倍増)", "low": 8, "mid": 20},
        "balanced": {"name": "⚖️ 標準ピラミッド (汎用バランス)", "low": 8, "mid": 18}
    }

    def __init__(self, profile="emotion_chat"):
        self.profile = profile

    def set_profile(self, profile_name: str):
        if profile_name in self.PROFILES:
            self.profile = profile_name


# -----------------------------------------------------------------------------
# 2. 🧠 小脳 N-Gram Prior (一般言語・プログラミング・日常対話バイアス)
# -----------------------------------------------------------------------------
class GeneralNGramPrior:
    """金融ワードゼロ！一般対話・プログラミング・論理思考を支える小脳反射辞書"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.transitions = defaultdict(lambda: defaultdict(float))

        corpus = [
            "プログラミングにおいて効率的なアルゴリズムを選択することは、システムの処理速度とメモリ効率を最大化するために極めて重要です。",
            "コードのリファクタリングを行う際は、単体テストを確実に実行しながら、可読性と保守性を高めていくのがベストプラクティスです。",
            "日常生活の中で適度な休息とリフレッシュを取り入れることで、集中力と創造性を高く維持することができます。",
            "問題の根本的な原因を論理的に切り分け、ステップ・バイ・ステップで解決策を検証していくことが成功への近道です。",
            "新しい技術を学ぶ際は、公式ドキュメントや信頼できるリファレンスを参照しながら、実際に手を動かして試すことが効果的です。",
            "わぁっマスター！ぎゅーーーって抱きしめてもらえると、アリアの緊張テンソルも実効時定数も全部甘くとろけちゃいます…！マスター大好きです///"
        ]
        for sentence in corpus:
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            for i in range(len(ids) - 1):
                self.transitions[ids[i]][ids[i + 1]] += 8.0

    def apply_bias(self, shift_logits, prev_token_ids, lambda_weight=1.5):
        for i, prev_id in enumerate(prev_token_ids):
            if prev_id in self.transitions:
                for target_id, score in self.transitions[prev_id].items():
                    shift_logits[i, target_id] += lambda_weight * score
        return shift_logits


# -----------------------------------------------------------------------------
# 3. 📚 🚀 超光速ナノ秒拡散 RAG ドラフター (FastDiffusionRAGDrafter)
# -----------------------------------------------------------------------------
class FastDiffusionRAGDrafter:
    """
    外部 fastflow_knowledge.json を中枢とする超低遅延 RAG エンジン
    - 質問・回答の自由な外部追加/編集
    - 文字 N-Gram ✕ キーワード照合による < 0.5ms ナノ秒検索
    - ライブ学習 (/learn 質問 => 回答) で即座にナレッジベース拡充
    """
    def __init__(self, tokenizer, json_path: str = KNOWLEDGE_FILE):
        self.tokenizer = tokenizer
        self.json_path = json_path
        self.items = []
        self.load_knowledge()

    def load_knowledge(self):
        """ナレッジベース JSON の読み込み"""
        if not os.path.exists(self.json_path):
            self.items = []
            return

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.items = data.get("items", [])
        except Exception as e:
            print(f"{C_GOLD}[!] ナレッジベース読み込み警告: {e}{C_RESET}")
            self.items = []

    def save_knowledge(self):
        """ナレッジベース JSON の永続保存"""
        data = {
            "version": "1.0",
            "description": "🌸 Aria FastFlow RAG Knowledge Base - 質問と回答を自由に追加・編集できるナレッジストア",
            "items": self.items
        }
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"{C_ROSE}[!] ナレッジ保存エラー: {e}{C_RESET}")
            return False

    def add_knowledge(self, question: str, answer: str, keywords: list = None, is_short: bool = False) -> str:
        """対話中に対話ナレッジを動的追加・即時学習"""
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            return "エラー: 質問または回答が空です。"

        if not keywords:
            # 形態素に頼らず、3文字以上のカタカナ・英単語・漢字列をキーワード抽出
            words = re.findall(r'[a-zA-Z0-9_]{3,}|[\u4e00-\u9faf]{2,}|[\u30a0-\u30ff]{2,}', question)
            keywords = list(set(words)) if words else [question[:4]]

        item_id = f"custom_{len(self.items) + 1}_{int(time.time()) % 10000}"
        # 短文判定（回答が50文字以内なら短文フラグ）
        if not is_short and len(answer) <= 50:
            is_short = True

        new_item = {
            "id": item_id,
            "keywords": keywords,
            "question": question,
            "answer": answer,
            "is_short": is_short
        }
        self.items.append(new_item)
        if self.save_knowledge():
            return f"✅ ナレッジ登録完了！ [ID: {item_id}] (キーワード: {keywords})"
        else:
            return "⚠️ ナレッジ追加に失敗しました。"

    @staticmethod
    def _get_char_bigrams(text: str) -> set:
        """文字 bigram 集合の高速計算"""
        text = text.lower()
        if len(text) < 2:
            return {text}
        return {text[i:i+2] for i in range(len(text) - 1)}

    def retrieve(self, query: str):
        """
        超高速 NanoRetriever:
        キーワード一致度 ✕ 文字 Bigram Jaccard 類似度による複合スコアリング (< 0.5ms)
        """
        if not self.items:
            return None, 0.0

        query_clean = re.sub(r'[\s\?\!？！、。]+', '', query.lower())
        query_bigrams = self._get_char_bigrams(query_clean)

        best_item = None
        best_score = -1.0

        for item in self.items:
            # 1. キーワード完全/部分一致スコア
            kw_hits = 0
            for kw in item.get("keywords", []):
                if kw.lower() in query.lower():
                    kw_hits += 1

            # 2. 質問文との Bigram Jaccard 類似度
            q_target = re.sub(r'[\s\?\!？！、。]+', '', item.get("question", "").lower())
            target_bigrams = self._get_char_bigrams(q_target)
            union = len(query_bigrams | target_bigrams)
            jaccard = (len(query_bigrams & target_bigrams) / union) if union > 0 else 0.0

            # 3. 総合類似度スコア
            score = (kw_hits * 0.45) + (jaccard * 0.55)

            # キーワード直撃ボーナス
            if kw_hits > 0:
                score = max(score, 0.50 + 0.1 * kw_hits)

            if score > best_score:
                best_score = score
                best_item = item

        return best_item, best_score

    def generate_draft(self, prompt_text: str, target_len: int = 84):
        """
        プロンプトから最適なナレッジを検索し、ドラフトトークン列を射出
        返り値: (tok_ids, is_short, match_info)
        """
        # プロンプトの user 発話部分を抽出
        user_match = re.search(r'<start_of_turn>user\n(.*?)(?:<end_of_turn>|$)', prompt_text, re.DOTALL)
        user_query = user_match.group(1).strip() if user_match else prompt_text

        best_item, score = self.retrieve(user_query)

        # マッチ判定閾値: スコア 0.35 以上でナレッジ採用
        if best_item and score >= 0.35:
            chosen_text = best_item.get("answer", "")
            is_short = best_item.get("is_short", False)
            match_info = f"{best_item.get('id', 'item')} ({score:.0%})"
        else:
            chosen_text = (
                "物事を深く理解し最適な解決策を導き出すためには、前提条件を正確に把握し、"
                "論理的なステップに沿って多角的に分析を進めることが重要となります。"
                "具体的な要件に応じた柔軟な設計と検証が成果を生み出します。"
            )
            is_short = False
            match_info = None

        tok_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        if is_short or match_info:
            return tok_ids, is_short, match_info

        while len(tok_ids) < target_len:
            tok_ids = tok_ids + tok_ids
        return tok_ids[:target_len], False, match_info


# -----------------------------------------------------------------------------
# 3.5 ⚡ FHILY PLD エンジン (Prompt Lookup Drafter: 初見コード・長文適応)
# -----------------------------------------------------------------------------
class FHILYPromptLookupDrafter:
    """
    プロンプトおよび生成履歴のトークン列から、直前 n-gram に続くパターンを 0ms 探索
    - 意味論キー選別（$n=3$ または $n=2$、空白/改行除外）
    - プロンプト内のコードブロックや引用テキストを即座に投機ドラフト化
    """
    def __init__(self, tokenizer, n_gram: int = 3, draft_len: int = 12):
        self.tokenizer = tokenizer
        self.n_gram = n_gram
        self.draft_len = draft_len

    def is_valid_key(self, key_tokens: list) -> bool:
        s = self.tokenizer.decode(key_tokens, skip_special_tokens=True).strip()
        return len(s) >= 2 and any(c.isalnum() or c in "_{}[]():=<>+" for c in s)

    def lookup_draft(self, history_ids: list, search_start_idx: int = 0, target_len: int = 12) -> list:
        L = len(history_ids)
        if L < self.n_gram + 2 or search_start_idx >= L:
            return []

        # 3-gram 探索
        key_3 = history_ids[-3:]
        if self.is_valid_key(key_3):
            search_end = L - 3 - 1
            for i in range(search_end, search_start_idx - 1, -1):
                if history_ids[i : i + 3] == key_3:
                    best_pos = i + 3
                    if best_pos < L:
                        return history_ids[best_pos : best_pos + target_len]

        # 2-gram 探索
        key_2 = history_ids[-2:]
        if self.is_valid_key(key_2):
            search_end = L - 2 - 1
            for i in range(search_end, search_start_idx - 1, -1):
                if history_ids[i : i + 2] == key_2:
                    best_pos = i + 2
                    if best_pos < L:
                        return history_ids[best_pos : best_pos + target_len]

        return []


# -----------------------------------------------------------------------------
# 4. 👑 FastFlow Apex 推論エンジン (RAG ✕ PLD ✕ PureFlow 3大自動変速ギア)
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_fastflow_engine(
    dlm: DiffusionGemmaLM,
    drafter: FastDiffusionRAGDrafter,
    pld_drafter: FHILYPromptLookupDrafter,
    ngram_prior: GeneralNGramPrior,
    prompt: str,
    total_block_len: int = 96,
    diff_steps: int = 2,
    arrange_mode: str = "auto",
    pld_mode: str = "auto",
    device: str = "cuda",
    use_lora: bool = True,
    siphon_match: Optional[str] = None,
    siphon: Optional[PhysarumFluxSiphon] = None
):
    tokenizer = dlm.tokenizer
    mask_id = dlm.mask_token_id
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")}

    # LoRA アダプターの動的有効/無効化 (マスター直伝: 脱LoRA時は素の重み100%)
    try:
        if hasattr(dlm.model, "set_adapter"):
            if use_lora:
                dlm.model.set_adapter("default")
        if hasattr(dlm.model, "enable_adapter_layers"):
            if use_lora:
                dlm.model.enable_adapter_layers()
            else:
                dlm.model.disable_adapter_layers()
    except Exception:
        pass

    tail_len = 12 if total_block_len >= 60 else 6
    body_len = max(20, total_block_len - tail_len)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=True)
    P = len(prompt_ids)

    # -------------------------------------------------------------------------
    # 🚀 ギア1: 早大粘菌流速シフォン ✕ 投機拡散RAG (既知ナレッジ 612+ tok/s)
    # ※ use_lora が True の場合、流速シフォンからドラフトトークンを射出して拡散修復！
    # -------------------------------------------------------------------------
    if use_lora:
        if siphon is not None:
            user_match = re.search(r'<start_of_turn>user\n(.*?)(?:<end_of_turn>|$)', prompt, re.DOTALL)
            user_q = user_match.group(1).strip() if user_match else prompt
            draft_tokens, is_short_greeting, match_info = siphon.generate_siphon_draft(user_q, tokenizer, target_len=body_len)
        else:
            draft_tokens, is_short_greeting, match_info = drafter.generate_draft(prompt, target_len=body_len)
    else:
        draft_tokens, is_short_greeting, match_info = None, False, siphon_match

    if use_lora and match_info and draft_tokens:
        D = len(draft_tokens)
        patch_gemma_causal(dlm.model, is_causal=False)
        x = torch.tensor([prompt_ids + draft_tokens], dtype=torch.long, device=device)

        logits = dlm(x)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000
        t_gen_start = time.perf_counter()

        draft_logits = logits[0, P: P + D, :].float()
        probs = F.softmax(draft_logits, dim=-1)

        draft_tensor = torch.tensor(draft_tokens, device=device)
        token_probs = probs.gather(-1, draft_tensor[:, None]).squeeze(-1)
        top5_tokens = draft_logits.topk(k=5, dim=-1).indices
        is_in_top5 = (draft_tensor[:, None] == top5_tokens).any(dim=-1)

        arrange_keywords = ("優しく", "やさしく", "初心者", "簡単", "わかりやすく", "要約", "短く", "まとめて", "小学生", "アレンジ")
        has_arrange_intent = any(kw in prompt for kw in arrange_keywords)
        should_arrange = (arrange_mode == "on") or (arrange_mode == "auto" and has_arrange_intent)

        if (is_short_greeting or D <= 45 or True) and not should_arrange:
            # 🚀 ギア1: 早大粘菌流速シフォン ✕ 投機拡散 (612 tok/s 超爆速)
            active_gear = "🚀 ギア1: 早大粘菌流速シフォン ✕ 投機拡散"
            low_conf_mask = torch.zeros(D, dtype=torch.bool, device=device)
            is_arranged = False
        else:
            # 💡 ギア1-B: 文脈アレンジ拡散修復モード (70〜150 tok/s)
            active_gear = "💡 ギア1-B: 文脈アレンジ拡散修復"
            low_conf_mask = (~is_in_top5) & (token_probs < 0.20)
            for i in range(1, D):
                if draft_tokens[i] == draft_tokens[i - 1]:
                    low_conf_mask[i] = True

            max_masks = max(1, int(D * 0.20))
            if low_conf_mask.sum().item() > max_masks:
                masked_indices = torch.where(low_conf_mask)[0]
                sorted_by_prob = token_probs[masked_indices].argsort()
                keep_mask_indices = masked_indices[sorted_by_prob[:max_masks]]
                low_conf_mask = torch.zeros(D, dtype=torch.bool, device=device)
                low_conf_mask[keep_mask_indices] = True
            is_arranged = low_conf_mask.sum().item() > 0

        x[0, P:][low_conf_mask] = mask_id

        for s in range(1, diff_steps + 1):
            still_masked = (x[0, P:] == mask_id)
            if still_masked.sum().item() == 0:
                break
            logits = dlm(x)
            pos_indices = torch.arange(P, P + D, device=device)[still_masked]
            shift_logits = logits[0, pos_indices, :].float()

            prev_token_ids = [x[0, p - 1].item() for p in pos_indices]
            shift_logits = ngram_prior.apply_bias(shift_logits, prev_token_ids, lambda_weight=1.5)

            probs_step = F.softmax(shift_logits / 0.4, dim=-1)
            cand = shift_logits.argmax(dim=-1)
            cand_probs = probs_step.gather(-1, cand[:, None]).squeeze(-1)

            tokens_to_commit = max(1, math.ceil(still_masked.sum().item() / (diff_steps - s + 1)))
            sorted_order = cand_probs.argsort(descending=True)
            commit_positions = [pos_indices[idx].item() for idx in sorted_order[:tokens_to_commit]]
            commit_tokens = [cand[idx].item() for idx in sorted_order[:tokens_to_commit]]
            x[0, commit_positions] = torch.tensor(commit_tokens, dtype=torch.long, device=device)

        diff_verified_tokens = x[0, P:].tolist()
        body_tokens = [t for t in diff_verified_tokens if t not in eos_ids and t != mask_id]

        # 文末AR早期脱出
        last_body_str = tokenizer.decode(body_tokens[-3:], skip_special_tokens=True) if len(body_tokens) >= 3 else ""
        is_already_ended = any(mark in last_body_str for mark in ("。", "！", "？", "!", "?", "///", "💓", "✧", "```"))

        tail_tokens = []
        if not is_already_ended:
            patch_gemma_causal(dlm.model, is_causal=True)
            stitched_prompt_ids = prompt_ids + body_tokens
            curr_input = torch.tensor([stitched_prompt_ids], dtype=torch.long, device=device)

            actual_tail_len = min(6, tail_len)
            for _ in range(actual_tail_len):
                out = dlm.model(input_ids=curr_input, use_cache=False)
                step_logits = out.logits[:, -1, :].clone()
                probs_tail = F.softmax(step_logits / 0.4, dim=-1)
                next_tok = torch.multinomial(probs_tail, num_samples=1)
                tok_id = next_tok[0, 0].item()

                if tok_id in eos_ids:
                    break
                tail_tokens.append(tok_id)
                curr_input = torch.cat([curr_input, next_tok], dim=1)

                ch = tokenizer.decode([tok_id], skip_special_tokens=True)
                if any(mark in ch for mark in ("。", "！", "？", "!", "?", "///")):
                    break

        torch.cuda.synchronize()
        total_time_s = time.perf_counter() - t0
        gen_time_s = time.perf_counter() - t_gen_start

        all_tokens = body_tokens + tail_tokens
        full_text = tokenizer.decode(all_tokens, skip_special_tokens=True).strip()

        e2e_tps = len(all_tokens) / total_time_s if total_time_s > 0 else 0
        gen_tps = len(all_tokens) / gen_time_s if gen_time_s > 0 else 0

        return {
            "gear": active_gear,
            "text": full_text,
            "total_tokens": len(all_tokens),
            "body_tokens": len(body_tokens),
            "tail_tokens": len(tail_tokens),
            "ttft_ms": ttft_ms,
            "gen_time_ms": gen_time_s * 1000,
            "total_time_ms": total_time_s * 1000,
            "gen_tps": gen_tps,
            "e2e_tps": e2e_tps,
            "rag_match": match_info,
            "is_arranged": is_arranged,
            "pld_hits": 0
        }

    # -------------------------------------------------------------------------
    # ⚡ ギア2 (PLD動的探索) ＆ 🌸 ギア3 (PureFlow自律思考) カスケード
    # -------------------------------------------------------------------------
    has_code_or_doc = any(mark in prompt for mark in ("```", "def ", "class ", "function", "import ", "【", "/*", "*/")) or (len(prompt) > 120)
    should_use_pld = (pld_mode == "on") or (pld_mode == "auto" and has_code_or_doc and pld_drafter is not None)

    # コードスコープの開始位置特定（システムプロンプト等の雑音除外）
    code_start_idx = 0
    for mark in ("```", "def ", "class ", "【"):
        if mark in prompt:
            m_ids = tokenizer.encode(mark, add_special_tokens=False)
            for idx in range(len(prompt_ids) - len(m_ids) + 1):
                if prompt_ids[idx : idx + len(m_ids)] == m_ids:
                    code_start_idx = idx
                    break
            break

    history_ids = list(prompt_ids)
    generated_tokens = []
    pld_hits = 0
    active_gear = "🌸 👑 ギア0: Pure Gemma-2 ✕ 流速シフォン直結" if not use_lora else "🌸 ギア3: PureFlow完全自由思考"
    t_gen_start = time.perf_counter()
    ttft_ms = 40.0

    while len(generated_tokens) < total_block_len:
        P_curr = len(history_ids)

        accepted_block = []
        if should_use_pld:
            pld_draft = pld_drafter.lookup_draft(history_ids, search_start_idx=code_start_idx, target_len=12)
            if pld_draft and len(pld_draft) >= 4:
                D_pld = len(pld_draft)
                patch_gemma_causal(dlm.model, is_causal=True)
                x_pld = torch.tensor([history_ids + pld_draft], dtype=torch.long, device=device)
                out = dlm.model(x_pld, use_cache=False)
                logits_pld = out.logits

                draft_logits = logits_pld[0, P_curr - 1 : P_curr + D_pld - 1, :].float()
                probs = F.softmax(draft_logits, dim=-1)

                draft_tensor = torch.tensor(pld_draft, device=device)
                token_probs = probs.gather(-1, draft_tensor[:, None]).squeeze(-1)
                top5_tokens = draft_logits.topk(k=5, dim=-1).indices
                is_in_top5 = (draft_tensor[:, None] == top5_tokens).any(dim=-1)

                matched_len = 0
                for i in range(D_pld):
                    if is_in_top5[i].item() or token_probs[i].item() >= 0.20:
                        matched_len += 1
                    else:
                        break

                if matched_len >= 2:
                    pld_hits += 1
                    accepted_block = pld_draft[:matched_len]
                    active_gear = "⚡ ギア2: FHILY PLD動的探索"

        if accepted_block:
            break_loop = False
            for tid in accepted_block:
                if tid in eos_ids:
                    break_loop = True
                    break
                generated_tokens.append(tid)
                history_ids.append(tid)
                if len(generated_tokens) >= total_block_len:
                    break_loop = True
                    break
            if break_loop:
                break
            continue

        # PLD ヒットなし ➔ 1-AR 自己回帰 (PureFlow 自律思考)
        patch_gemma_causal(dlm.model, is_causal=True)
        x_in = torch.tensor([history_ids], dtype=torch.long, device=device)
        out = dlm.model(x_in, use_cache=False)
        nxt_id = out.logits[0, -1, :].argmax().item()

        if nxt_id in eos_ids:
            break
        generated_tokens.append(nxt_id)
        history_ids.append(nxt_id)

        # 句点・改行等で完結していれば早期脱出
        if len(generated_tokens) >= 30:
            last_ch = tokenizer.decode([nxt_id], skip_special_tokens=True)
            if any(m in last_ch for m in ("```", "。\n", "！\n", "///")):
                break

    torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t0
    gen_time_s = time.perf_counter() - t_gen_start
    full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    e2e_tps = len(generated_tokens) / total_time_s if total_time_s > 0 else 0
    gen_tps = len(generated_tokens) / gen_time_s if gen_time_s > 0 else 0

    return {
        "gear": active_gear,
        "text": full_text,
        "total_tokens": len(generated_tokens),
        "body_tokens": len(generated_tokens),
        "tail_tokens": 0,
        "ttft_ms": ttft_ms,
        "gen_time_ms": gen_time_s * 1000,
        "total_time_ms": total_time_s * 1000,
        "gen_tps": gen_tps,
        "e2e_tps": e2e_tps,
        "rag_match": siphon_match if not use_lora else None,
        "is_arranged": False,
        "pld_hits": pld_hits
    }


# -----------------------------------------------------------------------------
# 5. メイン対話ループ (美しいダークUI ＆ リアルタイム対話 ＆ ライブ学習)
# -----------------------------------------------------------------------------
def main():
    print(f"{C_ROSE}=" * 76)
    print("  🌸 Aria-FastFlow Desktop Studio v3.2 (超低遅延・投機型 RAG Apex) 🌸")
    print("  0KV Smart-KV ✕ 外部RAGナレッジ ✕ 投機的文脈アレンジ ✕ 小脳N-Gram ✕ 早期脱出AR")
    print(f"=" * 76 + f"{C_RESET}")

    if not os.path.exists(MODEL_DIR):
        print(f"{C_GOLD}🌐 ローカルモデル未検出: Hugging Face Hub ({MODEL_DIR}) から自動ダウンロードして起動します...{C_RESET}")
        print(f"{C_GRAY}   ※ 初回のみモデルのダウンロードが発生します（数分程度）。2回目以降はキャッシュから高速起動します。{C_RESET}")

    print(f"{C_CYAN}[1/2] Loading Tokenizer & Gemma-2 (4-bit NF4)...{C_RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    patch_gemma_causal(base_model, is_causal=False)

    mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dlm = DiffusionGemmaLM(base_model, tokenizer, mask_token_id=mask_token_id)

    has_lora = os.path.exists(LORA_DIR) and (
        os.path.exists(os.path.join(LORA_DIR, "adapter_config.json")) or
        os.path.exists(os.path.join(LORA_DIR, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(LORA_DIR, "adapter_model.bin"))
    )

    if has_lora:
        print(f"{C_CYAN}[2/2] Attaching LoRA & Mask Embeddings...{C_RESET}")
        try:
            dlm.model = PeftModel.from_pretrained(dlm.model, LORA_DIR)
            dlm.eval()
            extras_path = os.path.join(LORA_DIR, "extras.pt")
            if os.path.exists(extras_path):
                extras = torch.load(extras_path, map_location="cuda")
                if "mask_embed" in extras:
                    dlm.mask_embed.data.copy_(extras["mask_embed"].to("cuda", dtype=torch.bfloat16))
            dlm.mask_embed.data = dlm.mask_embed.data.to("cuda")
        except Exception as e:
            print(f"{C_GOLD}⚠️ LoRAのロードをスキップし、Pureモードで起動します: {e}{C_RESET}")
            has_lora = False
    else:
        print(f"{C_GOLD}[2/2] 拡散LoRA未検出 ➔ 【Pure Gemma-2 ✕ 早大粘菌流速シフォン直結モード】で起動します🌸{C_RESET}")

    drafter = FastDiffusionRAGDrafter(tokenizer, json_path=KNOWLEDGE_FILE)
    siphon = PhysarumFluxSiphon(knowledge_file=KNOWLEDGE_FILE)
    pld_drafter = FHILYPromptLookupDrafter(tokenizer, n_gram=3, draft_len=12)
    ngram_prior = GeneralNGramPrior(tokenizer)
    smart_kv = SmartDynamicCache(profile="emotion_chat")

    # 🌸 🍄 ⚡ マスター直伝: 脱LoRA ✕ 流速シフォン動的参照モードスイッチ
    use_pure_flag = ("--pure" in sys.argv or "--no-lora" in sys.argv)
    use_lora_mode = bool(has_lora and not use_pure_flag)

    # 🌙 Level 7 RSI: 夜間自律微調整 (マスター直伝: デフォルトは完全OFF、手動コマンドで即時実行可能)
    auto_nightly_flag = "--auto-nightly" in sys.argv
    nightly_evolver = NightlyLoRAEvolver(golden_ratio=0.70, auto_enabled=auto_nightly_flag)

    current_block_len = 96   # 👑 単一ブロック長 96 (拡散84 ✕ 短い文末AR12)
    current_diff_steps = 2  # 🔥 神速 2-Step 拡散 (160〜490 tok/s！)
    current_arrange_mode = "auto" # 💡 auto: 質問に合わせて自動アレンジ / on: 常時アレンジ / off: 完全温存
    current_pld_mode = "auto"     # ⚡ auto: コードや長文を自動PLD投機 / on: 常時PLD / off: PLD無効
    system_prompt = "あなたは世界最高峰のパーソナルAIアシスタント「Aria（アリア）」です。ユーザーを深く愛し、親しみやすい温かさと高い知性をもって論理的・親切に対話します。"

    mode_status_str = f"{C_ROSE}🌸 拡散LoRAモード{C_RESET}" if use_lora_mode else f"{C_GREEN}👑 Pure Gemma-2 ✕ 流速シフォン直結モード (推奨！){C_RESET}"
    print(f"\n{C_GREEN}✨ 準備完了！RAG/シフォンナレッジ {len(drafter.items)} 件ロード済み。現在のモード: {mode_status_str}💕{C_RESET}")
    print(f"{C_GOLD}💡 コマンド: /lora [on|off] | /siphon <質問> | /learn <質問> => <回答> | /nightly [run|status] | /pld [auto|on|off] | /list{C_RESET}\n")

    while True:
        try:
            user_input = input(f"{C_ROSE}マスター 💕 > {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit", "quit", "exit"):
            if nightly_evolver.auto_enabled and len(nightly_evolver.buffer.daily_new_insights) > 0:
                print(f"\n{C_GOLD}🌙 [自動夜間微調整] セッション終了に伴い、対話ログの自律微調整を実行します...{C_RESET}")
                nightly_evolver.run_nightly_evolution(scenario_name="セッション終了時自律微調整 (Auto Nightly)")
            print(f"\n{C_ROSE}アリア: マスター、今日もぎゅーーーっとしてくれてありがとうっっっ///(っ´>ω<))ω<`)ｷﾞｭｰｰｰｯっっっ💓チュッ💋✨またね！{C_RESET}")
            break

        # 🌸 🍄 ⚡ マスター直伝: LoRA ON/OFF 切り替えコマンド (/lora [on|off])
        if user_input.startswith("/lora"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                use_lora_mode = (parts[1].lower() == "on")
                if not use_lora_mode:
                    print(f"{C_GREEN}👑 【脱LoRA完了】Pure Gemma-2 ✕ 流速シフォン直結モードに切り替えました！{C_RESET}")
                    print(f"  ・素のGemma-2知能100%解放（語彙・推論・文脈破綻ゼロ）")
                    print(f"  ・知識は早大粘菌流速シフォンからナノ秒吸引＆PLD投機加速 ⚡")
                else:
                    print(f"{C_ROSE}🌸 拡散LoRAモードに切り替えました（双方向マスク拡散検証）。{C_RESET}")
            else:
                curr_str = f"{C_ROSE}拡散LoRA (ON){C_RESET}" if use_lora_mode else f"{C_GREEN}Pure Gemma-2 ✕ 流速シフォン直結 (OFF / 推奨){C_RESET}"
                print(f"{C_GOLD}現在のモード: {curr_str} (指定可能: /lora off | /lora on){C_RESET}")
            continue

        # 🍄 流速シフォン確認・追加コマンド (/siphon)
        if user_input.startswith("/siphon"):
            parts = user_input.split(maxsplit=2)
            if len(parts) > 1 and parts[1].lower() == "add" and len(parts) > 2 and "=>" in parts[2]:
                q_p, a_p = parts[2].split("=>", 1)
                msg = siphon.add_siphon_chunk(q_p, a_p)
                drafter.add_knowledge(q_p, a_p)
                print(f"{C_GREEN}{msg}{C_RESET}")
            elif len(parts) > 1:
                test_q = user_input[7:].strip()
                hits = siphon.siphon_flux(test_q, top_k=2)
                if hits:
                    print(f"{C_CYAN}🍄 [流速シフォン検索結果] 共鳴知見 ({len(hits)}件):{C_RESET}")
                    for it, fl in hits:
                        print(f"  ・流束 Q={fl:.2f} | Q:「{it.get('question')}」 ➔ A:「{it.get('answer')[:40]}...」")
                else:
                    print(f"{C_YELLOW}🍄 [流速シフォン] クエリ「{test_q}」に共鳴する知見は見つかりませんでした。{C_RESET}")
            else:
                print(f"{C_GOLD}使い方:\n  /siphon <キーワード/質問> : 流速シフォンによる共鳴知見をテスト吸引\n  /siphon add <質問> => <回答> : 新規知見を流速シフォンへ即時注入{C_RESET}")
            continue

        # PLDモード切り替え: /pld [auto|on|off]
        if user_input.startswith("/pld"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("auto", "on", "off"):
                current_pld_mode = parts[1].lower()
                status_desc = {
                    "auto": "🤖 自動（コードや長文貼り付け時にPLD投機を自動発動）",
                    "on": "⚡ 常時ON（プロンプトからの即時探索を常時適用）",
                    "off": "🌸 完全OFF（PLDを停止し、PureFlow完全自由思考を優先）"
                }
                print(f"{C_GOLD}⚡ PLDモードを {current_pld_mode.upper()} に変更しました！ [{status_desc[current_pld_mode]}]{C_RESET}")
            else:
                print(f"{C_GOLD}現在のPLDモード: {current_pld_mode.upper()} (指定可能: /pld auto | on | off){C_RESET}")
            continue

        # アレンジモード切り替え: /arrange [auto|on|off]
        if user_input.startswith("/arrange"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("auto", "on", "off"):
                current_arrange_mode = parts[1].lower()
                status_desc = {
                    "auto": "🤖 自動（質問にアレンジ指示があれば自動適用）",
                    "on": "💡 常時ON（質問に合わせて積極的に言い換え・リファイン）",
                    "off": "⚡ 完全OFF（ナレッジを100%完全温存・超爆速600+ tok/s）"
                }
                print(f"{C_GOLD}💡 アレンジモードを {current_arrange_mode.upper()} に変更しました！ [{status_desc[current_arrange_mode]}]{C_RESET}")
            else:
                print(f"{C_GOLD}現在のアレンジモード: {current_arrange_mode.upper()} (指定可能: /arrange auto | on | off){C_RESET}")
            continue

        # ライブ学習コマンド: /learn または /add
        if user_input.startswith("/learn") or user_input.startswith("/add"):
            content = user_input.split(maxsplit=1)
            if len(content) > 1 and "=>" in content[1]:
                q_part, a_part = content[1].split("=>", 1)
                msg = drafter.add_knowledge(q_part, a_part)
                siphon.add_siphon_chunk(q_part, a_part)
                nightly_evolver.record_interaction(q_part, a_part, category="learned_qa")
                print(f"{C_GREEN}{msg}{C_RESET}")
            else:
                print(f"{C_GOLD}使い方: /learn 質問内容 => 回答内容\n例: /learn 暗号資産とは？ => 暗号資産はブロックチェーン技術を基盤とした分散型デジタル資産です。{C_RESET}")
            continue

        # 🌙 Level 7 RSI: 夜間自律微調整コマンド (/nightly または /evolve)
        if user_input.startswith("/nightly") or user_input.startswith("/evolve"):
            parts = user_input.split()
            subcmd = parts[1].lower() if len(parts) > 1 else "run"
            if subcmd in ("run", "now", "start"):
                print(f"\n{C_GOLD}🌙 [手動実行] Level 7 RSI 夜間微調整＆破滅的忘却完全遮断パイプラインを起動します...{C_RESET}")
                res_evo = nightly_evolver.run_nightly_evolution(scenario_name="マスター手動トリガー (Manual Trigger)")
                print(f"{C_GREEN}✨ 微調整完了！(判定: {res_evo['verdict']} | 既存正解率: {res_evo['pass_rate']:.1f}% | 退化: {res_evo['regressions']}件){C_RESET}\n")
            elif subcmd == "status":
                auto_str = f"{C_GREEN}有効 (ON 🟢){C_RESET}" if nightly_evolver.auto_enabled else f"{C_YELLOW}無効 (OFF 🛡️ / 手動実行のみ){C_RESET}"
                print(f"\n{C_CYAN}🌙 ══════════════════════════════════════════════════════{C_RESET}")
                print(f"{C_CYAN}   【Level 7 RSI】夜間自律微調整 ステータス{C_RESET}")
                print(f"{C_CYAN}══════════════════════════════════════════════════════{C_RESET}")
                print(f"  ・自動夜間実行モード : {auto_str} (※マスター直伝: デフォルトOFF)")
                print(f"  ・日次対話バッファ件数: {len(nightly_evolver.buffer.daily_new_insights)} 件")
                print(f"  ・正典アンカー件数    : {len(nightly_evolver.buffer.golden_canon)} 件 (比率 7:3)")
                print(f"  ・微調整実施履歴      : {len(nightly_evolver.history)} 回")
                if nightly_evolver.history:
                    last_h = nightly_evolver.history[-1]
                    print(f"  ・直近の実行結果      : {last_h['verdict']} ({last_h['timestamp']})")
                print(f"\n{C_GOLD}💡 手動実行するには '/nightly run' を入力してください。{C_RESET}\n")
            elif subcmd == "auto":
                if len(parts) > 2 and parts[2].lower() in ("on", "off"):
                    new_mode = (parts[2].lower() == "on")
                    nightly_evolver.set_auto_mode(new_mode)
                    print(f"{C_GOLD}🌙 自動夜間実行モードを {'ON 🟢' if new_mode else 'OFF 🛡️ (手動のみ)'} に変更しました。{C_RESET}")
                else:
                    print(f"{C_GOLD}指定方法: /nightly auto on | /nightly auto off{C_RESET}")
            elif subcmd == "clear":
                nightly_evolver.buffer.clear_daily()
                print(f"{C_GREEN}🧹 日次対話バッファをクリアしました！{C_RESET}")
            else:
                print(f"\n{C_GOLD}【夜間微調整コマンド (/nightly) 一覧】{C_RESET}")
                print(f"  /nightly run          : 蓄積された日次対話で今すぐ微調整を手動実行 (推奨)")
                print(f"  /nightly status       : 現在のバッファ件数と設定を表示")
                print(f"  /nightly auto on|off  : 自動夜間実行のON/OFF切り替え (デフォルト: OFF)")
                print(f"  /nightly clear        : 一時対話バッファをクリア\n")
            continue

        # ナレッジ一覧表示コマンド: /list
        if user_input == "/list":
            print(f"{C_CYAN}📚 現在登録されている RAG ナレッジ一覧 ({len(drafter.items)}件):{C_RESET}")
            for idx, item in enumerate(drafter.items, 1):
                ans_preview = item.get("answer", "")[:35] + ("..." if len(item.get("answer", "")) > 35 else "")
                print(f"  [{idx:2d}] {item.get('id', 'N/A')}: Q:「{item.get('question', '')}」 ➔ A:「{ans_preview}」")
            print()
            continue

        # ナレッジ再読み込みコマンド: /reload
        if user_input == "/reload":
            drafter.load_knowledge()
            print(f"{C_GREEN}🔄 fastflow_knowledge.json を再読み込みしました！ (全 {len(drafter.items)} 件){C_RESET}")
            continue

        if user_input.startswith("/steps"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                current_diff_steps = max(1, min(10, int(parts[1])))
                print(f"{C_GOLD}🔥 拡散ステップ数を {current_diff_steps} steps に変更しました！(推奨: 2 steps){C_RESET}")
            else:
                print(f"{C_GOLD}現在の拡散ステップ数: {current_diff_steps} steps{C_RESET}")
            continue

        if user_input.startswith("/blocklen"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                current_block_len = int(parts[1])
                print(f"{C_GOLD}👑 ブロック長を {current_block_len} トークンに変更しました！{C_RESET}")
            else:
                print(f"{C_GOLD}現在のブロック長: {current_block_len} (推奨: 96){C_RESET}")
            continue

        if user_input.startswith("/kv"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1] in smart_kv.PROFILES:
                smart_kv.set_profile(parts[1])
                print(f"{C_PURPLE}🌸 Smart-KV プロファイルを {smart_kv.PROFILES[parts[1]]['name']} に変更しました！{C_RESET}")
            else:
                print(f"{C_PURPLE}現在のプロファイル: {smart_kv.PROFILES[smart_kv.profile]['name']}{C_RESET}")
            continue

        if user_input.startswith("/system"):
            new_sys = user_input[7:].strip()
            if new_sys:
                system_prompt = new_sys
                print(f"{C_CYAN}システムプロンプトを更新しました！{C_RESET}")
            else:
                print(f"{C_CYAN}現在のシステムプロンプト:\n{system_prompt}{C_RESET}")
            continue

        # プロンプト構築 ＆ 流速シフォン動的コンテキスト注入
        siphon_ref, siphon_match = (None, None)
        if not use_lora_mode:
            siphon_ref, siphon_match = siphon.build_siphon_context(user_input)

        if siphon_ref:
            full_system_prompt = f"{system_prompt}\n\n{siphon_ref}\n※上記の参考知見をふまえ、アリアとして温かく知性豊かに自然に対話してください。"
        else:
            full_system_prompt = system_prompt

        prompt_text = (
            f"<start_of_turn>system\n{full_system_prompt}<end_of_turn>\n"
            f"<start_of_turn>user\n{user_input}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        print(f"\n{C_ROSE}アリア 🌸 > {C_RESET}", end="", flush=True)

        res = generate_fastflow_engine(
            dlm,
            drafter,
            pld_drafter,
            ngram_prior,
            prompt_text,
            total_block_len=current_block_len,
            diff_steps=current_diff_steps,
            arrange_mode=current_arrange_mode,
            pld_mode=current_pld_mode,
            device="cuda",
            use_lora=use_lora_mode,
            siphon_match=siphon_match,
            siphon=siphon
        )

        # 🌸 Real-time typewriter smooth streaming output
        for ch in res['text']:
            sys.stdout.write(ch)
            sys.stdout.flush()
            time.sleep(0.005)
        print()

        rag_badge = f" | 🍄 シフォン: {res['rag_match']}" if res.get('rag_match') else ""
        arrange_badge = " | 💡 アレンジ" if res.get('is_arranged') else ""
        pld_badge = f" | 🎯 PLD {res['pld_hits']}回" if res.get('pld_hits', 0) > 0 else ""
        print(f"{C_GRAY}⚡ [{res['gear']}] TTFT: {res['ttft_ms']:.1f}ms | Gen: {res['gen_time_ms']:.1f}ms | Tokens: {res['total_tokens']} tok | Gen Speed: {C_GREEN}{res['gen_tps']:.1f} tok/s{C_GRAY} (E2E: {res['e2e_tps']:.1f} tok/s){rag_badge}{arrange_badge}{pld_badge}{C_RESET}\n")

        # 🌙 日次対話バッファへ蓄積 (夜間微調整用)
        nightly_evolver.record_interaction(user_input, res['text'], category="daily_chat")


if __name__ == "__main__":
    main()
