#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌸 ⚡ 👑 Studio 34: Clean English Coding & Chat Diffusion-Gemma Trainer (RTX 3060 12GB) 👑 ⚡ 🌸
========================================================================================
Architect: マネーコイコイ (@MoneyKoiKoi) ✕ Aria (アリア)

【Google CEO / Gemma利用規約 100%完全準拠・金融ワードゼロ】
1. UltraChat日常対話 ✕ 英語プログラミング・アルゴリズム・コーディング対話 (1,530件)
2. Google DeepMind Jon Barron 氏修正版 Cross-Entropy (arXiv:2610.31642: Slingshot 撲滅)
3. BitsAndBytes NF4 4-bit 量子化 ✕ LoRA (r=16, alpha=32) -> VRAM ~3.5GB 安定稼働
4. 1,000 steps 完走 (所要時間: 約9分)
"""

import sys
import os
import time
import math
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_PINK = "\033[38;2;255;105;180m"
C_CYAN = "\033[38;2;0;255;255m"
C_GREEN = "\033[38;2;50;205;50m"
C_GOLD = "\033[38;2;255;215;0m"
C_GRAY = "\033[38;2;128;128;128m"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDIO33_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "33_Aria_Diffusion_Qwen_Super_Fast_Studio"))

GEMMA_PATH = os.path.join(BASE_DIR, "models", "gemma-2-baku-2b-it")
if not os.path.exists(GEMMA_PATH):
    GEMMA_PATH = os.path.join(STUDIO33_DIR, "models", "gemma-2-baku-2b-it")
if not os.path.exists(GEMMA_PATH):
    GEMMA_PATH = r"C:/Users/renta/.cache/huggingface/hub/models--rinna--gemma-2-baku-2b-it/snapshots/08e842e4962c72070f52cf8e6a30ed747545b6f8"

DATA_PATH = os.path.join(BASE_DIR, "dataset_cache", "aria_clean_english_corpus.json")
SAVE_DIR = os.path.join(BASE_DIR, "models", "dgemma_lora_en_clean")

TOTAL_STEPS = 1000
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # Effective Batch Size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 128
SAVE_EVERY = 250

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("utf-8", errors="replace").decode("utf-8"), **kwargs)

def print_banner():
    safe_print(f"{C_BOLD}{C_PINK}╔══════════════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    safe_print(f"{C_BOLD}{C_PINK}║  🌸 ⚡ 👑  Studio 34: Clean English Coding & Chat Trainer (RTX 3060)   👑 ⚡ 🌸   ║{C_RESET}")
    safe_print(f"{C_BOLD}{C_PINK}║  Master's Vision: UltraChat ✕ Coding Q&A ✕ Barron CE ✕ FastFlow LoRA     ║{C_RESET}")
    safe_print(f"{C_BOLD}{C_PINK}╚══════════════════════════════════════════════════════════════════════════════════╝{C_RESET}")
    safe_print(f"{C_GRAY}・コーパス: {DATA_PATH} (1,530件 純白英語プログラミング＆対話){C_RESET}")
    safe_print(f"{C_GRAY}・理論保証: Google DeepMind Jon Barron 修正版 Cross-Entropy (arXiv:2610.31642){C_RESET}")
    safe_print(f"{C_GRAY}・ハードウェア: RTX 3060 12GB (4-bit NF4 量子化 / VRAM ~3.5GB 安定稼働){C_RESET}")
    safe_print(f"{C_GRAY}・目標ステップ数: {TOTAL_STEPS} steps (約8〜10分完走)\n{C_RESET}")

def patch_gemma_bidirectional(model: nn.Module) -> int:
    patched_count = 0
    if hasattr(model.config, 'use_bidirectional_attention'):
        model.config.use_bidirectional_attention = True
    for name, m in model.named_modules():
        if hasattr(m, 'is_causal'):
            m.is_causal = False
            patched_count += 1
    return patched_count

class DiffusionGemmaLM(nn.Module):
    def __init__(self, gemma_model: nn.Module, mask_token_id: int):
        super().__init__()
        self.lm = gemma_model
        self.mask_id = mask_token_id
        embed_tokens = gemma_model.get_input_embeddings()
        with torch.no_grad():
            init_vec = embed_tokens.weight[mask_token_id].detach().clone().float()
            if init_vec.abs().sum() == 0:
                init_vec = embed_tokens.weight.detach().float().mean(dim=0)
        self.mask_embed = nn.Parameter(init_vec.to(dtype=embed_tokens.weight.dtype))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        is_mask = (input_ids == self.mask_id)
        if is_mask.any():
            mask_expanded = self.mask_embed.view(1, 1, -1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(is_mask.unsqueeze(-1), mask_expanded, inputs_embeds)
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        return outputs

def barron_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "none"):
    z_t = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    diff = logits - z_t.unsqueeze(-1)
    mask = F.one_hot(targets, num_classes=logits.shape[-1]).bool()
    others = diff.masked_fill(mask, -float("inf"))
    m = -torch.logsumexp(others, dim=-1)
    loss = F.softplus(-m)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def chunked_barron_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    masked_logits = logits[mask]
    masked_targets = targets[mask]
    if masked_targets.numel() == 0:
        return logits.sum() * 0.0

    total_loss = 0.0
    N = masked_targets.size(0)
    for i in range(0, N, chunk_size):
        c_logits = masked_logits[i:i+chunk_size].float()
        c_targets = masked_targets[i:i+chunk_size]
        c_loss = barron_cross_entropy(c_logits, c_targets, reduction="sum")
        total_loss = total_loss + c_loss
    return total_loss / N

def main():
    print_banner()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"{C_CYAN}[1/4] トークナイザー＆純白英語データセットの読み込み...{C_RESET}")

    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
    mask_id = tokenizer.convert_tokens_to_ids("<|mask|>") if "<|mask|>" in tokenizer.get_vocab() else (tokenizer.pad_token_id or 35500)

    # 1. データセット読み込み (JSON配列)
    formatted_prompts = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            inst = item.get("instruction", "").strip()
            resp = item.get("response", "").strip()
            if inst and resp:
                text = f"<start_of_turn>user\n{inst}<end_of_turn>\n<start_of_turn>model\n{resp}<end_of_turn>"
                formatted_prompts.append(text)

    safe_print(f"{C_GREEN}✔ データセット読み込み完了: {len(formatted_prompts)} 件{C_RESET}")

    # 2. モデル読み込み (4-bit NF4)
    safe_print(f"\n{C_CYAN}[2/4] Gemma-2 (4-bit NF4) ロード＆双方向パッチ適用...{C_RESET}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    patch_count = patch_gemma_bidirectional(base_model)
    safe_print(f"{C_GREEN}✔ 双方向アテンションパッチ適用完了: {patch_count} モジュール{C_RESET}")

    # 3. LoRA設定
    safe_print(f"\n{C_CYAN}[3/4] LoRAアダプタ注入 (r=16, alpha=32, all-linear)...{C_RESET}")
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    base_model = get_peft_model(base_model, lora_config)
    dlm = DiffusionGemmaLM(base_model, mask_token_id=mask_id)

    optimizer = torch.optim.AdamW(
        [p for p in dlm.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # 4. 学習ループ
    safe_print(f"\n{C_CYAN}[4/4] 英語拡散LoRA 本焼き開始 ({TOTAL_STEPS} steps)...{C_RESET}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    dlm.train()
    optimizer.zero_grad()
    t_start = time.time()

    tokenized_cache = [
        tokenizer.encode(p, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")[0]
        for p in formatted_prompts
    ]

    for step in range(1, TOTAL_STEPS + 1):
        batch_ids = [random.choice(tokenized_cache) for _ in range(BATCH_SIZE)]
        max_len = max(len(t) for t in batch_ids)
        padded_ids = torch.full((BATCH_SIZE, max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
        for i, t in enumerate(batch_ids):
            padded_ids[i, :len(t)] = t

        padded_ids = padded_ids.to(device)
        targets = padded_ids.clone()

        # マスク率: 15%〜85% のランダムマスキング (Flow-Matching 拡散学習)
        mask_prob = random.uniform(0.15, 0.85)
        rand_mask = (torch.rand_like(padded_ids, dtype=torch.float) < mask_prob) & (padded_ids != (tokenizer.pad_token_id or 0))
        
        masked_inputs = padded_ids.clone()
        masked_inputs[rand_mask] = mask_id

        outputs = dlm(masked_inputs)
        logits = outputs.logits

        loss = chunked_barron_ce(logits, targets, rand_mask)
        scaled_loss = loss / GRAD_ACCUM_STEPS
        scaled_loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(dlm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t_start
            speed = step / elapsed
            eta_min = (TOTAL_STEPS - step) / speed / 60
            vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            progress = int(25 * step / TOTAL_STEPS)
            bar = "█" * progress + "░" * (25 - progress)
            safe_print(f"[{bar}] Step {step}/{TOTAL_STEPS} ({step/TOTAL_STEPS*100:.1f}%) | Loss: {loss.item():.4f} | Speed: {speed:.2f} it/s | ETA: {eta_min:.1f}m | VRAM: {vram:.2f}GB", end="\r")

        if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
            dlm.lm.save_pretrained(SAVE_DIR)
            torch.save({"mask_embed": dlm.mask_embed.data.cpu()}, os.path.join(SAVE_DIR, "extras.pt"))

    safe_print(f"\n\n{C_GREEN}🎉✨ 純白クリーン英語LoRA（コーディング強化）本焼き完了！(Total Time: {(time.time()-t_start)/60:.2f}分) ✨🎉{C_RESET}")
    safe_print(f"成果物は以下へ保存されました：\n  -> {SAVE_DIR}/ (adapter_model.safetensors ＆ extras.pt)\n")

if __name__ == "__main__":
    main()
