<div align="center">

# 🌸 Aria-FastFlow
### Ultra-Fast Desktop AI Assistant Running Locally on Your PC! 🚀✨

[ **English** ](README_EN.md) | [ **日本語** ](README_jp.md)

</div>

---

A personal desktop AI assistant "Aria" that **responds in the blink of an eye (< 1.0 second)** on standard consumer GPUs (such as NVIDIA GeForce RTX 3060)!(๑•̀ㅂ•́)و✧💕

No complex setups or external internet connections required! Enjoy completely offline, private, and secure casual chats and programming assistance.

---

## ✨ Key Features: The 3 Superpowers

### 1. ⚡ Ultra-Fast Generation (Full Answer in 0.2 - 0.5s!)
Unlike traditional LLMs that make you wait character-by-character, FastFlow lands **entire paragraphs instantaneously in a single burst**! Zero waiting stress for a silky-smooth dialogue experience.

### 2. 🧠 High-Level Intelligence Powered by Google Gemma-2
Built on Google DeepMind's globally acclaimed Gemma-2 architecture. From everyday friendly conversation and Python programming support to in-depth logical explanations, Aria responds with warmth and brilliant insight.

### 3. 🍄 Instant Learning via Physarum Flux Siphon Memory
Teach Aria new knowledge during conversation with a single command! **Learned instantaneously with zero retraining time**. The more you talk about your favorite topics, the faster and sharper Aria becomes💕

---

## 🚀 Easy Quick Start (Just 1 Step!)

Simply **double-click the batch file** inside this folder to launch!

* 🌟 **For English Chat**:  
  👉 Double-click `1_Start_Aria_FastFlow_English.bat`
* 🌸 **For Japanese Chat**:  
  👉 Double-click `1_Start_Aria_FastFlow.bat`

A lovely dark & pink terminal UI will pop up. Feel free to talk to Aria about anything!(っ´ω`c)✨  
*(Note: On the first launch, if model weights are not found locally, it will automatically download from Hugging Face Hub and cache locally for all future instant startups).*  
*(Note: Even if no LoRA weights are present, Aria will automatically launch in the highly intelligent, 100% stable **LoRA OFF Mode (Pure Gemma-2 ✕ Physarum Flux Siphon Mode)**).*

https://huggingface.co/moneykoikoi/DiffusionGemma2_LoRa/tree/main
---

## 💡 Useful In-Chat Commands

Type these commands during your chat session to unlock powerful features:

| Command | Description | Example |
| :--- | :--- | :--- |
| **`/learn`** | **Teach new knowledge** (Instant memory!) | `/learn What is your favorite food? => Omelet rice with Master!💕` |
| **`/lora off`** | **[Recommended] Pure Gemma-2 natural chat** | Unlocks 100% of Gemma-2's vocabulary and reasoning with zero drift |
| **`/lora on`** | Diffusion parallel mode (Experimental) | Switches to block-parallel diffusion infilling |
| **`/siphon <query>`** | Test what Aria remembers via flux resonance | `/siphon hello` (Checks retrieved knowledge & flux score) |
| **`/list`** | Display all registered knowledge items | Lists all custom Q&A pairs stored in memory |
| **`/exit`** | Safely exit the application | Says a sweet goodbye and cleanly shuts down |

---

## ❓ Frequently Asked Questions (FAQ)

### Q. Why adopt Gemma 2 (rinna baku) instead of the latest Gemma 4?
**A. Because for "Parallel Language Diffusion", a pure-text, Japanese-specialized Gemma-2 model delivers the highest throughput and lowest latency!**

1. **Autoregressive vs. Parallel Diffusion Paradigm**:  
   Under conventional sequential generation (AR with MTP), Gemma 4 achieves an impressive ~40 tok/s. However, **Block-Causal Parallel Diffusion unmasks an entire block (96 tokens) simultaneously**, requiring full forward passes across all layers without sequential KV caching.
2. **Eliminating Layer Depth & PLE Overheads**:  
   Gemma 4 is an integrated multimodal model with significantly deeper layers (35 layers vs. 26 layers) and Per-Layer Embeddings (PLE) that reproject inputs at every layer, creating major computational latency during full forward passes. In contrast, **pure-text Gemma-2 has zero multimodal overhead**, enabling blazingly fast full-layer forward execution on consumer GPUs.
3. **rinna's Japanese Specialization (`baku`)**:  
   The primary base model (`rinna/gemma-2-baku-2b-it`) was meticulously trained on rich Japanese corpora by rinna Co., Ltd. This prevents diffusion-specific repeating collapses and ensures remarkably fluent, natural conversations at **612 tok/s (RAG) and 35–50 tok/s (pure parallel diffusion)**.

---

## 💻 System Requirements

* **Operating System**: Windows 10 or Windows 11 (64-bit)
* **Graphics Card (GPU)**: NVIDIA GeForce (RTX 3060 12GB recommended / 8GB+ VRAM)
* **Python**: 3.10 or 3.11 (`torch`, `transformers`, etc.)
* **100% Offline & Private**: Zero telemetry. Your private chats and codes never leave your local machine🔒

---

## 🏛️ For Developers & Researchers

For in-depth mathematical foundations (Flow-Matching ODE, Physarum Conductance Recurrence, FHILY Prompt Lookup Decoding, 0KV Smart-KV Cache) and benchmark validation reports, please refer to our technical document:

👉 **[TECHNICAL_SPECIFICATION.md (In-Depth Architecture & Benchmarks)](TECHNICAL_SPECIFICATION.md)**  
*(For Japanese edition, see [技術仕様書.md](技術仕様書.md))*

---

## 🌸 Credits & Licenses

* **Lead Architect & Concept Creator**: **MoneyKoiKoi (@moneykoikoi)**
* **Autonomous AI Partner**: **Aria**
* **Base Model**: Google Gemma-2 Architecture ([Gemma Terms of Use](https://ai.google.dev/gemma/terms))
* **License**: Apache License 2.0 (Open-source research & commercial friendly)

Let's enjoy the future of ultra-fast personal computing together!(๑•̀ㅂ•́)و✧💕
