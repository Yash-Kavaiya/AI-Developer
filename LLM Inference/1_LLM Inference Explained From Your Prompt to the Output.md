# LLM Inference: The Complete Foundation

> Notes on: *"What happens in the 1 second between pressing Enter and seeing the first word appear?"* — a walkthrough of the full LLM inference stack, from raw text to sampled token.

## 1. Training vs. Inference

Machine learning has two distinct phases:

| | Training | Inference |
|---|---|---|
| **What happens** | Model learns from data | Model is used to generate output |
| **Weights** | Constantly updating (gradient descent + backprop) | Frozen — no learning occurs |
| **Input** | Massive text corpora (books, web, code, articles) | A single user prompt |
| **Frequency** | Done once (occasionally repeated for new versions) | Every single request, all day, every day |
| **Cost profile** | Extremely expensive, but a one-time capital cost | Recurring, scales with usage |
| **Duration** | Weeks | ~Seconds |

**Key idea:** Training produces the model; inference is what you actually experience every time you chat with ChatGPT, Claude, or Gemini.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/ac0f1c46-0f39-4c24-8dad-298f96c79d52" />


## 2. The Economic Reality: Why Inference Matters

Inference isn't just a technical detail — it's the dominant financial reality of the AI industry.

- **OpenAI (2025):** ~$14 billion total estimated compute spend, with **62% going to inference**, not training. Inference cost is nearly equal to total revenue, contributing to an estimated **$8.5–9 billion annual cash burn**.
- **Anthropic (2025):** ~$6.8 billion estimated compute spend, **60% inference**. Inference spend came in **23% over internal forecast**, forcing a downward revision of projected gross margin from **50% down to 40%**.
- **Nvidia CEO Jensen Huang** has predicted AI inference demand will become **orders of magnitude larger** than it is today.
- **Reasoning models** (models that "think" before answering) consume **up to 150x more compute** than standard LLMs for the same query.

**Takeaway:** Faster, cheaper inference directly translates into lower operating costs, higher margins, and the ability to serve more users profitably. This is why inference optimization is a first-class engineering and business problem, not an academic afterthought.

---

## 3. Step 1 — Tokenization

Before a model can process text, the text must be converted into numbers.

- **Tokenization** splits raw text into chunks called **tokens**.
- A token is *roughly* 3–4 characters on average — not exactly a word, not exactly a letter.
- Common short words (`the`, `is`) are often a single token.
- Rare or complex words get split into multiple sub-word tokens.
  - Example: `unbelievable` → `un` | `believ` | `able`
- Each token is mapped to a **unique integer ID** from the model's vocabulary.
  - Example: `unbelievable` → `[1867, 374, 279]`
- **Vocabulary size** varies by model:
  - GPT-4 family: ~100,000 tokens
  - Llama 3: ~128,000 tokens

**Why it matters:** The number of tokens directly determines computational cost. A 1,000-token prompt costs roughly 1,000x more to process than a single-token prompt. This is exactly why LLM APIs bill **per token**, not per word.

---

## 4. Step 2 — Embedding Projection

Raw integer IDs are meaningless to the model on their own — it needs a rich, continuous representation.

- Each token ID is looked up in the **embedding table** (weight matrix `W_E`), a giant matrix where:
  - Each **row** corresponds to one vocabulary token.
  - Each row is a **vector** of numbers — typically **4,096 dimensions** for large models.
- Think of it like a dictionary, but instead of a text definition, each entry is a point in a 4,096-dimensional mathematical space encoding that token's *meaning*.
- After embedding, a sequence of tokens becomes a **matrix**.
  - Example: a 10-token input → a matrix of shape **[10, 4096]**.

### Positional Encoding
Since attention has no inherent sense of order, the model also adds **positional encoding** at this stage — information telling the model *where* each token sits in the sequence.

- Without it, the model cannot distinguish `"dog bites man"` from `"man bites dog"` — same tokens, different meaning, different order.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/8b62c7c8-ce8c-49f4-894e-9983f882a678" />

## 5. Step 3 — The Transformer Forward Pass

This is where the real computation happens. The embedded + positionally-encoded matrix flows through a stack of **transformer layers**.

- Example: **Llama 3 70B** has **80 layers**.
- Each layer has **two core components**:

### A. Self-Attention
- Every token "looks at" every other token and asks: *"How relevant are you to me right now?"*
- Mechanism (short version):
  1. Each token generates a **Query (Q)**, **Key (K)**, and **Value (V)** vector.
  2. Queries are matched against Keys to compute **attention scores**.
  3. Those scores are used to weight the Values.
  4. Each token comes out with a new, **context-aware representation**.

### B. Feed-Forward Neural Network (FFN)
- After attention, each token's representation passes independently through a small neural network.
- This is widely considered where the model's **factual knowledge, reasoning patterns, and language understanding** live.
- The FFN is typically **~4x wider** than the attention layer's internal dimension.

### Putting a Layer Together
A single transformer layer = **Attention → Add & Normalize → Feed-Forward → Add & Normalize**.

This entire block is repeated **N times** (80x for Llama 3 70B). Each pass refines the token representations further. At the very end, the **last token's final hidden state** — a single vector of 4,096 numbers — is what the model uses to predict the next token.



## 6. Step 4 — LM Head, Logits, and Sampling

### From Hidden State to Logits
- The last token's final hidden vector (4,096 numbers) is multiplied by the **language model head** — essentially the reverse operation of the embedding lookup.
- Output: a vector with one number per vocabulary entry (e.g., **128,000 numbers** for a 128K vocabulary).
- These raw numbers are called **logits**. Higher logit = model thinks that token is more likely to come next.

### Softmax
- **Softmax** converts logits into a probability distribution that sums to 1.

### Sampling Strategies
Once you have a probability distribution, you need to actually pick a token:

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/c6b7e293-d5b4-4611-97dd-9990a69b4326" />


| Method | Description |
|---|---|
| **Greedy decoding** | Always pick the single highest-probability token |
| **Temperature** | Scales logits *before* softmax. High temperature → more randomness/creativity. Low temperature → more deterministic/focused |
| **Top-p (nucleus sampling)** | Only sample from the smallest set of top tokens whose cumulative probability sums to `p` |
| **Top-k** | Only sample from the `k` highest-probability tokens |

These are the key hyperparameters used to tune an LLM's output behavior. The sampled token ID is appended to the sequence, and **the entire process repeats** for the next token.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/a732a249-a8ad-4de5-b7c8-0353c71536c0" />



## 7. Step 5 — Autoregressive Generation

This is the part that makes LLM inference uniquely challenging.

> LLMs generate text **one token at a time, sequentially**. Each token depends on all previous tokens — you cannot generate token 5 until you have tokens 1 through 4.

### The Generation Loop
1. Take the current sequence: `t1, t2, ..., tn`
2. Run a full forward pass through all layers.
3. Sample the next token `t(n+1)`.
4. Append it to the sequence.
5. Repeat from step 1.

This continues until the model emits a special **end-of-sequence** token or hits a **maximum length** limit.
<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/8fd2bb2e-c248-4123-ad20-6d1e2c73a4a9" />


### Two Distinct Phases

| Phase | What it does | Characteristics |
|---|---|---|
| **Prefill** | Processes the *entire* input prompt at once | All tokens processed **in parallel** — one big matrix operation. **Fast.** |
| **Decode** | Generates the response one token at a time | Strictly **sequential**. **Slow** — this is the bottleneck. |

**Example:** A 1,000-token prompt followed by a 500-token response = **1 fast prefill step** + **500 slow, sequential decode steps**.

This sequential nature is *the* fundamental reason inference is hard to speed up — and the reason nearly every optimization technique in this space exists.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/18554761-c7c5-45c3-aa30-a99e9ba399ed" />


## 8. Step 6 — The KV Cache

One of the most important optimizations in all of LLM inference — enabled by default in virtually every serving framework.

### The Problem
Without caching, generating each new token would require **recomputing attention over every previous token from scratch**, even though most of that work was already done in the previous step.

### The Solution
- The **KV cache** stores the **Key (K)** and **Value (V)** matrices computed for all previous tokens.
- When generating a new token, you only need to compute K and V for the **new token**, then concatenate it onto the cached values.
- The new token then attends to the full set of cached K/V pairs.

### Impact on Complexity

| | Per-token compute complexity |
|---|---|
| **Without KV cache** | O(n²) — recomputes everything, every step |
| **With KV cache** | O(n) — only computes the new token, dramatically faster |

### The Trade-off: Memory
- The KV cache isn't free — for a **70B parameter model with long context**, the cache alone can take **tens of gigabytes** of memory.
- Efficiently managing this cache (when to evict entries, how to compress them, how to share them across requests) is itself a deep, separate topic in inference optimization.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/f4aff9de-77cb-4865-bbc4-c90ba5e69bf5" />


## 9. The Full Inference Stack — Putting It All Together

```
User Input (raw text)
        │
        ▼
   Tokenizer  ──────────► Token IDs (integers)
        │
        ▼
   Embedding  ──────────► Word vector matrix + positional encoding
        │
        ▼
  N × Transformer Layers ──► (Self-Attention + Feed-Forward) × 80
        │
        ▼
    LM Head   ──────────► Logits (vocab-size vector, e.g. 128,000)
        │
        ▼
 Softmax + Sampling ────► Probabilities → next token
        │
        ▼
  Append token → loop back to decode step
```

Every stage maps to one or more real-world optimization techniques:

- **Tokenization / Embeddings** → vocabulary & embedding efficiency
- **Attention mechanism** → optimized/sparse attention implementations
- **KV Cache management** → eviction, compression, sharing
- **Quantization** → compressing model weights to reduce memory/compute
- **Speculative decoding** → speeding up the sequential decode loop
- **Batching / Continuous batching** → running multiple users through the pipeline simultaneously

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/66b1557e-ce00-4f6e-8f0e-3d283cd61a4f" />



## 10. Key Numbers to Remember

| Metric | Value |
|---|---|
| Avg. token length | ~3–4 characters |
| GPT-4 vocabulary size | ~100,000 tokens |
| Llama 3 vocabulary size | ~128,000 tokens |
| Typical embedding dimension | 4,096 (large models) |
| Llama 3 70B layer count | 80 layers |
| FFN width vs. attention | ~4x wider |
| Attention complexity w/o KV cache | O(n²) per token |
| Attention complexity w/ KV cache | O(n) per token |
| Reasoning model compute overhead | up to 150x standard LLM |
| OpenAI 2025 compute spend | ~$14B (62% inference) |
| Anthropic 2025 compute spend | ~$6.8B (60% inference) |

---

## 11. Why This Matters (Recap)

- Inference — not training — is the **recurring, dominant cost** of running an AI product.
- The **autoregressive, sequential decode loop** is the fundamental bottleneck: you cannot parallelize token-by-token generation the way you can parallelize prefill.
- Nearly every major inference optimization technique (KV cache, quantization, speculative decoding, batching, attention optimizations) exists specifically to attack **cost, latency, or memory** somewhere in this pipeline.
- Understanding this full stack at a high level is the foundation for going deeper into *why* certain steps are fast, why others are slow, and what "memory-bandwidth-bound" actually means in practice (the next topic in this series).

---

### Coming Up Next in the Series
- A deeper dive into GPU hardware: why certain steps are compute-bound vs. memory-bandwidth-bound.
- Deep dives into each optimization: attention mechanisms, KV cache management, quantization, speculative decoding, and batching/continuous batching.
