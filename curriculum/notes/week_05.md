# Lecture plan at a glance

The transformer is the common building block of the rest of the module (LLMs, text encoders, diffusion transformers, multimodal models, agents). Students should leave able to compute attention by hand, explain every part of a transformer block, distinguish the three architecture families, and implement attention and a small GPT. Attention maps and the character-level GPT results in the slides are real (`curriculum/assets/make_week_05.py`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | "What does *it* refer to?" (tired vs wide). Keep the sentence on the board. |
| 5–15 min | 5–7 | 1 · Why transformers? | Recurrence vs attention; comparison table. |
| 15–27 min | 8–10 | 2 · Tokens and position | Embeddings; permutation equivariance; sinusoidal, learned and rotary positions. |
| 27–65 min | 11–22 | 3 · Attention | Q, K, V; the equation; worked example on the board; √d_k; multi-head; causal mask; cross-attention; real BERT and GPT-2 heads; quiz. |
| 65–72 min | – | Break | |
| 72–94 min | 23–28 | 4 · Architecture | The block; three families; model sizes; architecture quiz; code. |
| 94–118 min | 29–37 | 5 · Training, efficiency, applications | Objectives; perplexity; real small GPT; n² cost and FlashAttention; KV cache; MoE; ViT/Whisper; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Do the three-token worked example (slide 14) live on the board with a calculator. Exam questions almost always include a small attention calculation.

<!-- pagebreak -->

# Lecture notes

## 1. Why transformers?

Before 2017, sequence modelling used **recurrent neural networks** (RNNs, LSTMs, GRUs), often with an attention layer added for translation. Recurrent networks have two limitations:

1. **Long-term dependencies.** Information about an early token must survive many recurrent steps to influence a late one; in practice it fades, and gradients vanish or explode over long sequences.
2. **No parallelism over time.** Step t needs the hidden state of step t − 1, so training cannot use the GPU's parallelism across a sequence.

The transformer (Vaswani et al., 2017) removes recurrence. **Self-attention** lets every token look at every other token directly, in one step, and all positions are computed in parallel. This made it possible to train on vastly more data, which is what turned language models into LLMs. The cost is that attention compares all pairs of tokens: compute and memory grow as $n^2$ with sequence length $n$.

![Recurrence vs attention](fig:rnn_vs_attention)

## 2. Tokens, embeddings and position

* **Tokenisation** (week 1): text → sub-word tokens with integer IDs (byte-pair encoding and variants).
* **Embedding:** a learned table of size $|V| \times d$ maps each ID to a $d$-dimensional vector (e.g. 50,257 × 768 in GPT-2). Tokens used in similar contexts get similar vectors (week 12 uses this for retrieval).
* **Positional information:** self-attention is **permutation-equivariant**: if the input tokens are shuffled, the outputs are shuffled in the same way. Without position information "dog bites man" and "man bites dog" are indistinguishable.

Three ways to add position:

| Method | Idea | Used in |
|---|---|---|
| Sinusoidal | Add fixed sine/cosine waves of geometrically spaced frequencies: $PE_{(p,2i)} = \sin(p/10000^{2i/d})$, $PE_{(p,2i+1)} = \cos(p/10000^{2i/d})$ | Original transformer |
| Learned absolute | A trainable vector per position up to a maximum length | BERT, GPT-2 |
| Rotary (RoPE) | Rotate query and key vectors by an angle proportional to position; their dot product depends on relative distance | LLaMA, Qwen, Mistral families and most current open LLMs |

RoPE can be rescaled to extend context length beyond the training length, one of the main techniques behind 128K–1M-token context windows.

![Sinusoidal positional encoding and rotary encoding](fig:positional)

## 3. Attention

### Queries, keys and values

Each token vector $x_i$ is projected three times with learned matrices: $q_i = x_i W_Q$ (what am I looking for?), $k_i = x_i W_K$ (what do I contain, for matching?), $v_i = x_i W_V$ (what do I pass on?). Stacking all tokens gives matrices $Q, K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V$$

* $QK^\top$ is an $n \times n$ table of compatibility scores (query $i$ · key $j$).
* Softmax is applied to each **row**, so each token's weights over all tokens sum to 1.
* Each output row is a weighted average of the value vectors: a new, context-aware representation.

![Queries, keys and values](fig:qkv)

### Worked example

Tokens *cat, sat, mat* with $Q = K$ whose rows are (1, 0), (0, 1) and (1, 1), $V$ whose rows are (1, 0), (0, 1) and (0.5, 0.5), and $d_k = 2$.

| Step | cat row | sat row | mat row |
|---|---|---|---|
| Scores $QK^\top/\sqrt{2}$ | [0.71, 0, 0.71] | [0, 0.71, 0.71] | [0.71, 0.71, 1.41] |
| exp | [2.03, 1, 2.03] | [1, 2.03, 2.03] | [2.03, 2.03, 4.11] |
| Weights (÷ row sum) | [0.40, 0.20, 0.40] | [0.20, 0.40, 0.40] | [0.25, 0.25, 0.50] |
| Output (weights × V) | [0.60, 0.40] | [0.40, 0.60] | [0.50, 0.50] |

Each output mixes the value vectors of the tokens the query attends to; *mat* attends most to itself because its query matches its own key best.

![Worked example as heatmaps](fig:attn_worked)

### Why divide by √d_k?

If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$ has variance $d_k$. For $d_k = 64$, scores have standard deviation 8, so the softmax is nearly one-hot (like temperature → 0 in week 1) and its gradients almost vanish. Dividing by $\sqrt{d_k}$ restores unit variance. The lab measures this directly.

### Multi-head attention

$$\mathrm{head}_i = \mathrm{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}), \qquad \mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W_O$$

Each of the $h$ heads works in dimension $d/h$, so the total cost equals one full-size head, but the model can attend to several kinds of relationship at once (syntax, coreference, previous token, …). In code, heads become an extra tensor dimension: `[batch, n, d] → [batch, h, n, d/h]`, attention is computed once in batch, then reshaped back. **Grouped-query attention (GQA)** lets several query heads share one key/value head to reduce memory (Llama 3 8B: 32 query heads, 8 KV heads).

![Multi-head attention](fig:multihead)

### Causal masking

Decoders generate left to right, so token $t$ may only attend to tokens $\leq t$. Before the softmax, add $-\infty$ to scores above the diagonal; their weights become exactly 0. A single forward pass then trains all $n$ next-token predictions of a sequence at once, each seeing only its own past.

![Bidirectional vs causal attention masks](fig:causal_mask)

### Self-attention vs cross-attention

In **self-attention** Q, K and V come from the same sequence. In **cross-attention** Q comes from one sequence and K, V from another: the decoder of an encoder–decoder model attends to the encoded input; Stable Diffusion's image latents attend to the prompt tokens (week 4); multimodal models fuse image and text features (weeks 9–10).

### Real attention heads and their interpretation

In pretrained BERT-base, the head with the strongest *it → animal* attention on the warm-up sentence was layer 9, head 11 (weight 0.68). GPT-2's heads are lower-triangular (causal) and show recognisable habits: attending to the previous token, to the first token (an "attention sink"), or to related earlier words.

**Read the causal experiment correctly.** In the lab, replacing *tired* with *wide* changes a token after *it*. GPT-2's representation and attention row at *it* cannot use either later adjective, so they should remain unchanged when the prefix through *it* is unchanged. This demonstrates the causal mask; it does not show that GPT-2 failed to reason about the complete sentence. To test use of the adjective, inspect a later position or a prediction made after the full sentence. BERT's bidirectional representation can use later words.

**Caution: attention is not explanation.** We *selected* one head out of 144 because it shows the pattern; most heads do not. Attention weights describe where information is mixed, but information also flows through value vectors, MLPs and the residual stream, and studies (e.g. Jain & Wallace, 2019) show attention can be changed without changing predictions. Faithful explanations need interventions such as ablation, probing or causal tracing.

![A real BERT head resolving "it"](fig:bert_attention)

![Real GPT-2 heads](fig:gpt2_attention)

## 4. The transformer architecture

### The block

A (pre-norm) transformer block computes

$$h = x + \mathrm{MHA}(\mathrm{LN}(x)), \qquad y = h + \mathrm{MLP}(\mathrm{LN}(h))$$

* **Attention** mixes information across tokens; the **feed-forward MLP** (typically 4× wider than $d$, with GELU or SwiGLU) processes each token separately and stores much of the model's factual knowledge.
* **Residual connections** let each sublayer learn a correction and let gradients flow through very deep stacks.
* **Layer normalisation** (often RMSNorm in modern models) keeps activations well scaled; pre-norm (normalise before each sublayer) trains more stably than the original post-norm.
* Stack $N$ identical blocks; add token + position embeddings at the input and a final normalisation + linear "unembedding" to vocabulary logits at the output.

![One transformer block](fig:block)

### Three families

| Family | Attention | Typical models | Best for |
|---|---|---|---|
| Encoder-only | Bidirectional | BERT, RoBERTa, embedding models | Classification, tagging, embeddings for search |
| Decoder-only | Causal | GPT, LLaMA, Claude, Gemini, Qwen, Mistral | Text generation; almost all chat LLMs |
| Encoder–decoder | Encoder bidirectional; decoder causal + cross-attention | T5, BART, Whisper, original transformer | Translation, summarisation, speech recognition |

A frequent misconception is that BERT is a chat model; it is an encoder and does not generate free text left to right.

![Three families of transformers](fig:three_archs)

### Scale of published models

| Model | Type | Layers | d_model | Heads | Params | Context |
|---|---|---|---|---|---|---|
| Transformer base (2017) | Enc–dec | 6 + 6 | 512 | 8 | 65 M | ~512 |
| BERT-base (2018) | Encoder | 12 | 768 | 12 | 110 M | 512 |
| GPT-2 small (2019) | Decoder | 12 | 768 | 12 | 124 M | 1,024 |
| GPT-3 (2020) | Decoder | 96 | 12,288 | 96 | 175 B | 2,048 |
| Llama 3 8B (2024) | Decoder | 32 | 4,096 | 32 (8 KV) | 8 B | 8K (128K in 3.1) |

Frontier proprietary models do not publish these figures; do not quote unofficial estimates.

### Attention in code

```python
def attention(q, k, v, causal=False):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    if causal:
        n = q.size(-2)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights
```

`torch.nn.functional.scaled_dot_product_attention` computes the same result with optimised kernels (FlashAttention) on GPUs.

## 5. Training, efficiency and applications

### Training objectives

* **Masked language modelling** (BERT): hide ~15% of tokens, predict them from both sides.
* **Next-token prediction** (GPT): predict each token from its past with a causal mask; the loss is cross-entropy = negative log-likelihood (week 1).
* **Span corruption / text-to-text** (T5): remove spans, generate them with the decoder.

All are **self-supervised**: labels come from the text itself, enabling training on trillions of tokens.

### Perplexity

$$\mathrm{PPL} = \exp\left(-\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(x_t \mid x_{<t})\right)$$

Perplexity is comparable only between models with the same tokeniser and test text, and low perplexity does not imply helpfulness, truthfulness or safety (week 7).

**Real result.** A 1.8 M-parameter character-level transformer (4 layers, 6 heads, d = 192, context 128) trained for 3000 steps on Tiny Shakespeare (9 minutes on a GTX 1650 laptop GPU) reduced the validation loss from 4.37 to 1.51 nats per character (perplexity 4.5; uniform guessing over 65 characters would give 65). Samples went from random characters (step 0) to word-like fragments with line structure (step 300) to Shakespeare-like dialogue with speaker names (final): "I think my heart is uncle. / MENENIUS: / My lord, I would leave you me for grace. / CORIOLANUS: / Come,-- / In the noble ribbour that …" It has learned spelling, format and style, not meaning.

![Validation loss of the lecture's character-level transformer](fig:chargpt_loss)

### Efficiency

* **Quadratic cost:** $n^2$ scores per head per layer. A single fp16 score matrix at 131,072 tokens would need tens of gigabytes.
* **FlashAttention** (Dao et al., 2022): exact attention computed in tiles in fast on-chip memory without materialising the $n \times n$ matrix; standard in current frameworks.
* **Restricted attention:** sliding-window and sparse patterns; **grouped-/multi-query attention** reduce KV memory; recurrent **state-space models** (e.g. Mamba) and hybrids offer linear-time alternatives.
* **KV cache:** during generation store the keys and values of earlier tokens; each new token costs one step of work instead of recomputing the prefix. Cache memory grows with context × layers × KV heads and dominates LLM serving memory (week 11).
* **Mixture-of-experts (MoE):** replace the MLP with many expert MLPs and a router that selects the top-k per token; large total parameter count, modest compute per token (Mixtral 8×7B: ~47 B total, ~13 B active; DeepSeek-V3: 671 B total, 37 B active). All experts must still be held in memory.
* **Long context:** RoPE scaling, efficient attention and careful training extend context to 128K–1M+ tokens, but cost and "lost in the middle" effects mean retrieval (week 12) is still valuable.

![Quadratic cost of attention](fig:complexity)

![The KV cache](fig:kv_cache)

![Mixture-of-experts routing](fig:moe)

### Applications beyond text

* **Vision Transformer (ViT):** 16 × 16 image patches as tokens; the image encoder in CLIP (week 9) and most multimodal models.
* **Speech:** Whisper is an encoder–decoder transformer from log-mel spectrograms to text.
* **Generation:** diffusion transformers (week 4); protein structure, weather, time series, robotics and code models.

![Image patches as tokens](fig:vit_patches)

# Common misconceptions

| Misconception | Correction |
|---|---|
| "Attention weights explain the model's decision." | They show information flow; faithful explanations need interventions. |
| "Transformers understand word order automatically." | Attention is order-blind; positional encodings add order. |
| "BERT is a chatbot." | BERT is an encoder for understanding and embeddings; chat models are decoders. |
| "More heads means more computation." | Heads split the dimension; total cost is roughly unchanged. |
| "The KV cache makes the model smarter." | It only avoids recomputation; outputs are identical. |
| "Mixture-of-experts models use all their parameters per token." | Only the selected experts run; all must be in memory. |
| "Perplexity tells you which chatbot is better." | It measures fit to text with a given tokeniser, not helpfulness or truth. |

# Responsible AI lens: attention, interpretability and cost

* **Explanations:** present attention maps as evidence of information flow at most; avoid cherry-picking heads; prefer controlled experiments.
* **Compute and energy:** transformer scale has real environmental and financial cost; choose the smallest adequate model and report compute.
* **Language fairness:** tokenisers split some languages into 3–5× more tokens (week 1 lab), raising cost and lowering quality for those speakers.

# Lab guide and answers

**Runtime:** Colab T4 (character GPT: 3,000 steps in a few minutes) or CPU (800 steps). Downloads: Tiny Shakespeare (about 1 MB), BERT-base and GPT-2 (about 1 GB total). **Hand-in:** passing checks, loss curve and perplexity, samples, attention plots, three written answers.

## TODOs

* **TODO 1** `scores = q @ k.transpose(-2, -1) / sqrt(d_k)`; `masked_fill(mask, -inf)`; `softmax(-1)`; return `weights @ v, weights`. The check reproduces the worked example and matches `F.scaled_dot_product_attention`.
* **Section 1.1:** the dot-product standard deviation grows as √d_k (≈ 2, 8, 22.6 for d_k = 4, 64, 512); unscaled softmax maxima approach 1 for large d_k, scaled ones stay moderate.
* **TODO 2** `torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)`.
* **Permutation test** prints `True`: attention is permutation-equivariant.
* **TODO 3** reshape `[b, n, d] → [b, h, n, d/h]` with `.view(b, n, h, dh).transpose(1, 2)`, attend, then `.transpose(1, 2).reshape(b, n, d)`. The causality check fails if the mask is not passed or heads are mixed up.
* **TODO 4** `math.exp(final_loss)`. Expect validation loss ≈ 1.5 nats/char (perplexity ≈ 4.5) after 3,000 GPU steps; ≈ 1.9–2.1 after 800 CPU steps.
* **TODO 5** `scores = bert_att[:, :, i_it, i_animal]`; `divmod(argmax, n_heads)`. The histogram shows that most heads give *animal* little weight.
* **GPT-2 adjective comparison:** unchanged attention at *it* is expected, because both alternative adjectives occur later and are masked from that position. Keep this separate from the question of whether attention alone explains a prediction.
* **KV cache:** expect a clear speed-up with the cache on (typically 2–5× on CPU for 200 tokens).

## Troubleshooting

* *Shape errors in TODO 3:* print shapes after each step; `view` needs contiguous tensors, so use `reshape` after `transpose`.
* *`output_attentions` returns None:* load the model with `attn_implementation="eager"` (fast kernels do not return weights).
* *Slow training on CPU:* reduce `STEPS`; the quality drops but the lesson holds.

# Practice questions with model answers

## Multiple choice

1. The softmax in attention is applied: **(a)** over the whole matrix; **(b)** over each row (each query's scores); **(c)** over each column; **(d)** to V. *Answer: (b).*
2. Dividing by √d_k prevents: **(a)** overfitting; **(b)** softmax saturation and vanishing gradients; **(c)** the causal mask from working; **(d)** long context. *Answer: (b).*
3. A decoder-only model uses: **(a)** bidirectional attention; **(b)** causal masking; **(c)** no attention; **(d)** only cross-attention. *Answer: (b).*
4. The KV cache: **(a)** improves accuracy; **(b)** stores earlier keys and values to speed up generation; **(c)** compresses the model; **(d)** replaces positional encoding. *Answer: (b).*

## Short answer

1. **Compute attention weights for one query with scaled scores [1, 1, 0].** (3 marks) *Model answer:* exp: 2.72, 2.72, 1.00; sum 6.44; weights 0.42, 0.42, 0.16.
2. **Explain why transformers need positional encoding and compare sinusoidal with rotary encodings.** (4 marks) *Model answer:* self-attention is permutation-equivariant, so order must be injected (1); sinusoidal adds fixed waves to embeddings, giving each absolute position a unique pattern (1.5); RoPE rotates queries and keys so scores depend on relative distance and can be rescaled for longer context (1.5).
3. **Describe the components of a transformer block and the role of residual connections.** (4 marks) *Model answer:* multi-head attention and feed-forward MLP sublayers, each with layer normalisation and a residual connection (2); residuals let sublayers learn corrections and let gradients flow through deep stacks (2).
4. **Why is attention cost quadratic and name two remedies.** (4 marks) *Model answer:* every query is compared with every key: n² scores per head and layer (2); FlashAttention (memory-efficient exact), sliding-window/sparse attention, grouped-query attention, state-space models (2 for two).

## Exam-style question

**"Explain how self-attention works, using a small numerical example, and discuss how multi-head attention, causal masking and positional encoding make the transformer suitable for language generation. Comment critically on the use of attention maps as explanations."** (20 marks)

*Marking guide:* Q, K, V and the attention equation (4); correct numerical example (4); multi-head attention (3); causal masking and its training benefit (3); positional encoding with one method explained (3); critical comment on attention as explanation (3).

# Glossary

| Term | Meaning |
|---|---|
| Self-attention | Each token forms a weighted average of all tokens' values using query–key scores |
| Query / key / value | Learned projections: what a token seeks / offers for matching / passes on |
| Scaled dot-product | Dividing scores by √d_k to keep the softmax trainable |
| Multi-head attention | Several attention heads in parallel, concatenated and projected |
| Causal mask | Blocks attention to future tokens in decoders |
| Cross-attention | Queries from one sequence, keys/values from another |
| Positional encoding | Information about token order (sinusoidal, learned, rotary) |
| Residual connection | Adding a sublayer's input to its output |
| Layer normalisation | Normalising activations per token (LayerNorm, RMSNorm) |
| Encoder / decoder | Bidirectional understanding stack / causal generating stack |
| Perplexity | exp(average negative log-likelihood per token) |
| KV cache | Stored keys/values of past tokens for fast generation |
| FlashAttention | Memory-efficient exact attention kernel |
| Mixture-of-experts | Router sends each token to a few of many expert MLPs |
| ViT | Vision Transformer: image patches as tokens |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Tunstall, von Werra & Wolf (2022) *Natural Language Processing with Transformers*: chapter 3, "Transformer Anatomy".
* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 9, "Transformers".
* Rothman, D. (2022) *Transformers for Natural Language Processing*: chapters 1–2 (supplementary).

**Videos**

* [3Blue1Brown: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) and [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)
* [Andrej Karpathy: Let's build GPT from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [StatQuest: Transformer Neural Networks, clearly explained](https://www.youtube.com/watch?v=zxQyTK8quyY)

**Papers and explainers**

* Vaswani et al. (2017) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Su et al. (2021) [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
* Dosovitskiy et al. (2020) [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
* [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Hugging Face blog: Mixture of Experts explained](https://huggingface.co/blog/moe)
* [Hugging Face LLM Course, chapter 1](https://huggingface.co/learn/llm-course/chapter1/1)

# Link to the group project

Every project will use a transformer, usually pretrained. Ask groups to identify which family their components belong to (e.g. an encoder for search embeddings, a decoder LLM for generation, a vision transformer for images) and to note context-length and memory constraints that will affect their design in weeks 11–12.
