# R2 · Week 05: attention and transformers

Checked **2026-10-10**. Scope: built slide titles and lab headings. Proposals need implementation and timing.

**Teaching explanations (adapt, then state the limit).**
- Context changes a word's representation. Embedding lookup starts without context; attention supplies other tokens' information. A bright heatmap does not establish human-like understanding. [3Blue1Brown](https://www.3blue1brown.com/lessons/attention/).
- Queries request information; keys match; values carry information. Replace this metaphor with learned projections immediately. Programmers do not assign heads grammatical jobs. [3Blue1Brown](https://www.3blue1brown.com/lessons/attention/).
- A covered answer sheet explains causal masking: hide future answers while training all positions together. Generation still produces tokens sequentially. This analogy is our adaptation. [Hugging Face explanation](https://huggingface.co/learn/llm-course/en/chapter1/4).

**Five misconceptions to correct.**

| Misconception | Correction |
|---|---|
| A token is always a word | Tokens can be pieces of words. |
| Attention inherently knows positions | Position information needs an explicit mechanism. |
| Queries, keys and values are input words | They are learned vector projections. |
| Causal masking forces sequential training | Masked training can evaluate positions in parallel. |
| Heatmaps explain the model's decision | Attention weights alone do not establish causality. |

The final caution has original experimental support: [Attention is not Explanation](https://arxiv.org/abs/1902.10186).

**Current practice.** Compare explicit `Q @ K.T`, row softmax and weighted values with PyTorch `scaled_dot_product_attention`. Its Boolean mask uses `True` for allowed; `MultiheadAttention.key_padding_mask` reverses this meaning. Set attention dropout to zero during evaluation. [PyTorch API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html). Separate learned absolute positions from RoPE's query/key rotations. [RoFormer](https://arxiv.org/abs/2104.09864). BERT/GPT-2 are historical inspection examples. Published Colab `torch 2.11.0`/`transformers 5.18.0`/`ipywidgets 7.7.1` require compatibility checks. [R0](platform_2026.md).

**Redraw.** For three tokens, show `Q,K:[3,2]` → scores `[3,3]` → mask/softmax `[3,3]` → multiply `V:[3,2]` → output `[3,2]`. Use paper-style rows throughout; 3Blue1Brown uses columns. Separate projections from the attention pipeline to avoid crossed arrows. Colour real operations by role, label tensor arrows, and show one dashed repeat arrow on the decoder diagram.

**Beginner barriers.** Titles stack matrix multiplication, softmax, head reshaping, residuals, perplexity, KV caching and experts. Lab parts then train a model and inspect two pretrained models. Explain a weighted average numerically before a matrix; print intermediate shapes before `.view()`/`.transpose()`; defer performance benchmarks and mixture-of-experts detail.

**CPU core proposal: target ≤15 minutes, unmeasured.** Hand-check three-token attention, test row sums and zero future weights, build a two-head block, and train a tiny character decoder on a bundled short corpus with capped updates. Temperature/mask widgets operate on small local examples (`skip-test`). Keep pretrained-head inspection and cache timing as stretches with measured stored figures; avoid downloads on the required path. Measure setup and computation before accepting the core.

**Gemini role.** Optional synthetic sentence-completion comparison; it cannot expose this lab's internal attention weights. Shared `USE_GEMINI=False`, local generator, cached few-call budget. R0's EEA/UK client-distribution caveat remains unresolved; no regional free-access promise. [R0](platform_2026.md).

**Further reading:** [attention paper](https://arxiv.org/abs/1706.03762), [HF explanation](https://huggingface.co/learn/llm-course/en/chapter1/4), [PyTorch API](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html), [RoFormer](https://arxiv.org/abs/2104.09864), [3Blue1Brown attention video](https://www.youtube.com/watch?v=eMlx5fFNoYc).

Handoff: research complete; primary sources checked. No lab/API/widget execution; CPU timing and Colab compatibility await B/V.
