# Lecture plan at a glance

A typing assistant sees a prefix and suggests a continuation. Students trace that example, calculate attention and build a small character transformer. They need basic Python, lists and tables, not prior deep learning.

| Time | Segment | Teaching action |
|---|---|---|
| 0–12 min | Hook, map and vocabulary | Ask what “it” refers to; trace the system. |
| 12–22 min | Why attention? | Compare direct routes with recurrent memory. |
| 22–34 min | Text, vectors and positions | Assign illustrative IDs; name matrix axes. |
| 34–64 min | Attention | Calculate one row; explain heads, masks and cross-attention. |
| 64–71 min | Break | Collect one unclear term from each group. |
| 71–87 min | Blocks and architectures | Follow residual additions; compare tasks. |
| 87–112 min | Learning, uncertainty and efficiency | Shift targets; read results; explain caching and non-text inputs. |
| 112–120 min | Lab preview and recap | Predict a notebook output; answer a check together. |

> **Teaching tip:** Before every diagram, ask what goes in, what comes out and what each arrow carries. Keep the numerical example visible when introducing its equation.

<!-- pagebreak -->

# Lecture notes

## 1. Begin with a useful prediction

A transformer turns an existing prefix into scores for possible next **tokens**. A token is one unit the model reads: a character, word piece or another represented item.

Our lab uses one character per token. ROMEO followed by a colon becomes character IDs. A vector is an ordered list of numbers. Token and position vectors pass through blocks that restrict reading to current and earlier positions. The final position produces next-character scores. Softmax turns scores into probabilities; sampling chooses an ID. Appending it makes the next prefix.

Fluent-looking continuation does not establish factual knowledge or coherent reasoning.

**Ask:** What often follows a speaker name and colon? Accept several answers: the model learns a distribution rather than one infallible rule.

## 2. Why replace recurrent steps?

A **recurrent neural network**, or RNN, reads step by step. Its hidden state is a learned memory vector passed forward. An **LSTM** learns when to retain or discard information.

A distant word affects a later position through many intermediate states. Information can weaken. Learning signals, called gradients, can become very small or large. These are long-term dependency and vanishing or exploding gradient problems. LSTMs help, but time steps remain dependent on earlier steps.

Attention connects allowed positions directly. Training can process many known input positions in parallel. Generation remains sequential because new predictions depend on generated prefixes.

![Recurrent memory and direct attention routes](fig:rnn_vs_attention)

Self-attention exchanges information within one sequence. Full attention considers every pair of token positions. Four positions give sixteen pairs; eight give sixty-four. Doubling length quadruples comparisons.

**Common mistake:** Direct access guarantees reasoning. It only makes information available; learned operations determine its use.

## 3. IDs, vectors and position

### A classroom ID example

This invented vocabulary explains shapes. It is not the lab vocabulary or a measured result.

| Character | Illustrative ID |
|---|---|
| a | 0 |
| c | 1 |
| s | 2 |
| t | 3 |

Cats becomes [1, 0, 3, 2]. An ID selects a row from an **embedding table**, a learned table of vectors. The ID is not a meaning or position.

A **feature** is one numerical entry in a vector. With width eight, three IDs produce shape [3, 8]: three positions, eight features. A batch [2, 3, 8] means two examples with those same position and feature axes.

A **tensor** is a numerical array that can have several named axes. Read its shape as a description, not a string to memorise.

### Why position is separate

Plain unrestricted self-attention is **permutation-equivariant**: shuffling input rows shuffles output rows the same way. It does not supply numbered coordinates. “Dog bites man” and “man bites dog” need order information.

| Method | Where position enters | Intuition |
|---|---|---|
| Sinusoidal encoding | Add fixed vectors to input embeddings | Clock hands change at different rates. |
| Learned absolute positions | Add a learned vector per position | Numbered seats have learned labels. |
| Rotary position embedding, RoPE | Rotate query/key feature pairs | Matching depends on relative rotation. |

![Fixed, learned and rotary position information](fig:positional)

CharGPT uses learned absolute positions; the original transformer used sinusoidal vectors. After the intuition, its optional rule is:

$$PE_{(p,2i)}=\sin(p/10000^{2i/d}), \qquad PE_{(p,2i+1)}=\cos(p/10000^{2i/d})$$

PE is the positional vector. p indexes position; i identifies a feature pair; d is vector width. Sine and cosine fill alternating features. The constant 10000 sets the frequency range.

RoPE rotates Q and K rather than adding another input vector. Its construction introduces relative position into comparisons. Extending position settings requires appropriate training and evaluation.

**Check:** Does vocabulary ID three always occupy position three? No: identity and position are separate.

## 4. Attention: match, then mix

### Queries, keys and values

Think of a library request. It matches catalogue descriptions; the useful books provide contents.

A **query** supplies learned matching features for a position seeking information. A **key** supplies matching features for a source. A **value** supplies information to transfer. A projection is a learned linear transformation of vector features. These features are not literal questions or human meanings.

Input table X is multiplied by learned matrices $W_Q$, $W_K$ and $W_V$, producing query table Q, key table K and value table V. In self-attention, they originate in one sequence.

Values bypass score calculation and softmax. Follow that separate branch.

![Q/K produce weights; V supplies the mixed information](fig:qkv)

### Worked example: one row first

These are deliberately chosen classroom vectors, not trained-model measurements. Cat, sat and mat label rows; the numbers are not learned word meanings.

| Row | Query Q | Key K | Value V |
|---|---|---|---|
| cat | [1, 0] | [1, 0] | [1, 0] |
| sat | [0, 1] | [0, 1] | [0, 1] |
| mat | [1, 1] | [1, 1] | [0.5, 0.5] |

**Step 1: compare.** A dot product multiplies corresponding entries, then adds them. Cat's query against the three keys gives [1, 0, 1].

**Step 2: scale.** Query/key width $d_k$ is two. Divide by $\sqrt{2}$, about 1.414, giving [0.707, 0, 0.707].

**Step 3: make weights.** Softmax exponentiates scores to approximately [2.028, 1, 2.028]. Divide by their sum, approximately 5.056. The weights are approximately [0.401, 0.198, 0.401].

**Step 4: mix values.** Calculate 0.401[1, 0] + 0.198[0, 1] + 0.401[0.5, 0.5]. The approximate output is [0.602, 0.398].

The output is a value mixture, not a selected key or vocabulary ID.

![Classroom scores, weights and value mixture](fig:attn_worked)

| Query | Scaled scores | Approximate weights | Approximate output |
|---|---|---|---|
| cat | [0.707, 0, 0.707] | [0.401, 0.198, 0.401] | [0.602, 0.398] |
| sat | [0, 0.707, 0.707] | [0.198, 0.401, 0.401] | [0.398, 0.602] |
| mat | [0.707, 0.707, 1.414] | [0.248, 0.248, 0.503] | [0.500, 0.500] |

Unrounded rows sum to one; rounded rows may differ slightly.

### Now read the equation

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Q, K and V are query, key and value tables. Superscript T means transpose: swap the final two axes. $d_k$ counts query/key features. Softmax runs across each query's source-position row.

For $n$ positions, $Q$ and $K$ are [$n$, $d_k$]. Values of width $d_v$ make $V$ [$n$, $d_v$]. Scores and weights are [$n$, $n$]; outputs are [$n$, $d_v$]. $n$ counts positions; $d_v$ counts value features.

**Why scale?** With independent, mean-zero, unit-variance entries, dot-product variance grows with $d_k$. At width 64, standard deviation is eight. Dividing by eight controls score scale and can prevent weak softmax gradients. It neither forces uniform weights nor masks future positions.

### Several heads

Each **head** learns its own Q/K/V projections in a smaller feature space. Concatenation joins outputs along the feature axis. A learned output projection mixes the joined features.

Width 32 with four heads gives eight features per head. [2, 10, 32] becomes [2, 4, 10, 8]: batch, heads, positions, features. Score tables are [2, 4, 10, 10]. The final shape returns to [2, 10, 32].

![Smaller parallel heads, joined and projected](fig:multihead)

After this shape example:

$$\mathrm{head}_i=\mathrm{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)})$$

$$\mathrm{MultiHead}(X)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O$$

X is the input table. i identifies a head; h counts heads. $W_Q$, $W_K$ and $W_V$ are its learned projections. $W_O$ mixes joined outputs. Concat joins features; dots mean continue through the remaining heads.

Main matrix-product work is similar at fixed total width. Naively stored score memory still depends on head count. Roles can be mixed or unclear; heads are not guaranteed human specialists.

### Causal masks

A **causal mask** prevents reading later positions. The current position remains allowed because it predicts the following token. Before softmax, blocked scores become negative infinity; their exponential is zero.

![Allowed and blocked source positions](fig:causal_mask)

For input cat and targets ats, c predicts a using only c. The a position predicts t using c and a. The t position predicts s using c, a and t. One pass trains all these predictions without leaking future inputs.

Our custom mask uses True for blocked. PyTorch SDPA uses True for allowed. Do not interchange them.

### Cross-attention

Self-attention takes Q/K/V from one sequence. **Cross-attention** takes queries from one sequence and keys/values from another.

A translation decoder supplies queries from its current states. Encoded source states supply K/V. Three query positions and five source positions give scores [3, 5], followed by three output vectors.

Diffusion image features can similarly read prompt vectors. Multimodal systems can connect image, text or audio representations this way.

## 5. Read model pictures honestly

The lecture summary uses measured BERT attention. Of all 144 heads, layer {{w05_bert_layer}}, head {{w05_bert_head}} gives the largest weight from it to animal: {{w05_attention_weight}}. This is the same rule as lab TODO 5, so students should find the same head. Choosing the strongest head is part of the method: the median head gives animal only {{w05_bert_median}}, and {{w05_bert_above_half}} of 144 heads exceed 0.5.

![Selected measured BERT connection, summarised for readability](fig:bert_summary)

The full matrix is evidence to inspect, not proof of a resolved pronoun. Value vectors, residual routes and other networks also affect representations.

![The selected BERT head's full table](fig:bert_matrix)

GPT-2 is a causal decoder. In the measured heads below, every square above the diagonal has weight 0.0: a token never reads a later token. Different heads still behave differently. Layer 1, head 1 spreads weight over several earlier tokens; layer 5, head 12 reads the previous token.

![Two measured GPT-2 heads with readable labels](fig:gpt2_causal_summary)

Changing later tired to wide cannot change GPT-2's earlier it row when the prefix through it stays unchanged. Both adjectives are blocked at that earlier position.

Test adjective use at a later position or a prediction after the complete sentence. Unchanged earlier attention is expected causality, not failed full-sentence reasoning.

A third measured head, layer 10, head 7, puts at least {{w05_gpt2_first_min}} of every later row's weight on the first token. Heads that park weight on one fixed position are common; they show again that a bright cell need not mark a meaningful link.

An **ablation** removes a component and measures its effect. Other controlled interventions modify a component while checking outputs. Such experiments can test importance; colour alone cannot.

## 6. A block: exchange information, then revise features

Attention exchanges information between positions. A **feed-forward network**, often called an MLP, changes features at each position separately. The same MLP weights serve every position within one layer.

A **residual connection** adds a sublayer's output to the vector entering its path. Input [1, 2] plus correction [0.3, −0.1] gives [1.3, 1.9]. The earlier representation remains available.

**Layer normalisation** adjusts feature scale within each token vector, with learned scale and shift. RMSNorm is a related normalization choice, not an identical operation.

![Our lab's pre-norm block with explicit additions](fig:block)

CharGPT normalises before sublayers, called **pre-norm**. The original transformer normalised after residual addition, called **post-norm**. After the numerical residual example:

$$h=x+\mathrm{MHA}(\mathrm{LN}(x)), \qquad y=h+\mathrm{MLP}(\mathrm{LN}(h))$$

x is the block input; h is its intermediate residual sum; y is output. MHA means multi-head attention; LN means layer normalisation; MLP means per-position feed-forward network. Plus signs add matching entries.

The lab MLP expands width 192 to 768, applies GELU and returns to 192. GELU is nonlinear, allowing more than a linear projection. Dropout omits features randomly during training as regularisation.

Stack blocks with the same structure but separate learned weights. Final normalization and a vocabulary head produce **logits**, unnormalised scores for possible output tokens.

### Follow the code

Here q, k and v are projected numerical arrays. This self-attention example assumes equal input sequence lengths.

~~~python
def attention(q, k, v, causal=False):
    scores = q @ k.transpose(-2, -1) / q.size(-1)**0.5
    if causal:
        n = q.size(-2)
        blocked = torch.ones(n, n, dtype=torch.bool,
                             device=q.device).triu(1)
        scores = scores.masked_fill(blocked, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights
~~~

`q.size(-1)` is the query/key width. `transpose` swaps the final axes of `k`. `triu(1)` marks positions strictly above the diagonal. `softmax(dim=-1)` normalises each source-position row. `device=q.device` keeps the mask beside the input, on CPU or GPU.

The lab passes an explicit blocked-position mask instead of this causal flag. PyTorch's built-in `scaled_dot_product_attention` implements the same core operation and selects a backend suited to the inputs.

## 7. Three architecture families

![Information routes in three families](fig:three_archs)

| Family | Information access | Typical task | Examples |
|---|---|---|---|
| Encoder-only | Both sides of supplied input | Classify, tag, form trained text embeddings | BERT, RoBERTa |
| Decoder-only | Current and earlier positions | Generate a continuation | GPT-2, Llama, Qwen |
| Encoder–decoder | Read source; generate with causal and source attention | Translate, summarise or transcribe | T5, BART, Whisper |

![Compare inputs, outputs and uses](fig:architecture_comparison)

Architecture is a useful guide, not an exclusive task rule. Decoders can translate through prompting. Retrieval can combine a trained text-embedding model for search and a decoder for answers.

Standard BERT is not a chat generator. Token lookup vectors are not automatically effective search representations; text embeddings need appropriate training and evaluation.

### Published sizes: optional reference

These are historical specifications, not results from our lab.

| Model | Layers | Width | Heads | Parameters |
|---|---|---|---|---|
| Transformer base | 6 encoder + 6 decoder | 512 | 8 | 65 million |
| BERT-base | 12 | 768 | 12 | 110 million |
| GPT-2 small | 12 | 768 | 12 | 124 million |
| GPT-3 | 96 | 12,288 | 96 | 175 billion |
| Llama 3 8B | 32 | 4,096 | 32 query; 8 KV | About 8 billion |

BERT-base has 512 input positions; GPT-2 small has 1,024; GPT-3 has 2,048. Llama 3's original 8B release has 8K, while Llama 3.1 extends to 128K. These are release-specific specifications, not guarantees of reliable context use.

Grouped-query attention shares K/V heads across query heads. Do not quote guesses for unpublished proprietary parameter counts or architectures.

## 8. Training supplies targets; generation extends a prefix

### Objectives from the text itself

**Masked language modelling:** BERT predicts selected tokens using both sides. The original recipe selects 15% of tokens; not every selected token becomes a mask symbol.

**Next-token prediction:** GPT-style training predicts each next token from its allowed prefix. CharGPT uses input cat and targets ats from cats.

**Span corruption:** T5 replaces spans with special markers, then trains the decoder to generate the removed text.

These objectives are **self-supervised** because targets come from the data itself.

### Trace one training step

1. `get_batch` provides input IDs and target IDs shifted one place ahead.
2. Token and position lookup produce input vectors. Targets bypass these operations.
3. Causal blocks produce contextual vectors.
4. The vocabulary head produces next-character scores at every position.
5. Cross-entropy compares scores with the known target IDs.
6. Backpropagation computes gradients: how parameter changes affect loss.
7. The optimizer uses gradients to update parameters; a fresh batch begins another step.

The target branch ends at the loss. Targets do not travel through the input representation path.

### Worked loss and perplexity

This is a classroom calculation, not a lab measurement. Suppose known next characters receive probabilities 0.5, 0.25 and 0.25.

Their negative natural logarithms are approximately 0.693, 1.386 and 1.386. Average loss is approximately 1.155 nats per character. Exponentiating gives perplexity approximately 3.175.

A **nat** uses the natural logarithm. **Perplexity** describes uncertainty equivalent to about 3.175 uniformly likely choices. It does not imply that exact candidate count at every position.

$$\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_{t=1}^{N} \, \log p_\theta(x_t\mid x_{<t})\right)$$

PPL is perplexity. N counts scored tokens; t indexes a token. x with subscript t is the known token; x with subscript before t is its preceding context. p with subscript $\theta$ (theta) is the model probability with parameters $\theta$. Log is the natural logarithm; exp is its inverse. The minus sign converts log-probability into loss.

Use text kept separate from training. Compare the same tokenizer, test text and procedure. Character and subword perplexity are different measurements. Ordinary causal perplexity is not directly defined for masked BERT.

### Stored measured lecture experiment

| Quantity | Stored measured value |
|---|---|
| Parameters | {{w05_params}} |
| Training steps | {{w05_steps}} |
| Recorded runtime | {{w05_minutes}} minutes on {{w05_device}} |
| Vocabulary | {{w05_vocab}} characters |
| Initial validation loss | {{w05_val_start}} nats per character |
| Final validation loss | {{w05_val_end}} nats per character |
| Final validation perplexity | {{w05_ppl_end}} |

These values come from the stored metrics, not a promised outcome for every runtime. A full run of the lab notebook on {{w05_lab_device}} ({{w05_lab_date}}) reached a final validation loss of {{w05_lab_val}} (perplexity {{w05_lab_ppl}}) in {{w05_lab_minutes}} minutes of training. Each check averages only {{w05_lab_batches}} random validation batches, so repeated runs differ slightly. The recorded experiment uses a four-layer model, six heads, width 192 and context 128.

![Stored measured validation curve](fig:chargpt_loss)

**Stored initial sample:**

{{w05_sample_start}}

**Stored final sample:**

{{w05_sample_end}}

These are generated text, not quotations from Shakespeare. Inspect spelling, speaker labels, punctuation and failures of meaning. Low loss alone does not establish factual knowledge or coherence.

### Generation loop

Encode a prefix, keep its final CTX IDs and run the model with fixed weights. Only final-position logits predict the next character. Temperature rescales scores; softmax makes probabilities; sampling chooses an ID. Append it and repeat.

Set eval mode to disable dropout. Use no_grad to avoid gradient recording. They do different jobs: no_grad alone leaves dropout active in a model still in training mode.

## 9. Efficiency and other modalities

### Save storage or restrict comparisons

Full attention makes n squared comparisons. The memory plot assumes one stored score matrix per head, with two bytes per entry. It is theoretical, not a device benchmark.

![Theoretical score-table memory and stated assumptions](fig:complexity)

**FlashAttention** calculates exact attention in tiles without storing the entire score table in GPU memory. It saves storage and memory traffic, not all quadratic arithmetic.

**Sliding-window attention** restricts attention to nearby positions. Sparse patterns add selected wider routes. They change which comparisons occur.

**Grouped-query attention** shares K/V heads to reduce cache memory. **State-space models**, such as Mamba, use another sequence-processing design; hybrids can combine this with attention.

Long context needs appropriate positions, training and testing. Available space does not guarantee reliable use of every detail. Retrieval can provide a shorter relevant context.

### Reuse earlier K/V

A **KV cache** stores processed tokens' keys and values at each layer. The newest known token supplies Q/K/V. Its query uses earlier and current K/V to predict a new token.

![Known input, cached representations and next-token prediction](fig:kv_cache)

Earlier K/V are reused rather than recomputed. The new query still compares against retained positions. Cache memory grows with context, layers, KV heads and feature width.

CharGPT recomputes its context. The optional GPT-2 experiment measures caching separately. Warm up both settings, synchronise GPU work around timing and generate the same token count. Do not promise a fixed speed-up.

### Select a few experts

A **mixture-of-experts**, or MoE, replaces a feed-forward layer with expert networks and a learned router. The router selects a few per token. Routing weights combine their outputs.

![Selected experts and weighted combination](fig:moe)

For scale, Mixtral 8×7B reports about 47 billion total parameters and 13 billion active per token. DeepSeek-V3 reports 671 billion total and 37 billion active. These are published model specifications, not our lab results.

Expert weights still need storage. Routing, load balancing and moving weights can add costs. Learned experts need not have identifiable human specialties.

### Images and speech as sequences

A **Vision Transformer**, or ViT, embeds image patches and supplies positions. Its encoder exchanges information between patch vectors. Patch size is a configuration choice, not always sixteen pixels.

![Patches become positioned vectors](fig:vit_patches)

Whisper encodes audio spectrogram features and decodes transcript text. A spectrogram describes sound energy across time and frequency.

Diffusion transformers process image-latent patches. Transformers also appear in coding, time series, proteins, weather and robotics. Operations may be reusable even when input representations differ.

# Common misconceptions and FAQ

| Mistaken claim or question | Correction |
|---|---|
| “Attention chooses one word.” | It forms a weighted mixture of allowed values. |
| “Keys carry the transferred information.” | Keys make scores; values supply the mixture. |
| “Zero score means blocked.” | exp(0) is one. Blocked scores are negative infinity before softmax. |
| “A decoder cannot read its current token.” | It reads that input to predict the following token. |
| “RoPE is added to input embeddings.” | It rotates query/key feature pairs. |
| “Heads and layers are the same.” | A layer contains heads; they are separate axes. |
| “Each head has one human specialty.” | Learned roles can be mixed or unclear. |
| “Residuals join feature lists.” | Residuals add matching entries; concatenation joins lists. |
| “All blocks share weights.” | Ordinary blocks share structure but learn separate weights. |
| “no_grad disables dropout.” | eval controls dropout; no_grad controls gradient recording. |
| “Unchanged early attention proves a later clue was missed.” | Future clues are blocked there; test a later position. |
| “Low perplexity means truthful answers.” | It measures text predictability for a particular tokenizer and procedure. |
| “Caching makes the model smarter.” | It avoids repeated work while using memory. |
| “Unselected experts need no storage.” | Their weights still require storage in the system. |
| “Long context guarantees reliable memory.” | Reliability must be measured on the intended task. |

# Responsible AI lens

Describe attention maps as routing evidence and disclose selected heads. Do not infer explanation from colour alone. Report experiment inputs, model and limitations.

Report compute costs honestly. Choose a model adequate for the learning task. Language-dependent token counts can affect context capacity and costs.

Separate classroom calculations from measured results. Keep actual failures in generated samples so students practise evaluating evidence.

# Lab guide and answers

**Runtime:** Free Colab T4 GPU, with a CPU fallback using fewer steps. **Downloads: Tiny Shakespeare** text and the separate BERT and GPT-2 inspection models. On the 4 GB GTX 1650 laptop GPU used to check this pack (9 October 2026), the complete notebook ran in about 10 minutes with the models and data already downloaded. Download and training time depend on network and hardware. **Hand-in:** checks, loss curve, measured perplexity, generated samples, attention plots and explanations.

Follow **read → predict → run → change → check**. Before each cell, name inputs, expected shape, expected output and a failure check.

| Function or object | Purpose | Check |
|---|---|---|
| attention | Compare Q/K; mix V | Rows sum to one; result matches the classroom example and PyTorch. |
| causal_mask | Mark later positions | Future weights are zero. |
| MultiHeadAttention | Split, attend and rejoin | Output returns to [batch, positions, width]. |
| Block | Normalise, attend, transform and add | Changed future inputs do not affect earlier outputs. |
| get_batch | Make inputs and shifted targets | Decode one pair and inspect its offset. |
| val_loss | Score separate text | Use evaluation mode; record this run's loss. |
| CharGPT.generate | Append sampled characters | Use eval mode; inspect temperature effects. |
| bert_att | Store pretrained attention | Name layer, head, query and source axes. |

## TODO and written answers

**TODO 1:** Calculate `q @ k.transpose(-2, -1)`, divide by the square root of the query/key width, mask blocked positions and apply a row softmax. Return `weights @ v` and `weights`.

**TODO 2:** Use `torch.triu` on a boolean square table with `diagonal=1`. Keep its device consistent with the other tensors.

**TODO 3:** Reshape [b, n, d] to [b, n, h, d/h], then transpose to [b, h, n, d/h]. Attend, transpose back and reshape. b counts examples; n positions; d width; h heads.

**TODO 4:** Apply `math.exp` to your final measured validation loss. Compare against this run's uniform-vocabulary baseline.

**TODO 5:** Find the maximum it-to-animal weight over layers and heads. Disclose that this selects the strongest head. With the lecture's library versions this is layer {{w05_bert_layer}}, head {{w05_bert_head}} ({{w05_attention_weight}}).

**Question 1:** Unrestricted attention without positions is permutation-equivariant. Sinusoidal and learned absolute vectors are added; RoPE rotates Q/K. A causal mask also changes allowed routes.

**Question 2:** Interpret your measured perplexity and inspect actual samples. Do not assume spelling, style or meaning improved without evidence.

**Question 3:** Attention alone does not explain an output. Selection and other network operations limit the inference. GPT-2's early row cannot use a later adjective.

## Troubleshooting

For shape errors, print named axes after every reshape and transpose. A transposed tensor may need reshape instead of view.

If attention weights are unavailable, load with attn_implementation="eager". Optimised backends may not return full weight tables.

SDPA chooses a backend appropriate to the inputs and hardware; a fast kernel is not guaranteed for every GPU or dtype. For its dropout option, explicitly pass dropout_p=0.0 during evaluation.

A repeated offline text fallback can overlap across train and validation content. Treat that run as a demonstration; its score does not establish generalisation to new text.

# Practice questions with model answers

## Multiple choice

1. Four queries attend over six source positions. The score table has shape: **(a)** [4, 6]; **(b)** [6, 4]; **(c)** [4, 4]; **(d)** [6, 6]. *Answer: (a); one row per query and one column per source position.*
2. A layer of width 32 is split into four attention heads. Each head works with vectors of width: **(a)** 2; **(b)** 4; **(c)** 8; **(d)** 32. *Answer: (c); [2, 10, 32] becomes [2, 4, 10, 8].*
3. FlashAttention mainly saves: **(a)** the pairwise query–key comparisons themselves; **(b)** the parameters of the attention layers; **(c)** the need for a causal mask in training; **(d)** writing the full score table to GPU memory. *Answer: (d); it still computes exact attention, in tiles.*
4. During generation, a KV cache stores: **(a)** keys and values of earlier tokens, for reuse; **(b)** queries of earlier tokens, for reuse; **(c)** earlier answers, to repeat them for the same prompt; **(d)** gradients from training, to update weights faster. *Answer: (a).*

## Short answer

1. **Scaled scores [2, 0, 0]: calculate the softmax weights.** (3 marks) *Model answer:* exponentiate to approximately [7.39, 1, 1] (1); divide by their sum, about 9.39 (1); the weights are approximately [0.787, 0.107, 0.107] (1).
2. **From the word "cats", give the training inputs and next-character targets, and say what the first input position can read.** (3 marks) *Model answer:* inputs c, a, t (1); targets a, t, s (1); with the causal mask the first input position can read c only (1).
3. **In "The animal avoided the street because it was tired", why can the attention at "it" in a causal model stay unchanged when "tired" becomes "wide"?** (2 marks) *Model answer:* the causal mask blocks "it" from attending to later positions (1), so changing a later word changes neither its scores nor its output (1).
4. **Contrast training with generation, and describe how you could test whether one attention head matters.** (4 marks) *Model answer:* training compares predictions with known targets and updates parameters (1); generation keeps parameters fixed and extends a prefix one token at a time (1); remove or change the head under controlled conditions (1) and measure the effect on outputs, because an attention picture shows weights, not causal importance (1).

## Exam-style question

**"Trace text through a transformer, calculate one attention row, and explain heads, masks and positions. Discuss what an attention picture can establish."** (20 marks)

*Marking guide:* inputs, token IDs and tensor shapes (4); one attention row: scores, scaling, softmax weights and the value mix (5); heads, the causal mask, positions and residual addition (5); training with shifted targets versus fixed-weight generation (3); limits of attention pictures, with selected evidence disclosed and no claim that attention alone proves reasoning (3).

# Glossary

| Term | Plain meaning |
|---|---|
| Token | One unit the model reads. |
| Tokenisation | Split text and assign vocabulary IDs. |
| Embedding | A learned vector; token tables and whole-text embeddings have different roles. |
| Vector | An ordered list of numbers. |
| Matrix | A numerical table with row and column axes. |
| Tensor | A numerical array with one or more axes. |
| Shape | The length of each axis. |
| Parameter | A learned number changed during training. |
| Query / key / value | Matching request / matching source / transferred information. |
| Dot product | Multiply matching entries and add their products. |
| Softmax | Turn scores into nonnegative weights summing to one. |
| Attention | Mix values using query-key weights. |
| Self-attention | Q/K/V come from one sequence. |
| Cross-attention | Queries read K/V from another sequence. |
| Head | One set of projections and its mixture. |
| Causal mask | Prevent a position reading later positions. |
| Positional encoding | Information about sequence positions. |
| RoPE | Position-dependent Q/K feature-pair rotations. |
| Residual connection | Add a sublayer output to its input route. |
| Layer normalisation | Adjust feature scale within a token vector. |
| Feed-forward MLP | A learned network applied separately at each position. |
| Logit | An unnormalised output score. |
| Target | A known answer used to score a prediction. |
| Cross-entropy | Loss based on probability assigned to known targets. |
| Gradient | How a parameter change affects loss. |
| Optimizer | A rule using gradients to update parameters. |
| Self-supervised | Targets come from the data itself. |
| Perplexity | exp of average negative log-probability. |
| KV cache | Stored K/V for processed tokens. |
| FlashAttention | Exact attention with less score-table storage and memory traffic. |
| Mixture-of-experts | A router chooses a few feed-forward networks per token. |
| ViT | Transformer reading image-patch vectors. |
| Spectrogram | Sound energy across time and frequency. |
| Ablation | Remove a component and measure its effect. |

# Readings, videos and further practice

Start with a visual explanation; then practise the numerical example. Consult official documentation when implementing code.

- [3Blue1Brown: Transformer overview](https://www.youtube.com/watch?v=wjZofJX0v4M).
- [3Blue1Brown: Attention step by step](https://www.youtube.com/watch?v=eMlx5fFNoYc).
- [Karpathy: Build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- [StatQuest: Transformer networks](https://www.youtube.com/watch?v=zxQyTK8quyY).
- [Hugging Face: Transformer architectures](https://huggingface.co/learn/llm-course/chapter1/6).
- [Stanford CS224N: 2026 transformer lecture](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf).
- [Jurafsky and Martin: Transformer chapter](https://web.stanford.edu/~jurafsky/slp3/9.pdf).

Primary technical references:

- [Vaswani et al.: Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Su et al.: RoFormer and RoPE](https://arxiv.org/abs/2104.09864).
- [Dao et al.: FlashAttention](https://arxiv.org/abs/2205.14135).
- [Jain and Wallace: Attention is not Explanation](https://aclanthology.org/N19-1357/).
- [Dosovitskiy et al.: Vision Transformer](https://arxiv.org/abs/2010.11929).
- [Radford et al.: Whisper](https://arxiv.org/abs/2212.04356).
- [Brown et al.: GPT-3](https://arxiv.org/abs/2005.14165).
- [Meta: The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).
- [Jiang et al.: Mixtral of Experts](https://arxiv.org/abs/2401.04088).
- [DeepSeek-AI: DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

Official implementation references:

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
- [Hugging Face attention backends](https://huggingface.co/docs/transformers/attention_interface).
- [Hugging Face: How caching works](https://huggingface.co/docs/transformers/cache_explanation).
- [Hugging Face: Perplexity](https://huggingface.co/docs/transformers/perplexity).

Descriptor book reading remains Tunstall, von Werra and Wolf, Natural Language Processing with Transformers, chapter 3; Foster, Generative Deep Learning, chapter 9; and Rothman, Transformers for Natural Language Processing, chapters 1–2. [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) offers a further visual recap.

# Link to the group project

Identify model roles: search representations, answer generation, image encoding or speech transcription. Draw inputs, operations and outputs before selecting a model. Record context limits, memory needs and evaluation questions for deployment and retrieval.

Attention connects distant tokens directly, supporting long-range dependencies without a recurrent chain.
