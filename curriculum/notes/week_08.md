# Lecture plan at a glance

This week is the core of MIMLO 3: adapting pretrained models by changing their weights. Students should leave able to decide when fine-tuning is appropriate, explain SFT, instruction tuning, RLHF, DPO and RL with verifiable rewards, compute LoRA parameter counts, fine-tune with PEFT/TRL, and evaluate base vs fine-tuned models honestly. The results in section 5 are real, from running this week's lab notebook in advance (saved to `curriculum/assets/week_08/results.json`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | Brand voice / weekly rates / 50,000 messages: prompt, retrieve or fine-tune? |
| 5–18 min | 5–8 | 1 · When to fine-tune | Comparison table; decision process; good and bad reasons. |
| 18–38 min | 9–14 | 2 · SFT and instruction tuning | Full vs SFT; loss with masking; instruction tuning; chat template; data preparation. |
| 38–63 min | 15–21 | 3 · Preference alignment | RLHF pipeline; reward model and KL objective; DPO; three approaches; AI feedback and failure modes; quiz. |
| 63–70 min | – | Break | |
| 70–95 min | 22–29 | 4 · PEFT | LoRA diagram and equations (worked count); lab model counts; QLoRA; method comparison; memory table; distillation. |
| 95–118 min | 30–37 | 5 · Domain adaptation | Workflow; pitfalls; real results and loss curve; evaluation; domain models; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Do the LoRA parameter count (16 × (4096 + 4096) = 131,072 vs 16,777,216) on the board; it is a common exam calculation.

<!-- pagebreak -->

# Lecture notes

## 1. When to fine-tune

| | Prompting | Retrieval (RAG) | Fine-tuning |
|---|---|---|---|
| Changes | Input | Input (adds documents) | Weights |
| Solves | Task description, format | Missing/changing knowledge | Behaviour, style, format, specialised skills |
| Data | A few examples | Document collection | Hundreds–thousands of labelled examples |
| Update | Edit text | Re-index | Re-train and re-evaluate |
| Traceability | High | High (citations) | Low |

**Decision process.** Start with a prompt and an evaluation set. If quality is insufficient, diagnose: missing or outdated knowledge → retrieval; behaviour, format, style or a specialised skill that examples cannot teach, or a need for a smaller cheaper model → fine-tuning. Combine approaches as needed.

**Good reasons:** consistent format/style; specialised classification or extraction; domain language; replacing a large prompted model with a small tuned one (cost, latency, privacy); reliable tool calling; new languages. **Bad reasons:** adding frequently changing facts; "fixing hallucination" in general (fine-tuning on new facts can increase it); tiny datasets without an evaluation set; licence restrictions.

![Decision process](fig:decision)

## 2. Supervised and instruction fine-tuning

**Full fine-tuning** updates all weights; **supervised fine-tuning (SFT)** describes training on (prompt, response) pairs and can be done with full fine-tuning or with adapters.

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{(x, y)} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

Prompt tokens are masked out of the loss (labels set to be ignored), so the model learns to produce responses, not to reproduce prompts. TRL does this automatically when data are provided as `prompt`/`completion` pairs.

**Instruction tuning** fine-tunes on many tasks expressed as instructions (FLAN, 2021; the SFT stage of InstructGPT, 2022), giving general instruction following, including for unseen tasks. **Chat templates** wrap messages in model-specific special tokens; training and inference must use the same template (`tokenizer.apply_chat_template`). **LIMA** (Zhou et al., 2023) showed that 1,000 carefully curated examples can yield strong assistant behaviour: quality beats quantity.

**Data preparation checklist:** correct, consistent labels; coverage of real input variety and edge cases; clean train/validation/test splits without near-duplicates (leakage); personal data removed and rights to use data confirmed; class balance considered; a datasheet documenting source, size, labelling and gaps.

## 3. Preference alignment

### RLHF

1. SFT on demonstrations.
2. Humans compare model answers to the same prompt.
3. Train a **reward model** $r_\phi$ on the comparisons (Bradley–Terry):

$$\mathcal{L}_{\mathrm{RM}} = -\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)$$

**Step 4.** Optimise the policy with RL (PPO), maximising reward while staying close to the reference model:

$$\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta\, D_{\mathrm{KL}}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right)$$

The KL penalty prevents **reward hacking** (exploiting reward-model flaws). In InstructGPT (Ouyang et al., 2022), people preferred a 1.3 B-parameter InstructGPT over the 175 B GPT-3.

![RLHF pipeline](fig:rlhf)

### Direct Preference Optimisation (DPO)

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)$$

DPO (Rafailov et al., 2023) optimises the same objective as RLHF under its assumptions without a separate reward model or RL loop: simpler and more stable. Variants include IPO, KTO, ORPO and SimPO.

### RL with verifiable rewards

When correctness can be checked automatically (maths answers, unit tests), the reward needs no human labels. **GRPO** (group relative policy optimisation, DeepSeek) samples several answers per problem and rewards those better than the group average. DeepSeek-R1 (2025) showed that extended step-by-step reasoning emerges from such training: this is how reasoning models are made.

### AI feedback and failure modes

**RLAIF / Constitutional AI** (Bai et al., 2022): a model critiques and ranks outputs against written principles, reducing the need for human labels. Failure modes: reward hacking (e.g. verbosity), sycophancy, and the question of whose preferences are encoded.

## 4. Parameter-efficient fine-tuning

### LoRA

$$h = W x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)$$

Only $A$ and $B$ are trained; $W$ is frozen; $B$ is initialised to zero so training starts from the base model. Trainable parameters per matrix: $r(d + k)$ instead of $dk$.

**Worked example:** $d = k = 4096$, $r = 16$: $16 \times 8192 = 131{,}072$ trainable vs $16{,}777{,}216$ frozen (0.78%).

After training, merge $BA$ into $W$ (no inference overhead) or keep adapters separate and swap them per task. Typical hyper-parameters: $r$ = 4–64, $\alpha \approx 2r$, dropout 0.05, learning rate $10^{-4}$–$3\times10^{-4}$, targets = attention projections (and sometimes MLP layers). The LoRA paper reported 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning of GPT-3 with comparable quality.

**The lab model:** {{LORA_PARAMS}}

![LoRA](fig:lora)

![Trainable parameters for the lab model](fig:param_compare)

### QLoRA

Store the frozen base in 4-bit NormalFloat (NF4) and train LoRA adapters in 16-bit; double quantisation and paged optimizers reduce memory further (Dettmers et al., 2023). QLoRA fine-tuned a 65 B model on a single 48 GB GPU; 7–8 B models fit on a free Colab T4.

### Other methods

| Method | Trained | Pros | Cons |
|---|---|---|---|
| LoRA | Low-rank updates | Strong, mergeable, swappable | Rank/target choices |
| QLoRA | LoRA on a 4-bit base | Large models on small GPUs | Slower; small quality cost |
| Adapters (Houlsby et al., 2019) | Bottleneck layers in each block | Modular | Extra latency |
| Prompt/prefix tuning | Soft prompt vectors | Tiny | Weaker on small models |
| Full fine-tuning | All weights | Maximum flexibility | Memory, cost |

### Memory rules of thumb (7–8 B model)

Full fine-tuning with Adam in mixed precision needs roughly 16 bytes per parameter (≈ 100+ GB); LoRA with a 16-bit base ≈ 18–24 GB; QLoRA ≈ 7–12 GB. Activations depend on sequence length and batch size.

### Knowledge distillation

A small **student** imitates a large **teacher** by matching its output distributions (soft labels; Hinton et al., 2015) or by fine-tuning on teacher-generated answers. DeepSeek-R1 was distilled into small Qwen and Llama models. Students inherit teachers' errors and biases; some API terms forbid training competing models on outputs.

## 5. Domain adaptation in practice

**Workflow:** define task and metric (held-out test set first) → prepare data → prompting baseline → LoRA SFT with validation monitoring → evaluate task metric, general abilities, safety and cost → iterate or deploy.

**Pitfalls:** overfitting (validation loss rises); catastrophic forgetting (loss of general ability); data leakage; chat-template mismatch; learning rate too high; weak evaluation.

### Real results from the lab (instructor run)

{{RESULTS_NOTES}}

![Prompting vs LoRA fine-tuning](fig:results_bar)

![Training and validation loss](fig:loss_curve)

### Evaluation after fine-tuning

Task metric on held-out data vs the prompting baseline (with margins of error); a general-capability regression suite; safety tests before and after; robustness (paraphrases, typos, new periods); cost and latency; expert review with a rubric.

### Domain models

BloombergGPT (2023; 50 B parameters, mixed financial and general data) is an example of heavy domain training. Today, strong general models plus retrieval often match or beat older domain models; fine-tuning is used mainly for format, cost and specialised skills. Domain data is usually sensitive: governance comes first.

# Common misconceptions

| Misconception | Correction |
|---|---|
| "Fine-tuning is how you add new facts." | Retrieval handles changing facts better; fine-tuning on new facts can increase hallucination. |
| "More epochs are better." | Overfitting and forgetting grow; monitor validation loss. |
| "LoRA is a different model architecture." | It adds a low-rank update to existing weights; it can be merged back. |
| "RLHF teaches the model facts." | It shapes preferences and behaviour; knowledge comes mainly from pre-training. |
| "DPO needs no data." | It needs preference pairs, but no reward model or RL loop. |
| "A fine-tuned model is as safe as its base." | Fine-tuning can erode safety; re-test. |
| "Good training loss means a good model." | Only held-out evaluation tells you. |

# Responsible AI lens: fine-tuning

* **Safety erosion:** fine-tuning aligned models, even on benign data, can weaken refusal behaviour (Qi et al., 2023). Re-run safety tests.
* **Memorisation and privacy:** remove personal data; test for regurgitation.
* **Bias amplification:** audit performance by group; examine label sources.
* **Licences and terms:** check base-model licences and API terms for fine-tuning and output use.
* **Documentation:** publish a model card for the adapter (data, evaluation, limitations, intended use).

# Lab guide and answers

**Runtime:** Colab T4 (training ≈ 5–8 min, evaluation ≈ 2–4 min per condition). CPU only for the smoke test (10 steps). Dataset: `mteb/banking77` (CC-BY-4.0 source data), 10 intents. **Hand-in:** comparison table, loss curve, confusion table, three answers.

## TODOs

* **TODO 1** `{"prompt": [{"role": "user", "content": f"{INSTRUCTION}\nMessage: {text}\nIntent:"}], "completion": [{"role": "assistant", "content": label}]}`.
* **TODO 2** `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM")`. Expect ≈ 0.4–0.5% trainable.
* **TODO 3** evaluate the tuned model with the zero-shot prompt and build the table.
* **TODO 4** sum file sizes in the adapter folder; model size = parameters × bytes per element.

## Expected results

See the real results above. Students should find zero-shot prompting around half correct with some invalid labels, few-shot prompting much better (the examples fix the format and show the boundaries), LoRA best with near-zero invalid outputs, remaining confusions between similar intents, and over-specialisation on general questions with the adapter on (fixed by switching it off).

## Troubleshooting

* *`SFTConfig` argument errors:* library versions change; this notebook uses `warmup_steps` (not `warmup_ratio`) and `max_length`.
* *Out of memory:* lower `per_device_train_batch_size` to 8 and add `gradient_accumulation_steps=2`.
* *Accuracy does not improve:* check that `completion` holds only the label and that the evaluation prompt matches the training prompt.

# Practice questions with model answers

## Multiple choice

1. LoRA trains: **(a)** all weights; **(b)** low-rank matrices added to frozen weights; **(c)** only the embedding layer; **(d)** a reward model. *Answer: (b).*
2. The KL term in RLHF: **(a)** speeds up training; **(b)** keeps the policy close to the reference model to limit reward hacking; **(c)** measures accuracy; **(d)** is used only in DPO. *Answer: (b).*
3. Which is the best use of fine-tuning? **(a)** Keeping up with daily news; **(b)** enforcing a consistent classification format for 50 subtle categories; **(c)** answering questions about this week's prices; **(d)** avoiding an evaluation set. *Answer: (b).*
4. QLoRA reduces memory mainly by: **(a)** removing layers; **(b)** storing the frozen base in 4-bit; **(c)** shorter prompts; **(d)** smaller vocabulary. *Answer: (b).*

## Short answer

1. **Compute the LoRA trainable parameters for a 2048 × 2048 matrix with rank 8, and the percentage.** (3 marks) *Model answer:* 8 × (2048 + 2048) = 32,768; 2048² = 4,194,304; 0.78%.
2. **Explain the difference between RLHF with PPO and DPO.** (4 marks) *Model answer:* RLHF trains a reward model on preferences and optimises the policy with RL under a KL penalty; DPO optimises the same objective directly with a loss on preference pairs, needing no reward model or RL loop; DPO is simpler and more stable.
3. **Describe two signs of overfitting in fine-tuning and two remedies.** (4 marks) *Model answer:* validation loss rises while training loss falls; held-out accuracy drops or general abilities degrade; remedies: fewer epochs/early stopping, lower learning rate or rank, more/diverse data, mixing general data.
4. **Why is retrieval preferred over fine-tuning for frequently changing facts?** (3 marks) *Model answer:* updates require only re-indexing; answers can cite sources; fine-tuning is slow to update, opaque and can increase hallucination on new facts.

## Exam-style question

**"A telecom company wants a model that classifies support tickets into 60 categories and drafts replies in its style. Propose and justify an adaptation strategy, including data, method, evaluation and risks."** (20 marks)

*Marking guide:* diagnosis of task needs and decision between prompting/RAG/fine-tuning (4); data preparation incl. splits, leakage, privacy (4); method: LoRA/QLoRA SFT for classification and style, optional DPO for reply preferences, retrieval for policies (4); evaluation: held-out metrics vs baseline, general/safety regression, human review, cost (5); risks: forgetting, safety erosion, bias, licences, documentation (3).

# Glossary

| Term | Meaning |
|---|---|
| Full fine-tuning | Updating all model weights |
| SFT | Supervised fine-tuning on prompt–response pairs |
| Instruction tuning | SFT on many instruction-formatted tasks |
| Chat template | Model-specific special-token format for messages |
| RLHF | Reinforcement learning from human feedback (reward model + PPO) |
| Reward model | Model predicting which response humans prefer |
| PPO | Proximal Policy Optimisation, the RL algorithm used in RLHF |
| DPO | Direct Preference Optimisation: preference loss without RL |
| GRPO / RLVR | Group relative policy optimisation / RL with verifiable rewards |
| Reward hacking | Exploiting flaws in a reward signal |
| LoRA | Low-rank adaptation: trainable low-rank update to frozen weights |
| QLoRA | LoRA on a 4-bit quantised base model |
| Adapter | Small trainable module inserted into a frozen network |
| Prompt tuning | Learning soft prompt vectors |
| Distillation | Training a student model to imitate a teacher |
| Catastrophic forgetting | Loss of prior abilities after fine-tuning |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Tunstall, von Werra & Wolf (2022) *NLP with Transformers*: chapters on fine-tuning (e.g. text classification, chapter 2; few-to-no labels, chapter 9).
* Ozdemir, S. (2023) *Quick Start Guide to LLMs*: chapters on fine-tuning and RLHF.

**Courses and documentation**

* [Hugging Face LLM Course: Supervised fine-tuning](https://huggingface.co/learn/llm-course/chapter11/1)
* [Hugging Face PEFT](https://huggingface.co/docs/peft/index) and [TRL](https://huggingface.co/docs/trl/index)
* [Hugging Face blog: Illustrating RLHF](https://huggingface.co/blog/rlhf)
* [Google ML Crash Course: LLMs](https://developers.google.com/machine-learning/crash-course/llm) (fine-tuning, distillation)

**Videos**

* [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (post-training, RLHF, reasoning)

**Papers**

* Ouyang et al. (2022) [InstructGPT](https://arxiv.org/abs/2203.02155); Rafailov et al. (2023) [DPO](https://arxiv.org/abs/2305.18290); Hu et al. (2021) [LoRA](https://arxiv.org/abs/2106.09685); Dettmers et al. (2023) [QLoRA](https://arxiv.org/abs/2305.14314); DeepSeek-AI (2025) [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

# Link to the group project

Groups whose evaluation (week 7) shows a behaviour or format gap may fine-tune a small open model with LoRA. The report must include the prompting baseline on the same held-out test set, the fine-tuning configuration, training/validation curves, general and safety regression checks, and a model card for the adapter. Groups that do not fine-tune must justify why prompting and/or retrieval suffice.
