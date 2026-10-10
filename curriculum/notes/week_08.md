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

Start with the vocabulary and overall flow. For each section below, read the simple explanation first, trace the worked example aloud, and ask students to name the input and output. Introduce the equation only after the operation makes sense. The detailed text retains the full syllabus, while the lab and checks show what each method can and cannot establish.

## 1. When to fine-tune

**Start here.** Diagnose the error before choosing a solution. Prompting changes the instructions; retrieval supplies source facts; fine-tuning changes learned behaviour. These methods can be combined. A final test set is needed before further training so you can tell whether the new behaviour helps.

**Small worked example.** A banking rate changes today: retrieve the current rate. The bank needs a stable ten-label routing format: compare prompts first, then consider training if errors persist.

| | Prompting | Retrieval (RAG) | Fine-tuning |
|---|---|---|---|
| Changes | Input | Input (adds documents) | Weights |
| Solves | Task description, format | Missing/changing knowledge | Behaviour, style, format, specialised skills |
| Data | A few examples | Document collection | Hundreds–thousands of labelled examples |
| Update | Edit text | Re-index | Re-train and re-evaluate |
| Traceability | High | High (citations) | Low |

**Decision process.** Try a prompt and define how to score it. Read the errors: missing or stale facts suggest document retrieval; inconsistent behaviour or a specialised task may justify fine-tuning. A small adapted model may also lower serving cost. Compare options with the same final test and combine them when useful.

**Good reasons to investigate fine-tuning:** stable output format, subtle classification/extraction, specialist language, reliable tool-call formatting or a measured need for a smaller model. **Poor reasons:** adding changing facts, expecting all false answers to disappear, training without a test, or using a model/data licence that does not permit the task.

![Decision process](fig:decision)

## 2. Supervised and instruction fine-tuning

**Start here.** Supervised fine-tuning means learning from inputs paired with desired answers. Full fine-tuning describes which weights change: all of them. These are different choices. This lab combines supervised examples with LoRA, which trains only added weights. The chat template puts messages into the format expected by the model.

**Small worked example.** For one correct answer token, predicted probability 0.8 gives loss −ln(0.8), about 0.223. Probability 0.2 gives about 1.609. Lower loss means the desired token received more probability; it is not a complete task score.

**Full fine-tuning** updates all weights; **supervised fine-tuning (SFT)** describes training on (prompt, response) pairs and can be done with full fine-tuning or with adapters.

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{(x, y)} \;\; \sum_{t=1}^{|y|} \, \log p_\theta(y_t \mid x, y_{<t})$$

The loss counts answer tokens in this setup. **Masking** means assigning ignored labels to the prompt tokens; the model still reads them as context. TRL supports this with `prompt`/`completion` pairs and the appropriate completion-loss setting. SFT can use other masking choices, so inspect the trainer rather than assuming every SFT run masks prompts.

**Instruction tuning** uses many instructions with desired replies, as in FLAN (2021) and InstructGPT (2022). It teaches general instruction-following patterns; new tasks still need evaluation. **Chat templates** add the model-specific role and end tokens. Use `tokenizer.apply_chat_template` consistently at training and prediction. LIMA (Zhou et al., 2023) reported strong results with 1,000 carefully reviewed examples; this demonstrates the value of useful data, not a universal minimum.

**Data preparation checklist:** correct, consistent labels; coverage of real input variety and edge cases; clean train/validation/test splits without near-duplicates (leakage); personal data removed and rights to use data confirmed; class balance considered; a datasheet documenting source, size, labelling and gaps.

## 3. Preference alignment

**Start here.** A preference pair contains two answers to the same prompt and a choice of which is better. RLHF first learns a scoring model and then updates the answer model. DPO uses the pairs directly. Verifiable-reward training uses checks such as known maths answers or code tests instead of asking people to rank every answer.

**Small worked example.** A rater prefers a concise correct answer over a longer incorrect one. The dataset records that choice; it does not assume length means quality. Ask which guideline the rater used and whose views the dataset represents.

### RLHF

1. SFT on demonstrations.
2. Humans compare model answers to the same prompt.
3. Train a **reward model** $r_\phi$ on the comparisons (Bradley–Terry):

$$\mathcal{L}_{\mathrm{RM}} = -\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)$$

**Step 4.** Optimise the policy with RL (PPO), maximising reward while staying close to the reference model:

$$\max_{\pi_\theta} \;\; \mathbb{E}_{y \sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta\, D_{\mathrm{KL}}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right)$$

The KL penalty discourages large changes from the starting model. It can reduce **reward hacking** (exploiting flaws in the scoring model), but does not prevent it. In InstructGPT (Ouyang et al., 2022), people preferred a 1.3 B-parameter InstructGPT over the 175 B GPT-3.

![RLHF pipeline](fig:rlhf)

### Direct Preference Optimisation (DPO)

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\right)$$

DPO (Rafailov et al., 2023) trains directly on preferred and rejected answers. Under the paper's assumptions it corresponds to the regularised RLHF objective, while removing separate reward-model fitting and the RL loop. It is often easier to implement, but still depends on preference quality and evaluation. Related approaches include IPO, KTO, ORPO and SimPO.

### RL with verifiable rewards

For tasks with automatic checks, such as known maths answers or code tests, rewards can come from those checks. **GRPO** (group relative policy optimisation) samples a group of answers and compares their rewards within that group. DeepSeek-R1 (2025) is an example of reasoning training with this approach. Reward quality and failure cases still matter.

### AI feedback and failure modes

**RLAIF / Constitutional AI** (Bai et al., 2022): a model critiques and ranks outputs against written principles, reducing the need for human labels. Failure modes: reward hacking (e.g. verbosity), sycophancy, and the question of whose preferences are encoded.

## 4. Parameter-efficient fine-tuning

**Start here.** LoRA adds a small correction to selected layers. The large base matrix stays fixed. One small matrix reduces the feature count, another expands it again, and their result is added to the original result. QLoRA also compresses the fixed base weights. Distillation instead teaches a smaller model to imitate a larger model.

**Small worked example.** A base layer with 8 inputs and 4 outputs has 8 × 4 = 32 weights. A rank-2 LoRA path has 2 × (8 + 4) = 24 trainable weights. With very large layers and small rank, the saving is much larger.

### LoRA

$$h = W x + \frac{\alpha}{r} B A x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)$$

Only $A$ and $B$ are trained; $W$ is frozen; $B$ is initialised to zero so training starts from the base model. Trainable parameters per matrix: $r(d + k)$ instead of $dk$.

**Worked example:** $d = k = 4096$, $r = 16$: $16 \times 8192 = 131{,}072$ trainable vs $16{,}777{,}216$ frozen (0.78%).

After training, merge $BA$ into $W$ (no inference overhead) or keep adapters separate and swap them per task. Typical hyper-parameters: $r$ = 4–64, $\alpha \approx 2r$, dropout 0.05, learning rate $10^{-4}$–$3\times10^{-4}$, targets = attention projections (and sometimes MLP layers). The LoRA paper reported 10,000× fewer trainable parameters and 3× less GPU memory than full fine-tuning of GPT-3 with comparable quality.

**The lab model:** {{LORA_PARAMS}}

![LoRA](fig:lora)

![Trainable parameters for the lab model](fig:param_compare)

### QLoRA

Store the frozen base in 4-bit NormalFloat (NF4) and train LoRA adapters in 16-bit; double quantisation and paged optimizers reduce memory further (Dettmers et al., 2023). QLoRA fine-tuned a 65 B model on a single 48 GB GPU; some 7–8 B configurations can fit on a 16 GB T4; check sequence length, batch, activations and measured peak memory.

### Other methods

| Method | Trained | Pros | Cons |
|---|---|---|---|
| LoRA | Low-rank updates | Strong, mergeable, swappable | Rank/target choices |
| QLoRA | LoRA on a 4-bit base | Large models on small GPUs | Hardware support; test speed and quality |
| Adapters (Houlsby et al., 2019) | Bottleneck layers in each block | Modular | Extra latency |
| Prompt tuning | Learned input vectors | Few trainable values | Task quality must be tested |
| Prefix tuning | Learned layer key/value prefixes | Few trainable values | Task quality must be tested |
| Full fine-tuning | All weights | Maximum flexibility | Memory, cost |

### Memory rules of thumb (7–8 B model)

Full fine-tuning with Adam in mixed precision needs roughly 16 bytes per parameter (≈ 120+ GB); LoRA with a 16-bit base ≈ 18–24 GB; QLoRA ≈ 7–12 GB. Activations depend on sequence length and batch size.

### Knowledge distillation

In **distillation**, a small student learns to imitate a larger teacher. It can learn from the teacher's probability distribution (**soft labels**; Hinton et al., 2015) or generated answers. DeepSeek-R1-distilled Qwen and Llama models are examples. The student may copy errors and bias. Check current data/model licences and provider terms before using outputs as training targets.

## 5. Domain adaptation in practice

**Start here.** Measure the best prompting baseline before training. Separate examples for learning, choosing settings and final evaluation. Watch validation loss, then inspect actual test replies. Also recheck general questions and safety, because learning one task can change other behaviour.

**Small worked example.** If training loss falls from 1.2 to 0.4 while validation loss rises from 0.9 to 1.3, investigate memorising the training data. Fewer passes, changed settings or more varied data may help; use the untouched test only for the final comparison.

**Workflow:** define task and metric (held-out test set first) → prepare data → prompting baseline → LoRA SFT with validation monitoring → evaluate task metric, general abilities, safety and cost → iterate or deploy.

**Pitfalls:** overfitting (validation loss rises); catastrophic forgetting (loss of general ability); data leakage; chat-template mismatch; learning rate too high; weak evaluation.

### Real results from the lab (instructor run)

{{RESULTS_NOTES}}

![Prompting vs LoRA fine-tuning](fig:results_bar)

![Training and validation loss](fig:loss_curve)

### Evaluation after fine-tuning

Task metric on held-out data vs the prompting baseline (with margins of error); a general-capability regression suite; safety tests before and after; robustness (paraphrases, typos, new periods); cost and latency; expert review with a rubric.

### Domain models

BloombergGPT (2023; 50 B parameters, mixed financial and general data) is an example of heavy domain training. Compare a general model plus retrieval with a specialist on the same domain test; fine-tuning is used mainly for format, cost and specialised skills. Domain data is usually sensitive: governance comes first.

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

**Runtime:** A GPU is recommended for the full lab. On the laptop CPU used to check this pack, training took about 76 minutes in the recorded run and about 53 minutes in a repeat run on 9 October 2026, which gave the same losses; the whole repeat notebook took about 84 minutes. Prepare a full CPU run before class or use the supplied recorded results for discussion. The tiny CPU smoke run uses six test messages, three training steps and batch size two to check that the pipeline executes. Its scores cannot establish model quality. Measure time on your hardware. Dataset: `mteb/banking77` (CC-BY-4.0 source data), 10 intents. **Hand-in:** comparison table, loss curve, confusion table, three answers.

## TODOs

* **TODO 1** `{"prompt": [{"role": "user", "content": f"{INSTRUCTION}\nMessage: {text}\nIntent:"}], "completion": [{"role": "assistant", "content": label}]}`.
* **TODO 2** `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM")`. Expect ≈ 0.4–0.5% trainable.
* **TODO 3** evaluate the tuned model with the zero-shot prompt and build the table.
* **TODO 4** sum file sizes in the adapter folder; model size = parameters × bytes per element.

## Expected results

See the real results above. The recorded experiment found a strong few-shot gain and a further LoRA gain; these are observations for this run, not guaranteed student outcomes. Inspect wrong and unparsed labels. The adapter-on/off question illustrates a possible behaviour change, not a complete test of earlier abilities.

## Troubleshooting

* *`SFTConfig` argument errors:* library versions change; this notebook uses `warmup_steps` (not `warmup_ratio`) and `max_length`.
* *Out of memory:* lower `per_device_train_batch_size` to 8 and add `gradient_accumulation_steps=2`.
* *Accuracy does not improve:* check that `completion` holds only the label and that the evaluation prompt matches the training prompt.

# Practice questions with model answers

## Multiple choice

1. LoRA trains: **(a)** all weights, with a smaller learning rate; **(b)** only the embedding layer and output head; **(c)** low-rank matrices added to frozen weights; **(d)** a reward model that scores the outputs. *Answer: (c).*
2. The KL term in RLHF: **(a)** speeds up training by removing the need for a reward model; **(b)** is used only in DPO and plays no part in PPO-based RLHF; **(c)** measures the policy's accuracy on the held-out preference data; **(d)** keeps the policy close to the reference model to limit reward hacking. *Answer: (d).*
3. Which is the best use of fine-tuning? **(a)** Enforcing a consistent classification format for 50 subtle categories; **(b)** keeping the model up to date with each day's news stories; **(c)** answering customer questions about this week's changing prices; **(d)** avoiding the need to build an evaluation set for the task. *Answer: (a).*
4. QLoRA reduces memory mainly by: **(a)** removing some of the base model's layers; **(b)** storing the frozen base in 4-bit; **(c)** using shorter prompts during training; **(d)** shrinking the tokenizer's vocabulary. *Answer: (b).*

## Short answer

1. **Compute the LoRA trainable parameters for a 2048 × 2048 matrix with rank 8, and the percentage.** (3 marks) *Model answer:* 8 × (2048 + 2048) = 32,768; 2048² = 4,194,304; 0.78%.
2. **Explain the difference between RLHF with PPO and DPO.** (4 marks) *Model answer:* RLHF trains a reward model on preferences and optimises the policy with RL under a KL penalty; DPO optimises the same objective directly with a loss on preference pairs, needing no reward model or RL loop; DPO is simpler and more stable.
3. **Describe two signs of overfitting in fine-tuning and two remedies.** (4 marks) *Model answer:* validation loss rises while training loss falls; held-out accuracy drops or general abilities degrade; remedies: fewer epochs/early stopping, lower learning rate or rank, more/diverse data, mixing general data.
4. **Why is retrieval preferred over fine-tuning for frequently changing facts?** (3 marks) *Model answer:* updates require only re-indexing; answers can cite sources; fine-tuning is slow to update, opaque and can increase hallucination on new facts.

## Exam-style question

**"A telecom company wants a model that classifies support tickets into 60 categories and drafts replies in its style. Propose and justify an adaptation strategy, including data, method, evaluation and risks."** (20 marks)

*Marking guide:* diagnosis of task needs and decision between prompting/RAG/fine-tuning (4); data preparation including splits, leakage, privacy (4); method: LoRA/QLoRA SFT for classification and style, optional DPO for reply preferences, retrieval for policies (4); evaluation: held-out metrics vs baseline, general/safety regression, human review, cost (5); risks: forgetting, safety erosion, bias, licences, documentation (3).

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
| Prompt tuning | Learning soft vectors at the input |
| Prefix tuning | Learning key/value prefixes at transformer layers |
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


# Detailed explanations behind the short slides

These explanations retain the detail moved off crowded slides. Read the short slide first, then use this reference for deeper discussion.

## Slide 4: Warm-up

- Which needs **knowledge** that changes often?

- Which needs **consistent behaviour or format**?

- Which is about **cost and latency** at high volume?

## Slide 6: Prompting vs. retrieval vs. fine-tuning

|  | Prompting | Retrieval (RAG) | Fine-tuning |
|---|---|---|---|
| Changes | The input | The input (adds documents) | The weights |
| Solves | Task description, format | Missing or changing knowledge | Behaviour, style, format, specialised skills |
| Data needed | A few examples | A document collection | Hundreds–thousands of labelled examples |
| Update cost | Edit text | Re-index documents | Re-train and re-evaluate |
| Traceability | High | High (citations) | Low (knowledge hidden in weights) |

**Key idea:** Use retrieved documents for changing facts. Fine-tuning changes learned behaviour and must be tested again.

## Slide 8: Good and bad reasons to fine-tune

**Good reasons**

- A task needs more consistent labels, style or format

- Prompts struggle with subtle category boundaries

- The model needs practice with specialist language

- A measured small-model adaptation reduces serving cost

- The task needs reliable tool-call formats or language support

**Bad reasons**

- You only need to add facts that change often

- You expect it to remove all false answers

- You lack enough useful examples or a separate test set

- You have no measured task problem to solve

- The model or data licence does not allow the intended use

## Slide 10: Full fine-tuning vs. supervised fine-tuning (SFT)

**Full fine-tuning**

- Describes **which weights** are updated: all of them

- Needs more memory for gradients and optimiser state

- Usually save a full trained model for each task

- Re-test older abilities after training

**Supervised fine-tuning (SFT)**

- Describes **what examples** are used: inputs with desired answers

- Can train all weights or only added adapters

- Learn by predicting the answer tokens

- This lab scores the answer tokens, while reading the prompt

## Slide 11: The SFT loss: next-token prediction on the answer

- Read one prompt x and its desired answer y. Raise the probability of each correct answer token.

- Ignore prompt tokens when adding up the error (**loss masking**); the model still reads them.

- Example: x = "What team handles a card payment?"; y = the correct banking intent label.

## Slide 12: Instruction tuning: from base model to assistant

- **Instruction tuning** uses many input instructions paired with useful answers (FLAN; InstructGPT).

- It teaches general instruction following; performance on new tasks still needs checking.

- A **chat template** adds the model's special role and end-of-message tokens. Use the same template at training and prediction.

- LIMA (2023) showed strong results with 1,000 carefully reviewed examples; useful data quality matters.

## Slide 13: What a chat template produces (Qwen family)

- Special tokens mark who is speaking and where a message ends.

- The **tokenizer** stores the template for that model family.

- Use tokenizer.apply_chat_template for training examples and new requests.

## Slide 14: Data preparation: where most of the quality comes from

- **Quality:** Check that labels and example answers are correct and consistent.

- **Diversity:** Include short, long, common and unusual inputs similar to real use.

- **Clean splits:** Use separate learning, settings-selection and final-test examples. Remove duplicates across them.

- **Privacy & rights:** Remove personal data where possible; check the data rights and base-model licence.

- **Balance:** Count examples per label. An uneven set can teach the model to favour frequent labels.

- **Document:** Record the data source, size, labelling method and known gaps in a datasheet.

## Slide 17: The reward model and the RL objective

- The **reward model** learns to give the preferred answer $y_w$ a higher score than the less-preferred answer $y_l$.

- The language model then generates replies that receive higher predicted reward.

- The **KL penalty** discourages a large change from the starting model. It reduces risk; it does not guarantee safe behaviour.

## Slide 18: Direct Preference Optimisation (DPO)

- DPO trains directly on a preferred and a rejected answer for the same prompt.

- Compare their probabilities with the fixed reference model; make the preferred answer relatively more likely.

- Under the paper's assumptions, this replaces reward-model fitting and the RL loop. Related methods: IPO, KTO, ORPO, SimPO.

## Slide 19: Three ways to learn from feedback

**RLHF (PPO)**

- Humans rank pairs; train a reward model; update the language model with PPO

- **PPO** is a reinforcement-learning update method

- Several training stages increase cost and complexity

**DPO and variants**

- Use ranked answer pairs directly in a preference loss

- No separate reward model or RL sampling loop

- Often simpler; still depends on data quality

**RL with verifiable rewards**

- Check maths answers or code tests automatically

- **GRPO** compares rewards within a sampled answer group

- Used in reasoning-model training, such as DeepSeek-R1

## Slide 20: AI feedback, and what can go wrong

- **RLAIF** uses AI feedback. In **Constitutional AI**, written principles guide critique and ranking (Anthropic, 2022).

- **Reward hacking** means improving the reward score without doing the intended job well.

- **Sycophancy** means agreeing with the user even when the user is wrong; preferences can reward it.

- Ask whose preferences were collected, under which instructions, and which groups were represented.

## Slide 24: LoRA in equations

- Keep the original matrix W fixed. Train two small matrices A and B to add a correction.

- For d = k = 4096 and r = 16: train 16 × (4096 + 4096) = 131,072 values, versus 16,777,216 in W.

- The correction is scaled by $\alpha/r$. For compatible layers, merge it into W after training or keep an adapter to switch tasks.

## Slide 25: How many weights does LoRA train? (the lab model)

- {{LORA_PARAMS}}

- Fewer trained weights need less optimiser memory and smaller saved adapters.

- The original LoRA paper reported 10,000 times fewer trained parameters and 3 times less GPU memory for its GPT-3 setup.

## Slide 26: QLoRA: fine-tune on top of a 4-bit model

- **QLoRA** stores the fixed base model with 4 bits per weight and trains LoRA adapters in higher precision.

- **NF4** is the base-weight number format. Double quantisation compresses scales; paged optimisers manage memory spikes.

- The paper trained a 65-billion-parameter model on a 48 GB GPU. Smaller models may fit on a 16 GB T4.

- Exact memory and quality depend on the model, sequence length and batch size. Measure them.

## Slide 27: Parameter-efficient methods compared

| Method | What is trained | Pros | Cons |
|---|---|---|---|
| LoRA | Low-rank updates to chosen weight matrices | Strong quality; mergeable; swappable adapters | Choose rank and target modules |
| QLoRA | LoRA on a 4-bit frozen base | Large models on small GPUs | Hardware support; test speed and quality |
| Adapters (Houlsby et al., 2019) | Small bottleneck layers inserted in each block | Modular; well studied | Extra inference latency |
| Prompt / prefix tuning | Prompt: input vectors; prefix: learned K/V at transformer layers | Tiny; one model, many tasks | Weaker on smaller models; less interpretable |
| Full fine-tuning | All weights | Maximum flexibility | Memory, cost, one full copy per task |

## Slide 28: Rough GPU memory to fine-tune a 7–8 B model

| Approach | Weights | Gradients + optimizer | Rough total | Hardware |
|---|---|---|---|---|
| Full fine-tuning (16-bit, Adam) | ≈ 15 GB | ≈ 100–110 GB | ≈ 120+ GB | Multiple data-centre GPUs |
| LoRA (16-bit base) | ≈ 15 GB | < 1 GB | ≈ 18–24 GB | One 24 GB GPU |
| QLoRA (4-bit base) | ≈ 4–5 GB | < 1 GB | ≈ 7–12 GB | Colab T4 (16 GB) |

**Key idea:** Rules of thumb only: activations depend on sequence length and batch size. Measure on your own setup.

## Slide 29: Knowledge distillation: big teacher, small student

- **Distillation** teaches a small "student" model to imitate a larger "teacher".

- Use the teacher's answer probabilities or its generated input-answer examples as training targets.

- It can reduce serving time and cost on the target task; the student may inherit errors and bias.

- Examples include DeepSeek-R1-distilled Qwen and Llama models. Check licences and provider terms before using outputs for training.

## Slide 31: A domain fine-tuning workflow

- Define task & metric: Inputs, outputs, success criteria; held-out test set first

- Prepare data: Clean, label, split, format with the chat template

- Baseline: Best prompting result on the same test set

- LoRA SFT: Small learning rate, few epochs, validation loss

- Evaluate & decide: Task metric, general-ability check, cost; iterate or deploy

## Slide 32: Common pitfalls

- **Overfitting:** The model learns training examples too closely. Rising validation loss is a warning; try fewer passes or more varied data.

- **Catastrophic forgetting:** The adapted model loses an earlier ability. Keep general checks and, where useful, mix in general training examples.

- **Data leakage:** A test example or near-duplicate appears in training, making the final score misleading.

- **Template mismatch:** Using a different chat format at prediction time can produce unexpected replies.

- **Learning rate:** The step size controls weight changes. Start small, monitor validation, and tune it on development data.

- **Weak evaluation:** A falling training loss alone does not show better answers. Use a separate test and inspect errors.

## Slide 33: Real results: prompting vs. LoRA fine-tuning (Banking77, 10 intents)

- {{RESULTS_SUMMARY}}

- Same base model (Qwen2.5-0.5B-Instruct), same held-out test messages, greedy decoding.

- The bar to beat is the **best prompt** (few-shot), not zero-shot: fine-tuning must earn its cost.

## Slide 34: Real training curve

- For this one-epoch run, answer-token training loss and validation loss both fell; the curve shows no clear overfitting.

- {{ADAPTER_SIZE}}

- A saved adapter contains the added weights; the base model is needed when loading it.

## Slide 35: Evaluating a fine-tuned model

| What to check | How |
|---|---|
| Task score | Use new labelled examples; compare accuracy, F1 or exact match with the best prompt. |
| Earlier abilities | Re-run a small general-question set or benchmark subset, such as MMLU. |
| Safety | Compare refusals and harmful-content responses before and after training. |
| New input styles | Test rewording, typos and messages from different channels or dates. |
| Time and cost | Measure reply length, seconds per request and hardware needs. |
| Human review | Ask a domain expert to score a sample using a written guide. |

## Slide 36: Domain-specific models: build, adapt or retrieve?

- Some models receive extensive domain training: BloombergGPT used financial and general data (50 B parameters, 2023).

- Other applications adapt a general model with specialist examples.

- Compare a strong prompted model plus retrieval before investing in further training.

- Specialist data can be sensitive; check privacy, permission and data governance before use.

## Slide 37: Responsible AI lens: fine-tuning

- **Safety can erode**: fine-tuning aligned models, even on benign data, can weaken refusals (Qi et al., 2023); re-test safety.

- **Memorisation**: fine-tuned models can reproduce training records; remove personal data, test for leakage.

- **Bias amplification**: skewed labels teach skewed decisions; audit performance by group.

- **Licences and terms**: base-model licences and API terms may restrict fine-tuning or use of outputs.

- **Documentation**: publish a model card for your adapter: data, evaluation, limits.

## Slide 38: Lab 8: LoRA fine-tuning for a banking domain task

- Prepare Banking77 messages for 10 **intents**: categories describing the customer's request. Keep a separate final test.

- Measure the unadapted model with zero-shot and few-shot prompts.

- Add LoRA paths with rank r = 16 using PEFT; count the trainable weights.

- Train with TRL's SFTTrainer; plot training and validation loss.

- Compare models on the same test; inspect mistaken labels in a confusion table.

- Check older abilities, turn the adapter off, and compare saved file sizes. Optional: QLoRA or DPO.

## Slide 39: Summary

- Start with a prompt; search documents for facts; train further when behaviour needs to change.

- SFT learns desired answers from input-answer pairs. Instruction tuning uses varied instructions.

- Use the model's chat template and separate train, validation and test examples.

- RLHF uses a reward model and RL; DPO learns directly from preferences; verified rewards can train reasoning.

- LoRA trains small added matrices; QLoRA also compresses the fixed base weights.

- Distillation teaches a small model to imitate a larger one.

- Compare with the best prompt on new examples; check lost abilities, safety and cost.

- Record the data, model licence, settings, scores and remaining limits.


## Implementation and recorded-run checks

SFT names the objective; full fine-tuning names which weights change. The lab explicitly sets completion_only_loss=True and the Qwen turn-end token, so prompt tokens are read but excluded from completion loss.

Prompt tuning adds learned vectors to input embeddings. Prefix tuning adds learned key/value prefixes at transformer layers. Keep those mechanisms separate. LoRA freezes the original base and trains low-rank matrices. QLoRA stores quantised base weights; its compute and adapters use higher precision. Memory depends on sequence, batch, activations and implementation.

The parameter chart reads the stored count and fails clearly if evidence is missing; it has no invented fallback or unmeasured rank 8 bar. Stored total includes adapters: original base count is total minus trainable. {{RUN_NOTE}} Evaluation adds more time. Falling validation loss is useful evidence but does not rule out all overfitting or lost abilities.


# Complete lecture code examples

## What a chat template produces (Qwen family)
```python
messages = [
    {"role": "system", "content": "You classify banking messages."},
    {"role": "user", "content": "I lost my card!"},
    {"role": "assistant", "content": "lost_or_stolen_card"},
]
print(tok.apply_chat_template(messages, tokenize=False))

# <|im_start|>system
# You classify banking messages.<|im_end|>
# <|im_start|>user
# I lost my card!<|im_end|>
# <|im_start|>assistant
# lost_or_stolen_card<|im_end|>
```
