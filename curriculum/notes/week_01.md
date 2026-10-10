# Lecture plan at a glance

This week sets up the vocabulary, the probability ideas and the habits of responsible use that every later week depends on. The pace is deliberately gentle: students arrive with different backgrounds, and the equations are short. The lab builds a real (tiny) generative model and runs a real open-weight language model.

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–8 min | 1–4 | Welcome and warm-up | Pair discussion: which AI tools helped or failed this week? Keep the "failed" list for section 4. |
| 8–30 min | 5–12 | 1 · The landscape | Definition, history, model vs. application, proprietary vs. open-weight, tools. Quiz at slide 12. |
| 30–48 min | 13–17 | 2 · Core concepts | Tokens, training vs. inference, the pre-training → post-training pipeline, scale. |
| 48–55 min | – | Break | |
| 55–90 min | 18–29 | 3 · Probabilistic foundations | Next-token distributions, chain rule, likelihood, worked example, temperature, decoding, generative vs. discriminative, latent variables. Quiz at slide 29. |
| 90–100 min | 30–32 | Model families and course map | Four families and the 12-week route. |
| 100–115 min | 33–37 | 4 · Responsible AI | Hallucination, six risks, five-step checklist, case discussion. |
| 115–120 min | 38–40 | Lab preview and wrap-up | Lab steps, key takeaways, resources. |

> **Teaching tip:** If time is short, compress slides 30–32 (the families return in weeks 2–6) rather than the probability section, which students need for the VAE maths next week.

<!-- pagebreak -->

# Lecture notes

## Start with one visible example

Write . e m m a . on the board. Ask the class to count its five adjacent pairs before naming a bigram. Count one pair in N, turn one row into P, then draw one character. Only then connect the same sequence to tokens and neural weights.

**Try it aloud.** A row [2, 1, 0] becomes [3, 2, 1] after smoothing, then [3/6, 2/6, 1/6]. Ask which entry supplies the most likely next character.

**Misconception to resolve.** Likelihood measures fit to a model, not factual truth. Lower temperature concentrates choices; it cannot check a citation.

**Explanation loop.** Explain one operation, ask students to predict its output, run it, compare the observation, then restate it in their own words. If predictions disagree, return to the same concrete row rather than adding new terminology.



## 1. The generative AI landscape

### What "generative" means

A **generative model** learns patterns in examples and uses them to create new examples. We write its model of possible data as $p(x)$: the probability of an example $x$. Outputs can be text, code, images, audio, video, 3-D shapes or molecules.

A **discriminative model** predicts a label $y$ for input $x$, written $p(y \mid x)$. A spam filter labels an email. A generative model writes an email.

Most systems in use today are **foundation models**: large models trained once on broad data and then adapted to many tasks. The term comes from a 2021 Stanford report. One foundation model can summarise, translate, answer questions and write code without a separate model for each task.

> **Key idea:** Generating means sampling from a learned probability distribution. VAEs, GANs, diffusion models and large language models (LLMs) are different ways of learning that distribution and sampling from it.

### A short history, and why it matters for this module

| Year | Milestone | Why it matters here |
|---|---|---|
| 2013 | Variational autoencoder (Kingma & Welling) | Latent variables and a bound on the likelihood (week 2) |
| 2014 | Generative adversarial network (Goodfellow et al.) | Adversarial training; sharp images (week 3) |
| 2017 | Transformer ("Attention Is All You Need") | The architecture behind modern LLMs (week 5) |
| 2018 | GPT-1 and BERT | Pre-train on raw text, then adapt (week 6) |
| 2020 | GPT-3; denoising diffusion (DDPM) | Few-shot prompting (week 7); diffusion (week 4) |
| 2021 | CLIP and DALL·E | Text–image alignment (week 9) |
| 2022 | Stable Diffusion (open weights); ChatGPT | Generative AI reaches the public |
| 2023 | GPT-4, Llama, Claude, Gemini | Multimodal models; strong open-weight models |
| 2024 | Reasoning models, video generation, Model Context Protocol (MCP) | Test-time compute; tools and agents (week 12) |
| 2025 | Open reasoning models (e.g. DeepSeek-R1); agent frameworks | Reinforcement learning for reasoning (week 8); agents |

Model names change every few months; the ideas in the left-hand column do not. Encourage students to learn the ideas and treat product names as examples.

### Model, application and tool

A **model** is the learned numerical system. An **application** is the product that uses it. For example, a chat app combines a model with instructions, tools and an interface. This distinction explains why two apps can use similar models but behave differently.

* **Model:** trained weights (parameters) that map an input to a probability distribution over outputs, e.g. a GPT, Llama or Gemini family model.
* **Application:** the model plus hidden **system instructions**, conversation memory, **retrieval** of documents or web results, **tools** (code execution, image generation, search), **safety filters** and a user interface. Two apps using the same model can behave very differently.
* **Tool / workflow:** an application embedded in a task, e.g. meeting summaries in a video-call platform or AI search in a library catalogue.

When a product "gets smarter overnight", ask which layer changed: the model, the instructions, the tools or the retrieval.

### Proprietary versus open-weight models

| | Proprietary (closed-weight) | Open-weight |
|---|---|---|
| Access | App or paid API; weights never released | Weights downloadable (subject to licence) |
| Examples (families) | GPT, Claude, Gemini | Llama, Qwen, Mistral, Gemma, DeepSeek |
| Strengths | Often the strongest general capability; no hardware needed | Privacy, control, can be fine-tuned and inspected; predictable cost at scale |
| Weaknesses | Data leaves your organisation; provider controls updates and retirements | You manage hardware, updates and safety; smaller models are less capable |

"Open-weight" is not the same as "open-source". Many open-weight releases do not publish training data or training code, and licences can restrict commercial use. In practice organisations combine both: a hosted frontier model for difficult tasks and a small open model for private or high-volume tasks. Week 11 returns to this decision.

### Everyday AI tools

Students will meet chat assistants (ChatGPT, Gemini, Claude, Microsoft Copilot, Le Chat), coding assistants (GitHub Copilot, Cursor, Claude Code, Gemini Code Assist), research and study tools (NotebookLM, deep-research modes, AI search engines), image and design generators (Adobe Firefly, Midjourney, Imagen), video and audio tools (Veo, Sora, ElevenLabs, Suno), and open models run locally through Hugging Face or Ollama. Products change monthly, so teach students to evaluate any tool with four questions: **which model, what data, which tools, and how are outputs verified?**

## 2. Core concepts: data, parameters, tokens

### Data and parameters

A neural network is a function with adjustable numbers called **parameters** (or weights). Modern LLMs have billions of them. **Training data** is the collection of examples used to set the parameters: web text, books, code, images, audio. Data quality, coverage and licensing shape what a model can do and which biases it inherits.

### Tokens

A **tokenizer** turns text into pieces called **tokens**, then gives each piece an integer ID. A token can be a word, part of a word or punctuation. The model reads those IDs. Each tokenizer has its own fixed **vocabulary**, its list of available pieces. The lab prints actual pieces so students can inspect them. One English token is often about three-quarters of a word, but count tokens with the chosen model.

Tokens matter in practice because the **context window**, token-priced API costs and generation speed depend on token counts. A sentence requiring more tokens can consume more context and cost more under token pricing. Token count alone does not establish the model's language quality; test the task. Multimodal systems also turn images and audio into model representations (weeks 9–10).

### Training versus inference

* **Training** searches for parameter values that make the training data likely, by computing a loss and adjusting parameters with gradient descent over many passes. It is expensive: large GPU clusters for weeks or months.
* **Inference** uses the fixed parameters to produce an output for a new input. For an LLM, each generated token requires one forward pass through the network.

> **Misconception:** "The chatbot learns from my conversation." Within a conversation the model only sees more context; its weights do not change. Some providers may later use conversations to train future models, which is a privacy question and why data-sharing settings matter.

### Pre-training, post-training and the application layer

1. **Data:** very large corpora, filtered for quality and with repeated copies removed.
2. **Pre-training:** predict the next token over trillions of tokens. The result is a **base model** that has absorbed a great deal of knowledge but only continues text.
3. **Post-training:** supervised **instruction tuning** on example conversations; **preference tuning** such as reinforcement learning from human feedback (RLHF) or direct preference optimisation (DPO); and reinforcement learning on problems with checkable answers to improve step-by-step reasoning. The result is an **assistant model**.
4. **Application:** system prompt, tools, retrieval, guardrails and interface.

A useful summary for students: *most of the knowledge comes from pre-training; most of the behaviour comes from post-training.* Weeks 5–6 cover pre-training, week 8 post-training, and weeks 11–12 the application layer.

### A sense of scale

GPT-3 (2020) had 175 billion parameters. Meta reported pre-training Llama 3 (2024) on more than 15 trillion tokens. Context windows grew from about 2,000 tokens (GPT-3) to around 2 million tokens in some 2024 models (Gemini 1.5 Pro). OpenAI reported 800 million weekly ChatGPT users in October 2025. Always attribute such figures to who reported them and when; they illustrate orders of magnitude, not current records. Bigger is not automatically better: well-trained smaller models now match much larger older ones.

## 3. Probabilistic foundations

### Probability distributions and sampling

A **probability distribution** lists the chances of possible outcomes. They add to one. After "The cat sat on the", a model might give mat probability 0.42, floor 0.17 and sofa 0.12, with the remaining 0.29 spread across other tokens.

**Sampling** chooses randomly using those chances. A token with probability 0.42 should appear in about 42% of many repeated choices from this fixed distribution. Generation chooses a token, adds it to the context and repeats.

Random sampling can give different outputs for the same prompt. Each is a possible model output, but may still be false or unsuitable. Check the answer against the task and reliable evidence. Repeat runs when evaluating how consistently a model behaves.

![Next-token probabilities for "The cat sat on the"](fig:next_token)

### The chain rule and autoregressive models

Any joint distribution over a sequence can be factorised exactly with the chain rule of probability:

**Reading the symbols.** $T$ is the number of tokens in the sequence. $x_t$ is the token at position $t$. The vertical bar means "given the earlier tokens". The product sign $\prod$ means multiply one probability for each position. For the two-token example "the cat", multiply 0.06 by 0.01 to get 0.0006. Here $T$ means sequence length; the temperature section uses the same letter for a different setting.

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

A model that predicts the next token given all previous ones therefore defines a probability for any sequence. Such a model is **autoregressive**: it generates by feeding its own outputs back as inputs. GPT-style LLMs are autoregressive transformers (weeks 5–6). The factorisation is exact; the modelling assumption is only in how well the network approximates each conditional.

### Likelihood and maximum likelihood

The **likelihood** of parameters $\theta$ is the probability the model assigns to the observed data. For $N$ independent training examples we work with the log-likelihood:

**Reading the symbols.** $\theta$ names the model's learned numbers. $N$ is the number of training examples, and $x^{(i)}$ is example number $i$. The sum sign $\sum$ means add a value for each example. $\log$ is the natural logarithm. "arg max" means choose the parameter values with the largest score. "arg min" means choose those with the smallest loss. The star on $\theta^{*}$ marks the chosen values.

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \, \log p_\theta(x^{(i)})$$

**Maximum likelihood estimation (MLE)** chooses the parameters that make the training data as probable as possible:

$$\theta^{*} = \arg\max_{\theta} \mathcal{L}(\theta) = \arg\min_{\theta} \left( -\frac{1}{N} \sum_{i=1}^{N} \, \log p_\theta(x^{(i)}) \right)$$

Small probabilities multiplied many times can become too small for a computer to represent. A **logarithm** turns these products into sums. Since log is increasing, making the probability larger also makes its log larger.

The loss is the average **negative log-likelihood (NLL)**. Lower loss means the model gives more probability to the observed targets. For known token targets this is cross-entropy. VAEs and diffusion use a likelihood bound. GANs use a different, adversarial objective.

**Simple example (MLE from counts).** If *a* is followed by *n* in 300 of 1,000 occurrences, the unsmoothed maximum-likelihood estimate of $p(n \mid a)$ is 0.3. The lab then adds one to each count before normalising: this Laplace smoothing changes the estimate and gives unseen pairs a nonzero chance.

### Worked example: the likelihood of a short sentence

| Step | Probability | Natural log |
|---|---|---|
| p(the) | 0.06 | −2.81 |
| p(cat \| the) | 0.01 | −4.61 |
| p(sat \| the cat) | 0.20 | −1.61 |
| p(on \| the cat sat) | 0.55 | −0.60 |
| **Whole sequence** | 0.06 × 0.01 × 0.20 × 0.55 = 6.6 × 10⁻⁵ | **−9.63** |

* Average NLL per token = 9.63 / 4 ≈ **2.41 nats**.
* **Perplexity** = exp(average NLL) = $e^{2.41}$ ≈ **11.1**. Interpretation: the model is as uncertain as if it were choosing uniformly among about 11 tokens at each step. Lower perplexity is better.
* The step *cat given the* contributes most to the loss; training on sentences like this one would raise that probability.

> **Teaching tip:** Ask students to check that the sum of logs equals the log of the product: ln(6.6 × 10⁻⁵) = ln 6.6 + ln 10⁻⁵ = 1.89 − 11.51 = −9.63.

### Temperature and decoding strategies

The network outputs a score called a **logit** $z_i$ for each token. The softmax function turns logits into probabilities, and the **temperature** $T > 0$ rescales them first:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**Worked example** with logits [2, 1, 0]:

| Temperature | exp values | Probabilities |
|---|---|---|
| T = 1 | 7.39, 2.72, 1.00 (sum 11.11) | 0.67, 0.24, 0.09 |
| T = 0.5 | 54.6, 7.39, 1.00 (sum 63.0) | 0.87, 0.12, 0.02 |
| T = 2 | 2.72, 1.65, 1.00 (sum 5.37) | 0.51, 0.31, 0.19 |

These values are rounded. A displayed row may add to 1.01; the probabilities before rounding add to exactly one, up to computer precision.

Lower positive temperature concentrates sampling choices; higher temperature spreads them. Neither setting verifies facts. The positive-temperature formula approaches greedy choice as T → 0. In this Transformers lab use `do_sample=False` for greedy decoding; do not divide by zero. Other common controls:

* **Top-k:** sample only among the k most likely tokens.
* **Top-p (nucleus):** sample among the smallest set of tokens whose cumulative probability reaches at least p (for example 0.9); the cut-off adapts to the model's confidence.
* **Seed:** a seed helps repeat sampling in a fixed setup; software, hardware and model revisions can still change outputs.

Decoding settings change **how we sample**, never what the model knows. Some reasoning models fix or ignore these settings, so always check the documentation. For experiments, record model, version, temperature, top-p, seed and date.

![The same logits at three temperatures](fig:temperature)

### Generative versus discriminative models

A **discriminative** model learns $p(y \mid x)$ directly: for classification this is a decision boundary. A **generative** model learns how the inputs themselves are distributed, $p(x)$ or $p(x \mid y)$, and can classify via Bayes' rule:

$$p(y \mid x) = \frac{p(x \mid y)\, p(y)}{\sum_{y'} p(x \mid y')\, p(y')}$$

A generative classifier learns what inputs in each class look like. Examples are naive Bayes and Gaussian discriminant analysis. A discriminative classifier, such as logistic regression, directly learns which class an input belongs to. Modelling the inputs is an extra task, but it also permits new samples.

A **condition** is information supplied to generation. In $p(x \mid c)$, the vertical bar reads "given". A text-to-image model creates an image given a prompt. A chat model creates a response given the conversation.

![Discriminative boundary versus generative density](fig:gen_vs_disc)

### Latent variables

A **latent variable** $z$ is a hidden numerical description behind an observation. It might capture aspects of pose, lighting or style, although learned dimensions need not have simple names. Draw a code from a simple distribution and let a **decoder** $g$ map it to output $x = g(z)$.

The integral below combines all the possible hidden codes. It asks how likely an image is after allowing for every code that could explain it.

$$p(x) = \int p(x \mid z)\, p(z)\, dz$$

Moving gradually through latent space often changes the output gradually. Inspect the samples because intermediate codes can still give poor or unclear results. Latent interpolation is an experiment in weeks 2 and 3. Computing this integral exactly is usually impossible for neural networks, which is the problem VAEs solve with a lower bound next week.

![A simple latent space mapped to structured data](fig:latent)

## 4. Generative model families

| Family | How it generates | Strengths | Weaknesses | Where it is used today |
|---|---|---|---|---|
| Autoregressive | Next token given previous tokens | Exact likelihood; excellent for text and code | Sequential, so slow for long outputs | LLMs, code models, speech tokens |
| VAE | Decode a sampled latent code | Useful codes; direct loss optimisation | Samples can blur; poor codes can remain | Compression inside latent diffusion; anomaly detection |
| GAN | Generator fools a discriminator | Sharp samples; one step, fast | Unstable training; mode collapse | Super-resolution; fast distilled generators |
| Diffusion / flow | Repeatedly update a noisy sample | Can give strong quality and coverage | Often many model calls; fast variants exist | Image, video and audio generation |

There is a triangle of trade-offs between **quality**, **diversity** and **speed**. Real systems combine families: Stable Diffusion uses a VAE to compress images, a diffusion model in the compressed space, and a transformer text encoder for the prompt.

![The four families at a glance](fig:families)

## 5. Responsible AI from day one

### Hallucination

A **hallucination** is a generated claim that is false or lacks evidence. Examples include an invented paper, a wrong number or a nonexistent code function. A language model learns plausible text patterns, so fluent wording can still be wrong.

**Grounding** means supplying relevant sources. Retrieval-augmented generation (RAG, week 12) finds documents to support an answer. Open and check its citations. Use calculators or code for numerical claims, record uncertainty and include human review when errors matter. Lower temperature can improve repeatability but does not verify facts.

### Six risk areas

* **Bias and fairness:** outputs reflect patterns and gaps in training data, e.g. stereotyped images of professions. Test outputs across groups.
* **Privacy:** personal or confidential data pasted into a tool may be stored or used for training; check settings and organisational policy (GDPR applies).
* **Intellectual property:** copyright in training data is contested in several jurisdictions; check output licences and attribute sources.
* **Transparency and law:** Article 50 of the EU AI Act applies from 2 August 2026. Duties differ by role. Tell people when they directly interact with AI unless this is obvious. Providers must machine-mark generated media, with exceptions such as standard editing. Deployers must disclose deepfakes and relevant AI-written public-interest text, subject to the editorial-review exception. For systems marketed before 2 August 2026, the 2 December 2026 transition concerns the Article 50(2) marking duty only. The Commission published guidelines on 31 July 2026. Read the official text for full scope and exceptions.
* **Academic integrity:** follow the module's AI-use policy, disclose how AI was used, and remember that the student is accountable for everything submitted.
* **Security and misuse:** prompt injection (weeks 7 and 12), deepfakes and harmful content; safety filters reduce but do not remove risk.

Other modules in the programme (Explainable and Emerging AI Technologies) cover regulation, compliance and assurance in depth. Here the focus is on recognising and testing the risks specific to generative models.

# Common misconceptions

| Misconception | Correction |
|---|---|
| "ChatGPT / Gemini / Claude is a model." | They are applications built on model families, adding instructions, memory, tools, retrieval and safety layers. |
| "The chatbot learns from our conversation." | Weights are fixed at inference; the conversation is only context. Future training on user data is a separate provider policy. |
| "Higher temperature makes the model smarter or more creative." | Temperature changes sampling only. It spreads probability to less likely choices; effects on error depend on the task. It adds no knowledge. |
| "A confident, detailed answer is probably correct." | Fluency is not evidence. Hallucinations are often the most fluent answers. |
| "Open-weight means open-source." | Weights may be released without data or training code, and licences can restrict use. |
| "Generative models are only for images and chat." | They also generate code, speech, music, video, molecules, synthetic data and more. |
| "Bigger models are always better." | Data quality, post-training and task fit matter; small models can be faster, cheaper, private and good enough. |
| "Discriminative models are old-fashioned." | For pure classification they are often better; generative models solve a harder problem. |

# Responsible AI lens: the five-step checklist

Use this checklist in every lab and in the group project. It also works as an exam answer structure for "how should an organisation use tool X?" questions.

1. **Purpose:** is AI appropriate and permitted for this task? Some assessed work may not allow it.
2. **Protect data:** no personal, confidential or proprietary data without approval; check retention and training settings.
3. **Prompt clearly:** task, context, output format and constraints (week 7).
4. **Verify:** check facts, numbers, citations and code; test edge cases and different user groups.
5. **Disclose and record:** say how AI was used; keep prompts, model name and version, settings and date so others can reproduce the result.

> **Discussion (case on slide 37):** A student asks a chat assistant for five references and two do not exist. *Mechanism:* the model generated plausible-looking references because they are likely text. *Responsibility:* the student who submitted unverified work. *Failed step:* verify, since every reference must be opened and checked. *Nuance:* tools that answer only from uploaded papers greatly reduce invented references, but can still misquote or misattribute, so checking remains necessary.

# Lab guide and answers

**Where the numbers come from.** The results below were recorded in {{W01_RUN}}; the run saved them to `curriculum/assets/week_01/results.json`. Students' numbers can differ slightly with another dataset download, model revision or hardware, so they should record their own dataset, model, settings and outputs. The recap arithmetic is illustrative.

**Runtime:** Google Colab with a T4 GPU is recommended, but every part runs on a laptop CPU. The main downloads are the SmolLM2 model (about 700 MB for the 360M version) and the names dataset (a fallback list is built in); dependency installation also needs access. On the laptop CPU used to check this pack (9 October 2026), the complete notebook ran in about 1 minute with the models and data already downloaded. **Hand-in:** completed notebook with five code TODOs, four written answers and the evidence table.

## Before the lab

* Students need a Google account (Colab) and can download this public model without a Hugging Face account. The optional Part 4 comparison uses a Gemini API key (availability and quotas depend on the account) from Google AI Studio, stored in Colab Secrets as `GEMINI_API_KEY`; students without a key use any chat assistant in the browser.
* Remind students never to paste API keys into notebooks or share notebooks that contain them.

## Part 1: a generative model from counts

* **TODO 1** counts bigrams: `N[stoi[a], stoi[b]] += 1`. The heat map shows strong transitions such as *.→a*, *a→n*, *n→.*; students should explain one of them.
* **TODO 2** averages the negative log-probabilities of the transitions. Recorded run: *anna* {{W01_NLL_ANNA}}, *emma* {{W01_NLL_EMMA}}, *xqzz* {{W01_NLL_XQZZ}}. Real-looking names are far more probable than random strings.
* Recorded dataset comparison ({{W01_NAMES}} names): **uniform {{W01_UNIFORM}}** nats per character (= ln 27), **unigram {{W01_UNIGRAM}}**, **bigram {{W01_BIGRAM}}**. Conditioning on the previous character reduces uncertainty; that is the whole idea of language modelling, and transformers extend the context from one character to thousands of tokens.
* **TODO 3** divides log-probabilities by T and renormalises. Expected behaviour is described in the model answer to Question 1.

## Part 2: generative versus discriminative

* Recorded held-out accuracy: logistic regression {{W01_ACC_LR}}, QDA {{W01_ACC_QDA}}. Similar accuracy is the point: the two models differ in what else they learn, not in how well they classify these points.
* **TODO 4:** `gen.multivariate_normal(genm.means_[0], genm.covariance_[0], 200)`. The stars should cover the class-0 cloud.
* Question 2 checks why the fitted Gaussian generative classifier can sample inputs, while the lab’s logistic classifier only estimates class probabilities.

## Part 3: a small LLM in action

* Recorded with SmolLM2's tokenizer: **{{W01_TOKENS}}** for the same sentence. Link this to cost, speed and context limits, and to fairness across languages. The `Ġ` symbol marks a leading space in byte-level BPE; Hindi appears as raw byte pieces because the vocabulary has few Devanagari tokens.
* **TODO 5:** `torch.softmax(logits / T, dim=-1)`. Recorded for "The cat sat on the": *mat* gets {{W01_MAT}}, while {{W01_RUNNERS_UP}} gain probability as T rises.
* Compare repeated outputs at T = 0.2 and T = 1.2. Record actual variation; a small number of outputs need not show a clear difference.

## Part 4: hallucination

* In the recorded run SmolLM2-360M invented a summary of the non-existent paper **both** without the honesty instruction ("{{W01_FICTION_PLAIN}}") and with it ("{{W01_FICTION_HONEST}}"). Sampling makes each run different, so students compare their own recorded answers. A search-enabled assistant may find supporting sources, but the sources and its use of them still require checking; no answer format proves factual accuracy.

## Troubleshooting

* *Model download is slow or fails:* re-run the cell; on Colab switch runtime off and on. On a slow network set `MODEL_ID` to the 135M model.
* *CUDA out of memory:* unlikely for this model; restart the runtime.
* *`google.colab` not found:* expected outside Colab; set the environment variable `GEMINI_API_KEY` instead or skip the optional cell.

# Practice questions with model answers

## Multiple choice

1. Which statement best describes a generative model? **(a)** It learns p(y | x), so it can predict a label for each input. **(b)** It only works on images, producing new pictures from noise. **(c)** It stores its training examples and retrieves the closest one. **(d)** It learns the distribution of the data and can sample new examples. *Answer: (d).*
2. Increasing temperature from 0.7 to 1.5 will: **(a)** update the weights; **(b)** make outputs more deterministic; **(c)** spread probability to less likely tokens; **(d)** increase the context window. *Answer: (c).*
3. The perplexity of a model whose average negative log-likelihood is ln 8 nats per token is: **(a)** 3; **(b)** 8; **(c)** ln 8; **(d)** 1/8. *Answer: (b), because perplexity = exp(average NLL).*
4. Which family does NOT train by (a bound on) maximum likelihood? **(a)** VAE; **(b)** autoregressive LLM; **(c)** GAN; **(d)** diffusion. *Answer: (c).*
5. A hospital wants patient notes summarised without data leaving its network. The most suitable starting point is: **(a)** an open-weight model hosted on its own servers; **(b)** a public chat app with a strong privacy policy; **(c)** a text-to-image model run on its own servers; **(d)** a hosted API with the temperature set to 0. *Answer: (a).*

## Short answer

1. **Explain the difference between a model and an application, using a chat assistant as the example.** (4 marks) *Model answer:* A model is the trained set of parameters that maps an input to a probability distribution over outputs (e.g. a GPT-family LLM). A chat assistant is an application that wraps the model with hidden system instructions, conversation memory, retrieval or search, tools such as code execution, safety filters and a user interface (2 marks for correct definitions). Consequently two applications on the same model can behave differently, and a product can change behaviour without the model changing (2 marks for implication).
2. **Compute the probability and the average negative log-likelihood of a 3-token sequence with next-token probabilities 0.5, 0.2 and 0.1.** (3 marks) *Model answer:* joint probability = 0.5 × 0.2 × 0.1 = 0.01; log-likelihood = ln 0.01 = −4.61; average NLL = 4.61 / 3 ≈ 1.54 nats; perplexity = $e^{1.54}$ ≈ 4.6.
3. **Why do LLMs hallucinate, and name two ways to reduce the risk with their limits.** (4 marks) *Model answer:* They are trained by maximum likelihood to produce plausible text, and a plausible false statement can be highly likely; they have no built-in check against the world (2 marks). Ways to reduce the risk: retrieval grounding (limit: retrieval can miss or the model can misread sources); tool use such as calculators or search (limit: tools can be misused or return wrong results); human review (limit: cost and expertise) (2 marks for two with limits).
4. **Distinguish generative from discriminative classifiers using Bayes' rule.** (4 marks) *Model answer:* A generative classifier models p(x | y) and p(y) and computes p(y | x) with Bayes' rule; a discriminative classifier models p(y | x) directly. Only the generative one can sample new x or score how typical an input is; the discriminative one usually needs fewer assumptions and is often more accurate for pure classification.

## Exam-style question

**"A university wants to let students use generative AI tools in coursework. Using concepts from this week, explain how these tools produce text, what can go wrong, and propose a policy with verification steps."** (15 marks)

*Marking guide:* explanation of next-token prediction and sampling with the chain rule (3); role of pre-training, post-training and the application layer (2); temperature and decoding and their effect on variability (2); hallucination mechanism with an example (2); at least three risk areas with specific examples (3); a policy built on purpose, data protection, prompting, verification and disclosure, including how students should document AI use (3). Strong answers distinguish model from application, avoid claiming that chatbots "look up" answers, and give concrete verification actions (open every reference; run code; cross-check figures).

# Glossary

| Term | Meaning |
|---|---|
| Generative model | A model of the data distribution p(x) (or p(x \| condition)) that can sample new examples |
| Discriminative model | A model of p(y \| x) that predicts labels without modelling the inputs |
| Foundation model | Large model trained on broad data and adapted to many tasks |
| Parameters (weights) | The adjustable numbers inside a neural network, learned in training |
| Token | A sub-word unit of text with an integer ID; models read and write tokens |
| Context window | The maximum number of tokens a model can take into account at once |
| Training / inference | Learning parameters from data / using fixed parameters to produce outputs |
| Pre-training / post-training | Next-token learning on broad data / instruction and preference tuning into an assistant |
| Likelihood | Probability the model assigns to observed data, as a function of the parameters |
| Maximum likelihood (MLE) | Choosing parameters that maximise the likelihood of the training data |
| Negative log-likelihood / cross-entropy | The training loss of LLMs; lower is better |
| Perplexity | exp(average NLL per token); an "effective number of choices" |
| Logit | Raw score before softmax |
| Temperature | Divides logits before softmax; controls how spread out sampling is |
| Top-k / top-p | Restrict sampling to the k most likely tokens / to the smallest set with total probability p |
| Latent variable | Unobserved cause of the data; sampled then decoded |
| Autoregressive model | Generates one element at a time, each conditioned on the previous ones |
| Hallucination | Fluent output that is false or unsupported |
| Open-weight model | A model whose trained weights are publicly downloadable |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed., O'Reilly: chapter 1, "Generative Modeling" (probabilistic framing, generative vs. discriminative, representation learning).
* Tunstall, von Werra & Wolf (2022) *Natural Language Processing with Transformers*, O'Reilly: chapter 1, "Hello Transformers" (optional preview of weeks 5–6).

**Videos (links last checked 9 October 2026)**

* [3Blue1Brown: Large Language Models explained briefly](https://www.youtube.com/watch?v=LPZh9BOjkQs): 8-minute visual introduction to next-token prediction.
* [Google Cloud Tech: Introduction to Generative AI](https://www.youtube.com/watch?v=G2fqAlgmoPo): accessible overview.
* [Andrej Karpathy: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g): pre-training, post-training and security risks.
* [Andrej Karpathy: How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw): practical tour of AI tools and their settings.

**Official learning pages**

* [Microsoft Learn: Introduction to generative AI and agents](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/)
* [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners): lessons 1–3, including "Using generative AI responsibly".
* [Google Skills: Introduction to Generative AI learning path](https://www.skills.google/paths/118)
* [OpenAI Academy](https://academy.openai.com/): free courses on using AI tools effectively.

**Responsible AI sources**

* [EU AI Act, Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) and the [European Commission guidelines (July 2026)](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems).
* [NIST AI 600-1: Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf): a structured list of generative-AI risks and actions.
* Huang et al. (2023) [A Survey on Hallucination in Large Language Models](https://arxiv.org/abs/2311.05232).

# Link to the group project

The group project (60%) asks teams of two or three to design, implement and evaluate a generative AI application for a real-world problem. This week, ask each student to write down **two candidate problem domains** and, for each, which model family would fit, what data would be involved and which risks from the six-risk list apply. Groups form by week 4. Every later lab produces a component (a trained model, an evaluation, a fine-tuned adapter, a multimodal pipeline, a deployed interface, a RAG index) that can be reused in the project.


# Detailed explanations behind the short slides

The lecture shows the essential idea first. These fuller explanations retain the supporting comparisons, examples and checks for revision.

## Slide 6: What is generative AI?

- **Generative AI** creates new text, code, images, speech or other content.

- A model learns patterns from examples. These patterns give probabilities for possible outputs.

- **Sampling** means choosing an output using those probabilities.

- Example: a classifier labels a photo "cat". A generator creates a new cat image.

- A **foundation model** learns from broad data and can support many tasks.

The main idea: learn patterns from examples, then use them to create new examples.

## Slide 8: Foundation models come in several kinds

**Language models.** Read text and code, then write more text or code. Examples include GPT, Claude, Gemini and Llama.

**Image & video generators.** Create an image or video from a prompt or reference image. We study diffusion in week 4.

**Speech & audio models.** Recognise speech, speak written text or generate sounds and music. See weeks 9-10.

**Multimodal assistants.** Work with more than one kind of data, such as text and images. Some also call software tools.

## Slide 9: Model, application, tool: three layers people confuse

A chat app can change its model, instructions or tools. Record which app and model you used.

| Layer | What it is | Example |
|---|---|---|
| Model | Learned numbers that turn inputs into output scores or probabilities | An LLM from the GPT, Llama or Gemini family |
| Application | A model with instructions, tools, stored context and an interface | A chat app or coding assistant |
| Tool / workflow | AI support built into a particular task | A meeting summary or library search |

## Slide 10: Proprietary vs. open-weight models

Choose using your task, cost, data needs, licence and available hardware.

**Proprietary (closed-weight).** 

- Use the provider's app or API, a software connection

- The provider runs the model and updates it

- You pay for access or usage

- Check where your data goes and how long it stays

- Examples: GPT, Claude and Gemini families

**Open-weight.** 

- Download the learned weights and run the model

- Choose your own machine or cloud server

- Adapt the weights by fine-tuning (week 8)

- Manage hardware, updates and safety checks

- Read the licence: open weights do not always mean open source

## Slide 11: Everyday AI tools you will meet

Products change monthly. Judge any tool with the same questions: which model, what data, which tools, how verified?

**Chat assistants.** ChatGPT, Gemini, Claude, Microsoft Copilot, Le Chat: general writing, analysis and Q&A.

**Coding assistants.** GitHub Copilot, Cursor, Claude Code, Gemini Code Assist: complete, explain and edit code.

**Research & study.** NotebookLM, deep-research modes, AI search engines: work over sources you supply or search.

**Image & design.** Image generators and editors inside design tools, e.g. Adobe Firefly, Midjourney, Imagen.

**Video & audio.** Text-to-video, voice and music tools, e.g. Veo, Sora, ElevenLabs, Suno.

**Open models you run.** Hugging Face and Ollama let you run open-weight models on your own machine (weeks 6 and 11).

## Slide 14: Tokens: how models see text

- A **token** is a text piece with a vocabulary number: a word, part of a word or punctuation.

- A tokenizer splits text into these pieces. Different models split the same text differently.

- For English, one token is often about three-quarters of a word. Count tokens for your model.

- Input limits, usage costs and output length often depend on the token count.

- Models also turn images and audio into numerical pieces (weeks 9–10).

## Slide 15: Training vs. inference

Training changes learned numbers. Inference uses those numbers to produce an output.

**Training.** 

- Use examples to adjust **weights**, the model's learned numbers

- A **loss** measures prediction error

- An optimiser changes weights to reduce that error

- Large model training needs substantial time and computing power

**Inference.** 

- Use weights already learned during training

- Send a prompt and compute output scores

- For text, choose a token and repeat

- An ordinary chat request changes the context, while weights stay fixed

## Slide 16: From raw data to an assistant: pre-training, post-training, application

- **Pre-training** learns broad patterns. A base model mainly continues text.

- **Post-training** uses instruction examples and feedback to improve assistant behaviour (week 8).

**Data.** Collect text, code or images and remove poor or repeated examples

**Pre-training.** Predict next tokens to learn broad language patterns

**Post-training.** Learn from instructions and feedback about better answers

**Assistant model.** Use the adapted model to follow user requests

**Application.** Add instructions, tools, source retrieval and a user interface

## Slide 19: An LLM outputs a probability distribution over the next token

- The context is the text already provided or generated.

- The model gives each possible next token a probability. All probabilities add to 1.

- Choose one token, add it to the context and predict again.

- Different choices can produce different answers. Each answer still needs checking.

## Slide 20: The chain rule: generating one token at a time

- Multiply the probabilities of each next token to get the probability of the whole sequence.

- Example: 0.5 × 0.2 × 0.1 = 0.01 for three successive choices.

- Autoregressive means predicting each next item from the items already present. The chain-rule factorisation is exact.

## Slide 21: Likelihood and maximum likelihood training

- **Likelihood** asks how much probability the model gives the observed examples.

- **Maximum likelihood training** adjusts weights to give real examples more probability.

- Taking logs turns products into sums. We minimise their negative average: **negative log-likelihood**, or cross-entropy for token targets.

## Slide 22: Worked example: the likelihood of a short sentence

Average negative log-likelihood = 9.63 / 4 ≈ 2.41 nats per token. Perplexity = $e^{2.41}$ ≈ 11: the model is as uncertain as choosing among ~11 equally likely tokens.

| Step | Next-token probability | log p (natural) |
|---|---|---|
| p(the) | 0.06 | −2.81 |
| p(cat / the) | 0.01 | −4.61 |
| p(sat / the cat) | 0.20 | −1.61 |
| p(on / the cat sat) | 0.55 | −0.60 |
| Sequence | 0.06 × 0.01 × 0.20 × 0.55 = 6.6 × 10⁻⁵ | sum = −9.63 |

## Slide 23: Temperature reshapes the distribution before sampling

- A **logit** is a raw score. **Softmax** converts scores into probabilities that add to 1.

- For T > 0, divide logits by temperature T before softmax. Lower T favours stronger scores. Higher T spreads the choices.

- For scores [2, 1, 0], probabilities are about [0.67, 0.24, 0.09] at T = 1 and [0.87, 0.12, 0.02] at T = 0.5.

## Slide 25: Decoding strategies: how we pick the next token

- **Greedy**: choose the highest-probability token each time. Repetition can still occur.

- **Temperature sampling**: rescale scores, then choose randomly from the probabilities.

- **Top-k**: keep only the k most likely tokens before sampling.

- **Top-p**: keep the smallest likely group whose probabilities reach p, such as 0.9.

- **Random seed**: sets the random sequence used in an experiment. Record it with the model and settings.

Decoding is the rule for selecting tokens. Report the rule and its settings.

## Slide 27: Connecting the two views with Bayes' rule

- A **generative classifier** models inputs within each class, p(x | y), and how common each class is, p(y).

- Bayes' rule combines these to predict the class of a new input.

- A **discriminative classifier** directly predicts p(y | x). It does not model how to create new inputs.

Generative AI can also create an output given a prompt: p(output | prompt).

## Slide 28: Latent variables: simple hidden causes, complex data

- A **latent variable** z is a hidden numerical description, such as image style or pose.

- Draw a simple random code z. A network g turns it into an output x.

- The integral combines the possible hidden codes to describe how likely x is.

- Weeks 2–4 use hidden codes or noisy states to generate images.

## Slide 31: Comparing the four families

| Family | How it generates | Strengths | Weaknesses | Where used today |
|---|---|---|---|---|
| Autoregressive | Choose the next item from previous items | Can score sequences; useful for text/code | Long outputs need repeated steps | LLMs, code and speech tokens |
| VAE | Draw a hidden code and decode it | Organised codes; often steady training | Basic image samples may be blurry | Latent compression; anomaly detection |
| GAN | Draw noise and run the trained generator | Sharp images; fast generation | Training can fail or lose variety | Super-resolution; fast generators |
| Diffusion / flow | Refine noise over several steps | Strong image quality and variety | More computation per sample | Image, video and audio generation |

## Slide 34: Hallucination: fluent but false

- A **hallucination** is a generated claim that is false or lacks evidence.

- Example: an answer names a paper that does not exist.

- The model learns plausible text patterns. Plausible wording can still be wrong.

- Find reliable sources, open the citations and check numbers or code with appropriate tools.

- Record uncertainty and use human review when errors matter.

A fluent answer needs evidence. Confidence in the wording does not prove correctness.

## Slide 35: Six risks to check with any generative AI system

**Bias & fairness.** Outputs reflect patterns and gaps in training data, e.g. stereotyped images of professions. Test across groups.

**Privacy.** Do not paste personal or confidential data into tools without approval. Check retention and training settings.

**Intellectual property.** Training-data copyright is contested; check output licences and attribute your sources.

**Transparency & law.** EU AI Act Article 50: disclose direct AI interaction. Provider marking and deployer disclosure duties differ. See notes.

**Academic integrity.** Follow the module's AI-use policy, disclose use, and remember you are accountable for what you submit.

**Security & misuse.** Prompt injection, deepfakes and harmful content: safety filters reduce but do not remove risk.

## Slide 36: A five-step checklist for using AI tools

Use this checklist in every lab and in your group project. It is also a good exam answer structure.

**Purpose.** Is AI appropriate and allowed for this task?

**Protect data.** No personal or confidential data without approval

**Prompt clearly.** Task, context, format, constraints (week 7)

**Verify.** Check facts, numbers, citations, code; test edge cases

**Disclose & record.** Say how AI was used; keep prompts and model versions

## Slide 37: Case discussion

- What mechanism produced the fake references?

- Who is responsible for the error in the submitted review?

- Which checklist step failed, and what process would have prevented it?

- Would a research tool that answers only from uploaded papers remove the risk entirely?

## Slide 38: Lab 1: probability, sampling and a first look at an LLM

Google Colab (T4 GPU) or local Jupyter; everything also runs on a CPU

Completed notebook with five code TODOs, four written answers and the evidence table

- Prepare Colab and inspect safe key storage. The local model also runs on CPU.

- Count neighbouring character pairs in names. Build probabilities and calculate name scores.

- Sample names at temperatures 0.5, 1.0 and 1.5. Explain what changed.

- Compare a generative and discriminative classifier on 2-D points. Sample new points.

- Run a small language model. Inspect token pieces, next-token chances and generated text.

- Test a fictional-paper prompt and verify claims. Complete the evidence table.

## Slide 39: Summary

- Generative models learn patterns and create new examples by sampling.

- A model supplies learned weights. An application adds instructions, tools and an interface.

- Training changes weights. Inference uses fixed weights with the current input.

- Tokens are text pieces. Next-token probabilities combine to give sequence probability.

- Likelihood measures fit to examples. Temperature changes how we sample.

- Generators model possible data. Discriminative classifiers predict labels.

- Autoregressive, VAE, GAN and diffusion models generate in different ways.

- Check evidence, protect data and disclose AI use.
