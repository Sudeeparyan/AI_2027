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

## 1. The generative AI landscape

### What "generative" means

A **generative model** learns the probability distribution of its training data, written $p(x)$, well enough to **draw new samples** that look like that data. The outputs can be text, code, images, audio, speech, music, video, 3-D shapes or even molecules. A **discriminative model** (such as a classifier) instead learns $p(y \mid x)$: given an input, which label? A spam filter is discriminative; a model that writes a new email is generative.

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

Students often say "ChatGPT is a model". It is not; it is an **application**. The distinction matters in the exam and in the project.

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

Language models do not read words or characters directly. A **tokenizer** splits text into sub-word pieces called **tokens** and maps each to an integer ID in a fixed vocabulary (typically 32,000–260,000 entries). Common words are usually one token; rare words, names and many non-English words are split into several. A useful rule of thumb for English is that one token is about three-quarters of a word.

Tokens matter in practice because the **context window** (how much text the model can see at once), **pricing** (per million input and output tokens) and **speed** (tokens per second) are all measured in tokens. Languages that need more tokens per sentence are more expensive to process and often handled less well. Images and audio are also converted to tokens in multimodal models (weeks 9–10).

### Training versus inference

* **Training** searches for parameter values that make the training data likely, by computing a loss and adjusting parameters with gradient descent over many passes. It is expensive: large GPU clusters for weeks or months.
* **Inference** uses the fixed parameters to produce an output for a new input. For an LLM, each generated token requires one forward pass through the network.

> **Misconception:** "The chatbot learns from my conversation." Within a conversation the model only sees more context; its weights do not change. Some providers may later use conversations to train future models, which is a privacy question and why data-sharing settings matter.

### Pre-training, post-training and the application layer

1. **Data:** very large corpora, filtered for quality and deduplicated.
2. **Pre-training:** predict the next token over trillions of tokens. The result is a **base model** that has absorbed a great deal of knowledge but only continues text.
3. **Post-training:** supervised **instruction tuning** on example conversations; **preference tuning** such as reinforcement learning from human feedback (RLHF) or direct preference optimisation (DPO); and reinforcement learning on problems with checkable answers to improve step-by-step reasoning. The result is an **assistant model**.
4. **Application:** system prompt, tools, retrieval, guardrails and interface.

A useful summary for students: *most of the knowledge comes from pre-training; most of the behaviour comes from post-training.* Weeks 5–6 cover pre-training, week 8 post-training, and weeks 11–12 the application layer.

### A sense of scale

GPT-3 (2020) had 175 billion parameters. Meta reported pre-training Llama 3 (2024) on more than 15 trillion tokens. Context windows grew from about 2,000 tokens (GPT-3) to around 2 million tokens in some 2024 models (Gemini 1.5 Pro). OpenAI reported 800 million weekly ChatGPT users in October 2025. Always attribute such figures to who reported them and when; they illustrate orders of magnitude, not current records. Bigger is not automatically better: well-trained smaller models now match much larger older ones.

## 3. Probabilistic foundations

### Probability distributions and sampling

A **probability distribution** assigns a probability to every possible outcome, and the probabilities sum to one. For an LLM, the outcomes are the tokens in the vocabulary. Given the context *"The cat sat on the"*, a model might give *mat* 0.42, *floor* 0.17, *sofa* 0.12 and so on. **Sampling** draws one outcome at random according to these probabilities. Generation is a loop: sample a token, append it to the context, compute the next distribution, repeat until an end token.

Because generation is sampling, the same prompt can give different answers, each a valid sample. This is a feature for creative work and a challenge for evaluation (week 7).

![Next-token probabilities for "The cat sat on the"](fig:next_token)

### The chain rule and autoregressive models

Any joint distribution over a sequence can be factorised exactly with the chain rule of probability:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1})$$

A model that predicts the next token given all previous ones therefore defines a probability for any sequence. Such a model is **autoregressive**: it generates by feeding its own outputs back as inputs. GPT-style LLMs are autoregressive transformers (weeks 5–6). The factorisation is exact; the modelling assumption is only in how well the network approximates each conditional.

### Likelihood and maximum likelihood

The **likelihood** of parameters $\theta$ is the probability the model assigns to the observed data. For $N$ independent training examples we work with the log-likelihood:

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log p_\theta(x^{(i)})$$

**Maximum likelihood estimation (MLE)** chooses the parameters that make the training data as probable as possible:

$$\theta^{*} = \arg\max_{\theta} \mathcal{L}(\theta) = \arg\min_{\theta} \left( -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(x^{(i)}) \right)$$

We use logarithms because products of many small probabilities underflow to zero on a computer, while sums of logs are numerically stable; and because the logarithm is increasing, the maximiser does not change. Minimising the average **negative log-likelihood (NLL)** is the same as minimising the **cross-entropy** between the data and the model, the loss printed during LLM training. VAEs (week 2), autoregressive models and diffusion models are all trained with likelihood or a bound on it; GANs (week 3) are the notable exception.

**Simple example (MLE from counts).** If the character *a* is followed by *n* in 300 of 1,000 occurrences, the maximum-likelihood estimate of $p(n \mid a)$ is 0.3. Counting and normalising *is* maximum likelihood for this kind of model, which is exactly what students do in Part 1 of the lab.

### Worked example: the likelihood of a short sentence

| Step | Probability | Natural log |
|---|---|---|
| p(the) | 0.06 | −2.81 |
| p(cat \| the) | 0.01 | −4.61 |
| p(sat \| the cat) | 0.20 | −1.61 |
| p(on \| the cat sat) | 0.55 | −0.60 |
| **Whole sequence** | 0.06 × 0.01 × 0.20 × 0.55 = 6.6 × 10⁻⁵ | **−9.63** |

* Average NLL per token = 9.63 / 4 ≈ **2.41 nats**.
* **Perplexity** = exp(average NLL) = e^2.41 ≈ **11.1**. Interpretation: the model is as uncertain as if it were choosing uniformly among about 11 tokens at each step. Lower perplexity is better.
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

Low temperature sharpens the distribution (more predictable, less diverse); high temperature flattens it (more diverse, more errors); T → 0 is **greedy decoding**. Other common controls:

* **Top-k:** sample only among the k most likely tokens.
* **Top-p (nucleus):** sample among the smallest set of tokens whose probabilities sum to p (for example 0.9); the cut-off adapts to the model's confidence.
* **Seed:** fixing the random seed makes sampling reproducible.

Decoding settings change **how we sample**, never what the model knows. Some reasoning models fix or ignore these settings, so always check the documentation. For experiments, record model, version, temperature, top-p, seed and date.

![The same logits at three temperatures](fig:temperature)

### Generative versus discriminative models

A **discriminative** model learns $p(y \mid x)$ directly: for classification this is a decision boundary. A **generative** model learns how the inputs themselves are distributed, $p(x)$ or $p(x \mid y)$, and can classify via Bayes' rule:

$$p(y \mid x) = \frac{p(x \mid y)\, p(y)}{\sum_{y'} p(x \mid y')\, p(y')}$$

Classic generative classifiers are naive Bayes and Gaussian discriminant analysis; logistic regression is the classic discriminative one. Generative models are harder to learn (a 512 × 512 colour image has 786,432 numbers to model), but they can create new examples, detect unusual inputs with low $p(x)$, and learn from unlabelled data. Modern generative AI learns **conditional** distributions $p(x \mid c)$ where the condition $c$ is a label, a text prompt or another image: a text-to-image model learns $p(\text{image} \mid \text{prompt})$; a chat model learns $p(\text{response} \mid \text{conversation})$.

![Discriminative boundary versus generative density](fig:gen_vs_disc)

### Latent variables

A **latent variable** $z$ is an unobserved cause of the data, such as the pose, lighting or style behind an image. The generative story is: sample a simple $z$ (usually from a standard Gaussian), then map it through a network $g$ (a decoder or generator) to a complex data point $x = g(z)$. The data distribution is a mixture over all possible hidden causes:

$$p(x) = \int p(x \mid z)\, p(z)\, dz$$

Moving smoothly through latent space changes the output smoothly, which is why latent interpolation is a favourite experiment in weeks 2 and 3. Computing this integral exactly is usually impossible for neural networks, which is the problem VAEs solve with a lower bound next week.

![A simple latent space mapped to structured data](fig:latent)

## 4. Generative model families

| Family | How it generates | Strengths | Weaknesses | Where it is used today |
|---|---|---|---|---|
| Autoregressive | Next token given previous tokens | Exact likelihood; excellent for text and code | Sequential, so slow for long outputs | LLMs, code models, speech tokens |
| VAE | Decode a sample from a latent space | Smooth latent space; stable training | Samples can be blurry | Compression inside latent diffusion; anomaly detection |
| GAN | Generator fools a discriminator | Sharp samples; one step, fast | Unstable training; mode collapse | Super-resolution; fast distilled generators |
| Diffusion / flow | Remove noise step by step | Highest quality and diversity | Many steps (being reduced) | Image, video and audio generation |

There is a triangle of trade-offs between **quality**, **diversity** and **speed**. Real systems combine families: Stable Diffusion uses a VAE to compress images, a diffusion model in the compressed space, and a transformer text encoder for the prompt.

![The four families at a glance](fig:families)

## 5. Responsible AI from day one

### Hallucination

LLMs are trained to produce **plausible** continuations, not verified facts. A **hallucination** is output that is fluent and confident but false or unsupported: invented references, wrong numbers, fake quotes, non-existent functions in code. It is more likely for rare facts, recent events, exact figures, long chains of reasoning and questions with false premises. Mitigations include grounding answers in sources (retrieval-augmented generation, week 12) and checking the citations, using tools such as calculators, code execution and search, lowering temperature for factual tasks, prompting the model to state uncertainty, and keeping a human reviewer for anything that matters. None of these reduces the risk to zero.

### Six risk areas

* **Bias and fairness:** outputs reflect patterns and gaps in training data, e.g. stereotyped images of professions. Test outputs across groups.
* **Privacy:** personal or confidential data pasted into a tool may be stored or used for training; check settings and organisational policy (GDPR applies).
* **Intellectual property:** copyright in training data is contested in several jurisdictions; check output licences and attribute sources.
* **Transparency and law:** since 2 August 2026, **Article 50 of the EU AI Act** requires providers and deployers to tell people when they are interacting with an AI system and to disclose AI-generated or manipulated content such as deepfakes; generative systems must mark outputs in a machine-readable way (systems already on the market before August 2026 have until 2 December 2026 for this marking). The European Commission published guidelines on these obligations in July 2026.
* **Academic integrity:** follow the module's AI-use policy, disclose how AI was used, and remember that the student is accountable for everything submitted.
* **Security and misuse:** prompt injection (weeks 7 and 12), deepfakes and harmful content; safety filters reduce but do not remove risk.

Other modules in the programme (Explainable and Emerging AI Technologies) cover regulation, compliance and assurance in depth. Here the focus is on recognising and testing the risks specific to generative models.

# Common misconceptions

| Misconception | Correction |
|---|---|
| "ChatGPT / Gemini / Claude is a model." | They are applications built on model families, adding instructions, memory, tools, retrieval and safety layers. |
| "The chatbot learns from our conversation." | Weights are fixed at inference; the conversation is only context. Future training on user data is a separate provider policy. |
| "Higher temperature makes the model smarter or more creative." | Temperature changes sampling only. It increases diversity and errors; it adds no knowledge. |
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

**Runtime:** Google Colab with a T4 GPU is recommended, but every part runs on a laptop CPU. The only download is the SmolLM2 model (about 700 MB for the 360M version) and the names dataset (a fallback list is built in). **Hand-in:** completed notebook with five code TODOs, four written answers and the evidence table.

## Before the lab

* Students need a Google account (Colab) and should create a free Hugging Face account. The optional Part 4 comparison needs a free Gemini API key from Google AI Studio, stored in Colab Secrets as `GEMINI_API_KEY`; students without a key use any chat assistant in the browser.
* Remind students never to paste API keys into notebooks or share notebooks that contain them.

## Part 1: a generative model from counts

* **TODO 1** counts bigrams: `N[stoi[a], stoi[b]] += 1`. The heat map shows strong transitions such as *.→a*, *a→n*, *n→.*; students should explain one of them.
* **TODO 2** averages the negative log-probabilities of the transitions. Measured on the full dataset: *anna* 1.91 nats (perplexity 6.7), *emma* 2.51 (12.4), *xqzz* 4.89 (132.5). Real-looking names are far more probable than random strings.
* Dataset comparison (32,033 names, measured): **uniform 3.30** nats per character (= ln 27), **unigram 2.82**, **bigram 2.46**. Conditioning on the previous character reduces uncertainty; that is the whole idea of language modelling, and transformers extend the context from one character to thousands of tokens.
* **TODO 3** divides log-probabilities by T and renormalises. Expected behaviour is described in the model answer to Question 1.

## Part 2: generative versus discriminative

* Both classifiers reach the same test accuracy on this data (measured: 0.928 each).
* **TODO 4:** `gen.multivariate_normal(genm.means_[0], genm.covariance_[0], 200)`. The stars should cover the class-0 cloud.
* Question 2 checks that students can explain *why* only the generative model can sample.

## Part 3: a small LLM in action

* The tokenizer part shows the same sentence costs **11 tokens in English, 33 in Irish and 56 in Hindi** with SmolLM2's tokenizer (measured). Link this to cost, speed and context limits, and to fairness across languages. The `Ġ` symbol marks a leading space in byte-level BPE; Hindi appears as raw byte pieces because the vocabulary has few Devanagari tokens.
* **TODO 5:** `torch.softmax(logits / T, dim=-1)`. Measured for "The cat sat on the": *mat* gets 1.00 at T = 0.5, 0.95 at T = 1.0 and 0.56 at T = 1.5, with *windows*, *couch* and *chair* gaining probability as T rises.
* The generation cell shows near-identical answers at T = 0.2 and varied ones at T = 1.2.

## Part 4: hallucination

* In our test run SmolLM2-360M invented a detailed summary of the non-existent paper **both** with and without the honesty instruction (e.g. "presents a novel quantum algorithm called 'Gradient Folding'"). Small models follow such instructions unreliably. Larger hosted models more often say they cannot find the paper, especially with search enabled, but can still fabricate details; students must check.

## Troubleshooting

* *Model download is slow or fails:* re-run the cell; on Colab switch runtime off and on. On a slow network set `MODEL_ID` to the 135M model.
* *CUDA out of memory:* unlikely for this model; restart the runtime.
* *`google.colab` not found:* expected outside Colab; set the environment variable `GEMINI_API_KEY` instead or skip the optional cell.

# Practice questions with model answers

## Multiple choice

1. Which statement best describes a generative model? **(a)** It learns p(y | x) to predict labels. **(b)** It learns the distribution of the data and can sample new examples. **(c)** It retrieves stored examples. **(d)** It only works on images. *Answer: (b).*
2. Increasing temperature from 0.7 to 1.5 will: **(a)** update the weights; **(b)** make outputs more deterministic; **(c)** spread probability to less likely tokens; **(d)** increase the context window. *Answer: (c).*
3. The perplexity of a model whose average negative log-likelihood is ln 8 nats per token is: **(a)** 3; **(b)** 8; **(c)** ln 8; **(d)** 1/8. *Answer: (b), because perplexity = exp(average NLL).*
4. Which family does NOT train by (a bound on) maximum likelihood? **(a)** VAE; **(b)** autoregressive LLM; **(c)** GAN; **(d)** diffusion. *Answer: (c).*
5. A hospital wants patient notes summarised without data leaving its network. The most suitable starting point is: **(a)** a public chat app; **(b)** an open-weight model hosted on its own servers; **(c)** a text-to-image model; **(d)** raising the temperature. *Answer: (b).*

## Short answer

1. **Explain the difference between a model and an application, using a chat assistant as the example.** (4 marks) *Model answer:* A model is the trained set of parameters that maps an input to a probability distribution over outputs (e.g. a GPT-family LLM). A chat assistant is an application that wraps the model with hidden system instructions, conversation memory, retrieval or search, tools such as code execution, safety filters and a user interface (2 marks for correct definitions). Consequently two applications on the same model can behave differently, and a product can change behaviour without the model changing (2 marks for implication).
2. **Compute the probability and the average negative log-likelihood of a 3-token sequence with next-token probabilities 0.5, 0.2 and 0.1.** (3 marks) *Model answer:* joint probability = 0.5 × 0.2 × 0.1 = 0.01; log-likelihood = ln 0.01 = −4.61; average NLL = 4.61 / 3 ≈ 1.54 nats; perplexity = e^1.54 ≈ 4.6.
3. **Why do LLMs hallucinate, and name two mitigations with their limits.** (4 marks) *Model answer:* They are trained by maximum likelihood to produce plausible text, and a plausible false statement can be highly likely; they have no built-in check against the world (2 marks). Mitigations: retrieval grounding (limit: retrieval can miss or the model can misread sources); tool use such as calculators or search (limit: tools can be misused or return wrong results); human review (limit: cost and expertise) (2 marks for two with limits).
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

**Videos (checked September 2026)**

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

* [EU AI Act, Article 50: transparency obligations](https://artificialintelligenceact.eu/article/50/) and the [European Commission guidelines (July 2026)](https://digital-strategy.ec.europa.eu/en/policies/guidelines-ai-transparency-obligations).
* [NIST AI 600-1: Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf): a structured list of generative-AI risks and actions.
* Huang et al. (2023) [A Survey on Hallucination in Large Language Models](https://arxiv.org/abs/2311.05232).

# Link to the group project

The group project (60%) asks teams of two or three to design, implement and evaluate a generative AI application for a real-world problem. This week, ask each student to write down **two candidate problem domains** and, for each, which model family would fit, what data would be involved and which risks from the six-risk list apply. Groups form by week 4. Every later lab produces a component (a trained model, an evaluation, a fine-tuned adapter, a multimodal pipeline, a deployed interface, a RAG index) that can be reused in the project.
