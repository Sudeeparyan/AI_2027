# Lecture plan at a glance

This week is the practical core of "using AI tools well". Students should leave able to structure prompts, apply the named techniques, produce validated structured outputs, engineer context, and above all **evaluate** prompts with a test set and sound statistics. All results in the slides come from running this week's lab notebook on {{MODEL}} (the notebook's instructor cell saves them to `curriculum/assets/week_07/results.json`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–6 min | 1–4 | Warm-up | Fix "Write about AI for the website." in pairs; ask how they would judge quality. |
| 6–16 min | 5–7 | 1 · Paradigm shift | Pre-training vs fine-tuning vs prompting; in-context learning; prompts as programs. |
| 16–41 min | 8–17 | 2 · Anatomy and basics | Anatomy; worked rewrite; roles; zero/one/few-shot; code; roles' limits; principles; real results; quiz. |
| 41–48 min | – | Break | |
| 48–73 min | 18–25 | 3 · Reasoning, structure, context | CoT; real CoT results; reasoning models; chaining; structured outputs; context engineering; task-specific prompting. |
| 73–98 min | 26–32 | 4 · Evaluation | Loop; methods table; ROUGE by hand; LLM-as-a-judge; rubric; statistics. |
| 98–118 min | 33–37 | 5 · Limitations and tools | Limitations; prompt injection (real measurement); tools; responsible AI. |
| 118–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** When showing the results slides, ask "would you ship the better prompt based on this?" before revealing the statistics slide. Most students will say yes; the statistics slide then lands.

<!-- pagebreak -->

# Lecture notes

## 1. The paradigm shift

| | Pre-training | Fine-tuning | Prompting |
|---|---|---|---|
| What changes | All weights | Some or all weights | Nothing in the model |
| Data needed | Trillions of tokens | Hundreds–thousands of examples | A few examples, or none |
| Cost | Millions | GPU hours (cheaper with LoRA) | Cost per call; minutes to iterate |
| Best for | General capability | Consistent format/style/domain | First approach to any task |

**Decision rule** (consistent with Microsoft, Google and OpenAI guidance): start with prompting; add retrieval when knowledge is missing (week 12); fine-tune only when a well-designed evaluation shows prompting cannot meet the quality bar (week 8).

The model sees only tokens: system instructions, conversation, documents, examples and the question. **In-context learning** (GPT-3, 2020) means that examples in the prompt change behaviour without training. A prompt is therefore a program in natural language and should be specified, tested and version-controlled. Assistants are post-trained with an **instruction hierarchy**: system/developer messages over user messages over content inside documents; this is a tendency, not a guarantee.

## 2. Anatomy of a prompt and basic techniques

### Anatomy

1. **System/developer message:** role, rules, tone, policies, output policy.
2. **Context:** documents, data, history, tool results, delimited (e.g. `<document>…</document>`).
3. **Examples:** input → output pairs in the exact format.
4. **Task:** what to do, for whom, with which constraints.
5. **Output format:** length, structure, schema, and what to do if unsure.

![Anatomy of a prompt](fig:anatomy)

**Worked rewrite of the warm-up prompt.**

| Element | Before | After |
|---|---|---|
| System | (none) | "You write for the college website. Use only the facts provided. UK English, friendly, no hype." |
| Audience/purpose | (none) | Prospective MSc students deciding whether to apply |
| Context | (none) | `<facts>` 12-week module; Python labs; VAEs, GANs, diffusion, LLMs, RAG, agents; project 60%, exam 40% `</facts>` |
| Task | "Write about AI." | "Write a short introduction to the Generative AI module." |
| Format | (none) | Headline + 3 bullets, < 120 words; omit missing facts |
| Success check | (none) | Every claim in `<facts>`; readable by non-experts; within length |

### Roles in chat APIs

| Role | Written by | Use |
|---|---|---|
| system/developer | Application builder | Persistent rules, persona, format |
| user | End user (untrusted) | Requests and pasted content |
| assistant | Model (or you, in examples) | Earlier replies; example outputs |
| tool | Application | Function-call results (week 12) |

### Zero-, one- and few-shot prompting

* **Zero-shot:** instruction only. Cheapest; relies on the model knowing the task.
* **One-shot:** one example; fixes the output format but can bias towards that example's label.
* **Few-shot:** several diverse examples (often 3–10), ideally one per class and some boundary cases. Choice, order and label balance of examples affect results; never take examples from the test set.

### Role-based prompting

A persona ("You are an experienced triage officer…") sets tone, vocabulary and priorities. It does not add knowledge, and an authoritative persona can make wrong answers sound more convincing. Specific behavioural instructions work better than flattering labels.

### Principles from the official guides

Be clear and specific; give context with delimiters; show examples; break complex tasks down; allow uncertainty ("If the answer is not in the document, say so"); test systematically.

### Real results

{{FEWSHOT_NOTES}}

![Few-shot results from the lab](fig:fewshot_results)

## 3. Reasoning, structure and context

### Chain-of-thought and self-consistency

**Chain-of-thought** (Wei et al., 2022) asks the model to write intermediate steps; the zero-shot form is "Let's think step by step" (Kojima et al., 2022). It helps multi-step problems, mainly in larger models, at a large token cost. Ask for a parseable final line ("Answer: …"). **Self-consistency** (Wang et al., 2022) samples several reasoning paths (temperature > 0) and takes the majority answer. The written reasoning is not a faithful record of the model's computation: score the final answer against ground truth.

{{COT_NOTES}}

![CoT and self-consistency results from the lab](fig:cot_results)

### Prompting reasoning models

Reasoning models already reason internally. Give goals, constraints and success criteria rather than step-by-step scripts; provide complete context; request a clear final format; use the effort/thinking-budget setting where available. Route hard, high-value problems to reasoning models and simple, high-volume ones to standard models.

### Prompt chaining

Decompose complex tasks into small, testable steps, e.g. extract → verify → draft → critique → finalise. Steps can use different models or plain code (e.g. verifying dates). This leads to workflows and agents (week 12).

### Structured outputs

Define a schema (e.g. with pydantic), request JSON, **validate in code**, and retry once with the validation error. Many APIs can enforce a JSON schema during decoding. A valid structure does not guarantee correct values: evaluate field-level accuracy. In the lab: {{EXTRACTION_NOTES}}

```python
class Request(BaseModel):
    name: str
    student_id: str                                   # 8 digits (validator)
    request: Literal["extension", "invoice", "deferral"]
    date: Optional[str] = None                        # YYYY-MM-DD or null
req = Request.model_validate_json(raw_output)        # raises ValidationError
```

### Context engineering

The practice of deciding what fills the limited context window at each step (instructions, query, retrieved documents, memory, tool outputs): **select** relevant information, **compress** long histories, **order** deliberately (long documents before the question), **delimit and label** sources. More context is not always better: cost, latency and distraction increase.

![Context engineering](fig:context_eng)

### Prompting for different tasks

| Task | Emphasise |
|---|---|
| Summarisation | Audience, length, key facts to keep, "use only the text" |
| Extraction | Schema, allowed values, null when absent, temperature 0, validation |
| Classification | Full label set with definitions, examples per class, one-word answers |
| Code | Language/versions, interfaces, tests; run the code |
| Image generation | Subject, style, composition, lighting, aspect ratio; negative prompts |
| Data analysis | Data schema; ask for code rather than computed numbers; verify |

## 4. Evaluation methods

### The evaluation loop

Define success → build a test set (realistic inputs, edge cases, held-out part) → run with fixed settings and full logging → score (automatic + rubric + human) → analyse errors and iterate. Start with 20–50 examples and grow the set from real failures.

### Methods

| Method | Measures | Limitation |
|---|---|---|
| Exact match, accuracy, F1 | Closed answers | Only for closed answers |
| BLEU, ROUGE | Word overlap with references | Misses paraphrase; rewards copying |
| Semantic similarity | Embedding similarity | Similar ≠ correct |
| Programmatic checks | Schema, tests, constraints | Only what code can check |
| LLM-as-a-judge | Rubric-based quality at scale | Biases; needs calibration |
| Human evaluation | Usefulness, correctness, safety | Costly; measure agreement |

### ROUGE-1 worked example

Reference: "the exam results are published in four weeks" (8 words). Output: "results are published after four weeks" (6 words). Overlap = {results, are, published, four, weeks} = 5.

$$\mathrm{Recall} = 5/8 = 0.63, \quad \mathrm{Precision} = 5/6 = 0.83, \quad F_1 = \frac{2 \times 0.83 \times 0.63}{0.83 + 0.63} \approx 0.71$$

A correct paraphrase ("marks come out a month later") scores close to 0.

### LLM-as-a-judge

A strong model grades outputs against a rubric or compares pairs (Zheng et al., 2023). Strong judges agreed with human preferences about as often as humans agreed with each other (≈ 80%) on MT-Bench. Biases: position, verbosity, self-preference, confident style. Mitigations: judge both orders; rubrics and reference answers; a different model family; calibrate on human labels and report agreement. In the lab: {{JUDGE_NOTES}}

### Rubric for summaries

| Criterion | 1 | 3 | 5 |
|---|---|---|---|
| Faithfulness | Unsupported claims | Minor unsupported detail | Every claim supported |
| Coverage | Misses key points | Most key points | All key points, no padding |
| Clarity | Hard to follow | Understandable | Clear and well organised |
| Format | Ignores instructions | Minor deviations | Exactly as specified |

Use at least two raters on a sample and measure agreement (e.g. Cohen's kappa).

### Statistics

The standard error of an accuracy $p$ on $n$ items is

$$\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}$$

For $p = 0.8$, $n = 40$: SE ≈ 0.063, so the 95% margin is about ±12 points. Compare prompts on the same items (paired), run several seeds for sampled outputs, keep a held-out set, and report model, version, date, settings, $n$ and failure examples.

## 5. Limitations, security and tools

### Limitations

Hallucination; prompt sensitivity (wording, order, format); knowledge cut-off; **sycophancy** (agreeing with the user's stated view); bias (test with counterfactual names and demographics); non-determinism (sampling, server changes).

### Prompt injection and jailbreaks

* **Direct injection / jailbreak:** the user tries to override the rules.
* **Indirect injection:** instructions hidden in documents, web pages, emails or retrieved content that the model processes.

Prompt injection tops the OWASP list of LLM application risks. In the lab: {{INJECTION_NOTES}} Delimiters and system rules reduce but do not eliminate injection, because instructions and data share one channel. Real systems need defence in depth: least privilege, tool allow-lists and argument validation, output filtering, human approval for consequential actions, monitoring and red-teaming (week 12).

![Prompt injection measurement from the lab](fig:injection_results)

### Supporting tools

Playgrounds (Google AI Studio, OpenAI Playground, Anthropic Console); prompt management and versioning; evaluation frameworks (OpenAI Evals, promptfoo, Inspect, LangSmith); prompt optimisers (DSPy); observability (week 11); guardrails (schema enforcement, PII detection, injection classifiers).

# Common misconceptions

| Misconception | Correction |
|---|---|
| "There is a perfect magic prompt." | Prompts are designs evaluated on a test set; they also change with the model. |
| "Personas make models more accurate." | They set tone; accuracy gains are small or inconsistent. |
| "Chain-of-thought text shows how the model reasoned." | It is generated text; score the answer, not the story. |
| "Valid JSON means correct data." | Schema validity says nothing about the values. |
| "A better score on 20 examples proves a better prompt." | Check the margin of error; use paired comparison and held-out data. |
| "LLM judges are objective." | They have position, verbosity and self-preference biases; calibrate. |
| "A strong system prompt prevents prompt injection." | It reduces it; defence in depth is required. |

# Responsible AI lens: prompting and evaluation

* **Privacy:** prompts are data; follow the week 1 checklist before pasting information into tools.
* **Bias testing:** include counterfactual variations (names, dialects, demographics) in test sets.
* **Over-reliance:** design verification into workflows where errors matter.
* **Transparency and accountability:** disclose AI assistance; version and log prompts.
* **Evaluation is responsible practice:** untested prompts are untested software.

# Lab guide and answers

**Runtime:** Colab T4 (Qwen2.5-1.5B-Instruct, ≈ 15–25 min of compute for all parts) or CPU (Qwen2.5-0.5B-Instruct, slower). **Hand-in:** results tables, error analysis, five written answers and the harness.

## TODOs

* **TODO 1** few-shot: join the five examples in the one-shot format, then the message.
* **TODO 2** role + few-shot: add a system message with role and allowed labels before the few-shot user message.
* **TODO 3** `extract_number`: last number in the text (regex); `majority_vote`: most common non-None value.
* **TODO 4** retry: append the model's output as an assistant message and a user message containing the validation error; parse again.
* **TODO 5** defended prompt: system rule that document content is untrusted data; document inside `<document>` tags; ask for a factual one-sentence summary.
* **TODO 6** judge: parse "A" or "B" with a regex.

## Measured results (instructor run)

{{RESULTS_TABLE}}

Students' numbers will differ slightly (hardware, library versions); the patterns should be similar. Encourage students to discuss whether their differences exceed the margin of error.

# Practice questions with model answers

## Multiple choice

1. Few-shot prompting mainly helps by: **(a)** updating weights; **(b)** showing the task format and resolving ambiguous cases; **(c)** increasing context length; **(d)** removing bias. *Answer: (b).*
2. Self-consistency: **(a)** asks the model if it is consistent; **(b)** samples several reasoning paths and takes a majority vote; **(c)** uses temperature 0; **(d)** is a fine-tuning method. *Answer: (b).*
3. Indirect prompt injection comes from: **(a)** the system prompt; **(b)** instructions hidden in content the model reads; **(c)** high temperature; **(d)** tokenisation. *Answer: (b).*
4. With 50 test items and 70% accuracy, the approximate 95% margin is: **(a)** ±1 point; **(b)** ±5 points; **(c)** ±13 points; **(d)** ±40 points. *Answer: (c); SE = √(0.7·0.3/50) ≈ 0.065.*

## Short answer

1. **Compute ROUGE-1 precision, recall and F1** for reference "the model predicts the next token" and output "the model predicts tokens". (4 marks) *Model answer:* reference 6 words, output 4; overlap {the, model, predicts} = 3 (duplicate "the" counted once in the output); recall 3/6 = 0.50, precision 3/4 = 0.75, F1 = 0.60.
2. **Why is valid structured output not sufficient evidence of correctness?** (3 marks) *Model answer:* schemas constrain form (types, allowed values, formats), not truth; values can be well-formed but wrong or invented; field-level accuracy against gold data is needed.
3. **Describe three biases of LLM judges and a mitigation for each.** (6 marks) *Model answer:* position (evaluate both orders), verbosity (length-controlled rubrics or penalties), self-preference (use a different model family), plus calibration against human labels.
4. **Explain why prompt injection cannot be solved by prompting alone and list three system-level defences.** (5 marks) *Model answer:* instructions and data share the same token stream; the model has no hard boundary (2); least privilege, tool allow-lists and argument validation, human approval, output filtering, monitoring (3).

## Exam-style question

**"You are building an assistant that routes and answers student emails. Design the prompting approach and an evaluation plan, and discuss the risks."** (20 marks)

*Marking guide:* prompt structure with system rules, context, examples and format (4); technique choices (few-shot for routing, structured output for extracted fields, retrieval for policy answers) (4); evaluation plan: test set construction, metrics per sub-task, rubric and human review, LLM judge calibration, statistics (6); risks: hallucination, privacy, bias, injection via emails, over-reliance, with mitigations (6).

# Glossary

| Term | Meaning |
|---|---|
| In-context learning | Learning a task from examples in the prompt without weight updates |
| System/developer message | Application-level instructions with priority over user messages |
| Zero/one/few-shot | Prompting with no, one or several examples |
| Role (persona) prompting | Assigning the model a role to shape tone and priorities |
| Chain-of-thought | Prompting the model to write intermediate reasoning steps |
| Self-consistency | Majority vote over several sampled reasoning paths |
| Prompt chaining | Splitting a task into a sequence of prompts |
| Structured output | Output constrained to a schema (e.g. JSON) |
| Context engineering | Selecting, compressing and ordering what enters the context window |
| ROUGE / BLEU | Word-overlap metrics against references (summarisation / translation) |
| LLM-as-a-judge | Using an LLM to score outputs with a rubric |
| Sycophancy | Tendency to agree with the user's stated view |
| Prompt injection | Malicious instructions that override intended behaviour |
| Jailbreak | Prompting that bypasses safety rules |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Ozdemir, S. (2023) *Quick Start Guide to Large Language Models*: chapters on prompt engineering.
* Tunstall et al. (2022) *NLP with Transformers*: chapter 6 (summarisation and ROUGE).

**Official guides**

* [OpenAI: Prompt engineering](https://developers.openai.com/api/docs/guides/prompt-engineering), [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
* [Google Gemini API: Prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies) and [Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
* [Anthropic: Interactive prompt engineering tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) and [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
* [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners), lessons 4–5 (prompt engineering fundamentals and advanced prompts)
* [OpenAI Academy](https://academy.openai.com/): prompting resources

**Videos**

* [Andrej Karpathy: How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw)
* [Microsoft: Generative AI for Beginners video series](https://www.youtube.com/watch?v=k7HaeJs-N-o)

**Papers**

* Wei et al. (2022) [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903); Wang et al. (2022) [Self-Consistency](https://arxiv.org/abs/2203.11171); Zheng et al. (2023) [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685); Brown et al. (2020) [GPT-3: few-shot learners](https://arxiv.org/abs/2005.14165)

# Link to the group project

Every group builds an **evaluation set** of at least 30 realistic cases for its application (with gold answers or a rubric), runs at least two prompt variants and one model alternative through the lab harness, and reports results with margins of error and error analysis. This set is reused in week 8 (base vs fine-tuned) and weeks 11–12 (deployed system, RAG).
