# -*- coding: utf-8 -*-
# Open this file in VS Code with the Jupyter extension, or run cells in order.
# The notebook contains embedded diagrams. Standalone PNGs are in Diagrams/.

# %% [markdown]
# # Week 1 Lab: Probability, sampling and a first look at an LLM
#
# **Module:** Generative AI (MSc in Artificial Intelligence) · **Time:** 2 hours · **Learning outcomes:** MIMLO 1, MIMLO 4
#
# In this lab you will:
#
# 1. build a tiny **generative model** of names from counts, measure its **likelihood** and sample from it at different **temperatures**;
# 2. compare a **generative** and a **discriminative** classifier on the same data, and generate new data from the generative one;
# 3. run a small **open-weight language model**, look inside its next-token probabilities and change the decoding settings;
# 4. probe **hallucination** and compare a small open model with a large hosted assistant;
# 5. record your evidence and apply the five-step responsible-use checklist.
#
# | Part | Topic | Suggested time |
# |---|---|---|
# | 0 | Setup | 10 min |
# | 1 | A generative model from counts | 35 min |
# | 2 | Generative vs. discriminative | 20 min |
# | 3 | A small LLM in action | 35 min |
# | 4 | Hallucination and tool comparison | 15 min |
# | 5 | Evidence and reflection | 5 min |

# %% [markdown]
# > **INSTRUCTOR VERSION — contains solutions. Do not distribute before the lab.**
#
# > **How to run this notebook**
# > - **Google Colab (recommended):** File ▸ Upload notebook, then Runtime ▸ Change runtime type ▸ **GPU** if available. A T4 is sufficient for the GPU examples. Free GPU access varies. Run cells top to bottom with Shift+Enter.
# > - **Local Jupyter / VS Code:** Python 3.10+; run the install cell once. Read this week's runtime note. Model downloads and training can take longer on CPU, and some full experiments need a GPU.
# > - **API keys (optional cells only):** store keys in Colab ▸ 🔑 Secrets or an environment variable. Never paste a key into a notebook you share.
# > - Cells marked **TODO** are yours to complete before running dependent cells. Questions marked ✍️ need a short written answer.
# > - Read each diagram by following its numbered blocks. The solid arrows carry data to the next block. A dashed arrow shows a step that repeats.

# %% [markdown]
# ## This week's place in the course
#
# ![Course map](Diagrams/beginner_course_map.png)

# %% [markdown]
# ## Before coding: follow the complete flow
#
# A **model** describes possible outputs. **Training** builds that description;
# **sampling** uses it to draw an output. This lab starts with a table you can
# inspect, then applies the same probability ideas to a neural language model.
#
# ![Learn a name model, then generate](Diagrams/beginner_overview.png)
#
# Read the blocks left to right: `words` supplies names; `N` stores adjacent
# character counts; `P` stores next-character probabilities; `sample_name`
# repeatedly chooses a character. The sampled character selects the next row.
# `avg_nll` scores a supplied name instead of generating one.
#
# ![Follow the Week 1 experiments](Diagrams/beginner_lab.png)
#
# | Diagram block | Code to find | First observation |
# |---|---|---|
# | Build the table | `N`, `P` | Matrix rows mean previous characters; each probability row sums to 1. |
# | Score and sample | `avg_nll`, `sample_name` | Scoring receives a name; generation creates one. |
# | Compare classifiers | `disc`, `genm` | QDA learns an input distribution that can be sampled. |
# | Inspect tokens | `tok`, `model` | Token pieces and integer IDs describe the same input. |
# | Inspect generation | `next_token_probs`, `chat` | A next-token distribution differs from a complete answer. |
# | Verify and record | `fictional_prompt`, the evidence table | Record observed claims and how you checked them. |
#
# **Pause and predict:** after generating `e`, which probability row should be
# read next? Point to the block that repeats before running the sampler.
# Work through the TODOs yourself; the diagrams describe roles and data flow.

# %% [markdown]
# ## Read, run, change, check
#
# **Read:** start with the count table, where every number has a visible meaning.
# A **probability** is a number from 0 to 1; 0.25 means a one-in-four chance.
# A **distribution** lists the chances of all possible choices; they add to 1.
# A **sample** is one choice drawn using those chances. A **seed** fixes the
# random starting state so you can repeat an experiment.
#
# **Run:** complete one TODO, run its cell, then inspect the output before
# running cells that depend on it. The setup creates names used by later cells.
# **Change:** vary only `T` in the name sampler first; leave `N` and `P` alone.
# **Check:** every row of `P` must add to 1. A generated name is a sample,
# so another run may give a different name without being a coding error.
#
# In formulas, `p` means probability, the bar in `p(next | previous)` means
# "given", and the product sign means "multiply all the step probabilities".
# `log` is the natural logarithm. **NLL** is a penalty for unlikely known
# answers; lower is better on the same data. **Nats** are its units.

# %% [markdown]
# ## Part 0 · Setup
#
# Run the next cell once. On Google Colab most packages are already installed, so it finishes quickly.

# %% [markdown]
# **Purpose:** Install the small set of lab packages.
# **Why now:** The next cells import these libraries.
# **Expected observation:** An installation completes or reports a dependency error.

# %%
import subprocess as _install_process
import sys as _install_sys
_install_process.check_call([_install_sys.executable, '-m', 'pip'] + ['install', '-q', 'transformers', 'accelerate', 'scikit-learn', 'matplotlib'])

# %% [markdown]
# **Purpose:** Choose device and repeatable random state.
# **Why now:** Later experiments share this setup.
# **Expected observation:** Printed device is CPU or CUDA; neither outcome is a conceptual result.

# %%
import math
import os
import random
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch

# SMOKE mode is only used by the instructor's automatic test: tiny settings so every cell runs quickly.
SMOKE = os.environ.get("GENAI_LAB_SMOKE") == "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| device:", DEVICE)
if DEVICE == "cpu":
    print("No GPU found. That is fine for this lab; the language model part will just be a little slower.")

# %% [markdown]
# ## Part 1 · A generative model from counts
#
# We model names **one character at a time**. A *bigram* model assumes each character depends only on the previous one:
#
# $$p(\text{name}) = \prod_{t} p(c_t \mid c_{t-1})$$
#
# The special token `.` marks the start and end of a name, so `emma` is modelled as `. e m m a .`
#
# ### 1.1 Load the data

# %% [markdown]
# **Purpose:** Load example names.
# **Why now:** Counts need a visible training dataset.
# **Expected observation:** A name list and its size; fallback use is printed.

# %%
URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
FALLBACK = """emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily elizabeth
mila ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope layla riley
zoey nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe leah hazel violet
aurora savannah audrey brooklyn bella claire skylar lucy paisley everly anna caroline nova genesis
liam noah william james oliver benjamin elijah lucas mason logan alexander ethan jacob michael
daniel henry jackson sebastian aiden matthew samuel david joseph carter owen wyatt john jack luke
jayden dylan grayson levi isaac gabriel julian mateo anthony jaxon lincoln joshua christopher""".split()

try:
    with urllib.request.urlopen(URL, timeout=15) as f:
        words = f.read().decode("utf-8").split()
    print("Downloaded", len(words), "names")
except Exception as e:
    words = FALLBACK
    print("Download failed, using the built-in list of", len(words), "names:", e)

if SMOKE:
    words = words[:2000]
print(words[:10])

# %% [markdown]
# ### 1.2 Count bigrams
#
# Build the vocabulary (26 letters plus `.`), then count how often each character follows each other character.
#
# **TODO 1:** fill the count matrix `N`, where `N[i, j]` is the number of times character `j` follows character `i`.
#
# **Read the shapes:** `N` has `V` rows and `V` columns. `stoi` translates a
# character to its index; `itos` translates an index back to a character.
# Adding start and end markers makes both beginning and stopping learnable.

# %% [markdown]
# **Purpose:** Count neighbouring character pairs.
# **Why now:** This is the learned part of the first model.
# **Expected observation:** The completed count matrix contains nonnegative counts.
#
# ![Learn probabilities from names](Diagrams/beginner_training.png)

# %%
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)
print("vocabulary size:", V)

N = np.zeros((V, V), dtype=np.int64)
for w in words:
    seq = ["."] + list(w) + ["."]
    for a, b in zip(seq, seq[1:]):
        N[stoi[a], stoi[b]] += 1

print("total bigrams counted:", N.sum())
print("most common first letters:", [itos[j] for j in np.argsort(-N[0])[:5]])

# %% [markdown]
# **Purpose:** View the learned count table.
# **Why now:** Axes explain the row and column meanings.
# **Expected observation:** A heat map of log(1 + counts).

# %%
plt.figure(figsize=(7, 6))
plt.imshow(np.log1p(N), cmap="Purples")
plt.xticks(range(V), [itos[i] for i in range(V)])
plt.yticks(range(V), [itos[i] for i in range(V)])
plt.xlabel("next character")
plt.ylabel("previous character")
plt.title("log(1 + bigram count)")
plt.show()

# %% [markdown]
# ### 1.3 From counts to probabilities (maximum likelihood)
#
# The maximum-likelihood estimate of $p(c_t = j \mid c_{t-1} = i)$ is simply the fraction of times `j` follows `i`.
# We add 1 to every count (**add-one smoothing**) so that no bigram gets probability exactly zero.
# This is a smoothed estimate: without the added counts, row normalisation is
# the ordinary maximum-likelihood estimate. `axis=1` totals each row, and
# `keepdims=True` keeps a column shape so division applies to every row entry.
# For example, counts `[2, 1, 0]` become `[3, 2, 1]` after smoothing, then
# probabilities `[3/6, 2/6, 1/6]`. The added counts are a small safety margin
# for combinations missing from this dataset.

# %% [markdown]
# **Purpose:** Turn counts into next-character chances.
# **Why now:** Sampling requires a distribution in each row.
# **Expected observation:** All row sums are one, up to rounding.

# %%
P = (N + 1) / (N + 1).sum(axis=1, keepdims=True)
print("each row sums to 1:", np.allclose(P.sum(axis=1), 1.0))
print("p(next | '.') for a, e, z:", round(P[0, stoi["a"]], 3), round(P[0, stoi["e"]], 3), round(P[0, stoi["z"]], 3))

# %% [markdown]
# **TODO 2:** write `avg_nll(word, P)`, the **average negative log-likelihood per character** of a word, including the final `.`.
# For `emma` you average over five predictions: `.→e, e→m, m→m, m→a, a→.`

# %% [markdown]
# **Purpose:** Score supplied names.
# **Why now:** Separate evaluating a known name from creating one.
# **Expected observation:** A lower NLL means greater probability under this table.

# %%
def avg_nll(word, P):
    # Read-only scoring: this function does not update the learned table P.
    seq = ["."] + list(word) + ["."]
    logps = [math.log(P[stoi[a], stoi[b]]) for a, b in zip(seq, seq[1:])]
    return -sum(logps) / len(logps)


for w in ["emma", "anna", "xqzz"]:
    print(f"{w:6s} average NLL = {avg_nll(w, P):.3f} nats  (perplexity {math.exp(avg_nll(w, P)):.1f})")

# %% [markdown]
# Now compare three models on the whole dataset. Lower negative log-likelihood means the model finds real names more probable.
#
# * **uniform**: every next character equally likely, $-\log(1/27)$
# * **unigram**: next character ignores the previous one
# * **bigram**: our model
#
# This scores the same names used to build the table. It is a learning check,
# not evidence of performance on new names. A stronger evaluation would set
# aside some names before counting and score that separate set afterwards.

# %% [markdown]
# **Purpose:** Compare simple probability models.
# **Why now:** Conditioning on one previous character adds information.
# **Expected observation:** Three training-set scores; these are not held-out evaluation.

# %%
uni = (N.sum(axis=0) + 1) / (N.sum() + V)
P_uni = np.tile(uni, (V, 1))
total = {"uniform": [], "unigram": [], "bigram": []}
for w in words:
    total["uniform"].append(math.log(V))
    total["unigram"].append(avg_nll(w, P_uni))
    total["bigram"].append(avg_nll(w, P))
for k, v in total.items():
    print(f"{k:8s}: {np.mean(v):.3f} nats per character")

# %% [markdown]
# ### 1.4 Sampling with temperature
#
# **TODO 3:** complete `sample_name`. Starting from `.`, repeatedly
# 1. take the row of probabilities for the current character,
# 2. apply temperature: $p_i \propto p_i^{1/T}$ (this is the same as dividing log-probabilities by $T$ before the softmax),
# 3. sample the next character, stop when you sample `.`.
#
# **Check one step:** use the previous character to choose a row, then draw
# one index from that row. `T` must be positive in this function. `T=1` leaves
# the learned probabilities unchanged; lower `T` favours common choices more.
# Stop at `.` or at `max_len`, so the loop cannot run forever.

# %% [markdown]
# **Purpose:** Generate names using the fixed table.
# **Why now:** The latest sampled letter chooses the next row.
# **Expected observation:** Name lists at three temperatures; N and P stay fixed.
#
# ![Generate a name from a fixed table](Diagrams/beginner_inference.png)

# %%
rng = np.random.default_rng(SEED)


def sample_name(P, T=1.0, max_len=20):
    # i is the previous character's index; index 0 represents the '.' marker.
    out, i = [], 0
    for _ in range(max_len):
        logits = np.log(P[i]) / T
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        i = rng.choice(V, p=p)
        if i == 0:
            break
        out.append(itos[i])
    return "".join(out)


name_samples = {}  # keep the samples so you can compare temperatures afterwards
for T in [0.5, 1.0, 1.5]:
    name_samples[T] = [sample_name(P, T) for _ in range(10)]
    print(f"T={T}:", ", ".join(name_samples[T]))

# %% [markdown]
# ✍️ **Question 1.** Describe how the names change as the temperature goes from 0.5 to 1.5. Link your answer to what temperature does to the distribution.
#
# **✅ Model answer:**
# At T = 0.5 common character transitions get more probability, so repeated patterns are more likely. At T = 1.0 we sample from the smoothed table P. At T = 1.5 rare transitions receive a larger share, so unusual combinations become more likely. Describe the actual lengths and spellings you observed: temperature alone does not guarantee short or pronounceable names. It changes how we sample, while the learned counts stay the same. The bigram model sees only one previous character, so plausible pairs can still produce an odd whole name. Models with longer context, such as the transformer in week 5, can learn longer patterns.

# %% [markdown]
# ## Part 2 · Generative vs. discriminative
#
# We fit two classifiers to the same 2-D data:
#
# * **Logistic regression** models $p(y \mid x)$ directly (discriminative).
# * **Quadratic discriminant analysis (QDA)** fits a Gaussian $p(x \mid y)$ for each class plus $p(y)$, then classifies with Bayes' rule (generative).
#
# Here `x` is a point with two coordinates and `y` is its class, 0 or 1.
# A **Gaussian** is a bell-shaped distribution. Its **mean** gives the centre;
# its **covariance** describes the spread and how the coordinates vary together.
# Both models predict a class. Only QDA also learns where points of that class
# tend to lie, which gives us a distribution from which to draw new points.

# %% [markdown]
# **Purpose:** Fit two classifiers to the same points.
# **Why now:** Their similar prediction task isolates different modelling choices.
# **Expected observation:** Two held-out accuracies; neither is guaranteed to win.

# %%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

gen = np.random.default_rng(1)
Xa = gen.multivariate_normal([-1.2, 0.3], [[0.6, 0.25], [0.25, 0.4]], 300)
Xb = gen.multivariate_normal([1.1, -0.2], [[0.5, -0.2], [-0.2, 0.6]], 300)
X = np.vstack([Xa, Xb])
y = np.array([0] * 300 + [1] * 300)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)

disc = LogisticRegression().fit(Xtr, ytr)
genm = QuadraticDiscriminantAnalysis(store_covariance=True).fit(Xtr, ytr)
print(f"logistic regression test accuracy: {disc.score(Xte, yte):.3f}")
print(f"QDA (generative) test accuracy:     {genm.score(Xte, yte):.3f}")

# %% [markdown]
# **TODO 4:** use the fitted generative model to **create 200 new points of class 0**. QDA stores the class means in `genm.means_` and the covariance matrices in `genm.covariance_`.

# %% [markdown]
# **Purpose:** Draw new class-0 points.
# **Why now:** QDA learned a class-conditional Gaussian.
# **Expected observation:** Stars spread around the learned class-0 mean.

# %%
new_points = gen.multivariate_normal(genm.means_[0], genm.covariance_[0], 200)

plt.figure(figsize=(6, 5))
plt.scatter(*Xa.T, s=8, alpha=0.4, label="real class 0")
plt.scatter(*Xb.T, s=8, alpha=0.4, label="real class 1")
plt.scatter(*new_points.T, s=14, marker="*", color="black", label="generated class 0")
plt.legend()
plt.title("A generative classifier can also generate")
plt.show()

# %% [markdown]
# ✍️ **Question 2.** Both classifiers have similar accuracy. Why can only one of them generate new data? When would you still prefer the discriminative model?
#
# **✅ Model answer:**
# QDA learns a full model of the inputs for each class, $p(x \mid y)$, so we can sample from it. Logistic regression learns only $p(y \mid x)$, the boundary between classes, and says nothing about where inputs lie, so there is nothing to sample from. The discriminative model is often preferred when the only goal is classification: it makes fewer assumptions (here QDA assumes Gaussian classes), needs fewer parameters, and usually performs better when the generative assumptions are wrong or the input is high-dimensional (e.g. images). Generative models are preferred when we need to create data, detect unusual inputs (low $p(x)$), handle missing inputs or learn from unlabelled data.

# %% [markdown]
# ## Part 3 · A small open LLM in action
#
# We load **SmolLM2**, a small open-weight model from Hugging Face. It is tiny compared with frontier assistants, which makes its behaviour (and its mistakes) easy to study. It runs on a CPU; a GPU makes it faster.

# %% [markdown]
# **Purpose:** Load a pretrained language model.
# **Why now:** We inspect inference before any complete answer.
# **Expected observation:** Model name, parameter count and vocabulary size.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct" if SMOKE else "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"{MODEL_ID}: {n_params / 1e6:.0f} million parameters, vocabulary of {tok.vocab_size} tokens")

# %% [markdown]
# ### 3.1 Tokens
# How does the tokenizer split text? Compare an English sentence with the same sentence in another language.
# A **tokenizer** turns text pieces into integer IDs the model can read.
# One token may be a word, part of a word or punctuation. More tokens here
# means more pieces to process; it does not directly measure translation quality.

# %% [markdown]
# **Purpose:** Inspect tokenizer pieces.
# **Why now:** A token is an indexed text piece.
# **Expected observation:** Different sentences can require different numbers of IDs.

# %%
sentences = {
    "English": "Generative models learn a probability distribution over data.",
    "Irish": "Foghlaimíonn samhlacha giniúnacha dáileadh dóchúlachta thar shonraí.",
    "Hindi": "जनरेटिव मॉडल डेटा पर प्रायिकता वितरण सीखते हैं।",
}
token_counts = {}
for lang, s in sentences.items():
    ids = tok(s)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    token_counts[lang] = len(ids)
    print(f"{lang:8s} {len(ids):3d} tokens | {pieces[:12]}")

# %% [markdown]
# ### 3.2 Inside the model: next-token probabilities
# The model outputs a **logit** for every token in its vocabulary. Softmax turns logits into probabilities.
#
# **TODO 5:** complete `next_token_probs` so it applies a temperature `T` to the logits before the softmax.
#
# **Read the output:** model logits have batch, sequence-position, and
# vocabulary dimensions. Selecting `[0, -1]` asks about the next token after
# this single prompt, leaving one score per vocabulary item. This function
# inspects one prediction; `chat` below performs the repeated generation loop.

# %% [markdown]
# **Purpose:** Expose one next-token distribution.
# **Why now:** Scores must become probabilities before sampling.
# **Expected observation:** Top candidates shift in concentration with positive T.
#
# ![One language-model generation step](Diagrams/beginner_mechanism.png)

# %%
@torch.no_grad()
def next_token_probs(prompt, T=1.0):
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    logits = model(**ids).logits[0, -1]  # logits for the next position
    return torch.softmax(logits / T, dim=-1)


prompt = "The cat sat on the"
top_tokens = {}
for T in [0.5, 1.0, 1.5]:
    probs = next_token_probs(prompt, T)
    top = torch.topk(probs, 5)
    top_tokens[T] = [(tok.decode([int(i)]), p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
    print(f"T={T}: " + ", ".join(f"{t!r}: {p:.2f}" for t, p in top_tokens[T]))

# %% [markdown]
# ### 3.3 Decoding settings change the output, not the model
# We ask the same question several times with a low and a high temperature.
# `top_p` keeps a group of likely next-token choices before sampling. The
# **prompt** is the input text; the **completion** is the generated continuation.
# Read each definition for accuracy as well as noticing variation in its wording.

# %% [markdown]
# **Purpose:** Generate complete answers.
# **Why now:** Repeated prediction extends the context.
# **Expected observation:** Low and high temperature outputs; inspect actual accuracy.

# %%
def chat(question, temperature=0.7, top_p=0.9, max_new_tokens=60, system=None):
    # The chat template adds role markers; generated output also contains the input.
    # Decoding slices off those input tokens so the returned text is only the answer.
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=16 if SMOKE else max_new_tokens, do_sample=temperature > 0,
                         temperature=temperature if temperature > 0 else None, top_p=top_p if temperature > 0 else None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


q = "Give a one-sentence definition of generative AI."
answers = {}
for T in [0.2, 1.2]:
    print(f"--- temperature {T}")
    answers[T] = [chat(q, temperature=T) for _ in range(2 if SMOKE else 3)]
    for k, answer in enumerate(answers[T], 1):
        print(f"[{k}]", answer)

# %% [markdown]
# ## Part 4 · Hallucination and tool comparison
#
# The paper below **does not exist**. Watch what a small model does when asked about it, then see whether an instruction to admit uncertainty helps.

# %% [markdown]
# **Purpose:** Test an unsupported-paper question.
# **Why now:** Plausible language needs external evidence.
# **Expected observation:** The response may invent, refuse or express uncertainty.

# %%
fictional_prompt = "Summarise the main findings of the 2019 paper 'Quantum Gradient Folding for Transformers' by Smith and Okafor."
plain_answer = chat(fictional_prompt, temperature=0.7, max_new_tokens=90)
honest_answer = chat(fictional_prompt, temperature=0.7, max_new_tokens=90,
                     system="If you are not certain that a paper exists, say so clearly instead of guessing.")
print("Plain question:\n", plain_answer)
print("\nWith an honesty instruction:\n", honest_answer)

# %% [markdown]
# ### Optional: compare with a large hosted model (free Gemini API key)
#
# 1. Create a free key at **Google AI Studio** (aistudio.google.com) with your own account.
# 2. In Colab, open 🔑 **Secrets**, add `GEMINI_API_KEY`, and allow this notebook to use it.
# 3. Run the cell. Without a key it simply skips; you can paste the same prompt into any chat assistant instead.

# %% [markdown]
# **Purpose:** Optionally compare a hosted assistant.
# **Why now:** Application tools and settings can change behaviour.
# **Expected observation:** A response only if a key is configured; API availability can vary.

# %%
def get_key(name="GEMINI_API_KEY"):
    try:
        from google.colab import userdata  # only available on Colab
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


key = None if SMOKE else get_key()
if not key:
    print("No GEMINI_API_KEY found: skipping the hosted-model comparison. Use a chat assistant in your browser instead.")
else:
    try:
        from google import genai
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-genai"])
        from google import genai
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(model="gemini-flash-latest", contents=fictional_prompt)
    print(resp.text)

# %% [markdown]
# ✍️ **Question 3.** What did the small model do with the fictional paper? Did the honesty instruction help? How did the large hosted model (or a chat assistant) behave? Which step of the five-step checklist catches this problem?
#
# **✅ Model answer:**
# Describe what your run did: it may invent details, admit uncertainty or refuse. If it asserts unsupported paper details, that is hallucination; plausible wording is not a source. The honesty instruction sometimes helps a little, but small models follow it unreliably. Large hosted assistants more often say they cannot find the paper, especially when they have a search tool, but they can still fabricate details, so their answer must also be checked. The **Verify** step catches it: search for the paper in Google Scholar or on the publisher's site before using any claim. Recording the model, settings and date (**Disclose & record**) lets others reproduce the test.

# %% [markdown]
# **Instructor tooling:** save this run's measured results for the teaching notes.
# **Why now:** Every number quoted in the notes must come from a recorded run.
# **Expected observation:** "results saved" when `GENAI_RESULTS_PATH` is set; otherwise nothing.

# %%
if os.environ.get("GENAI_RESULTS_PATH"):
    import json
    from datetime import datetime, timezone
    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(), "run_mode": "smoke" if SMOKE else "full",
        "names": len(words), "vocab": V, "device": DEVICE, "model_id": MODEL_ID, "model_params": n_params,
        "name_nll": {w: avg_nll(w, P) for w in ["emma", "anna", "xqzz"]},
        "dataset_nll": {k: float(np.mean(v)) for k, v in total.items()},
        "name_samples": {str(T): s for T, s in name_samples.items()},
        "accuracy": {"logistic_regression": disc.score(Xte, yte), "qda": genm.score(Xte, yte)},
        "token_counts": token_counts,
        "top_tokens": {str(T): rows for T, rows in top_tokens.items()},
        "answers": {str(T): rows for T, rows in answers.items()},
        "fictional_paper": {"plain": plain_answer, "honesty_instruction": honest_answer},
    }
    with open(os.environ["GENAI_RESULTS_PATH"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
    print("results saved")

# %% [markdown]
# ## Part 5 · Evidence table and reflection
#
# Complete the table with **two** of your experiments (double-click to edit).
#
# | Task | Model and version | Settings (T, top-p, seed) | What happened | Correct? How did you check? | One improvement |
# |---|---|---|---|---|---|
# | | | | | | |
# | | | | | | |
#
# ✍️ **Question 4.** Apply the five-step checklist (purpose, protect data, prompt clearly, verify, disclose) to one way you plan to use AI tools in this module.
#
# **✅ Model answer:**
# Example: using a chat assistant to explain a PyTorch error in the week 2 lab. **Purpose:** allowed by the module policy for learning, not for writing assessed text. **Protect data:** the error message contains no personal data; do not paste API keys or file paths with names. **Prompt clearly:** include the error, the code line and what I expected. **Verify:** run the suggested fix and check the loss still decreases; read the PyTorch docs for the function. **Disclose & record:** note in the project log that the fix came from an assistant, with the model name and date.
#
# **Before next week:** watch the 3Blue1Brown video *Large Language Models explained briefly* and read Foster (2023), chapter 1.

# %% [markdown]
# ## Week 1: what we learned
#
# ![Class recap](Diagrams/beginner_recap.png)
#
# Illustrative sequence: 0.5 × 0.2 × 0.1 = 0.01. A probable sequence can still contain a false claim.
#
# **Explain without looking:** If temperature changes, which changes: sampling choices or learned weights?
#
# **Instructor answer:** Sampling choices change. The weights stay fixed during this generation.
