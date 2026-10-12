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
# CPU computation is the core route. The shared setup supplies `DEVICE`, `SEED`,
# `SMOKE`, `check()` and `ask_model()`. Keep its optional switches off initially.
# Complete each TODO until its check prints ✅ before running dependent cells.
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
# ## Before coding: follow the complete flow
#
# A **model** describes possible outputs. **Training** builds that description;
# **sampling** uses it to draw an output. This lab starts with a table you can
# inspect, then applies the same probability ideas to a neural language model.
#
# ![Learn a name model, then generate](fig:beginner_overview)
#
# Read the blocks left to right: `words` supplies names; `N` stores adjacent
# character counts; `P` stores next-character probabilities; `sample_name`
# repeatedly chooses a character. The sampled character selects the next row.
# `avg_nll` scores a supplied name instead of generating one.
#
# ![Score a known name using learned probabilities](fig:beginner_lab)
#
# This scoring flow combines a supplied name with P; it leaves P unchanged.
# The classifier uses separate two-dimensional points, and the LLM uses separate text inputs.
# The table below lists those additional experiments; it is not a connected data flow.
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
# Run the install cell, then the shared setup injected immediately after it.
# Select the same notebook kernel when reopening this lab.
#
# > **Core:** confirm that imports and the readiness checks work.
# > **Change one thing:** change one readiness probability, then explain its effect on the sum.
# > **Optional stretch:** inspect the printed package versions and find one library's official documentation.
# > **Common errors:** a missing package usually means the wrong kernel or an install cell that was skipped.
#
# **Predict:** which list index selects the first item, and what should the three probabilities total?

# %% [markdown]
# **Purpose:** Install the small set of lab packages.
# **Why now:** The next cells import these libraries.
# **Expected observation:** An installation completes or reports a dependency error.
#

# %% tags=["colab-install"]
%pip install -q transformers accelerate scikit-learn matplotlib

# %% [markdown]
# **Purpose:** Import lab-specific tools and check Python readiness.
# **Why now:** Later cells use arrays, indexing and probability distributions.
# **Expected observation:** A length-three array, its shape, and three ✅ checks.
#

# %%
import math
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import torch

# The shared setup already chose the device and seeded these libraries.
readiness_items = ["first", "second", "third"]
readiness_probs = np.array([0.2, 0.3, 0.5])
print("Readiness probabilities:", readiness_probs, "| shape:", readiness_probs.shape)
check(readiness_items[0] == "first", "Index 0 selects the first item.", "Python indexes start at 0.")
check(readiness_probs.shape == (3,), "The array has three entries.", "(3,) means a one-dimensional array of length 3.")
check(np.isclose(readiness_probs.sum(), 1.0), "These probabilities sum to 1.", "Add the three probabilities; a distribution must total 1.")

# %% [markdown]
# ## Part 1 · A generative model from counts
#
# > **Core:** complete TODOs 1–3, then score and sample names from the learned table.
# > **Change one thing:** change temperature; predict which next-character choices become more concentrated.
# > **Optional stretch:** reserve some names before counting and score that separate set afterwards.
# > **Common errors:** rows mean previous characters; columns mean next characters. Include both `.` markers.
#
# **Predict:** how many adjacent pairs appear in `.emma.`? Which row follows a sampled `e`?
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
#

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
# **Hint:** use `stoi[a]` for the row and `stoi[b]` for the column; increase that entry by one.
#
# **Read the shapes:** `N` has `V` rows and `V` columns. `stoi` translates a
# character to its index; `itos` translates an index back to a character.
# Adding start and end markers makes both beginning and stopping learnable.

# %% [markdown]
# **Purpose:** Count neighbouring character pairs.
# **Why now:** This is the learned part of the first model.
# **Expected observation:** The completed count matrix contains nonnegative counts.
#
# ![Learn probabilities from names](fig:beginner_training)
#

# %%
# Keep all 26 letters even when the fallback list omits a rare letter such as q.
chars = sorted(set("".join(words)) | set("abcdefghijklmnopqrstuvwxyz"))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)
print("vocabulary size:", V)

N = np.zeros((V, V), dtype=np.int64)
for w in words:
    seq = ["."] + list(w) + ["."]
    for a, b in zip(seq, seq[1:]):
        ### BEGIN SOLUTION
        N[stoi[a], stoi[b]] += 1
        ### END SOLUTION

def check_bigrams(counts):
    # Every name of length L contributes L + 1 neighbouring pairs.
    expected_pairs = sum(len(word) + 1 for word in words)
    correct = counts.shape == (V, V) and np.all(counts >= 0) and counts.sum() == expected_pairs
    # Encode ordered pairs as flat IDs; bincount independently checks each table entry.
    pair_ids = [stoi[a] * V + stoi[b] for word in words
                for a, b in zip("." + word, word + ".")]
    reference = np.bincount(pair_ids, minlength=V * V).reshape(V, V)
    correct = correct and np.array_equal(counts, reference)
    return check(correct, "TODO 1 counted each ordered previous/next pair exactly once.",
                 "Rows are previous characters; columns are next characters. Include both boundary pairs.")

print("N shape:", N.shape)
assert check_bigrams(N), "Complete TODO 1 and rerun this cell before continuing."
print("total bigrams counted:", N.sum())
print("most common first letters:", [itos[j] for j in np.argsort(-N[0])[:5]])

# %% [markdown]
# **Purpose:** View the learned count table.
# **Why now:** Axes explain the row and column meanings.
# **Expected observation:** A heat map of log(1 + counts).
#

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
#

# %%
P = (N + 1) / (N + 1).sum(axis=1, keepdims=True)
print("P shape:", P.shape, "| row-total shape:", P.sum(axis=1, keepdims=True).shape)
check(np.allclose(P.sum(axis=1), 1.0), "Every row of P sums to 1.", "Use axis=1 and keepdims=True when dividing by each row total.")
print("p(next | '.') for a, e, z:", round(P[0, stoi["a"]], 3), round(P[0, stoi["e"]], 3), round(P[0, stoi["z"]], 3))

# %% [markdown]
# **TODO 2:** write `avg_nll(word, P)`, the **average negative log-likelihood per character** of a word, including the final `.`.
# For `emma` you average over five predictions: `.→e, e→m, m→m, m→a, a→.`
# **Hint:** take the natural log of each transition probability, average the five logs, then change the sign.

# %% [markdown]
# **Purpose:** Score supplied names.
# **Why now:** Separate evaluating a known name from creating one.
# **Expected observation:** A lower average NLL means a greater geometric-mean probability per predicted character. Whole-name likelihood also depends on name length.
#

# %%
def avg_nll(word, P):
    # Read-only scoring: this function does not update the learned table P.
    seq = ["."] + list(word) + ["."]
    ### BEGIN SOLUTION
    logps = [math.log(P[stoi[a], stoi[b]]) for a, b in zip(seq, seq[1:])]
    return -sum(logps) / len(logps)
    ### END SOLUTION

def check_avg_nll(score_function):
    # This explicit five-transition example catches a missing end marker.
    e, m, a = stoi["e"], stoi["m"], stoi["a"]
    expected = -(math.log(P[0, e]) + math.log(P[e, m]) + math.log(P[m, m])
                 + math.log(P[m, a]) + math.log(P[a, 0])) / 5
    try:
        observed = score_function("emma", P)
        uniform = score_function("emma", np.full((V, V), 1 / V))
        correct = observed is not None and uniform is not None and np.isfinite(observed)
        correct = correct and np.isclose(observed, expected) and np.isclose(uniform, math.log(V))
    except (TypeError, ValueError, KeyError, IndexError):
        correct = False
    return check(correct, "TODO 2 scores all five transitions and the uniform reference correctly.",
                 "Return a number: negative mean of the log probabilities, including a→. at the end.")

assert check_avg_nll(avg_nll), "Complete TODO 2 and rerun this cell before continuing."

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
#

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
# **Hint:** divide `np.log(P[i])` by positive `T`, exponentiate after subtracting the maximum, and normalise.
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
# ![Generate a name from a fixed table](fig:beginner_inference)
#

# %%
rng = np.random.default_rng(SEED)


def sample_name(P, T=1.0, max_len=20):
    # i is the previous character's index; index 0 represents the '.' marker.
    if T <= 0:
        raise ValueError("T must be positive when sampling; try T=1.0.")
    out, i = [], 0
    for _ in range(max_len):
        ### BEGIN SOLUTION
        logits = np.log(P[i]) / T
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        i = rng.choice(V, p=p)
        ### END SOLUTION
        if i == 0:
            break
        out.append(itos[i])
    return "".join(out)

def check_sample_name(sample_function):
    # A deterministic reference table follows .→e→m→a→. exactly.
    reference = np.eye(V)
    route = [0, stoi["e"], stoi["m"], stoi["a"], 0]
    for previous, following in zip(route, route[1:]):
        reference[previous] = 0
        reference[previous, following] = 1
    # A check should not consume the random draws reserved for the real experiment.
    saved_state = rng.bit_generator.state
    try:
        with np.errstate(divide="ignore"):
            observed = sample_function(reference, T=1.0, max_len=20)
        correct = observed == "ema"
    except (TypeError, ValueError, IndexError):
        correct = False
    finally:
        rng.bit_generator.state = saved_state
    return check(correct, "TODO 3 follows the selected row and stops at the end marker.",
                 "Update i from rng.choice(V, p=p); append itos[i] unless i is the . marker.")

assert check_sample_name(sample_name), "Complete TODO 3 and rerun this cell before continuing."

name_samples = {}  # keep the samples so you can compare temperatures afterwards
for T in [0.5, 1.0, 1.5]:
    name_samples[T] = [sample_name(P, T) for _ in range(10)]
    print(f"T={T}:", ", ".join(name_samples[T]))

# %% [markdown]
# **Purpose:** Explore temperature without downloading a model or retraining.
# **Why now:** A slider makes the change in one probability row visible.
# **Expected observation:** A chart of first-letter probabilities and five sampled names.
# If widgets do not display, edit the temperature argument in the preceding sampling cell.
# Predict the effect before moving the slider. Counts and learned probabilities remain fixed.

# %% tags=["skip-test"]
try:
    import ipywidgets as widgets
    from IPython.display import display

    def view_temperature(T):
        logits = np.log(P[0]) / T
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        plt.figure(figsize=(8, 3))
        plt.bar([itos[index] for index in range(V)], probabilities)
        plt.xlabel("First character (or immediate end marker)")
        plt.ylabel("Probability after temperature")
        plt.title(f"T = {T:.1f}; the learned table is unchanged")
        plt.show()
        print("Sampled names:", [sample_name(P, T=T) for _ in range(5)])

    slider = widgets.FloatSlider(value=1.0, min=0.2, max=2.0, step=0.1,
                                 description="T", continuous_update=False)
    display(widgets.interactive(view_temperature, T=slider))
except ImportError:
    print("Widgets unavailable. Change numeric T in sample_name(P, T=1.0) and compare the printed samples.")

# %% [markdown]
# ✍️ **Question 1.** Describe how the names change as the temperature goes from 0.5 to 1.5. Link your answer to what temperature does to the distribution.
#
# <!-- BEGIN ANSWER -->
# At T = 0.5 common character transitions get more probability, so repeated patterns are more likely. At T = 1.0 we sample from the smoothed table P. At T = 1.5 rare transitions receive a larger share, so unusual combinations become more likely. Describe the actual lengths and spellings you observed: temperature alone does not guarantee short or pronounceable names. It changes how we sample, while the learned counts stay the same. The bigram model sees only one previous character, so plausible pairs can still produce an odd whole name. Models with longer context, such as the transformer in week 5, can learn longer patterns.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 2 · Generative vs. discriminative
#
# > **Core:** fit both classifiers and complete TODO 4 to generate class-0 points.
# > **Change one thing:** change one diagonal covariance entry while keeping both classifiers' training data identical.
# > **Optional stretch:** try non-Gaussian class data and explain where QDA's assumption breaks.
# > **Common errors:** class 0 uses `means_[0]` and `covariance_[0]`; each generated point needs two coordinates.
#
# **Predict:** where should new class-0 points cluster? Which fitted model describes their spread?
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
#

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
print("X shape:", X.shape, "| y shape:", y.shape, "| train/test:", Xtr.shape, Xte.shape)

disc = LogisticRegression().fit(Xtr, ytr)
genm = QuadraticDiscriminantAnalysis(store_covariance=True).fit(Xtr, ytr)
print(f"logistic regression test accuracy: {disc.score(Xte, yte):.3f}")
print(f"QDA (generative) test accuracy:     {genm.score(Xte, yte):.3f}")

# %% [markdown]
# **TODO 4:** use the fitted generative model to **create 200 new points of class 0**. QDA stores the class means in `genm.means_` and the covariance matrices in `genm.covariance_`.
# **Hint:** call `gen.multivariate_normal` with the class-0 mean, class-0 covariance and sample count 200.

# %% [markdown]
# **Purpose:** Draw new class-0 points.
# **Why now:** QDA learned a class-conditional Gaussian.
# **Expected observation:** Stars spread around the learned class-0 mean.
#

# %%
### BEGIN SOLUTION
new_points = gen.multivariate_normal(genm.means_[0], genm.covariance_[0], 200)
### END SOLUTION
### STUB new_points = np.zeros((200, 2))  # replace with samples from the class-0 Gaussian

def check_generated_points(points):
    correct = points.shape == (200, 2) and np.all(np.isfinite(points))
    correct = correct and np.all(points.var(axis=0) > 0)
    return check(correct, "TODO 4 produced 200 varying points with two coordinates each.",
                 "Sample from genm.means_[0] and genm.covariance_[0]; a table of zeros is not a Gaussian sample.")

print("New point shape:", new_points.shape)
assert check_generated_points(new_points), "Complete TODO 4 and rerun this cell before continuing."
plt.figure(figsize=(6, 5))
plt.scatter(*Xa.T, s=8, alpha=0.4, label="real class 0")
plt.scatter(*Xb.T, s=8, alpha=0.4, label="real class 1")
plt.scatter(*new_points.T, s=14, marker="*", color="black", label="generated class 0")
plt.legend()
plt.title("A generative classifier can also generate")
plt.show()

# %% [markdown]
# ✍️ **Question 2.** Compare the two held-out accuracies. Why can only one classifier generate new input data? When would you prefer the discriminative model?
#
# <!-- BEGIN ANSWER -->
# QDA learns a full model of the inputs for each class, $p(x \mid y)$, so we can sample from it. Logistic regression learns only $p(y \mid x)$, the boundary between classes, and says nothing about where inputs lie, so there is nothing to sample from. The discriminative model is often preferred when the only goal is classification: it makes fewer assumptions (here QDA assumes Gaussian classes), needs fewer parameters, and usually performs better when the generative assumptions are wrong or the input is high-dimensional (e.g. images). Generative models are preferred when we need to create data, detect unusual inputs (low $p(x)$), handle missing inputs or learn from unlabelled data.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 3 · A small open LLM in action
#
# ![How an assistant reaches its fixed weights and application](fig:lecture_flow_18)
#
# This lab loads a pre-trained assistant; it does not repeat the training stages.
#
# > **Core:** inspect tokens, complete TODO 5 and compare short generations from SmolLM2-135M-Instruct.
# > **Change one thing:** change temperature while keeping the question and model fixed.
# > **Optional stretch:** try SmolLM2-360M-Instruct after completing the core, then record the changed checkpoint.
# > **Common errors:** a download error does not describe model quality. Check the connection and checkpoint spelling.
#
# **Predict:** will one token always equal one word? Will changing temperature change the weights?
#
# We use **SmolLM2-135M-Instruct**, an open-weight model documented by Hugging Face.
# It runs on CPU and is instruction-tuned: additional training prepared it for chat instructions.
# It still predicts one token at a time. Small size makes this a teaching model, not a reliability benchmark.

# %% [markdown]
# **Purpose:** Load a pretrained language model.
# **Why now:** We inspect inference before any complete answer.
# **Expected observation:** Model name, parameter count and vocabulary size.
#

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
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
#

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
    print(f"{lang:8s} ID shape {(len(ids),)} | {pieces[:12]}")

# %% [markdown]
# ### 3.2 Inside the model: next-token probabilities
# The model outputs a **logit** for every token in its vocabulary. Softmax turns logits into probabilities.
#
# **TODO 5:** complete `next_token_probs` so it applies a temperature `T` to the logits before the softmax.
# **Hint:** convert the final-position logits to float32, divide by positive `T`, then apply `torch.softmax(..., dim=-1)`.
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
# ![One language-model generation step](fig:beginner_mechanism)
#

# %%
@torch.no_grad()
def next_token_probs(prompt, T=1.0):
    if T <= 0:
        raise ValueError("T must be positive for a probability distribution; try T=1.0.")
    ids = tok(prompt, return_tensors="pt").to(DEVICE)
    all_logits = model(**ids).logits
    print("Input ID shape:", tuple(ids["input_ids"].shape), "| logit shape:", tuple(all_logits.shape))
    logits = all_logits[0, -1]  # One score for each vocabulary entry at the next position.
    ### BEGIN SOLUTION
    # Float32 keeps probability arithmetic precise when model scores use bfloat16.
    return torch.softmax(logits.float() / T, dim=-1)
    ### END SOLUTION

def check_next_token_probs(probability_function):
    try:
        low = probability_function("The cat sat on the", T=0.5)
        high = probability_function("The cat sat on the", T=1.5)
        correct = isinstance(low, torch.Tensor) and isinstance(high, torch.Tensor)
        correct = correct and low.shape == (model.config.vocab_size,) and high.shape == low.shape
        if correct:
            correct = bool(torch.isfinite(low).all() and torch.isfinite(high).all()
                           and (low >= 0).all() and (high >= 0).all()
                           and torch.isclose(low.sum(), torch.tensor(1.0, device=DEVICE))
                           and torch.isclose(high.sum(), torch.tensor(1.0, device=DEVICE))
                           and low.max() > high.max())
    except (TypeError, ValueError, RuntimeError, AttributeError):
        correct = False
    return check(correct, "TODO 5 returns a distribution; lower T concentrates the strongest prediction.",
                 "Return softmax(logits / T, dim=-1), with one probability per vocabulary entry and total 1.")

assert check_next_token_probs(next_token_probs), "Complete TODO 5 and rerun this cell before continuing."

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
#

# %%
def chat(question, temperature=0.7, top_p=0.9, max_new_tokens=32, system=None):
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
# > **Core:** compare the two local responses and separate claims from supporting evidence.
# > **Change one thing:** add the uncertainty instruction; keep the fictional question unchanged.
# > **Optional stretch:** choose the shared Gemini or OpenAI API switch if eligible, then compare the actual response using the same criteria.
# > **Common errors:** fluent language and confident wording do not prove that a cited paper exists.
#
# **Predict:** will the model question the invented citation? How could you verify any claimed findings?
#
# The paper below **does not exist**. Watch what a small model does when asked about it, then see whether an instruction to admit uncertainty helps.

# %% [markdown]
# **Purpose:** Test an unsupported-paper question.
# **Why now:** Plausible language needs external evidence.
# **Expected observation:** The response may invent, refuse or express uncertainty.
#

# %%
fictional_prompt = "Summarise the main findings of the 2019 paper 'Quantum Gradient Folding for Transformers' by Smith and Okafor."
plain_answer = chat(fictional_prompt, temperature=0.7, max_new_tokens=64)
honest_answer = chat(fictional_prompt, temperature=0.7, max_new_tokens=64,
                     system="If you are not certain that a paper exists, say so clearly instead of guessing.")
print("Plain question:\n", plain_answer)
print("\nWith an honesty instruction:\n", honest_answer)

# %% [markdown]
# ### Optional: compare with Gemini or the OpenAI API
#
# Read the shared setup's regional and account guidance first. Free eligibility and quota are not guaranteed.
# Gemini API-client distribution has a separate paid-service condition in the EEA, UK and Switzerland.
# The entire local lab works without a key or billing. Keep Gemini on the eligible free route.
#
# Store `GEMINI_API_KEY` or `OPENAI_API_KEY` in Colab Secrets or the local environment, never in a code cell.
# Set exactly one of `USE_GEMINI` or `USE_OPENAI` to `True` in shared setup and rerun it.
# OpenAI API usage may incur charges, separately from ChatGPT subscriptions; the local task needs no API.
# Only this invented paper question is sent. The helper never prints your key and makes no request by default.

# %% [markdown]
# **Purpose:** Optionally compare a hosted assistant.
# **Why now:** Application tools and settings can change behaviour.
# **Expected observation:** No API request by default; a response only after opting in with a configured key. API availability can vary.
#

# %%
hosted_answer = ask_model(fictional_prompt, temperature=0.7, max_output_tokens=128)
if hosted_answer is None:
    comparison_answer = honest_answer
    comparison_model = MODEL_ID
    print("No hosted response: continue using the local honesty-instruction response.")
else:
    comparison_answer = hosted_answer
    comparison_model = OPENAI_MODEL if USE_OPENAI else GEMINI_MODEL
    print("Optional hosted response:\n", comparison_answer)
print("Comparison checkpoint:", comparison_model)

# %% [markdown]
# ✍️ **Question 3.** What did the small model do with the fictional paper? Did the honesty instruction help? How did the large hosted model (or a chat assistant) behave? Which step of the five-step checklist catches this problem?
#
# <!-- BEGIN ANSWER -->
# Describe what your actual responses did: they may invent details, admit uncertainty or refuse.
# Unsupported details about this invented paper are hallucinations; plausible wording is not a source.
# Compare the plain and honesty-instruction responses using the same evidence criterion.
# If the selected API returned no response, record that and use the local comparison; do not claim a hosted result.
# One question cannot establish which model family is more reliable.
# **Verify** catches unsupported claims: check the actual publication record before using a citation.
# **Disclose & record** stores the checkpoint, settings and date so others can understand your comparison.
# <!-- END ANSWER -->

# %% [markdown] tags=["solution-only"]
# **Instructor tooling:** save this run's measured results for the teaching notes.
# **Why now:** Every number quoted in the notes must come from a recorded run.
# **Expected observation:** "results saved" when `GENAI_RESULTS_PATH` is set; otherwise nothing.

# %% tags=["solution-only"]
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
# ![Trace responsible use from purpose to recorded evidence](fig:lecture_flow_41)
#
# > **Core:** record two experiments with the setting, observed output and a stated check.
# > **Change one thing:** tighten one success criterion, then decide whether the same output still meets it.
# > **Optional stretch:** reserve a new example before editing your prompt and test your final choice once.
# > **Common errors:** report observed results; never fill the table with an expected or invented outcome.
#
# **Predict:** would the same output meet a stricter success criterion? State the criterion before judging it.
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
# <!-- BEGIN ANSWER -->
# Example: using a chat assistant to explain a PyTorch error in the week 2 lab. **Purpose:** allowed by the module policy for learning, not for writing assessed text. **Protect data:** the error message contains no personal data; do not paste API keys or file paths with names. **Prompt clearly:** include the error, the code line and what I expected. **Verify:** run the suggested fix and check the loss still decreases; read the PyTorch docs for the function. **Disclose & record:** note in the project log that the fix came from an assistant, with the model name and date.
# <!-- END ANSWER -->
#
# ## End-of-lab self-check
#
# 1. For `emma`, how many bigrams include both boundary markers? What shape does `N` have?
# <!-- BEGIN ANSWER -->
# Five: `.→e`, `e→m`, `m→m`, `m→a` and `a→.`. N has shape (V, V), with a previous-character row and next-character column.
# <!-- END ANSWER -->
#
# 2. Does changing temperature retrain `N`, `P` or the language-model weights?
# <!-- BEGIN ANSWER -->
# No. Temperature changes the sampling distribution derived from fixed probabilities or logits; the learned parameters remain unchanged.
# <!-- END ANSWER -->
#
# 3. Why can QDA generate input points while logistic regression alone cannot?
# <!-- BEGIN ANSWER -->
# QDA models p(x given y), the distribution of input points in each class. Logistic regression models p(y given x), class probabilities for a supplied input.
# <!-- END ANSWER -->
#
# 4. Why is a token ID neither a whole word nor a probability? What is one next-token output's shape?
# <!-- BEGIN ANSWER -->
# An ID is an index into the tokenizer vocabulary. Its text piece can be punctuation, part of a word or a word. One final-position distribution has one entry per vocabulary item.
# <!-- END ANSWER -->
#
# 5. What evidence supports a response about a paper? What should you record if the selected API is unavailable?
# <!-- BEGIN ANSWER -->
# Check that the cited paper exists and that it supports the claimed findings. Record that no hosted response was obtained and identify the local checkpoint actually used. Confidence or valid wording alone is insufficient.
# <!-- END ANSWER -->
#
# **Before next week:** watch the 3Blue1Brown video *Large Language Models explained briefly* and read Foster (2023), chapter 1.
