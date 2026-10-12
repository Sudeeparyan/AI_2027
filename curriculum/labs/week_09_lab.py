# %% [markdown]
# # Week 9 Lab: Multimodal embeddings with CLIP (and a little audio)
#
# **Module:** Generative AI (MSc in Artificial Intelligence) · **Time:** 2 hours · **Learning outcomes:** MIMLO 1, 2, 4
#
# CLIP places images and captions in one shared vector space. You will measure what that space can do and where it fails:
#
# 1. Embed **images and texts** and read a similarity table.
# 2. **Zero-shot classification** on CIFAR-10, and how the text template changes accuracy.
# 3. **Text-to-image retrieval** and precision@k.
# 4. Compute the **contrastive loss** that trains CLIP, on fixed embeddings.
# 5. See the **modality gap** in a 2-D projection.
# 6. **Speech → text → image:** transcribe audio with Whisper, measure word error rate, then search the gallery.
#
# **CPU core:** frozen CLIP ViT-B/32 encodes 1,000 CIFAR-10 test images once; Whisper-base transcribes ten short
# audiobook clips. The teaching notes give the measured CPU time on the course laptop; the final cell prints your own.
# **Optional stretch:** SigLIP (a sigmoid-loss relative of CLIP) on the same gallery. CLIP is evaluated, not trained:
# no cell updates its weights.
#
# Use the injected setup's `DEVICE`, `SMOKE`, `SEED`, `check()` and `ask_model()`. Keep `USE_GEMINI = False` and
# `USE_OPENAI = False` for the local core. The optional hosted cell sends three short text requests only; OpenAI API
# use is billed separately from a ChatGPT subscription.
#
# | Part | Activity | Suggested class time |
# |---|---|---|
# | 0 | Recall vocabulary; predict one similarity | 10 min |
# | 1 | Embeddings and the similarity table | 20 min |
# | 2 | Zero-shot classification | 20 min |
# | 3 | Retrieval and precision@k | 15 min |
# | 4 | Contrastive loss and temperature | 20 min |
# | 5 | Modality gap | 10 min |
# | 6 | Speech → text → image; self-checks | 25 min |
#
# Class time includes discussion and coding; it is not a measured compute time.

# %% [markdown]
# ## Start here: follow the two encoder paths
#
# A **modality** is a type of data, such as images, text or audio. Separate **encoders** turn each image and each
# caption into a vector called an **embedding**. After normalisation, a dot product measures how similar two
# directions are; the highest score picks the best match.
#
# ![Two encoders, one shared space](fig:beginner_overview)
#
# ## Your map from experiments to code
#
# | Step | Find this code | Observe this |
# |---|---|---|
# | Load gallery | `subset`, `images`, `labels` | One label per image |
# | Build vectors | `embed_images`, `embed_texts` | Shapes `[N, 512]` and norms near 1 |
# | Classify images | `zero_shot_accuracy` | Accuracy for each text template |
# | Search gallery | `precision_at_k` | Returned pictures and precision@k |
# | Inspect learning | `clip_loss` | Correct versus shuffled losses |
# | Add speech | `asr`, `wer` | Reference words, transcript and retrieved classes |
#
# **Read:** an **embedding** is a numerical description of an input. **Cosine similarity** compares directions; it is a
# match score, not a probability. **Predict → run → change → check:** predict each table before running it, change one
# setting at a time, and run every `check_…()` cell.

# %% [markdown]
# **Purpose:** Install the model, data, evaluation and widget libraries in Colab.
# **Why now:** Later cells import Transformers, Datasets, soundfile, scikit-learn and ipywidgets.
# **Expected output:** Installation messages; a prepared local environment can skip this cell.

# %% tags=["colab-install"]
%pip install -q transformers datasets soundfile scikit-learn matplotlib pandas ipywidgets

# %% [markdown]
# **Purpose:** Import tools, run three readiness checks and load the frozen CLIP model.
# **Why now:** Every later part reuses the same encoders.
# **Expected output:** Three ✅ lines, then the device and CLIP's parameter count (about 151 million).

# %%
import io
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch.nn.functional as F
from datasets import Audio, load_dataset
from transformers import CLIPModel, CLIPProcessor

core_started = time.perf_counter()
a, b = torch.tensor([3.0, 4.0]), torch.tensor([4.0, 3.0])
ready = check(abs(float(F.normalize(a, dim=0).norm()) - 1) < 1e-6, "Normalising [3, 4] gives a vector of length 1.",
              "F.normalize divides a vector by its length.")
ready &= check(abs(float(F.normalize(a, dim=0) @ F.normalize(b, dim=0)) - 0.96) < 1e-6,
               "Cosine similarity of [3, 4] and [4, 3] is (12 + 12) / 25 = 0.96.",
               "Normalise both vectors, then take the dot product.")
ready &= check(torch.zeros(6, 512).T.shape == (512, 6), "Transposing a [6, 512] table gives [512, 6].",
               ".T swaps the two axes of a matrix.")
assert ready, "Review these three ideas before continuing."

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("device:", DEVICE, "| CLIP parameters:", f"{sum(p.numel() for p in clip.parameters()) / 1e6:.0f} M", "| smoke:", SMOKE)

# %% [markdown]
# ## Part 1 · One space for images and text
#
# CLIP has an **image encoder** (a Vision Transformer) and a **text encoder** (a transformer). Each maps its input to a
# 512-number vector in the **same space**, trained so that matching image–caption pairs point in similar directions.
#
# > **Core:** normalise the embeddings and read the 6 × 6 similarity table.
# > **Change one thing:** rewrite one caption (for example "a cat sleeping" → "a sleeping kitten") and compare its column.
# > **Optional stretch:** add a seventh caption that matches no image and check its highest score.
# > **Common errors:** forgetting `F.normalize` gives norms far from 1, and dot products then mix length with direction.
#
# **TODO 1:** complete `embed_images` and `embed_texts` so they return **L2-normalised** embeddings.
# **Hint:** `F.normalize(feats, dim=-1)` divides each row by its length.
# The image table is `[N, 512]`; six captions form `[6, 512]`. Chosen image rows times the transposed caption table
# gives `[6, 6]`: one row per image, one column per caption.

# %% [markdown]
# **Purpose:** Load the gallery, encode images and captions, check the norms and show the similarity table.
# **Why now:** Every later experiment compares these vectors.
# **Expected output:** Shapes, one ✅ line and a heatmap whose diagonal should be bright.

# %%
cifar = load_dataset("uoft-cs/cifar10", split="test")
CLASSES = cifar.features["label"].names
N = 100 if SMOKE else 1000
subset = cifar.shuffle(seed=0).select(range(N))
images = [im.convert("RGB") for im in subset["img"]]
labels = np.array(subset["label"])


@torch.no_grad()
def embed_images(imgs, batch=64):
    # Keep input order: row i must still represent image i after batching.
    out = []
    for i in range(0, len(imgs), batch):
        inputs = proc(images=imgs[i:i + batch], return_tensors="pt").to(DEVICE)
        feats = clip.visual_projection(clip.vision_model(pixel_values=inputs["pixel_values"]).pooler_output)
        ### BEGIN SOLUTION
        out.append(F.normalize(feats, dim=-1).cpu())
        ### END SOLUTION
        ### STUB out.append(feats.cpu())  # TODO 1: normalise each row
    return torch.cat(out)


@torch.no_grad()
def embed_texts(texts):
    # The attention mask distinguishes real text tokens from padding.
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    feats = clip.text_projection(clip.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output)
    ### BEGIN SOLUTION
    return F.normalize(feats, dim=-1).cpu()
    ### END SOLUTION
    ### STUB return feats.cpu()  # TODO 1: normalise each row


started = time.perf_counter()
img_emb = embed_images(images)
print(f"image embeddings {tuple(img_emb.shape)} in {time.perf_counter() - started:.0f} s")
captions = ["a photo of a red car", "a small bird on a branch", "a cat sleeping", "a ship at sea",
            "a horse in a field", "an airplane in the sky"]
txt_emb = embed_texts(captions)
print("text embeddings", tuple(txt_emb.shape))
def check_embeddings():
    check(torch.allclose(img_emb.norm(dim=-1), torch.ones(len(img_emb)), atol=1e-4)
          and torch.allclose(txt_emb.norm(dim=-1), torch.ones(len(txt_emb)), atol=1e-4),
          "TODO 1: every image and caption vector has length 1, so dot products are cosine similarities.",
          "Apply F.normalize(feats, dim=-1) in both functions before returning.")

check_embeddings()

show_idx = [int(np.where(labels == CLASSES.index(c))[0][0]) for c in ["automobile", "bird", "cat", "ship", "horse", "airplane"]]
sim = img_emb[show_idx] @ txt_emb.T
fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]})
ax[0].imshow(np.hstack([np.array(images[i].resize((64, 64))) for i in show_idx])); ax[0].axis("off")
ax[0].set_title("images (left to right = rows 0–5)")
im = ax[1].imshow(sim.numpy(), cmap="Purples"); ax[1].set_xticks(range(6))
ax[1].set_xticklabels([c[:14] for c in captions], rotation=45, ha="right")
ax[1].set_ylabel("image row"); ax[1].set_title("cosine similarity"); plt.colorbar(im); plt.tight_layout(); plt.show()

# %% [markdown]
# ## Part 2 · Zero-shot classification
#
# No training: embed one text per class and give each image the class whose text is most similar. **Zero-shot** means
# no training on this dataset; CLIP was trained on image–caption pairs elsewhere. A **template** is wording with a slot,
# such as `a photo of a {}.`.
#
# > **Core:** compare three template choices.
# > **Change one thing:** add your own template to the ensemble and rerun only this part.
# > **Optional stretch:** Compare a template ensemble chosen on a separate development gallery; keep this test gallery for one final check.
# > **Common errors:** averaging unit vectors gives a shorter vector, so normalise again after the average.
#
# **TODO 2:** complete `zero_shot_accuracy(templates)`: for each class, embed every template filled with the class name,
# **average** those embeddings, **re-normalise**, predict the most similar class text for each image, and return
# `(accuracy, predictions)`.
# **Hint:** `F.normalize(embed_texts([t.format(c) for t in templates]).mean(dim=0), dim=-1)`.
#
# ### Zero-shot classification route
#
# ![Zero-shot classification route](fig:beginner_lab)
#
# **Predict before running:** could changing only the words change an image's predicted class?

# %% [markdown]
# **Purpose:** Classify every gallery image with three template choices and check the function.
# **Why now:** Zero-shot classification is retrieval with one text per class.
# **Expected output:** Three accuracy lines and one ✅ line.

# %%
def zero_shot_accuracy(templates):
    ### BEGIN SOLUTION
    class_emb = torch.stack([F.normalize(embed_texts([t.format(c) for t in templates]).mean(dim=0), dim=-1)
                             for c in CLASSES])
    pred = (img_emb @ class_emb.T).argmax(dim=1).numpy()
    return float((pred == labels).mean()), pred
    ### END SOLUTION
    ### STUB return 0.0, np.zeros(len(labels), dtype=int)  # TODO 2: average, re-normalise, predict


TEMPLATES = {
    "label only: '{}'": ["{}"],
    "'a photo of a {}.'": ["a photo of a {}."],
    "ensemble of 5 templates": ["a photo of a {}.", "a blurry photo of a {}.", "a low resolution photo of a {}.",
                                "a close-up photo of a {}.", "a pixelated photo of a {}."],
}
results = {}
for name, templates in TEMPLATES.items():
    acc, pred = zero_shot_accuracy(templates)
    results[name] = acc
    print(f"{name:28s} accuracy = {acc:.3f}")
last_pred = pred
def check_zero_shot():
    expected_classes = torch.stack([F.normalize(embed_texts([t.format(c) for t in TEMPLATES["ensemble of 5 templates"]]).mean(dim=0), dim=-1) for c in CLASSES])
    expected_predictions = (img_emb @ expected_classes.T).argmax(dim=1).numpy()
    check(pred.shape == labels.shape and np.array_equal(pred, expected_predictions) and abs(acc - float((pred == labels).mean())) < 1e-9 and acc > 0.5,
          "TODO 2: one prediction per image, and accuracy is the fraction matching the labels.",
          "Return (accuracy, predictions); predictions come from argmax over the class-text similarities.")

check_zero_shot()

# %% [markdown]
# **Purpose:** Compare predicted and reference classes.
# **Why now:** An average accuracy hides which classes are confused.
# **Expected output:** A 10 × 10 confusion matrix; read the bright off-diagonal cells.

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels, last_pred, labels=range(10))
plt.figure(figsize=(6.5, 5.5)); plt.imshow(cm, cmap="Purples")
plt.xticks(range(10), CLASSES, rotation=60); plt.yticks(range(10), CLASSES); plt.xlabel("predicted"); plt.ylabel("true")
plt.title("Zero-shot CLIP on CIFAR-10"); plt.colorbar(); plt.show()

# %% [markdown]
# ✍️ **Question 1.** CLIP was never trained on CIFAR-10 labels. Why does zero-shot classification work, and why does the
# template matter? Which classes are confused, and why might that be?
#
# <!-- BEGIN ANSWER -->
# CLIP learned from about 400 million image–caption pairs to place images near the captions that describe them. A class name turned into a caption ("a photo of a cat") lands near images of cats, so choosing the most similar class text is a classifier with no task-specific training. The template matters because CLIP's text encoder was trained on natural captions, not isolated words: "a photo of a {}" resembles its training text, and averaging several templates reduces the variance of any single phrasing. Read your confusion matrix: confusions usually involve visually similar classes at CIFAR's tiny 32 × 32 resolution (upscaled to 224 × 224), such as cat and dog or deer and horse.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 3 · Text-to-image retrieval and precision@k
#
# Retrieval is the same maths in the other direction: embed a query text and rank all images by similarity.
# **Precision@k** is the fraction of the top k results that are relevant. If four of the first five results for a car
# query are cars, precision@5 is 4/5 = 0.8. It does not say how many cars in the whole gallery were found (that is recall).
#
# > **Core:** compute precision@1, @5 and @10 for the ten class queries.
# > **Change one thing:** change `query` to a description of your own and inspect the six returned images.
# > **Optional stretch:** Measure recall@k too, dividing hits by the number of relevant gallery images.
# > **Common errors:** ranking along the wrong axis returns the best query for each image instead of the best images.
#
# **TODO 3:** complete `precision_at_k(scores, gallery_labels, k)`: `scores` has one row per class query and one column per
# image; for each row take the top-k image columns, compute the fraction whose label equals the row's class, and average
# over rows.
# **Hint:** `scores.topk(k, dim=1).indices` gives the top-k image indices for every query row.

# %% [markdown]
# **Purpose:** Implement precision@k, check it on a tiny known table, then measure it on the gallery.
# **Why now:** Retrieval quality is measured on ranked results, not on single scores.
# **Expected output:** One ✅ line, three precision values and a strip of six retrieved images.

# %%
def precision_at_k(scores, gallery_labels, k=10):
    ### BEGIN SOLUTION
    topk = scores.topk(k, dim=1).indices.numpy()
    return float(np.mean([(gallery_labels[topk[c]] == c).mean() for c in range(scores.shape[0])]))
    ### END SOLUTION
    ### STUB return 0.0  # TODO 3: average the fraction of relevant images in each row's top k


toy_scores = torch.tensor([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.3, 0.7]])
def check_precision():
    check(abs(precision_at_k(toy_scores, np.array([0, 1, 1, 1]), k=2) - 0.75) < 1e-9,
          "TODO 3: on the toy table, query 0 gets 1 of 2 relevant and query 1 gets 2 of 2, so precision@2 = 0.75.",
          "Use scores.topk(k, dim=1).indices, then compare gallery_labels at those indices with the row number.")

check_precision()

class_queries = embed_texts([f"a photo of a {c}." for c in CLASSES])
query_scores = class_queries @ img_emb.T      # [10 queries, N images]
precision = {k: precision_at_k(query_scores, labels, k) for k in (1, 5, 10)}
for k, value in precision.items():
    print(f"precision@{k} = {value:.3f}")

query = "a red sports car on a road"
top = (embed_texts([query]) @ img_emb.T)[0].topk(6).indices
plt.figure(figsize=(9, 1.8)); plt.imshow(np.hstack([np.array(images[i].resize((96, 96))) for i in top])); plt.axis("off")
plt.title(f"top-6 images for: '{query}'"); plt.show()

# %% [markdown]
# ## Part 4 · The contrastive loss that trains CLIP
#
# For a batch of $N$ matching pairs with normalised image vectors $u_i$ and text vectors $v_i$, the scores are
# $s_{ij} = u_i \cdot v_j / \tau$. CLIP minimises the **symmetric cross-entropy**: image $i$ should pick text $i$, and
# text $i$ should pick image $i$:
#
# $$\mathcal{L} = \frac{1}{2}\left(\mathrm{CE}(S, \mathrm{diag}) + \mathrm{CE}(S^\top, \mathrm{diag})\right)$$
#
# `N` is the number of pairs, $\tau$ (tau) is a positive temperature, and `S` has shape `[N, N]` with the intended pairs on
# its diagonal. These calls only **measure** the loss of pretrained embeddings; there is no optimiser or backward pass.
#
# ### Score one batch of pairs
#
# ![Score one batch of pairs](fig:beginner_mechanism)
#
# > **Core:** compare the loss for correct and shuffled pairs.
# > **Change one thing:** move the temperature slider and watch one row's probabilities sharpen or flatten.
# > **Optional stretch:** Find an off-diagonal caption that also describes its image and explain the false-negative problem.
# > **Common errors:** using `logits` in both cross-entropy terms scores only one direction; the second needs `logits.T`.
#
# ### How CLIP was trained
#
# ![One contrastive training step](fig:beginner_training)
#
# CLIP's original training repeated this step over about 400 million web pairs, with 32,768 pairs per batch;
# gradients updated both encoders. This lab only measures the loss on six pairs and never updates CLIP.
#
# **TODO 4:** implement `clip_loss(u, v, tau)`.
# **Hint:** `logits = (u @ v.T) / tau`, targets `torch.arange(len(u))`, then average `F.cross_entropy` of `logits` and `logits.T`.

# %% [markdown]
# **Purpose:** Implement the symmetric loss, check it on a known case and measure it on six pairs.
# **Why now:** The loss explains why matching pairs end up close in the shared space.
# **Expected output:** One ✅ line, then the correct, shuffled and uniform-guess losses.

# %%
def clip_loss(u, v, tau=0.07):
    ### BEGIN SOLUTION
    logits = (u @ v.T) / tau
    target = torch.arange(len(u))
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
    ### END SOLUTION
    ### STUB return torch.tensor(0.0)  # TODO 4: symmetric cross-entropy over rows and columns


# Three orthogonal unit pairs at tau = 1: each row is softmax([1, 0, 0]), so the loss is ln(e + 2) − 1 ≈ 0.551.
def check_contrastive_loss():
    probe_u = torch.tensor([[1., 0.], [0., 1.], [.6, .8]])
    probe_v = torch.tensor([[.8, .6], [1., 0.], [0., 1.]])
    probe_scores = probe_u @ probe_v.T / .3
    targets = torch.arange(3)
    expected = .5 * (F.cross_entropy(probe_scores, targets) + F.cross_entropy(probe_scores.T, targets))
    check(torch.allclose(clip_loss(probe_u, probe_v, tau=.3), expected) and abs(float(clip_loss(torch.eye(3), torch.eye(3), tau=1.0)) - (np.log(np.e + 2) - 1)) < 1e-5,
          "TODO 4: three orthogonal pairs at tau = 1 give ln(e + 2) − 1 ≈ 0.551.",
          "Divide the score table by tau, then average F.cross_entropy over rows (logits) and columns (logits.T).")

check_contrastive_loss()

u, v = embed_images([images[i] for i in show_idx]), embed_texts(captions)
shuffle = torch.randperm(len(v), generator=torch.Generator().manual_seed(SEED))
loss_correct, loss_shuffled = float(clip_loss(u, v)), float(clip_loss(u, v[shuffle]))
print(f"loss, correct pairs:  {loss_correct:.3f}")
print(f"loss, shuffled pairs: {loss_shuffled:.3f}")
print(f"loss of a uniform guess = ln(N) = {np.log(len(u)):.3f}")

# %% [markdown]
# ### Change one setting without recomputing embeddings
#
# **Purpose:** See how the temperature changes one image's probabilities over the six captions.
# **Why now:** Temperature controls how sharply the softmax prefers the best match.
# **Expected output:** A slider; moving it prints the six probabilities for the car image. Automated runs skip this cell.

# %% tags=["skip-test"]
try:
    import ipywidgets as widgets

    def show_temperature(tau=0.07):
        probs = torch.softmax(sim[0] / tau, dim=0).numpy()
        print(f"tau = {tau:.3f}: " + ", ".join(f"{c[:12]} {p:.2f}" for c, p in zip(captions, probs)))

    widgets.interact(show_temperature, tau=widgets.FloatLogSlider(value=0.07, base=10, min=-2.5, max=0, step=0.1,
                                                                  continuous_update=False))
except ImportError:
    print("ipywidgets is not available; call show_temperature(0.01) … show_temperature(1.0) instead.")

# %% [markdown]
# ✍️ **Question 2.** Why is the loss low for correct pairs and high for shuffled pairs? What role do the other captions
# in the batch play, and why did CLIP train with very large batches (32,768)?
#
# <!-- BEGIN ANSWER -->
# Each image must select its intended caption among the batch candidates, and each caption must select its image. Correct pairs usually receive higher similarity than mismatched ones, so their cross-entropy is smaller; compare your measured losses, because approximate captions do not guarantee that every diagonal entry wins, and a random shuffle can leave some pairs in place. The other captions act as negatives. Larger batches give more negatives per step, which makes the task harder and the learned space more discriminative, although some negatives may describe another image well (false negatives). The CLIP paper used 32,768 pairs per batch. SigLIP replaces the batch-wide softmax with an independent sigmoid loss for every pair.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 5 · The modality gap
#
# Project image and class-text embeddings together to 2-D with PCA (principal component analysis). PCA is a compressed
# view: nearby plotted points need not preserve every distance in the original 512-number space.
#
# > **Core:** read the plot and the two mean cosine similarities.
# > **Change one thing:** project only images, then only texts, and compare the spread.
# > **Optional stretch:** Compare PCA with another projection; do not interpret either as the original vector geometry.
# > **Common errors:** reading the 2-D plot as exact distances; check claims with the full-space cosine values.

# %% [markdown]
# **Purpose:** Project image and text vectors to two dimensions and compare mean similarities.
# **Why now:** Images and texts share a space but still sit in different regions of it.
# **Expected output:** A scatter plot and two mean cosine values printed below it.

# %%
from sklearn.decomposition import PCA

M = min(300, len(img_emb))
all_emb = torch.cat([img_emb[:M], class_queries]).numpy()
xy = PCA(n_components=2, random_state=0).fit_transform(all_emb)
plt.figure(figsize=(8.5, 5.5))
plt.scatter(xy[:M, 0], xy[:M, 1], c=labels[:M], cmap="tab10", s=10, alpha=0.7, label="images")
plt.scatter(xy[M:, 0], xy[M:, 1], c=range(10), cmap="tab10", s=180, marker="*", edgecolor="black", label="class texts")
kinds = plt.legend(loc="upper center")
plt.gca().add_artist(kinds)
colours = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i), label=c) for i, c in enumerate(CLASSES)]
plt.legend(handles=colours, title="class", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
plt.title("Images and texts occupy different regions (the 'modality gap')"); plt.tight_layout(); plt.show()
mean_image_image = float((img_emb[:200] @ img_emb[:200].T).mean())
mean_image_text = float((img_emb[:200] @ class_queries.T).mean())
print(f"mean cosine similarity: image–image {mean_image_image:.3f} | image–text {mean_image_text:.3f}")

# %% [markdown]
# ## Part 6 · Speech → text → image
#
# **Whisper** is an encoder–decoder transformer that turns audio (a log-mel spectrogram) into text. **Word error rate
# (WER)** = (substitutions + deletions + insertions) / number of reference words, found with an edit distance on words.
# Reference "a red car" and transcript "a blue car" need one substitution, so WER is 1/3. WER can exceed 1 when many
# words are inserted.
#
# > **Core:** transcribe ten clips, compute WER, then use one transcript as a search query.
# > **Change one thing:** record your own spoken description, load it with `soundfile`, and search with its transcript.
# > **Optional stretch:** try `openai/whisper-tiny` and compare WER and time.
# > **Common errors:** a sample-rate mismatch garbles transcripts; Whisper expects 16,000 samples per second.
#
# ### Speech to gallery search
#
# ![Speech to gallery search](fig:beginner_inference)
#
# **TODO 5:** complete `wer(reference, hypothesis)` with a word-level edit-distance table.
# **Hint:** `d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))`.

# %% [markdown]
# **Purpose:** Implement WER, check it on known cases, then transcribe ten clips.
# **Why now:** Transcript errors enter the image search directly, so measure them first.
# **Expected output:** One ✅ line, a table of references, transcripts and WER, and the mean WER.

# %%
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny" if SMOKE else "openai/whisper-base",
               device=0 if DEVICE == "cuda" else -1)
speech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").cast_column("audio", Audio(decode=False))


def load_audio(ex):
    arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
    return {"raw": arr.astype(np.float32), "sampling_rate": sr}


def wer(reference, hypothesis):
    r, h = reference.lower().split(), hypothesis.lower().replace(",", "").replace(".", "").split()
    ### BEGIN SOLUTION
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(1, len(r))
    ### END SOLUTION
    ### STUB return 0.0  # TODO 5: fill the edit-distance table and divide by the reference length


def check_wer():
    check(abs(wer("a red car", "a blue car") - 1 / 3) < 1e-9 and abs(wer("the cat sat", "the cat sat down") - 1 / 3) < 1e-9
          and wer("a b c", "") == 1.0,
          "TODO 5: one substitution, one insertion and three deletions give WER 1/3, 1/3 and 1.",
          "Initialise the first row and column with 0, 1, 2, …, then take the cheapest of delete, insert or substitute.")

check_wer()

rows, started = [], time.perf_counter()
for ex in speech.select(range(3 if SMOKE else 10)):
    hyp = asr(load_audio(ex))["text"].strip()
    rows.append({"reference": ex["text"].lower()[:70], "whisper": hyp[:70], "WER": round(wer(ex["text"], hyp), 3)})
df_asr = pd.DataFrame(rows)
asr_seconds = time.perf_counter() - started
print(f"mean WER: {df_asr.WER.mean():.3f} over {len(df_asr)} clips in {asr_seconds:.0f} s")
df_asr

# %% [markdown]
# **Purpose:** Draw one clip's sound energy over time and frequency.
# **Why now:** Whisper reads a related representation (a log-mel spectrogram), not the raw waveform.
# **Expected output:** An illustrative spectrogram with linear frequency bins.

# %%
arr = load_audio(speech[0])["raw"]
plt.figure(figsize=(10, 2.5)); plt.specgram(arr, Fs=16000, NFFT=400, noverlap=240, cmap="magma")
plt.xlabel("time (s)"); plt.ylabel("frequency (Hz)"); plt.title("Illustrative audio spectrogram: linear frequency bins"); plt.show()

# %% [markdown]
# **Purpose:** Use recognised words as a gallery search query and compare with a typed description.
# **Why now:** The pipeline chains two models, so a transcript error changes the search.
# **Expected output:** The transcript and the classes of the top-5 images for each query.
#
# The LibriSpeech clips are audiobook sentences rather than image descriptions, so compare the spoken query with a typed
# descriptive query.

# %%
spoken = asr(load_audio(speech[1]))["text"]
print("transcript:", spoken)
retrieved = {}
for text_query in [spoken, "a photo of a ship on the water"]:
    top = (embed_texts([text_query]) @ img_emb.T)[0].topk(5).indices
    retrieved[text_query[:60]] = [CLASSES[labels[i]] for i in top]
    print(f"query: {text_query[:60]!r} -> classes of top-5 images: {retrieved[text_query[:60]]}")

# %% [markdown]
# ✍️ **Question 3.** What are the weak points of a speech → text → image pipeline compared with a model that embeds audio
# directly? Give one advantage of the pipeline.
#
# <!-- BEGIN ANSWER -->
# Errors compound: every transcription error propagates into retrieval, and information that is not words (tone, emotion, background sounds, speaker identity) is lost when audio becomes text. Each stage adds time and cost. A model with a shared audio–image–text space, or an end-to-end multimodal model, can use that non-verbal information and avoid cascaded errors. Advantages of the pipeline: each component can be replaced and tested on its own (WER and retrieval are measured separately), the intermediate transcript is human-readable and auditable, and strong off-the-shelf models exist for each step.
# <!-- END ANSWER -->

# %% [markdown]
# ### Optional: captions written by a hosted model
#
# **Purpose:** Ask one hosted model for three short photo captions and use them as retrieval queries (three calls at most).
# **Why now:** Better query wording can improve retrieval without touching the image encoder.
# **Expected output:** With both switches False (default), a message that the local queries stand. With one switch
# enabled and a key available, three captions and the classes they retrieve. Only text is sent.

# %%
hosted = []
for c in ["cat", "ship", "truck"][: 1 if SMOKE else None]:
    caption = ask_model(f"Write one short, plain photo caption (under 12 words) of a typical {c}. Reply with the caption only.",
                        temperature=0.2, max_output_tokens=30)
    if caption is None:
        break
    top = (embed_texts([caption.strip()]) @ img_emb.T)[0].topk(5).indices
    hosted.append({"class": c, "caption": caption.strip(), "top-5 classes": [CLASSES[labels[i]] for i in top]})
if hosted:
    display(pd.DataFrame(hosted))
else:
    print("No hosted captions (switches off, no key or no response). The local class queries above are the result.")

# %% [markdown]
# ## Optional · SigLIP
#
# SigLIP replaces CLIP's softmax contrastive loss with a **sigmoid** loss on every image–text pair. Try zero-shot
# classification with `google/siglip-base-patch16-224` (use `AutoModel` and `AutoProcessor`, `padding="max_length"`) and
# compare accuracy with Part 2.
#
# ## Evidence for your report
#
# | Experiment | Setting | Result |
# |---|---|---|
# | Zero-shot CIFAR-10 | label only / template / ensemble | |
# | Retrieval | precision@1, @5, @10 | |
# | Contrastive loss | correct vs shuffled | |
# | Whisper | mean WER | |
#
# **Purpose:** Collect this run's results with their settings and print the elapsed time.
# **Why now:** Reported numbers need the model, gallery size and run mode beside them.
# **Expected output:** A summary line and the elapsed seconds; a JSON file only when the teacher sets GENAI_RESULTS_PATH.

# %%
elapsed_seconds = time.perf_counter() - core_started
run_results = {
    "run": {"clip": "openai/clip-vit-base-patch32", "asr": "openai/whisper-tiny" if SMOKE else "openai/whisper-base",
            "device": str(DEVICE), "smoke": bool(SMOKE), "seed": SEED, "n_images": int(len(img_emb)),
            "n_clips": int(len(df_asr)), "elapsed_seconds": elapsed_seconds},
    "zero_shot": results, "precision_at_k": precision,
    "loss_correct": loss_correct, "loss_shuffled": loss_shuffled, "loss_uniform": float(np.log(len(u))),
    "mean_cosine": {"image_image": mean_image_image, "image_text": mean_image_text},
    "mean_wer": float(df_asr.WER.mean()), "asr_examples": df_asr.head(3).to_dict("records"),
    "confusion": cm.tolist(), "classes": CLASSES,
}
print(f"ensemble accuracy {results['ensemble of 5 templates']:.1%} | precision@5 {precision[5]:.2f} | "
      f"mean WER {df_asr.WER.mean():.3f} | elapsed {elapsed_seconds:.0f} s")
results_target = os.environ.get("GENAI_RESULTS_PATH")
if results_target:
    target = Path(results_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Results exported to:", target)

# %% [markdown]
# ## Five self-checks
#
# 1. Why normalise embeddings before taking dot products?
# 2. What shape is the score table for 10 class queries and 1,000 images?
# 3. Four of the top five images for a "car" query are cars. What is precision@5, and what does it not tell you?
# 4. Why does the contrastive loss use both `logits` and `logits.T`?
# 5. A transcript changes "red car" to "bed car". Where does that error show up in the speech-to-image pipeline?
#
# <!-- BEGIN ANSWER -->
# 1. So the dot product measures direction only (cosine similarity), not vector length.
# 2. [10, 1000]: one row per query, one column per image.
# 3. 4/5 = 0.8; it does not say how many cars in the whole gallery were found (recall).
# 4. Rows score image-to-text matching and columns score text-to-image matching; CLIP trains both directions.
# 5. In the text query: CLIP embeds the wrong words, so the retrieved images can change.
# <!-- END ANSWER -->
