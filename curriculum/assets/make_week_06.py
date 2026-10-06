"""REAL results for week 6 (LLMs and families). CPU is fine (~10 minutes).

1. Tokenizer comparison across model families.
2. The same inputs through an encoder (BERT), an encoder-decoder (Flan-T5) and a decoder LLM (Qwen2.5).
3. Next-token distributions of a decoder LLM.
4. Perplexity of normal vs scrambled text (GPT-2).
5. A small reasoning model (Qwen3-0.6B) with thinking on vs off: see make_week_06_reasoning.py.

Run:  .venv-labs/Scripts/python.exe curriculum/assets/make_week_06.py
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, save  # noqa: E402

from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,  # noqa: E402
                          AutoTokenizer, pipeline)

OUT = Path(__file__).resolve().parent / "week_06"
OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
metrics = {}

# ------------------------------------------------------------ 1. tokenizers
texts = {
    "English": "Large language models predict the next token. Tokenisers split text into pieces.",
    "Irish": "Tuarann samhlacha móra teanga an chéad chomhartha eile. Roinneann tokenizers téacs ina phíosaí.",
    "Python code": "def softmax(z):\n    e = np.exp(z - z.max())\n    return e / e.sum()",
}
toks = {"GPT-2": "gpt2", "BERT": "bert-base-uncased", "Flan-T5": "google/flan-t5-base",
        "Qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct", "SmolLM2": "HuggingFaceTB/SmolLM2-360M-Instruct"}
counts, vocab = {}, {}
for name, mid in toks.items():
    t = AutoTokenizer.from_pretrained(mid)
    vocab[name] = len(t)
    counts[name] = {k: len(t(v, add_special_tokens=False)["input_ids"]) for k, v in texts.items()}
metrics["tokenizers"] = {"vocab": vocab, "counts": counts}
fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.16
for i, (name, c) in enumerate(counts.items()):
    xs = np.arange(len(texts)) + (i - 2) * w
    ax.bar(xs, list(c.values()), width=w, color=SERIES[i % len(SERIES)], label=f"{name} (vocab {vocab[name]:,})")
ax.set_xticks(range(len(texts)))
ax.set_xticklabels(list(texts))
ax.set_ylabel("number of tokens")
ax.set_title("Same text, different tokenisers", loc="left")
ax.legend(fontsize=10, ncol=2)
ax.grid(axis="x", visible=False)
save(fig, OUT / "tokenizers.png")

# ------------------------------------------------------------ 2. three families
fam = {}
fm = pipeline("fill-mask", model="bert-base-uncased")
fam["bert_fill_mask"] = [(r["token_str"], round(r["score"], 3)) for r in fm("The capital of Ireland is [MASK].")]
fam["bert_fill_mask_2"] = [(r["token_str"], round(r["score"], 3)) for r in fm("Generative models can [MASK] new images.")]
t5_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def t5_gen(p):
    ids = t5_tok(p, return_tensors="pt")
    with torch.no_grad():
        o = t5_model.generate(**ids, max_new_tokens=40)
    return t5_tok.decode(o[0], skip_special_tokens=True)


fam["t5"] = {p: t5_gen(p) for p in ["What is the capital of Ireland?", "Translate English to German: How old are you?",
                                     "Summarize: Generative models learn a probability distribution over data and can sample new examples that look like the training data."]}
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").to(DEV).eval()


def chat(model, tok, q, max_new=80, **kw):
    msgs = [{"role": "user", "content": q}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kw).to(DEV)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new = out[0, inp["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new), time.time() - t0


fam["qwen"] = {q: chat(qwen, qtok, q)[0] for q in ["What is the capital of Ireland?", "Translate English to German: How old are you?",
                                                    "Summarize in one sentence: Generative models learn a probability distribution over data and can sample new examples that look like the training data."]}
metrics["families"] = fam

# ------------------------------------------------------------ 3. next-token distributions
# A BASE (pre-trained, not instruction-tuned) model shows pure next-token prediction; instruct models given raw
# text tend to treat it as a fill-in-the-blank exercise and put mass on tokens such as " ______".
del qwen
if DEV == "cuda":
    torch.cuda.empty_cache()
btok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(DEV).eval()
metrics["next_token_model"] = "Qwen/Qwen2.5-0.5B (base)"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for a, prompt in zip(axes, ["The capital of Ireland is", "My favourite food is"]):
    ids = btok(prompt, return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = base(**ids).logits[0, -1]
    p = torch.softmax(logits.float(), -1)
    top = torch.topk(p, 8)
    labels = [repr(btok.decode([int(i)])) for i in top.indices]
    a.barh(labels[::-1], top.values.cpu().numpy()[::-1], color=[PRIMARY] * 7 + [ACCENT])
    a.set_title(f'"{prompt} ___"', loc="left")
    a.set_xlabel("probability")
    a.grid(axis="y", visible=False)
    metrics.setdefault("next_token", {})[prompt] = [(l, round(v, 3)) for l, v in zip(labels, top.values.tolist())]
save(fig, OUT / "next_token.png")

# ------------------------------------------------------------ 4. perplexity
gtok = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").eval()


def ppl(text):
    ids = gtok(text, return_tensors="pt")
    with torch.no_grad():
        loss = gpt2(**ids, labels=ids["input_ids"]).loss
    return math.exp(loss.item())


s = "The students trained a small language model and measured its perplexity on held-out text."
words = s.split()
random.seed(0)
shuffled = " ".join(random.sample(words, len(words)))
metrics["perplexity_gpt2"] = {"normal": ppl(s), "shuffled words": ppl(shuffled), "Irish": ppl(texts["Irish"]), "code": ppl(texts["Python code"])}

# ------------------------------------------------------------ 5. reasoning on/off
# Run make_week_06_reasoning.py afterwards: it adds metrics["reasoning"] (sampled decoding, three runs per mode).
previous = OUT / "metrics.json"
if previous.exists():  # keep an earlier reasoning result when only parts 1-4 are regenerated
    old = json.loads(previous.read_text(encoding="utf-8"))
    if "reasoning" in old:
        metrics["reasoning"] = old["reasoning"]

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
