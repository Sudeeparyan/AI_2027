"""REAL results for week 5 (transformers and attention).

1. Attention maps from pretrained BERT (bidirectional) and GPT-2 (causal).
2. A small character-level GPT trained on Tiny Shakespeare: loss curve, perplexity, samples during training.

Run:  .venv-gpu/Scripts/python.exe curriculum/assets/make_week_05.py
Outputs: curriculum/assets/week_05/*.png and metrics.json
"""
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from _style import ACCENT, INK, MUTED, PRIMARY, TEAL, save  # noqa: E402

OUT = Path(__file__).resolve().parent / "week_05"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "build" / "data"
DATA.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
metrics = {}
torch.manual_seed(0)

# ------------------------------------------------------------ 1. attention maps
from transformers import AutoModel, AutoTokenizer  # noqa: E402

sent = "The animal didn't cross the street because it was too tired."
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").eval()
enc = tok(sent, return_tensors="pt")
with torch.no_grad():
    out = bert(**enc, output_attentions=True)
toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
it = toks.index("it")
att = torch.stack(out.attentions)[:, 0]  # layers, heads, q, k
# find the head (in layers 5-9) where "it" attends most to "animal"
ani = toks.index("animal")
best = max(((l, h) for l in range(4, 10) for h in range(12)), key=lambda lh: att[lh[0], lh[1], it, ani].item())
metrics["bert_it_animal_head"] = {"layer": best[0] + 1, "head": best[1] + 1, "weight": att[best[0], best[1], it, ani].item()}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
a = axes[0]
im = a.imshow(att[best[0], best[1]].numpy(), cmap="Purples", vmin=0)
a.set_xticks(range(len(toks)))
a.set_xticklabels(toks, rotation=70, fontsize=10)
a.set_yticks(range(len(toks)))
a.set_yticklabels(toks, fontsize=10)
a.set_title(f"BERT layer {best[0] + 1}, head {best[1] + 1}: full attention matrix", loc="left", fontsize=13)
a.set_xlabel("key (attended to)")
a.set_ylabel("query (attending)")
a.grid(False)
b = axes[1]
w = att[best[0], best[1], it].numpy()
b.barh(range(len(toks))[::-1], w, color=[ACCENT if i == ani else PRIMARY for i in range(len(toks))])
b.set_yticks(range(len(toks))[::-1])
b.set_yticklabels(toks, fontsize=10)
b.set_title("Where does 'it' look? (one row)", loc="left", fontsize=13)
b.set_xlabel("attention weight")
b.grid(axis="y", visible=False)
save(fig, OUT / "bert_attention.png")

gtok = AutoTokenizer.from_pretrained("gpt2")
gpt2 = AutoModel.from_pretrained("gpt2", attn_implementation="eager").eval()
genc = gtok(sent, return_tensors="pt")
with torch.no_grad():
    gout = gpt2(**genc, output_attentions=True)
gt = [t.replace("Ġ", "·") for t in gtok.convert_ids_to_tokens(genc["input_ids"][0])]
gatt = torch.stack(gout.attentions)[:, 0]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
for a, (l, h) in zip(axes, [(0, 0), (4, 11), (9, 6)]):
    a.imshow(gatt[l, h].numpy(), cmap="Oranges", vmin=0)
    a.set_xticks(range(len(gt)))
    a.set_xticklabels(gt, rotation=80, fontsize=8.5)
    a.set_yticks(range(len(gt)))
    a.set_yticklabels(gt, fontsize=8.5)
    a.set_title(f"GPT-2 layer {l + 1}, head {h + 1}", loc="left", fontsize=12.5)
    a.grid(False)
save(fig, OUT / "gpt2_attention.png")

# ------------------------------------------------------------ 2. char-level GPT
path = DATA / "tinyshakespeare.txt"
if not path.exists():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = path.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train, val = data[:n], data[n:]
V, CTX, D, H, L = len(chars), 128, 192, 6, 4


def batch(split, bs=64):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - CTX - 1, (bs,))
    x = torch.stack([d[i:i + CTX] for i in ix])
    y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
    return x.to(DEV), y.to(DEV)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, H, batch_first=True, dropout=0.1)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(0.1))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, idx):
        T = idx.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8):
        for _ in range(n):
            logits = self(idx[:, -CTX:])[:, -1] / temperature
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


model = CharGPT().to(DEV)
metrics["chargpt_params"] = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
STEPS = 3000 if DEV == "cuda" else 1200
samples, curve = {}, []
t0 = time.time()
prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]], device=DEV)
for step in range(STEPS + 1):
    if step in (0, 300, STEPS):
        model.eval()
        torch.manual_seed(1)
        samples[step] = "".join(itos[i] for i in model.generate(prompt, 180)[0].tolist())
        model.train()
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            vl = np.mean([F.cross_entropy(model(x).view(-1, V), y.view(-1)).item() for x, y in (batch("val") for _ in range(10))])
        curve.append((step, vl))
        model.train()
    if step == STEPS:
        break
    x, y = batch("train")
    loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
metrics["chargpt"] = {"steps": STEPS, "device": DEV, "minutes": round((time.time() - t0) / 60, 1),
                      "val_loss_start": curve[0][1], "val_loss_end": curve[-1][1],
                      "val_ppl_end": math.exp(curve[-1][1]), "vocab": V}
metrics["chargpt_samples"] = samples
c = np.array(curve)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(c[:, 0], c[:, 1], color=PRIMARY, lw=2.5)
ax.axhline(math.log(V), ls="--", color=MUTED)
ax.text(c[-1, 0], math.log(V) + 0.05, f"uniform guess: ln {V} = {math.log(V):.2f}", ha="right", color=MUTED, fontsize=11)
ax.set_xlabel("training step")
ax.set_ylabel("validation loss (nats / character)")
ax.set_title(f"Character-level transformer on Tiny Shakespeare (final perplexity {math.exp(curve[-1][1]):.1f})", loc="left", fontsize=13)
save(fig, OUT / "chargpt_loss.png")

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
