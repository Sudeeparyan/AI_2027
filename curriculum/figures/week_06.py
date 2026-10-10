"""Week 6 figures: NLP evolution, scaling laws, LLM family tree, reasoning models, plus REAL results."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save as save_raster, role_box, role_legend, routed_arrow, save_editable_scene


def save(fig, path, transparent=False):
    if not transparent and all(not ax.axison and not ax.images for ax in fig.axes):
        save_editable_scene(fig, path)
    else:
        save_raster(fig, path, transparent=transparent)


def canvas(height=5):
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=.97)  # no empty side margins on slides
    ax.axis("off"); ax.set_xlim(0,13); ax.set_ylim(0,height)
    return fig, ax

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_06"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(figsize=(5, 5))
    for k in range(9):
        y = np.cumsum(rng.normal(0, 1, 60)) + k * 6
        ax.plot(y, color=SERIES[k % len(SERIES)], lw=2, alpha=0.8)
    ax.axis("off")
    save(fig, path, transparent=True)


def nlp_timeline(path):
    fig, ax = canvas(5.4)
    events=[("1990s", "Count phrases", "n-grams"), ("2013", "Word vectors", "word2vec"),
            ("2014–15", "Sequence memory", "RNN + attention"), ("2017", "Context attention", "Transformer"),
            ("2018", "Pre-train", "BERT / GPT"), ("2020", "Prompt examples", "GPT-3"),
            ("2022", "Assistant tuning", "ChatGPT"), ("2024+", "More input types", "Reasoning; tools")]
    for i,(year,title,example) in enumerate(events):
        row=i//4; col=i%4; x=.2+col*3.25; y=3.2-row*1.9
        role_box(ax,x,y,2.85,1.15,title+"\n"+example,"model",size=17)
        ax.text(x+1.425,y+1.28,year,ha="center",fontsize=17,color=INK,fontweight="bold")
        if col<3:routed_arrow(ax,[(x+2.85,y+.575),(x+3.25,y+.575)])
    routed_arrow(ax,[(11.95,3.2),(11.95,3.05),(.1,3.05),(.1,1.875),(.2,1.875)])
    ax.text(.2,5.15,"A sequence of ideas; earlier methods still have useful jobs",fontsize=19,color=INK,fontweight="bold")
    role_legend(ax,.12,size=16)
    save(fig,path)


def scaling(path):
    # Chinchilla parametric fit (Hoffmann et al., 2022, approach 3)
    E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28
    C = np.logspace(19, 25, 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1))
    for N, c in zip([1e8, 1e9, 1e10, 7e10], SERIES):
        D = C / (6 * N)
        L = E + A / N ** a + B / D ** b
        axes[0].semilogx(C, L, color=c, lw=2.2, label=f"{N / 1e9:g} B params" if N >= 1e9 else f"{N / 1e6:g} M params")
    Nopt = np.logspace(8, 12, 200)
    Lopt = [min(E + A / n ** a + B / (c_ / (6 * n)) ** b for n in Nopt) for c_ in C]
    axes[0].semilogx(C, Lopt, color=INK, lw=3, ls="--", label="best size per budget")
    axes[0].set_ylim(1.8, 4.2)
    axes[0].set_xlabel("training compute C (FLOPs) ≈ 6 N D", fontsize=18)
    axes[0].set_ylabel("predicted loss", fontsize=18)
    axes[0].set_title("Loss falls smoothly with compute", loc="left", fontsize=20)
    # The upper right is empty once each curve has flattened, so the legend sits there.
    axes[0].legend(fontsize=18, loc="upper right", handlelength=1.4, labelspacing=0.3)
    b_ = axes[1]
    Ns = np.logspace(8, 12, 50)
    b_.loglog(Ns, 20 * Ns, color=TEAL, lw=2.5, label="≈20 tokens/parameter (fit)")
    pts = [("GPT-3 (2020)", 175e9, 300e9), ("Chinchilla (2022)", 70e9, 1.4e12), ("Llama 3 8B (2024)", 8e9, 15e12)]
    for (n_, x, y), c in zip(pts, (ACCENT, PRIMARY, "#A21CAF")):
        b_.scatter([x], [y], s=90, color=c, zorder=3)
        b_.annotate(n_,(x,y),xytext={"GPT-3 (2020)": (0, -30), "Chinchilla (2022)": (-12, 10)}.get(n_, (-12, 0)),textcoords="offset points",fontsize=18,color=c,ha={"GPT-3 (2020)": "center", "Chinchilla (2022)": "right"}.get(n_, "right"),va="center" if "Llama" in n_ else "baseline")
    b_.set_xlabel("parameters N", fontsize=18)
    b_.set_ylabel("training tokens D", fontsize=18)
    b_.set_title("How much data per parameter?", loc="left", fontsize=20)
    b_.legend(fontsize=18, loc="lower right", handlelength=1.4)
    for ax_ in axes:
        ax_.tick_params(labelsize=18)
    save(fig, path)


def tokenizers(path):
    """Measured token counts (make_week_06.py) for the same three texts, redrawn at slide size."""
    tk = json.loads((ASSETS / "metrics.json").read_text(encoding="utf-8"))["tokenizers"]
    names = list(tk["counts"])
    texts = list(tk["counts"][names[0]])
    fig, ax = plt.subplots(figsize=(13, 3.9))
    fig.subplots_adjust(left=.07, right=.7, bottom=.15, top=.87)
    width = .8 / len(names)
    for i, (name, c) in enumerate(zip(names, SERIES)):
        xs = np.arange(len(texts)) + (i - (len(names) - 1) / 2) * width
        values = [tk["counts"][name][t] for t in texts]
        ax.bar(xs, values, width * .92, color=c, label=f"{name} (vocabulary {tk['vocab'][name]:,})")
        for x, v in zip(xs, values):
            ax.text(x, v + .8, str(v), ha="center", va="bottom", fontsize=17, color=INK)
    ax.set_xticks(range(len(texts)), texts, fontsize=19)
    ax.set_ylabel("tokens", fontsize=19)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylim(0, max(max(v.values()) for v in tk["counts"].values()) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Measured tokens for the same three texts", loc="left", fontsize=20)
    ax.legend(fontsize=17, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Tokeniser",
              title_fontsize=17, alignment="left")
    save(fig, path)


def next_token(path):
    """Measured top-8 next tokens from the base model (make_week_06.py), with readable labels."""
    m = json.loads((ASSETS / "metrics.json").read_text(encoding="utf-8"))

    def label(token):
        text = token[1:-1] if token[:1] == token[-1:] == "'" else token
        if text == "\\n":
            return "(new line)"
        return "·" + text[1:] if text.startswith(" ") else text

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
    fig.subplots_adjust(left=.13, right=.98, bottom=.26, top=.88, wspace=.42)
    for ax, (prompt, top) in zip(axes, m["next_token"].items()):
        names = [label(t) for t, _ in top][::-1]
        probs = [v for _, v in top][::-1]
        ax.barh(names, probs, color=[ACCENT if n.strip("·_") == "" else PRIMARY for n in names], height=.7)
        ax.set_title(f"“{prompt} …”", loc="left", fontsize=19)
        ax.tick_params(axis="y", labelsize=17)
        ax.tick_params(axis="x", labelsize=17)
        ax.set_xlabel("probability", fontsize=17)
        ax.grid(axis="y", visible=False)
    fig.text(.02, .03, f"{m['next_token_model']}. · marks a leading space; orange bars are blank markers such as “____”.",
             fontsize=17, color=TEXT)
    save(fig, path)


def family_tree(path):
    fig, ax = canvas(5.4)
    role_box(ax,5.1,4.35,2.8,.7,"Transformer","model",size=20)
    columns=[(.2,"Encoder-only",["Read all supplied text","BERT / ModernBERT","Masks; labels; embeddings"]),
             (4.5,"Encoder–decoder",["Read source; generate output","T5 / Flan-T5: text → text","Whisper: audio → transcript"]),
             (8.8,"Decoder-only",["Continue a token prefix","Qwen / Llama / Gemma","GPT-style text generation"])]
    for x,title,items in columns:
        role_box(ax,x,3.1,4,.8,title,"model",size=20)
        routed_arrow(ax,[(6.5,4.35),(6.5,4.15),(x+2,4.15),(x+2,3.9)])
        for j,item in enumerate(items):ax.text(x+.1,2.7-j*.52,item,fontsize=16,color=TEXT)
    ax.text(.2,.8,"Architecture and access are separate. Proprietary internals may be undisclosed.",fontsize=16,color=MUTED)
    role_legend(ax,.15,size=16)
    save(fig,path)


def reasoning(path):
    fig, ax = canvas(4.8)
    ax.text(.2,4.45,"Extra inference computation is a budget to evaluate",fontsize=19,color=INK,fontweight="bold")
    role_box(ax,.2,3.0,2.3,.85,"Prompt","data",size=21)
    role_box(ax,4.0,3.0,4.2,.85,"Generate answer","model",size=21)
    role_box(ax,9.7,3.0,2.8,.85,"Check answer","output",size=21)
    routed_arrow(ax,[(2.5,3.425),(4,3.425)]); routed_arrow(ax,[(8.2,3.425),(9.7,3.425)])
    role_box(ax,.2,1.45,2.3,1.1,"Prompt","data",size=21)
    role_box(ax,4,1.45,4.2,1.1,"Extra computation\n(budgeted thinking)","model",size=20)
    role_box(ax,9.7,1.45,2.8,1.1,"Check final\nanswer","output",size=20)
    routed_arrow(ax,[(2.5,2.0),(4,2.0)]); routed_arrow(ax,[(8.2,2.0),(9.7,2.0)])
    ax.text(.2,.8,"Check correctness, completion, tokens and seconds; more thinking can still fail.",fontsize=16,color=TEXT)
    role_legend(ax,.15,size=16)
    save(fig,path)


FIGURES = {
    "cover": cover, "nlp_timeline": nlp_timeline, "scaling": scaling, "family_tree": family_tree, "reasoning": reasoning,
    "tokenizers": tokenizers, "next_token": next_token,
}
