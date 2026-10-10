"""Week 5: original teaching schematics and unchanged measured evidence.

Numerical worked examples are labelled as examples. Empirical summaries read
the existing metrics.json; original measured PNGs are copied byte for byte.
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (INK, MUTED, TEXT, ROLE_COLORS, ROLE_FILLS, arrow, box,
                    role_box, role_legend, routed_arrow, save as save_raster, save_editable_scene)

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_05"
BLUE, GREEN, ORANGE, PINK = [ROLE_COLORS[r] for r in ("data", "model", "loss", "output")]


def save(fig,path,transparent=False):
    if not transparent and all(not ax.axison and not ax.images for ax in fig.axes):
        save_editable_scene(fig,path)
    else:
        save_raster(fig,path,transparent=transparent)


def canvas(height=4.8, width=13):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.02, right=.98, bottom=.03, top=.97)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def flow(ax, items, y=1.8, w=2.05, gap=.52, labels=None, size=18):
    """One left-to-right row: only real operations; edge-to-edge arrows."""
    for i, (label, role) in enumerate(items):
        x = .14 + i * (w + gap)
        role_box(ax, x, y, w, 1.25, label, role, size=size)
        if i:
            arrow(ax, x - gap, y + .625, x, y + .625, color=GREEN)
            if labels:
                ax.text(x - gap / 2, y + 1.52, labels[i - 1], ha="center", va="center", fontsize=17, color=BLUE)


def cover(path):
    n = 12
    # An illustrative causal pattern, not a result from a trained model.
    a = np.tril(np.ones((n, n)))
    a = a / a.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(a, cmap="Blues", vmin=0, vmax=1)
    ax.axis("off")
    save(fig, path, transparent=True)


def rnn_vs_attention(path):
    fig, ax = canvas(5.6)
    ax.text(.1, 5.2, "Two ways to pass information", fontsize=22, color=INK, weight="bold")
    ax.text(.1, 4.87, "Recurrence: process tokens one after another", fontsize=19, color=INK)
    for i, text in enumerate(["animal", "crossed", "street", "because", "it"]):
        x = .15 + i * 2.57
        role_box(ax, x, 3.64, 2.03, .85, text + "\nupdate state", "model", size=18)
        if i:
            arrow(ax, x - .54, 4.065, x, 4.065, color=GREEN)
    ax.text(.1, 3.18, "Self-attention: one token can directly use an allowed earlier token", fontsize=19, color=INK)
    flow(ax, [("it + earlier\ntoken vectors", "data"), ("Compare\nquery and keys", "model"),
              ("Make row\nweights", "model"), ("Mix value\nvectors", "model"), ("new it vector", "output")], y=1.45,
         labels=["vectors", "scores", "weights", "mixture"], size=18)
    routed_arrow(ax,[(1.165,1.45),(1.165,1.12),(8.875,1.12),(8.875,1.45)],color=BLUE,
                 label="values from token vectors",label_xy=(5.0,1.12),size=17)
    ax.text(.15, .67, "Illustrative route only. A causal decoder cannot use words after it.", fontsize=17, color=TEXT)
    role_legend(ax,.15,size=18)
    save(fig, path)


def qkv(path):
    fig, ax = canvas(4.6)
    flow(ax, [("Input X\n3 × 2", "data"), ("Project into\nQ, K and V", "model"),
              ("Scores:\nQKᵀ ÷ √2", "model"), ("Mask then\nrow softmax", "model"), ("Weights × V\n3 × 2", "output")],
         labels=["X", "Q and K", "3 × 3 scores", "3 × 3 weights"], size=18)
    # V bypasses the matching calculation and enters the final mixture.
    routed_arrow(ax, [(3.735, 1.8), (3.735, 1.1), (11.445, 1.1), (11.445, 1.8)],
                 color=BLUE, label="V: the information to pass onward (3 × 2)", label_xy=(7.5, .79), size=18)
    ax.text(.1, 4.05, "Query matches keys; the resulting weights mix values.", fontsize=21, weight="bold", color=INK)
    role_legend(ax, .2, size=18)
    save(fig, path)


def attn_worked(path):
    toks = ["cat", "sat", "mat"]
    q = np.array([[1., 0.], [0., 1.], [1., 1.]])
    scores = q @ q.T / np.sqrt(2)
    weights = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)
    values = np.array([[1., 0.], [0., 1.], [.5, .5]])
    output = weights @ values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.subplots_adjust(left=.06, right=.97, bottom=.30, top=.84, wspace=.45)
    for ax, matrix, title in [(axes[0], scores, "1. Compare Q and K (scores)"), (axes[1], weights, "2. Softmax each row (weights)")]:
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.5)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=20,
                        color="white" if matrix[i,j] > .9 else INK, weight="bold")
        ax.set_xticks(range(3), toks, fontsize=18)
        ax.set_yticks(range(3), toks, fontsize=18)
        ax.set_title(title, fontsize=20, loc="left")
        ax.set_xlabel("key position", fontsize=18)
        ax.set_ylabel("query position", fontsize=18)
        ax.grid(False)
    fig.text(.08, .06, f"Illustrative unmasked example: V = [[1,0], [0,1], [0.5,0.5]]; cat output ≈ [{output[0,0]:.2f}, {output[0,1]:.2f}].", fontsize=18, color=TEXT)
    save(fig, path)


def causal_mask(path):
    toks = ["cat", "sat", "mat"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.subplots_adjust(left=.06, right=.96, bottom=.30, top=.83, wspace=.4)
    for ax, allowed, title in [(axes[0], np.ones((3,3)), "Encoder: all input positions"), (axes[1], np.tril(np.ones((3,3))), "Causal decoder: present and past")]:
        ax.imshow(allowed, cmap="Blues", vmin=0, vmax=1.6)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "use" if allowed[i,j] else "block", ha="center", va="center", fontsize=20, color=INK)
        ax.set_xticks(range(3), toks, fontsize=18)
        ax.set_yticks(range(3), toks, fontsize=18)
        ax.set_xlabel("key: input that may be used", fontsize=18)
        ax.set_ylabel("query: current position", fontsize=18)
        ax.set_title(title, fontsize=20, loc="left")
        ax.grid(False)
    fig.text(.08, .06, "Blue = allowed. White = blocked: set its score to −∞ before softmax, so its weight becomes 0.", fontsize=18, color=TEXT)
    save(fig, path)


def multihead(path):
    fig, ax = canvas(4.6)
    flow(ax, [("Input X\nT × d", "data"), ("Split h heads\neach d/h wide", "model"),
              ("Run attention\nin each head", "model"), ("Join outputs\nT × d", "model"), ("Project with\noutput matrix", "output")],
         labels=["vectors", "h Q/K/V sets", "h outputs", "joined features"], size=18)
    ax.text(.1, 4.05, "Several learned views, then one joined representation", fontsize=21, color=INK, weight="bold")
    ax.text(.2, 1.0, "Example: d = 12 and h = 3 gives 4 features per head. The joined output has 12 again.", fontsize=18, color=TEXT)
    role_legend(ax, .25, size=18)
    save(fig, path)


def positional(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.subplots_adjust(left=.02, right=.98, bottom=.16, top=.8, wspace=.18)
    for ax in axes:
        ax.axis("off"); ax.set_xlim(0,6); ax.set_ylim(0,3)
    a, b = axes
    a.set_title("Add a position vector", fontsize=21, loc="left")
    role_box(a,.1,1.5,2.5,.9,"token [1, 0]", "data", size=20)
    role_box(a,3.2,1.5,2.6,.9,"position [0, 1]", "data", size=20)
    a.text(2.9,1.95,"+",fontsize=25,ha="center",color=GREEN)
    a.text(.15,.55,"new input = [1, 1]\nIllustrative vectors; lab positions are learned.",fontsize=18,color=TEXT)
    b.set_title("RoPE rotates Q and K",fontsize=21,loc="left")
    role_box(b,.1,1.5,2.55,.9,"Q and K\nat position p","model",size=20)
    role_box(b,3.35,1.5,2.5,.9,"rotate by\nposition angle","model",size=20)
    arrow(b,2.65,1.95,3.35,1.95,color=GREEN)
    b.text(.15,.55,"Relative position affects Q–K comparisons.\nRoPE is a different method from adding vectors.",fontsize=18,color=TEXT)
    fig.text(.02,.1,"Without positions, self-attention cannot tell cat sat from sat cat by order alone.",fontsize=18,color=TEXT)
    legend=fig.add_axes([.02,.015,.96,.07]);legend.set_xlim(0,13);legend.set_ylim(0,.4);legend.axis("off")
    role_legend(legend,.03,size=18)
    save(fig,path)


def block(path):
    fig, ax = canvas(4.8)
    role_box(ax,.12,1.95,2.05,1.35,"token +\nposition\nembeddings","data",size=18)
    role_box(ax,2.8,1.95,2.7,1.35,"LayerNorm →\nMulti-head\nself-attention","model",size=18)
    role_box(ax,6.45,1.95,2.85,1.35,"LayerNorm →\nFeed-forward MLP\n(4× wider)","model",size=18)
    role_box(ax,10.25,1.95,2.5,1.35,"→ next block\n(× N)","output",size=18)
    centres=[(5.98,2.625),(9.78,2.625)]
    for x,y in centres:
        ax.add_patch(plt.Circle((x,y),.20,fc="white",ec=GREEN,lw=1.8,zorder=4))
        ax.text(x,y,"+",ha="center",va="center",fontsize=21,color=GREEN,zorder=5)
    for xa,xb in [(2.17,2.8),(5.5,5.78),(6.18,6.45),(9.3,9.58),(9.98,10.25)]:
        arrow(ax,xa,2.625,xb,2.625,color=GREEN)
    # The arrows are too short for labels; the caption says once that each carries a T × d table.
    routed_arrow(ax,[(1.145,3.3),(1.145,3.8),(5.98,3.8),(5.98,2.825)],color=BLUE,
                 label="copy of the input (residual)",label_xy=(3.6,3.8),size=18)
    routed_arrow(ax,[(5.98,2.425),(5.98,1.4),(9.78,1.4),(9.78,2.425)],color=BLUE,
                 label="copy of the input (residual)",label_xy=(7.85,1.4),size=18)
    ax.text(.1,4.33,"Pre-norm block: learn a change, then add the input back",fontsize=21,weight="bold",color=INK)
    ax.text(.2,.75,"Every arrow carries a T × d table. LayerNorm rescales features; the MLP works per position.",fontsize=18,color=TEXT)
    role_legend(ax,.22,size=18)
    save(fig,path)


def architecture_comparison(path):
    fig, ax = canvas(4.8)
    columns=[("Encoder", "All input positions", "label / vector", "BERT"),
             ("Decoder", "Present + past", "next token", "GPT-2 / CharGPT"),
             ("Encoder–decoder", "Encoder input +\ndecoder past", "output sequence", "T5 / translation")]
    for i,(name,allowed,out,example) in enumerate(columns):
        x=.15+i*4.3
        role_box(ax,x,3.43,4.05,.65,name,"model",size=22)
        ax.text(x+.18,2.92,"Can use: " + allowed,fontsize=19,color=TEXT,va="top")
        ax.text(x+.18,1.65,"Produces: " + out,fontsize=19,color=TEXT)
        ax.text(x+.18,.95,"Example: " + example,fontsize=19,color=TEXT)
        ax.text(x+.18,.6,["Useful for search / classification","Useful for text continuation","Useful for input → output tasks"][i],fontsize=17,color=TEXT)
    ax.text(.1,4.5,"Choose the attention rule for the task",fontsize=22,weight="bold",color=INK)
    role_legend(ax,.1,size=18)
    save(fig,path)


def three_archs(path):
    """How information flows in the three families (the next slide compares their uses)."""
    fig,ax=canvas(5.7)
    rows=[("Encoder-only\n(BERT)",[("Whole input\nsentence","data"),("Encoder: each\ntoken sees all","model"),
                                   ("One vector\nper token","output"),("Label or\nsearch vector","output")]),
          ("Decoder-only\n(GPT-2)",[("Prefix\nso far","data"),("Decoder: sees\npresent + past","model"),
                                    ("Next-token\nprobabilities","output"),("Sample and\nappend token","model")]),
          ("Encoder–\ndecoder (T5)",[("Source\ntext","data"),("Encoder: sees\nwhole source","model"),
                                     ("Decoder: past +\ncross-attention","model"),("Output\nsequence","output")])]
    w,gap,h=2.3,.42,1.0
    for r,(name,items) in enumerate(rows):
        y=4.45-r*1.75
        ax.text(.1,y+h/2,name,fontsize=18,color=INK,weight="bold",va="center")
        for i,(label,role) in enumerate(items):
            x=2.35+i*(w+gap)
            role_box(ax,x,y,w,h,label,role,size=18)
            if i:
                arrow(ax,x-gap,y+h/2,x,y+h/2,color=GREEN)
        if r==1:  # generation repeats with the appended token
            routed_arrow(ax,[(2.35+3*(w+gap)+w/2,y),(2.35+3*(w+gap)+w/2,y-.32),(2.35+w/2,y-.32),(2.35+w/2,y)],
                         color=ORANGE,dashed=True,label="repeat with the longer prefix",label_xy=(6.43,y-.32),size=17)
    role_legend(ax,.08,size=18)
    save(fig,path)


def kv_cache(path):
    fig,ax=canvas(5.2)
    items=[("Known prefix\nThe cat sat", "data"), ("Read sat\n(known token)", "data"),
           ("Compute Q/K/V\nfor sat", "model"), ("Attend to K/V\nfor The cat sat", "model"), ("Next-token\nprobabilities", "output")]
    flow(ax,items,y=2.15,labels=["last token", "sat vector", "new Q/K/V", "last output"],size=18)
    routed_arrow(ax,[(1.165,2.15),(1.165,1.5),(8.875,1.5),(8.875,2.15)],color=BLUE,
                 label="cached K/V for The and cat (earlier passes)",label_xy=(6.7,1.18),size=18)
    ax.text(.1,4.6,"The query comes from the last known token, never an unknown future token.",fontsize=20,color=INK,weight="bold")
    ax.text(.15,.68,"Append sat's new K/V. CharGPT recomputes its context; a cache reuses earlier K/V.",fontsize=18,color=TEXT)
    role_legend(ax,.15,size=18)
    save(fig,path)


def moe(path):
    fig,ax=canvas(4.8)
    role_box(ax,.15,1.95,1.7,1.15,"token vector","data",size=18)
    role_box(ax,2.45,1.95,1.7,1.15,"router:\nchoose 2","model",size=18)
    role_box(ax,5.15,2.75,2.45,.85,"expert MLP A","model",size=18)
    role_box(ax,5.15,1.35,2.45,.85,"expert MLP B","model",size=18)
    role_box(ax,8.55,1.95,2.0,1.15,"weighted sum","model",size=18)
    role_box(ax,11.1,1.95,1.7,1.15,"new vector","output",size=18)
    arrow(ax,1.85,2.525,2.45,2.525,color=GREEN)
    routed_arrow(ax,[(4.15,2.85),(4.65,2.85),(4.65,3.175),(5.15,3.175)],color=GREEN)
    routed_arrow(ax,[(4.15,2.2),(4.65,2.2),(4.65,1.775),(5.15,1.775)],color=GREEN)
    routed_arrow(ax,[(7.6,3.175),(8.03,3.175),(8.03,2.85),(8.55,2.85)],color=GREEN)
    routed_arrow(ax,[(7.6,1.775),(8.03,1.775),(8.03,2.2),(8.55,2.2)],color=GREEN)
    arrow(ax,10.55,2.525,11.1,2.525,color=GREEN)
    for x,y,t in [(2.15,3.25,"vector"),(4.66,3.57,"vector"),(4.66,1.07,"vector"),
                  (8.03,3.57,"result"),(8.03,1.07,"result"),(10.83,3.25,"vector")]:
        ax.text(x,y,t,fontsize=17,ha="center",color=BLUE)
    routed_arrow(ax,[(3.3,3.1),(3.3,4.0),(9.55,4.0),(9.55,3.1)],color=BLUE,
                 label="router mixture weights",label_xy=(6.3,4.0),size=17)
    ax.text(.1,4.45,"Mixture of experts: only selected networks run for each token",fontsize=21,color=INK,weight="bold")
    ax.text(.2,.7,"Unselected experts do no work for this token. The sum uses the router's weights.",fontsize=18,color=TEXT)
    role_legend(ax,.22,size=18)
    save(fig,path)


def vit_patches(path):
    fig,ax=canvas(4.8)
    flow(ax,[("RGB photo\n32 × 32","data"),("4 patches\n16 × 16","model"),
             ("Flatten\n768 values","model"),("Project\n+ position","model"),("Encoder\nmix tokens","model"),
             ("Mean pool\nthen head","model"),("Scores:\nflower\nother","output")],
         w=1.42,gap=.38,labels=["pixels","patches","vectors","tokens","features","logits"],size=17)
    ax.text(.1,4.12,"A vision transformer treats image patches as tokens",fontsize=21,color=INK,weight="bold")
    ax.text(.2,.96,"Illustrative RGB example: 16 × 16 × 3 = 768 numbers per patch before projection.",fontsize=18,color=TEXT)
    role_legend(ax,.22,size=18)
    save(fig,path)


def complexity(path):
    n=np.array([512,2048,8192,32768,131072])
    gb=n.astype(float)**2*2/1e9
    fig,ax=plt.subplots(figsize=(13,4.6))
    fig.subplots_adjust(left=.11,right=.86,bottom=.31,top=.86)
    ax.loglog(n,gb,marker="o",color=ORANGE,lw=2.5)
    for i,(x,y) in enumerate(zip(n,gb)):
        label_y = y*.25 if i == 0 else y*.5
        ax.text(x*1.15,label_y,f"{y*1e3:.3g} MB" if y<1 else f"{y:.3g} GB",ha="left",va="top",fontsize=18)
    ax.set_xlim(300,8e5);ax.set_ylim(1e-5,200)
    ax.set_xlabel("sequence length n (tokens)",fontsize=18)
    ax.set_ylabel("one score matrix (GB)",fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_title("Calculated example: 4× more tokens make the n × n table 16× larger",loc="left",fontsize=20)
    fig.text(.11,.03,"One head, 2 bytes per score (fp16); excludes other tensors. Efficient kernels need not store the full table.",fontsize=17,color=TEXT)
    save(fig,path)


def bert_summary(path):
    """The head the lab's TODO 5 rule selects: largest it -> animal weight over all 144 heads."""
    result=json.loads((ASSETS/"bert_attention.json").read_text(encoding="utf-8"))
    weight=result["it_to_animal"]
    fig,ax=plt.subplots(figsize=(13,4.6))
    fig.subplots_adjust(left=.25,right=.95,bottom=.26,top=.78)
    ax.barh([1,0],[weight,1-weight],color=[BLUE,GREEN],height=.52)
    ax.set_yticks([1,0],["animal", "all other positions\n(aggregate)"],fontsize=20)
    ax.set_xlim(0,1); ax.tick_params(axis="x",labelsize=18)
    ax.set_xlabel("attention weight from the query it",fontsize=20)
    for y,v in [(1,weight),(0,1-weight)]:
        ax.text(v+.025,y,f"{v:.3f}",va="center",fontsize=21,color=INK)
    ax.set_title(f"Measured BERT head with the largest it → animal weight: layer {result['layer']}, head {result['head']}",
                 fontsize=21,loc="left")
    fig.text(.06,.06,"Chosen from all 144 heads, as in lab TODO 5. Other positions are summed. A weight is not understanding.",
             fontsize=18,color=TEXT)
    save(fig,path)


def bert_matrix(path):
    """Full table of the selected BERT head at a size the Word notes can print legibly."""
    result=json.loads((ASSETS/"bert_attention.json").read_text(encoding="utf-8"))
    toks=result["tokens"]; it=toks.index("it")
    fig,ax=plt.subplots(figsize=(9,8))
    fig.subplots_adjust(left=.2,right=.9,bottom=.2,top=.92)
    image=ax.imshow(result["weights"],cmap="Purples",vmin=0,vmax=1)
    ax.set_xticks(range(len(toks)),toks,rotation=60,ha="right",rotation_mode="anchor",fontsize=17)
    ax.set_yticks(range(len(toks)),toks,fontsize=17)
    ax.add_patch(plt.Rectangle((-.5,it-.5),len(toks),1,fill=False,ec=ORANGE,lw=2.5))
    ax.set_xlabel("key: token read",fontsize=18); ax.set_ylabel("query: current token",fontsize=18)
    ax.set_title(f"BERT layer {result['layer']}, head {result['head']}: full table (row it outlined)",loc="left",fontsize=18)
    ax.grid(False)
    bar=fig.colorbar(image,ax=ax,fraction=.04,pad=.02)
    bar.set_label("measured weight",fontsize=17); bar.ax.tick_params(labelsize=16)
    save(fig,path)


def gpt2_causal_summary(path):
    """Readable view of two measured GPT-2 heads (numbers from make_week_05_gpt2.py)."""
    data=json.loads((ASSETS/"gpt2_attention.json").read_text(encoding="utf-8"))
    toks=[t.lstrip("·") for t in data["tokens"]]
    fig,axes=plt.subplots(1,2,figsize=(13,6.4))
    fig.subplots_adjust(left=.1,right=.88,bottom=.25,top=.9,wspace=.32)
    for ax,head,note in zip(axes,data["heads"][:2],["mixes several earlier tokens","looks at the previous token"]):
        image=ax.imshow(head["weights"],cmap="Oranges",vmin=0,vmax=1)
        ax.set_xticks(range(len(toks)),toks,rotation=60,ha="right",rotation_mode="anchor",fontsize=17)
        ax.set_yticks(range(len(toks)),toks,fontsize=17)
        ax.set_title(f"Layer {head['layer']}, head {head['head']}: {note}",loc="left",fontsize=18)
        ax.set_xlabel("key: token read",fontsize=17)
        ax.grid(False)
    axes[0].set_ylabel("query: current token",fontsize=17)
    bar=fig.colorbar(image,ax=axes,fraction=.025,pad=.02)
    bar.set_label("measured weight",fontsize=17); bar.ax.tick_params(labelsize=17)
    fig.text(.1,.025,f"Every square above the diagonal (a later token) has weight {data['max_future_weight']:.1f} "
             f"in all {data['layers']*data['heads_per_layer']} heads.",fontsize=18,color=TEXT)
    save(fig,path)


FIGURES={"cover":cover,"rnn_vs_attention":rnn_vs_attention,"qkv":qkv,"attn_worked":attn_worked,
         "causal_mask":causal_mask,"multihead":multihead,"positional":positional,"block":block,
         "three_archs":three_archs,"architecture_comparison":architecture_comparison,"kv_cache":kv_cache,
         "moe":moe,"vit_patches":vit_patches,"complexity":complexity,
         "bert_summary":bert_summary,"bert_matrix":bert_matrix,"gpt2_causal_summary":gpt2_causal_summary,
         **{name:asset(name) for name in ["chargpt_loss"]}}
