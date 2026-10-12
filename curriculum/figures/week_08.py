"""Week 8 figures: fine-tuning diagrams, plus stored results from a full CPU run of the earlier, larger
version of the lab (curriculum/assets/week_08/results.json). The CPU core exports results_cpu_core.json."""
import json
import textwrap
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

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_08"
ORANGE = "#FFF1E6"


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.25), 0.55, 0.55, fc="#312E81", ec="#C7D2FE", lw=2))
    ax.text(0.325, 0.525, "W\nfrozen", ha="center", va="center", color="#C7D2FE", fontsize=20, fontweight="bold")
    ax.text(0.68, 0.52, "+", fontsize=34, color="white", va="center")
    ax.add_patch(plt.Rectangle((0.78, 0.25), 0.08, 0.55, fc=ACCENT, ec="white"))
    ax.add_patch(plt.Rectangle((0.9, 0.72), 0.55 * 0.18, 0.08, fc=TEAL, ec="white"))
    ax.text(0.82, 0.18, "B", ha="center", color="white", fontsize=18, fontweight="bold")
    ax.text(0.95, 0.85, "A", ha="center", color="white", fontsize=18, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    save(fig, path, transparent=True)


def decision(path):
    fig, ax = canvas(5.0)
    blocks=[(.1,2.6,2.3,.95,"Prompt +\ntest set","data"),
            (3.1,2.6,2.3,.95,"Meets\nthe bar?","loss"),
            (5.9,4.05,3.2,.85,"Ship +\nmonitor","output"),
            (5.9,2.6,3.2,.95,"Missing\nfacts?","loss"),
            (9.8,2.6,3.0,.95,"Retrieval\n(week 12)","tool"),
            (5.9,.95,3.2,.95,"Behaviour or\ncost gap?","loss"),
            (9.8,.95,3.0,.95,"LoRA\nfine-tune","model"),
            (3.1,.95,2.3,.95,"Review task\nand data","tool")]
    for x,y,w,h,label,role in blocks:role_box(ax,x,y,w,h,label,role,size=17)
    routed_arrow(ax,[(2.4,3.075),(3.1,3.075)],label="score",label_xy=(2.75,3.35),size=16)
    routed_arrow(ax,[(5.4,3.4),(5.6,3.4),(5.6,4.475),(5.9,4.475)],label="yes",label_xy=(5.3,3.95),size=16)
    routed_arrow(ax,[(5.4,3.075),(5.9,3.075)],label="no",label_xy=(5.65,2.8),size=16)
    routed_arrow(ax,[(9.1,3.075),(9.8,3.075)],label="yes",label_xy=(9.45,3.35),size=16)
    routed_arrow(ax,[(7.5,2.6),(7.5,1.9)],label="no",label_xy=(7.85,2.25),size=16)
    routed_arrow(ax,[(9.1,1.425),(9.8,1.425)],label="yes",label_xy=(9.45,1.7),size=16)
    routed_arrow(ax,[(5.9,1.425),(5.4,1.425)],label="no",label_xy=(5.65,1.7),size=16)
    # A role key would call the question boxes "loss / update"; say what the colours mean here instead.
    ax.text(.1,.3,"Orange boxes are questions; follow yes or no.",fontsize=16,color=TEXT)
    save(fig,path)


def rlhf(path):
    fig, ax = canvas(4.6)
    blue = "#0072B2"
    steps=[(.1,"1. SFT","Demonstrations","model"),(3.5,"2. Compare","Ranked pairs","data"),
           (6.9,"3. Reward\nmodel","Preference score","loss"),(10.3,"4. PPO\nupdate","Reward − KL","model")]
    for x,title,sub,role in steps:
        role_box(ax,x,2.55,2.6,1.05,title,role,size=20)
        ax.text(x+1.3,2.1,sub,ha="center",fontsize=16,color=TEXT)
    for x,label in zip((2.7,6.1,9.5),("replies","pairs","scores")):
        routed_arrow(ax,[(x,3.075),(x+.8,3.075)])
        ax.text(x+.4,3.8,label,ha="center",va="bottom",fontsize=16,color=blue)
    role_legend(ax,.3,size=16)
    save(fig,path)


def lora(path):
    fig, ax = canvas(4.8)
    blue = "#0072B2"
    role_box(ax,.2,2.15,1.4,.9,"Input x","data",size=21)
    role_box(ax,3.2,3.2,2.3,1.0,"W\nfrozen","model",size=20)
    role_box(ax,3.2,1.0,1.8,1.0,"A\ntrainable","loss",size=19)
    role_box(ax,6.3,1.0,2.3,1.0,"B · α/r\ntrainable","loss",size=18)
    role_box(ax,9.4,2.15,1.4,.9,"Add","model",size=21)
    role_box(ax,11.5,2.15,1.2,.9,"h","output",size=24)
    routed_arrow(ax,[(1.6,2.6),(2.3,2.6),(2.3,3.7),(3.2,3.7)])
    routed_arrow(ax,[(1.6,2.6),(2.3,2.6),(2.3,1.5),(3.2,1.5)])
    ax.text(2.75,3.8,"x [k]",ha="center",va="bottom",fontsize=16,color=blue)
    routed_arrow(ax,[(5.5,3.7),(10.1,3.7),(10.1,3.05)])
    ax.text(7.8,3.8,"W x [d]",ha="center",va="bottom",fontsize=16,color=blue)
    routed_arrow(ax,[(5.0,1.5),(6.3,1.5)])
    ax.text(5.65,1.6,"A x [r]",ha="center",va="bottom",fontsize=16,color=blue)
    routed_arrow(ax,[(8.6,1.5),(10.1,1.5),(10.1,2.15)])
    ax.text(9.3,1.6,"B(Ax) [d]",ha="center",va="bottom",fontsize=16,color=blue)
    routed_arrow(ax,[(10.8,2.6),(11.5,2.6)])
    ax.text(11.15,2.7,"[d]",ha="center",va="bottom",fontsize=16,color=blue)
    role_legend(ax,4.45,size=16)
    save(fig,path)


def param_compare(path):
    r=results()  # fail clearly if the measured evidence is unavailable
    trainable=int(r["trainable"]); base=int(r["total"])-trainable
    if base<=0 or trainable<=0:raise ValueError("Recorded parameter counts must be positive")
    labels=["Original base weights\n(full fine-tuning)","Recorded LoRA weights\n(r=16; q,k,v,o)"]
    values=[base,trainable]
    fig,ax=plt.subplots(figsize=(12,4.3))
    ax.barh(labels[::-1],values[::-1],color=[ACCENT,PRIMARY]); ax.set_xscale("log")
    for i,value in enumerate(values[::-1]):
        ax.text(value*1.12,i,f"{value:,} ({100*value/base:.2f}% of base)",va="center",fontsize=18)
    ax.set_xlabel("parameters that would be trained (log scale)",fontsize=18)
    ax.set_title("Qwen2.5-0.5B: exact counts from the stored full run",loc="left",fontsize=19)
    ax.set_xlim(1e6,base*40); ax.tick_params(labelsize=18); ax.grid(axis="y",visible=False)
    save(fig,path)


def results_bar(path):
    r = results()["comparison"]
    names = ["Base\nzero-shot", "Base\nfew-shot", "LoRA"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].bar(names, [x["accuracy"] for x in r], color=[MUTED, PRIMARY, ACCENT])
    ax[1].bar(names, [x["invalid"] for x in r], color=[MUTED, PRIMARY, ACCENT])
    top_invalid = max(0.1, max(x["invalid"] for x in r) * 1.4)  # its own scale, so small rates stay visible
    for a, key, t, top in ((ax[0], "accuracy", "Accuracy on held-out test messages", 1.1),
                           (ax[1], "invalid", "Unparsed intent replies", top_invalid)):
        for i, x in enumerate(r):
            a.text(i, x[key] + 0.02 * top, f"{x[key]:.1%}" if key == "accuracy" else f"{x[key]:.1%}", ha="center", fontsize=19)
        a.set_ylim(0, top)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        a.set_title(t, loc="left", fontsize=19)
        a.grid(axis="x", visible=False)
        a.tick_params(labelsize=19)
    save(fig, path)


def loss_curve(path):
    r = results()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(r["loss"]["step"], r["loss"]["loss"], color=PRIMARY, lw=2, label="training loss")
    if r.get("eval"):
        ax.plot(r["eval"]["step"], r["eval"]["eval_loss"], color=ACCENT, lw=2, marker="o", label="validation loss")
    ax.set_xlabel("step", fontsize=17)
    ax.set_ylabel("loss on completion tokens", fontsize=17)
    where = "a laptop CPU" if r["device"] == "cpu" else "a GPU"
    ax.set_title(f"LoRA SFT on {r['n_train']:,} examples ({r['train_minutes']:.0f} min on {where})", loc="left",
                 fontsize=17)
    ax.legend(fontsize=17)
    ax.tick_params(labelsize=17)
    save(fig, path)


FIGURES = {"cover": cover, "decision": decision, "rlhf": rlhf, "lora": lora, "param_compare": param_compare,
           "results_bar": results_bar, "loss_curve": loss_curve}
