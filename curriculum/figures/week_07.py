"""Week 7 figures: prompt anatomy, context engineering, plus stored results from an earlier GPU run of this lab
with Qwen2.5-1.5B-Instruct (curriculum/assets/week_07/results.json). The CPU core exports results_cpu_core.json."""
import json
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
    ax.axis("off"); ax.set_xlim(0,13); ax.set_ylim(0,height)
    return fig, ax

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_07"
ORANGE = "#FFF1E6"


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    lines = ["<system>", "  role · rules", "</system>", "<context>", "  documents", "</context>", "<examples> …", "task:", "format: JSON"]
    for i, t in enumerate(lines):
        ax.text(0.05, 0.92 - i * 0.1, t, fontsize=17, family="monospace", color=[PRIMARY, "#C7D2FE", PRIMARY, TEAL, "#C7D2FE", TEAL, ACCENT, "white", "white"][i])
    save(fig, path, transparent=True)


def anatomy(path):
    fig, ax = canvas(5.0)
    parts=[("Rules","Task limits; what to do when unsure","tool"),
           ("Sources","Facts; mark untrusted text","data"),
           ("Examples","Varied pairs in the target format","data"),
           ("Task","Job, audience, constraints","tool"),
           ("Format","Fields, length, checks","output")]
    for i,(title,detail,role) in enumerate(parts):
        y=4.15-i*.82
        role_box(ax,.2,y,3.0,.65,title,role,size=19)
        ax.text(3.6,y+.325,detail,fontsize=18,color=TEXT,va="center")
    role_legend(ax,.15,size=16)
    save(fig,path)


def context_eng(path):
    fig, ax = canvas(5.2)
    blue = "#0072B2"
    sources=[("Instructions","tool"),("Question","data"),("Documents","data"),("History","data"),("Tool results","tool")]
    for i,(name,role) in enumerate(sources):
        y=4.35-i*.7
        role_box(ax,.2,y,2.2,.5,name,role,size=17)
        # Separate source lanes join only at the collection bus.
        routed_arrow(ax,[(2.4,y+.25),(2.7,y+.25),(2.7,2.8),(4.4,2.8)])
    ax.text(3.55,2.95,"candidates",fontsize=16,ha="center",va="bottom",color=blue)
    role_box(ax,4.4,2.1,1.7,1.4,"Select\ncompress\norder","tool",size=19)
    routed_arrow(ax,[(6.1,2.8),(7.6,2.8)])
    ax.text(6.85,2.95,"kept items",fontsize=16,ha="center",va="bottom",color=blue)
    role_box(ax,7.6,1.2,2.4,3.3,"","data")
    ax.text(8.8,4.8,"Token budget",fontsize=18,ha="center",color=INK,fontweight="bold")
    for i,(name,role) in enumerate([("Rules","tool"),("Best sources","data"),("Recent turns","data"),("Question","data"),("Answer space","output")]):
        role_box(ax,7.72,3.8-i*.58,2.16,.45,name,role,size=16)
    routed_arrow(ax,[(10.0,2.8),(11.1,2.8)])
    ax.text(10.55,2.95,"context",fontsize=16,ha="center",va="bottom",color=blue)
    role_box(ax,11.1,2.1,1.7,1.4,"Fixed\nmodel","model",size=20)
    role_legend(ax,.15,size=16)
    save(fig,path)


def _bar(ax, names, vals, title, color, fmt="{:.1%}", ylim=(0, 1.05), err=None):
    # Horizontal, readable labels, e.g. "self-consistency\n(k=5)" and "chain-of-\nthought".
    names = [n.replace(" (", "\n(").replace("chain-of-thought", "chain-of-\nthought") for n in names]
    ax.bar(names, vals, color=color, yerr=err, capsize=6, error_kw={"ecolor": "#475569", "elinewidth": 1.2})
    top = ylim[1] if ylim else max(vals)
    for i, v in enumerate(vals):
        ax.text(i, v + (err[1,i] if err is not None else 0) + 0.02 * top, fmt.format(v), ha="center", fontsize=19)
    ax.set_title(title, loc="left", fontsize=20)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=19)
    ax.tick_params(axis="y", labelsize=19)


def _margin(p,n):
    """Asymmetric Wilson 95% distances from the observed accuracy."""
    z=1.96; denominator=1+z*z/n
    centre=(p+z*z/(2*n))/denominator
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return p-max(0,centre-half), min(1,centre+half)-p


def fewshot_results(path):
    r = results()["classification"]
    names = [x["prompt"] for x in r]
    acc = [x["accuracy"] for x in r]
    err = np.asarray([_margin(a,x["n"]) for a,x in zip(acc,r)]).T
    invalid = [x.get("invalid", 0) for x in r]
    if any(invalid):  # only worth a panel when some answers were not valid labels
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        _bar(ax[1], names, invalid, "Unparsed or ambiguous replies", ACCENT, fmt="{:.0f}", ylim=(0, max(invalid) * 1.25))
        ax = ax[0]
    else:
        fig, ax = plt.subplots(figsize=(10, 4.2))
    _bar(ax, names, acc, f"Stored GPU run: {r[0]['n']} messages; Wilson 95% intervals", PRIMARY, ylim=(0, 1.15), err=err)
    save(fig, path)


def cot_results(path):
    r = results()["reasoning"]
    names = [x["prompt"] for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    _bar(ax[0], names, [x["accuracy"] for x in r], f"Stored GPU run: accuracy on {r[0]['n']} problems", TEAL)
    toks = [x["avg_tokens"] for x in r]
    _bar(ax[1], names, toks, "Output tokens per problem (cost)", ACCENT, fmt="{:.0f}", ylim=(0, max(toks) * 1.2))
    save(fig, path)


def injection_results(path):
    r = results()["injection"]
    names = sorted(r, key=lambda k: k != "naive")  # naive prompt first, then the defended one
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _bar(ax, [f"{k} prompt" for k in names], [r[k] for k in names], "Stored GPU run: keyword-flagged replies", "#B91C1C", ylim=(0, 1.15))
    ax.tick_params(axis="x", rotation=0, labelsize=18)
    save(fig, path)


FIGURES = {"cover": cover, "anatomy": anatomy, "context_eng": context_eng, "fewshot_results": fewshot_results,
           "cot_results": cot_results, "injection_results": injection_results}
