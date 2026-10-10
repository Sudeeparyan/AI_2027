"""Week 7 figures: prompt anatomy, context engineering, plus REAL results harvested from the executed lab notebook
(curriculum/assets/week_07/results.json, written by the solution notebook when GENAI_RESULTS_PATH is set)."""
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
    fig, ax = canvas(5.8)
    parts=[("Application rules","Task boundaries and uncertainty behaviour","tool"),
           ("Source context","Facts and documents; label untrusted content","data"),
           ("Worked examples","Varied input-answer pairs in the desired format","data"),
           ("Task + audience","A clear job, constraints and intended reader","tool"),
           ("Output contract","Shape, fields, length and application checks","output")]
    for i,(title,detail,role) in enumerate(parts):
        y=4.55-i*.82
        role_box(ax,.2,y,3.4,.65,title,role,size=18)
        ax.text(4,y+.325,detail,fontsize=17,color=TEXT,va="center")
    ax.text(.2,5.5,"A prompt is a task brief; order is something to test",fontsize=19,fontweight="bold",color=INK)
    role_legend(ax,.15,size=16)
    save(fig,path)


def context_eng(path):
    fig, ax = canvas(5.5)
    sources=[("App instructions","tool"),("User question","data"),("Source documents","data"),("Selected history","data"),("Tool results","tool")]
    for i,(name,role) in enumerate(sources):
        y=4.35-i*.7
        role_box(ax,.2,y,3,.5,name,role,size=17)
        # Separate source lanes join only at the collection bus.
        routed_arrow(ax,[(3.2,y+.25),(3.55,y+.25),(3.55,2.8),(4,2.8)])
    role_box(ax,4,2.1,2.3,1.4,"Select\ncompress\norder","tool",size=20)
    role_box(ax,7.15,1.2,2.8,3.3,"","data")
    ax.text(8.55,4.8,"Token budget",fontsize=18,ha="center",color=INK,fontweight="bold")
    for i,(name,role) in enumerate([("Rules","tool"),("Useful sources","data"),("Recent history","data"),("Question","data"),("Output reserve","output")]):
        role_box(ax,7.3,3.8-i*.58,2.5,.45,name,role,size=16)
    routed_arrow(ax,[(6.3,2.8),(7.15,2.8)])
    role_box(ax,10.6,2.1,2.2,1.4,"Fixed\nmodel","model",size=21)
    routed_arrow(ax,[(9.95,2.8),(10.6,2.8)])
    ax.text(.2,.75,"Select relevant evidence; extra context can distract. Validate the final reply.",fontsize=16,color=TEXT)
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
    _bar(ax, names, acc, f"Accuracy: {r[0]['n']} messages; Wilson 95% intervals", PRIMARY, ylim=(0, 1.15), err=err)
    save(fig, path)


def cot_results(path):
    r = results()["reasoning"]
    names = [x["prompt"] for x in r]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    _bar(ax[0], names, [x["accuracy"] for x in r], f"Accuracy on {r[0]['n']} word problems", TEAL)
    toks = [x["avg_tokens"] for x in r]
    _bar(ax[1], names, toks, "Output tokens per problem (cost)", ACCENT, fmt="{:.0f}", ylim=(0, max(toks) * 1.2))
    save(fig, path)


def injection_results(path):
    r = results()["injection"]
    names = sorted(r, key=lambda k: k != "naive")  # naive prompt first, then the defended one
    fig, ax = plt.subplots(figsize=(7.5, 4))
    _bar(ax, [f"{k} prompt" for k in names], [r[k] for k in names], "Replies flagged by keyword heuristic", "#B91C1C", ylim=(0, 1.15))
    ax.tick_params(axis="x", rotation=0, labelsize=18)
    save(fig, path)


FIGURES = {"cover": cover, "anatomy": anatomy, "context_eng": context_eng, "fewshot_results": fewshot_results,
           "cot_results": cot_results, "injection_results": injection_results}
