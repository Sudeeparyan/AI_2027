"""Week 12 figures: RAG and agent diagrams plus REAL results harvested from the executed lab notebook."""
import json
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

from _style import role_box, role_legend, routed_arrow, save_editable_scene, ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_12"
ORANGE = "#FFF1E6"
GREEN = "#E6F4F1"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def cover(path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(3):
        ax.add_patch(plt.Rectangle((0.1 + i * 0.06, 0.52 - i * 0.06), 0.38, 0.4, facecolor="white", edgecolor=PRIMARY, lw=3))
    for j in range(4):
        ax.plot([0.26, 0.56], [0.72 - j * 0.07, 0.72 - j * 0.07], color="#C7D2FE", lw=4)
    ax.add_patch(plt.Circle((0.72, 0.3), 0.2, color=ACCENT))
    ax.text(0.72, 0.3, "⚙", ha="center", va="center", fontsize=60, color="white")
    ax.annotate("", xy=(0.6, 0.45), xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3))
    save(fig, path, transparent=True)


def _compact_legend(ax, roles, repeat=False, flow=True):
    labels={"data":"Data","model":"Model","tool":"Tool","check":"Check","output":"Output"}
    x=.15
    for role in roles:
        role_box(ax,x,.12,.22,.22,"",role,size=18)
        ax.text(x+.34,.23,labels[role],fontsize=18,color=TEXT,va="center")
        x+=1.45
    if flow:
        routed_arrow(ax,[(x,.23),(x+.35,.23)])
        ax.text(x+.43,.23,"Flow",fontsize=18,color=TEXT,va="center");x+=1.4
    if repeat:
        routed_arrow(ax,[(x,.23),(x+.35,.23)],dashed=True)
        ax.text(x+.43,.23,"Repeat",fontsize=18,color=TEXT,va="center")


def rag_pipeline(path):
    fig,ax=_diagram_canvas(5.1)
    top=[(.15,"Split\nKeep IDs","data"),(4.5,"Embed\nFixed model","model"),(8.85,"Store\nVectors","tool")]
    for x,label,role in top:role_box(ax,x,3.55,2.6,.95,label,role,size=19)
    for left,right,label in [(.15,4.5,"text"),(4.5,8.85,"vectors[N,384]")]:
        routed_arrow(ax,[(left+2.6,4.02),(right,4.02)],label=label,label_xy=((left+2.6+right)/2,4.28),size=18)
    low=[(.15,"Read\nQuestion","data"),(3.4,"Retrieve\nTop k","tool"),(6.65,"Generate\nCite; abstain","model"),(9.9,"Check\nQuotes","tool")]
    for x,label,role in low:role_box(ax,x,1.7,2.6,.95,label,role,size=19)
    for left,right,label in [(.15,3.4,"query"),(3.4,6.65,"passages"),(6.65,9.9,"answer")]:
        routed_arrow(ax,[(left+2.6,2.17),(right,2.17)],label=label,label_xy=((left+2.6+right)/2,2.96),size=18)
    routed_arrow(ax,[(10.15,3.55),(10.15,3.2),(4.7,3.2),(4.7,2.65)],label="index",label_xy=(7.2,3.2),size=18)
    ax.text(.15,4.77,"Indexing",fontsize=20,color=INK)
    _compact_legend(ax,("data","model","tool"));_diagram_finish(fig,path)


def chunking(path):
    fig,ax=_diagram_canvas(3.8)
    for x,label in [(.15,"Split words"),(4.55,"Slide windows"),(8.95,"Attach IDs")]:role_box(ax,x,1.55,3.3,.9,label,"data",size=21)
    for left,right,label in [(.15,4.55,"words"),(4.55,8.95,"passages")]:routed_arrow(ax,[(left+3.3,2),(right,2)],label=label,label_xy=((left+3.3+right)/2,2.95),size=18)
    ax.text(.15,3.36,"Illustrative word windows; two-word overlap",fontsize=20,color=INK)
    ax.text(4.65,1.05,"[0:8]  [6:14]  [12:20]",fontsize=20,color=TEXT)
    ax.text(8.95,.78,"IDs retain source",fontsize=18,color=TEXT)
    _compact_legend(ax,("data",));_diagram_finish(fig,path)


def agent_loop(path):
    fig,ax=_diagram_canvas(5.3)
    xs=[.15,2.3,4.45,6.6,8.75,10.9]
    labels=[("Read","data"),("Propose","model"),("Parse","tool"),("Decide","tool"),("Execute","tool"),("Record","output")]
    for x,(label,role) in zip(xs,labels):role_box(ax,x,2.75,1.75,.85,label,role,size=20)
    for left,right,label in zip(xs,xs[1:],["context","text","parsed calls","call","observation"]):routed_arrow(ax,[(left+1.75,3.17),(right,3.17)],label=label,label_xy=((left+1.75+right)/2,3.86),size=18)
    role_box(ax,6.6,1.05,2.4,.8,"Return answer","output",size=19)
    routed_arrow(ax,[(7.47,2.75),(7.47,1.85)],label="Final answer",label_xy=(8.9,2.2),size=18)
    routed_arrow(ax,[(11.77,3.6),(11.77,4.55),(3.17,4.55),(3.17,3.6)],dashed=True,label="observation",label_xy=(7.3,4.55),size=18)
    ax.text(.15,.55,"Execute validates; refusals skip the tool.",fontsize=18,color=TEXT)
    _compact_legend(ax,("data","model","tool","output"),repeat=True);_diagram_finish(fig,path)


def workflows(path):
    fig,ax=_diagram_canvas(3.1)
    for i,(head,sub) in enumerate([("Chain","Calls in order"),("Route","Choose handler"),("Merge","Parallel results"),("Delegate","Worker tasks"),("Review","Critique; revise")]):
        x=.15+i*2.57;role_box(ax,x,1.25,2.25,.85,head,"tool",size=21);ax.text(x+1.125,.88,sub,ha="center",fontsize=19,color=TEXT)
    ax.text(.15,2.55,"Workflow: code paths. Agent: chosen steps.",fontsize=20,color=INK)
    _compact_legend(ax,("tool",),flow=False);_diagram_finish(fig,path)


def mcp(path):
    fig,ax=_diagram_canvas(3.5)
    labels=[("Discover","tool"),("Validate","tool"),("Invoke","tool"),("Read result","output")]
    for i,(label,role) in enumerate(labels):role_box(ax,.15+i*3.25,1.4,2.65,.9,label,role,size=21)
    for i,label in enumerate(["schema","checked call","observation"]):routed_arrow(ax,[(2.8+i*3.25,1.85),(3.4+i*3.25,1.85)],label=label,label_xy=(3.1+i*3.25,2.75),size=18)
    ax.text(.15,3.12,"Client / host",fontsize=20,color=INK);ax.text(6.65,3.12,"Server tool",fontsize=20,color=INK)
    ax.text(.15,.7,"Compatibility does not grant trust.",fontsize=19,color=TEXT)
    _compact_legend(ax,("tool","output"));_diagram_finish(fig,path)


def agent_trace(path):
    res=json.loads((ASSETS/"results.json").read_text(encoding="utf-8"))
    trace=res["agent_trace"]
    search=json.loads(re.search(r"search_handbook\((\{.*?\})\)",trace[0]["content"]).group(1))["query"]
    expression=json.loads(re.search(r"calculator\((\{.*?\})\)",trace[1]["content"]).group(1))["expression"].replace(" ","")
    value=trace[1]["content"].split(" -> ",1)[1]
    fig,ax=_diagram_canvas(3.5)
    for x,label,role in [(.15,"Search\n"+search,"tool"),(4.55,"Calculate\n"+expression,"tool"),(8.95,"Return\n"+value,"output")]:role_box(ax,x,1.45,3.3,1.1,label,role,size=20)
    for left,right,label in [(.15,4.55,"assessment"),(4.55,8.95,"result")]:routed_arrow(ax,[(left+3.3,2),(right,2)],label=label,label_xy=((left+3.3+right)/2,2.94),size=18)
    ax.text(.15,.7,"Archived observations; complete trace in notes.",fontsize=19,color=TEXT)
    _compact_legend(ax,("tool","output"));_diagram_finish(fig,path)


def _diagram_canvas(height):
    fig, ax = plt.subplots(figsize=(13, height))
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.04, top=0.97)
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, height)
    return fig, ax


def _sequence(ax, labels, y, width=12.6, height=0.95, size=18):
    gap = 0.28
    block = (width - gap * (len(labels) - 1)) / len(labels)
    for i, (label, role) in enumerate(labels):
        x = 0.15 + i * (block + gap)
        role_box(ax, x, y, block, height, label, role, size=size)
        if i < len(labels) - 1:
            arrow(ax, x + block, y + height / 2, x + block + gap, y + height / 2)


def _diagram_finish(fig, path):
    save_editable_scene(fig, path)


def retrieval_results(path):
    r = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    methods = list(r["retrieval"])
    import numpy as np
    fig, ax = plt.subplots(figsize=(13, 3.9))
    x = np.arange(len(methods))
    for index, (field, label, colour) in enumerate([
        ("recall@1", "Doc hit@1", PRIMARY),
        ("recall@3", "Doc hit@3", TEAL),
        ("MRR", "Doc MRR", ACCENT),
    ]):
        values = [r["retrieval"][name][field] for name in methods]
        bars = ax.bar(x + (index - 1) * 0.26, values, 0.26, label=label, color=colour)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], fontsize=18, padding=3)  # as in the slide text
    ax.set_xticks(x, ["BM25", "Dense", "Hybrid", "Re-ranked"], fontsize=18)
    ax.set_yticks([0,.5,1]); ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1.22)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.24), fontsize=18)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, path)


def rag_results(path):
    r = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    rows = r["summary"]
    names = list(rows)
    values = [rows[name]["accuracy (answerable)"] for name in names]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    bars = ax.barh(range(len(values)), values, color=[MUTED, PRIMARY])
    ax.bar_label(bars, labels=[f"{value:.0%}" for value in values], fontsize=22, padding=7)
    ax.set_yticks(range(len(values)), ["Without retrieval", "RAG"], fontsize=22)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    from matplotlib.ticker import PercentFormatter
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticks([0,.5,1]); ax.tick_params(axis="x", labelsize=18)
    ax.set_xlabel("Archived answer accuracy", fontsize=20)
    where = r.get("gpu") or "GPU" if str(r["device"]).startswith("cuda") else "CPU"
    ax.set_title(f"Stored run: {r['llm'].split('/')[-1]} on {where.replace('NVIDIA GeForce ', '')}", fontsize=18, pad=14)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


FIGURES = {"cover": cover, "rag_pipeline": rag_pipeline, "chunking": chunking, "agent_loop": agent_loop, "workflows": workflows, "mcp": mcp,
           "agent_trace": agent_trace, "lab_retrieval": retrieval_results, "lab_rag": rag_results}
