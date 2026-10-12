"""Week 9 figures: multimodal diagrams plus REAL results from the executed lab notebook
(curriculum/assets/week_09/results.json and lab_*.png harvested by tools/harvest_figures.py)."""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import role_box, role_legend, routed_arrow, save_editable_scene, ACCENT, INK, MUTED, PRIMARY, SERIES, TEAL, TEXT, TINT, arrow, box, save

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "week_09"
ORANGE = "#FFF1E6"


def asset(name):
    def make(path):
        shutil.copy(ASSETS / f"{name}.png", path)
    return make


def results():
    return json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))


def cover(path):
    rng = np.random.default_rng(9)
    fig, ax = plt.subplots(figsize=(5, 5))
    a = rng.normal(0, 1, (60, 2)) * 0.5 + [-1, 0.6]
    b = rng.normal(0, 1, (60, 2)) * 0.5 + [1, -0.6]
    for i in range(0, 60, 4):
        ax.plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="#C7D2FE", alpha=0.3, lw=1)
    ax.scatter(*a.T, s=60, color=PRIMARY, alpha=0.85)
    ax.scatter(*b.T, s=60, color=ACCENT, alpha=0.85, marker="s")
    ax.axis("off")
    save(fig, path, transparent=True)


def modalities(path):
    fig, ax = _diagram_canvas(4.8)
    rows = [('Text\nTokens', 'IDs'), ('Image\nPatches', 'pixels'), ('Audio\nFrames', 'features'), ('Video\nFrames+time', 'sequence')]
    for i, (label, flow) in enumerate(rows):
        y = 3.35 - i * .88
        role_box(ax, .15, y, 2.55, .8, label, 'data', size=19)
        target = 3.6 - i * .63
        routed_arrow(ax, [(2.7, y+.4),(5.05,target)], label=flow, label_xy=(3.7,y+.46), size=17)
    role_box(ax, 5.05, 1.25, 3.0, 2.65, 'Separate\nencoders', 'model', size=21)
    routed_arrow(ax, [(8.05,2.6),(9.4,2.6)], label='vectors',label_xy=(8.73,2.9),size=17)
    role_box(ax,9.4,1.65,3.3,1.85,'Compare or\ncombine', 'output',size=21)
    ax.text(.15,4.35,'Alignment makes vectors comparable',fontsize=20,color=INK)
    role_legend(ax,.08,size=16)
    _diagram_finish(fig,path)




def fusion(path):
    fig, ax = _diagram_canvas(3.1)
    for x, title in [(.15,'Early fusion'),(4.55,'Late fusion'),(8.95,'Cross-attention')]:
        ax.text(x,2.65,title,fontsize=20,fontweight='bold',color=INK)
    for x,left,right,flow in [(.15,'Image tokens','Text tokens','tokens'),(4.55,'Image encoder','Text encoder','vectors'),(8.95,'Image K/V','Text queries','K/V, Q')]:
        role_box(ax,x,1.7,1.8,.65,left,'model' if x==4.55 else 'data',size=16)
        role_box(ax,x+2.05,1.7,1.8,.65,right,'model' if x==4.55 else 'data',size=16)
        arrow(ax,x+.9,1.7,x+1.25,1.15)
        arrow(ax,x+2.95,1.7,x+2.6,1.15)
        ax.text(x+1.92,1.4,flow,ha='center',fontsize=16,color=TEXT)
    for x,label in [(.6,'Joint model'),(5.,'Compare'),(9.4,'Attention')]:
        role_box(ax,x,.41,2.95,.74,label,'model',size=18)
    role_legend(ax,.05,size=16)
    _diagram_finish(fig,path)




def architectures(path):
    fig, ax = _diagram_canvas(4.9)
    panels = [(.15,2.8,'Dual encoder',['Image','Compare','Text'],'vectors'),(6.65,2.8,'Fusion encoder',['Image tokens','Text tokens','Joint encoder'],'tokens'),(.15,.7,'Encoder-decoder',['Encoder','Decoder'],'features'),(6.65,.7,'Vision + LLM',['Vision','Bridge','LLM'],'features')]
    for x,y,title,labels,flow in panels:
        ax.text(x,y+1.5,title,fontsize=20,fontweight='bold',color=INK)
        if title == 'Fusion encoder':
            role_box(ax,x,y+.43,2.1,.4,labels[0],'data',size=17)
            role_box(ax,x,y-.07,2.1,.4,labels[1],'data',size=17)
            role_box(ax,x+3.4,y,2.55,.8,labels[2],'model',size=18)
            arrow(ax,x+2.1,y+.63,x+3.4,y+.6)
            arrow(ax,x+2.1,y+.13,x+3.4,y+.2)
            ax.text(x+2.7,y+.39,flow,ha='center',fontsize=17,color=TEXT)
            continue
        gap=.55
        width=(5.95-gap*(len(labels)-1))/len(labels)
        for i,label in enumerate(labels):
            bx=x+i*(width+gap)
            role_box(ax,bx,y,width,.8,label,'model',size=18)
            if i < len(labels)-1:
                if title == 'Dual encoder' and i == 1:
                    arrow(ax,bx+width+gap,y+.4,bx+width,y+.4)
                else:
                    arrow(ax,bx+width,y+.4,bx+width+gap,y+.4)
        ax.text(x+2.975,y+1.04,flow,ha='center',fontsize=17,color=TEXT)
    _diagram_finish(fig,path)




def clip_matrix(path):
    rng = np.random.default_rng(1)
    n = 6
    m = rng.normal(0, 0.08, (n, n)) + 0.15 + np.eye(n) * 0.55
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.imshow(m, cmap="Purples", vmin=0, vmax=0.8)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ACCENT, lw=3))
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"text {j + 1}" for j in range(n)])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"image {j + 1}" for j in range(n)])
    ax.tick_params(labelsize=16)
    ax.set_title("Illustrative paired batch:\nmatching pairs lie on the diagonal", loc="left", fontsize=17)
    ax.grid(False)
    save(fig, path)


def zeroshot_pipeline(path):
    fig, ax = _diagram_canvas(3.6)
    for y, source, encoder in [(2.25, "Class prompts\n'a photo of a cat'", "Text encoder"),
                               (0.85, "New image", "Image encoder")]:
        role_box(ax, 0.15, y, 2.4, 0.9, source, "data", size=19)
        role_box(ax, 3.65, y, 2.0, 0.9, encoder, "model", size=19)
        routed_arrow(ax, [(2.55,y+.45),(3.65,y+.45)], label='text' if y==2.25 else 'pixels', label_xy=(3.10,y+1.0),size=16)
        routed_arrow(ax, [(5.65,y+.45),(6.80,2.0)], label='vectors',label_xy=(6.22,y+.83),size=16)
    role_box(ax, 6.8, 1.35, 2.3, 1.3, "Compare vectors\ncosine similarity", "model", size=19)
    role_box(ax, 9.85, 1.35, 2.85, 1.3, "Choose highest score\nfrom given classes", "output", size=19)
    routed_arrow(ax, [(9.1,2.0),(9.85,2.0)],label='scores',label_xy=(9.48,2.9),size=16)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def whisper_pipeline(path):
    fig, ax = _diagram_canvas(2.2)
    _sequence(ax, [("Waveform\n16 kHz", "data"), ("Log-mel\nfeatures", "data"),
                   ("Audio\nencoder", "model"), ("Text decoder\ncross-attention", "model"),
                   ("Transcript", "output")], 0.8, height=1.0, size=18)
    block = (12.6 - .28 * 4) / 5
    for i, label in enumerate(['samples', 'features', 'states', 'tokens']):
        x = .15 + i * (block + .28) + block + .14
        ax.text(x, 1.98, label, ha='center', fontsize=16, color=TEXT)
    role_legend(ax, 0.08, size=16)
    _diagram_finish(fig, path)



def zeroshot_results(path):
    r = results()["zero_shot"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    # Show each stored template as students would read it: "a photo of a cat." rather than 'a photo of a {}.'.
    names = [('class name only ("cat")' if k.startswith("label only") else '"' + k.strip("'").format("cat") + '"')
             if "{}" in k else k for k in r]
    vals = list(r.values())
    ax.barh(names[::-1], vals[::-1], color=[TEAL, PRIMARY, MUTED][: len(vals)])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=18)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(f"Accuracy: {results()['n_images']:,} CIFAR-10 images; 10 classes; chance 10%", fontsize=18)
    ax.set_title("Only the text prompt changed", loc="left", fontsize=19)
    ax.grid(axis="y", visible=False)
    save(fig, path)


def modality_gap(path):
    """Reframe the measured PCA image without recalculating any point.

    The historic notebook saved only PNG output, not projection coordinates.
    Crop the plotting area from that exact saved asset, retain its aspect
    ratio and supply a readable legend outside it. The small old key lies
    over empty background, so a white overlay hides only that old key.
    """
    image = plt.imread(ASSETS / "lab_modality_gap.png")
    if image.shape[:2] != (540, 839):
        raise ValueError("Recheck the historic PCA image crop before updating its wrapper")
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_axes([0.065, 0.10, 0.62, 0.85])
    plot = image[31:507, 54:699]
    ax.imshow(plot, interpolation="none")
    # Original key bounds in the cropped pixel coordinate system. No measured
    # point is covered: its nearest clouds are far left and far right.
    ax.add_patch(plt.Rectangle((260, 5), 126, 52, facecolor="white", edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PCA component 1", fontsize=20)
    ax.set_ylabel("PCA component 2", fontsize=20)
    key = fig.add_axes([0.725, 0.13, 0.26, 0.80])
    key.axis("off")
    key.text(0, 0.96, "Circle: image\nStar: class description", fontsize=18, color=TEXT, va="top")
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for index, name in enumerate(names):
        y = 0.70 - index * 0.068
        key.scatter([0.04], [y], color=plt.cm.tab10(index), s=95)
        key.text(0.13, y, name, fontsize=18, color=TEXT, va="center")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    save(fig, path)


def confusion_results(path):
    """Retain measured heatmap pixels and replace only the surrounding labels."""
    image = plt.imread(ASSETS / "lab_confusion.png")
    if image.shape[:2] != (551, 616):
        raise ValueError("Recheck the historic confusion-image crop")
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_axes([0.085, 0.15, 0.40, 0.80])
    matrix = image[31:435, 120:524]
    ax.imshow(matrix, interpolation="none")
    centres = (np.arange(10) + 0.5) * matrix.shape[1] / 10 - 0.5
    ax.set_xticks(centres, range(10), fontsize=18)
    ax.set_yticks(centres, range(10), fontsize=18)
    ax.set_xlabel("Predicted class ID", fontsize=20)
    ax.set_ylabel("True class ID", fontsize=20)
    ax.grid(False)
    bar = fig.add_axes([0.515, 0.15, 0.025, 0.80])
    bar.imshow(image[21:445, 549:571], aspect="auto", interpolation="none")
    bar.set_xticks([])
    bar.set_yticks([423, 346, 269, 192, 115, 38], ["0", "20", "40", "60", "80", "100"], fontsize=18)
    bar.yaxis.tick_right()
    bar.set_title("Count", fontsize=18, pad=12)
    bar.grid(False)
    key = fig.add_axes([0.66, 0.10, 0.32, 0.85])
    key.axis("off")
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for index, name in enumerate(names):
        key.text(0, 0.97 - index * 0.096, f"{index}  {name}", fontsize=18, color=TEXT, va="top")
    save(fig, path)



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


FIGURES = {
    "cover": cover, "modalities": modalities, "fusion": fusion, "architectures": architectures, "clip_matrix": clip_matrix,
    "zeroshot_pipeline": zeroshot_pipeline, "whisper_pipeline": whisper_pipeline, "zeroshot_results": zeroshot_results,
    "lab_modality_gap": modality_gap,
    "lab_confusion": confusion_results,
    **{n: asset(n) for n in ["lab_similarity", "lab_retrieval", "lab_spectrogram"]},
}
