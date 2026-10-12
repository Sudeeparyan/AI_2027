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


def _legend(ax, roles=("model",), repeat=False, y=.15):
    names={"data":"Data","model":"Model","loss":"Loss","output":"Output","metric":"Check","tool":"Tool"}
    x=.15
    for role in roles:
        role_box(ax,x,y,.24,.22,"",role,size=18)  # a FancyBbox swatch is kept in the editable PPT scene
        ax.text(x+.34,y+.11,names[role],va="center",fontsize=18,color=TEXT)
        x+=1.7
    ax.plot([x,x+.4],[y+.11,y+.11],color=GREEN)
    ax.text(x+.5,y+.11,"Flow",va="center",fontsize=18,color=TEXT)
    if repeat:
        ax.plot([x+2,x+2.4],[y+.11,y+.11],color=ORANGE,ls="--")
        ax.text(x+2.5,y+.11,"Repeat",va="center",fontsize=18,color=TEXT)


def rnn_vs_attention(path):
    fig,ax=canvas(5.4)
    ax.text(.1,5.0,"Illustrative information routes",fontsize=21,color=INK)
    ax.text(.1,4.6,"Recurrence",fontsize=19,color=INK)
    xs=[.15,4.45,8.75]
    for x,word,label in zip(xs,["animal","street","it"],["Update h1","Update h2","Update h3"]):
        role_box(ax,x,3.25,3.6,.9,label,"model",size=18)
        ax.text(x+1.8,4.35,word,ha="center",fontsize=18,color=TEXT)
        arrow(ax,x+1.8,4.25,x+1.8,4.15,color=BLUE)
    for x,right,label in zip(xs,xs[1:],["h1","h2"]):
        arrow(ax,x+3.6,3.7,right,3.7,color=GREEN)
        ax.text((x+3.6+right)/2,4.0,label,ha="center",fontsize=18,color=TEXT)
    ax.text(.1,2.65,"Attention",fontsize=19,color=INK)
    for x,label in zip(xs,["Project Q/K/V","Weight Q/K","Mix values"]):
        role_box(ax,x,1.35,3.6,.9,label,"model",size=18)
    for x,right,label in zip(xs,xs[1:],["Q/K","weights"]):
        arrow(ax,x+3.6,1.8,right,1.8,color=GREEN)
        ax.text((x+3.6+right)/2,2.33,label,ha="center",fontsize=18,color=TEXT)
    routed_arrow(ax,[(1.95,1.35),(1.95,.75),(10.55,.75),(10.55,1.35)],color=BLUE,label="V",label_xy=(6.3,.75),size=18)
    _legend(ax)
    save(fig,path)


def qkv(path):
    fig,ax=canvas(4.7)
    items=[("Read X","data"),("Project Q/K/V","model"),("Compare;\nscale","model"),("Mask;\nsoftmax","model"),("Mix values","output")]
    flow(ax,items,labels=["X[3,2]","Q/K[3,2]","scores[3,3]","weights[3,3]"],size=18)
    routed_arrow(ax,[(3.735,1.8),(3.735,1.0),(11.445,1.0),(11.445,1.8)],color=BLUE,label="V[3,2]",label_xy=(7.5,1.0),size=18)
    ax.text(.1,4.1,"Matching and carrying are separate",fontsize=21,color=INK)
    _legend(ax,("data","model","output"))
    save(fig,path)


def attn_worked(path):
    fig,ax=canvas(4.9)
    ax.text(.15,4.35,"Illustrative cat query [1,0]; keys [1,0], [0,1], [1,1]",fontsize=21,color=INK)
    blocks=[(.15,3.0,"Read inputs\nq and three keys","data"),(3.65,2.7,"Compare; scale\n[0.71,0,0.71]","model"),
            (6.85,2.9,"Row softmax\n[0.40,0.20,0.40]","model"),(10.25,2.6,"Mix values\n[0.60,0.40]","output")]
    for x,w,label,role in blocks:role_box(ax,x,2.3,w,1.2,label,role,size=19)
    for (x,w,_,_),(right,_,_,_),label in zip(blocks,blocks[1:],["q, K","scores","weights"]):
        arrow(ax,x+w,2.9,right,2.9,color=GREEN)
        ax.text((x+w+right)/2,3.75,label,ha="center",fontsize=18,color=BLUE)
    role_box(ax,6.85,.75,2.9,.9,"Values V\n[1,0], [0,1], [0.5,0.5]","data",size=17)
    routed_arrow(ax,[(9.75,1.2),(11.55,1.2),(11.55,2.3)],color=BLUE,label="V",label_xy=(10.65,1.2),size=18)
    _legend(ax,("data","model","output"),y=.12)
    save(fig,path)


def causal_mask(path):
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    fig.subplots_adjust(left=.07,right=.95,bottom=.22,top=.82,wspace=.35)
    for ax,allowed,title in [(axes[0],np.ones((3,3)),"Encoder"),(axes[1],np.tril(np.ones((3,3))),"Causal decoder")]:
        ax.imshow(allowed,cmap="Blues",vmin=0,vmax=1.6)
        for i in range(3):
            for j in range(3):ax.text(j,i,str(int(allowed[i,j])),ha="center",va="center",fontsize=23,color=INK)
        ax.set_xticks(range(3),["cat","sat","mat"],fontsize=18)
        ax.set_yticks(range(3),["cat","sat","mat"],fontsize=18)
        ax.set_title(title,fontsize=21,loc="left");ax.grid(False)
    fig.text(.07,.08,"Rows=query; columns=source. 1=allowed; 0=blocked.",fontsize=18,color=TEXT)
    save(fig,path)


def multihead(path):
    fig,ax=canvas(4.4)
    flow(ax,[("Project Q/K/V","model"),("Split heads","model"),("Attend","model"),("Merge outputs","model"),("Project output","output")],labels=["[T,3d]","[h,T,d/h]","[h,T,d/h]","[T,d]"],size=18)
    ax.text(.1,3.9,"Several learned mixtures",fontsize=21,color=INK)
    ax.text(.2,1.0,"Illustration: d=12; h=3; four features/head.",fontsize=18,color=TEXT)
    _legend(ax,("model","output"))
    save(fig,path)


def positional(path):
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    fig.subplots_adjust(left=.02,right=.98,bottom=.24,top=.83,wspace=.18)
    for ax in axes:ax.axis("off");ax.set_xlim(0,6);ax.set_ylim(0,3)
    a,b=axes
    a.set_title("Add position",fontsize=21,loc="left")
    role_box(a,.1,1.2,2.4,1.0,"Look up token\n[1,0]","model",size=18)
    role_box(a,3.45,1.2,2.4,1.0,"Add position\n[1,1]","model",size=18)
    arrow(a,2.5,1.7,3.45,1.7,color=GREEN)
    a.text(2.8,2.55,"vector",ha="center",fontsize=18,color=TEXT)
    arrow(a,4.65,.45,4.65,1.2,color=BLUE);a.text(4.65,.18,"position [0,1]",ha="center",fontsize=18,color=TEXT)
    b.set_title("RoPE",fontsize=21,loc="left")
    role_box(b,.1,1.2,2.4,1.0,"Project Q/K","model",size=18)
    role_box(b,3.45,1.2,2.4,1.0,"Rotate pairs","model",size=18)
    arrow(b,2.5,1.7,3.45,1.7,color=GREEN);b.text(2.8,2.55,"Q/K",ha="center",fontsize=18,color=TEXT)
    b.text(.1,.3,"Position changes comparison angles.",fontsize=18,color=TEXT)
    legend=fig.add_axes([.02,.015,.96,.07]);legend.set_xlim(0,13);legend.set_ylim(0,.4);legend.axis("off")
    _legend(legend)
    save(fig,path)


def block(path):
    fig,ax=canvas(4.8)
    xs=[.1+i*1.85 for i in range(7)]
    names=[("Input x","data"),("Normalise","model"),("Attend","model"),("Add","model"),("Normalise","model"),("MLP","model"),("Add","model")]
    for x,(label,role) in zip(xs,names):role_box(ax,x,2.0,1.5,1.1,label,role,size=18)
    for x,right in zip(xs,xs[1:]):
        arrow(ax,x+1.5,2.55,right,2.55,color=GREEN)
        ax.text((x+1.5+right)/2,3.35,"[T,d]",ha="center",fontsize=18,color=BLUE)
    centre=lambda i:xs[i]+.75
    routed_arrow(ax,[(centre(0),3.1),(centre(0),3.85),(centre(3),3.85),(centre(3),3.1)],color=BLUE,label="residual x[T,d]",label_xy=(3.6,3.85),size=18)
    routed_arrow(ax,[(centre(3),2.0),(centre(3),1.15),(centre(6),1.15),(centre(6),2.0)],color=BLUE,label="residual h[T,d]",label_xy=(9.2,1.15),size=18)
    ax.text(.1,4.35,"Pre-norm; residual additions",fontsize=21,color=INK)
    ax.text(.2,.65,"MLP acts per position.",fontsize=18,color=TEXT)
    _legend(ax,("data","model"))
    save(fig,path)


def architecture_comparison(path):
    fig,ax=canvas(4.7)
    rows=[("Encoder","Whole input","Label/vector"),("Decoder","Current; past","Continuation"),("Encoder–decoder","Source; output past","Output sequence")]
    for x,label in zip([.3,4.5,8.5],["Family","Reads","Produces"]):ax.text(x,3.9,label,fontsize=21,color=INK,weight="bold")
    for i,row in enumerate(rows):
        for x,value in zip([.3,4.5,8.5],row):ax.text(x,3.0-i*.85,value,fontsize=20,color=TEXT)
    ax.text(.3,.4,"Task guidance; evaluate the actual system.",fontsize=18,color=TEXT)
    save(fig,path)


def three_archs(path):
    fig,ax=canvas(6.7)
    rows=[("Encoder: BERT","Encode all","Select vector","IDs","vectors"),("Decoder: GPT-2","Decode causally","Sample; append","prefix","probabilities"),("Encoder–decoder: T5","Encode source","Decode cross+causal","source","encoded source")]
    for i,(name,left,right,incoming,edge) in enumerate(rows):
        y=5.25-i*2.05
        ax.text(.1,y+.5,name,fontsize=18,color=INK,va="center")
        role_box(ax,3.5,y,3.3,1.0,left,"model",size=18);role_box(ax,8.5,y,3.6,1.0,right,"model",size=18)
        arrow(ax,3.0,y+.5,3.5,y+.5,color=BLUE);ax.text(3.0,y+1.12,incoming,fontsize=18,ha="center",color=TEXT)
        arrow(ax,6.8,y+.5,8.5,y+.5,color=GREEN);ax.text(7.65,y+1.12,edge,fontsize=18,ha="center",color=TEXT)
        if i==1:routed_arrow(ax,[(10.3,y),(10.3,y-.38),(4.9,y-.38),(4.9,y)],color=ORANGE,dashed=True,label="extended prefix",label_xy=(7.6,y-.38),size=18)
    _legend(ax,repeat=True,y=.05)
    save(fig,path)


def kv_cache(path):
    fig,ax=canvas(5.3)
    xs=[.15,3.35,6.55,9.75]
    for x,label in zip(xs,["Read last token","Project Q/K/V","Attend","Score next token"]):role_box(ax,x,2.75,2.65,1.0,label,"model",size=18)
    for x,right,label in zip(xs,xs[1:],["sat vector","query q","last output"]):
        arrow(ax,x+2.65,3.25,right,3.25,color=GREEN);ax.text((x+2.65+right)/2,3.95,label,fontsize=18,ha="center",color=BLUE)
    role_box(ax,.15,.85,2.65,.9,"Cache:\nearlier K/V","data",size=18)
    role_box(ax,3.35,.85,2.65,.9,"Append K/V","model",size=18)
    arrow(ax,4.67,2.75,4.67,1.75,color=GREEN);ax.text(4.8,2.25,"new K/V",fontsize=18,color=BLUE,va="center")
    arrow(ax,2.8,1.3,3.35,1.3,color=GREEN)
    routed_arrow(ax,[(6.0,1.3),(7.87,1.3),(7.87,2.75)],color=BLUE,label="all K/V",label_xy=(6.95,1.3),size=18)
    ax.text(.1,4.75,"Current query uses a known token",fontsize=21,color=INK)
    _legend(ax,("data","model"),y=.15)
    save(fig,path)


def moe(path):
    fig,ax=canvas(4.9)
    role_box(ax,.8,1.9,2.2,1.0,"Route token","model",size=18)
    role_box(ax,4.4,2.7,2.7,.9,"Run expert A","model",size=18)
    role_box(ax,4.4,1.15,2.7,.9,"Run expert B","model",size=18)
    role_box(ax,9.0,1.9,2.7,1.0,"Weighted sum","model",size=18)
    for y,end in [(2.65,3.15),(2.15,1.6)]:routed_arrow(ax,[(3,y),(3.7,y),(3.7,end),(4.4,end)],color=GREEN)
    for y,end in [(3.15,2.65),(1.6,2.15)]:routed_arrow(ax,[(7.1,y),(8,y),(8,end),(9,end)],color=GREEN)
    ax.text(3.7,3.85,"vectors",fontsize=18,color=TEXT);ax.text(3.7,.7,"vectors",fontsize=18,color=TEXT)
    ax.text(7.6,3.85,"outputs",fontsize=18,color=TEXT);ax.text(7.6,.7,"outputs",fontsize=18,color=TEXT)
    routed_arrow(ax,[(1.9,2.9),(1.9,4.3),(10.35,4.3),(10.35,2.9)],color=BLUE,label="routing weights",label_xy=(6.3,4.3),size=18)
    arrow(ax,.1,2.4,.8,2.4,color=BLUE);arrow(ax,11.7,2.4,12.8,2.4,color=GREEN)
    ax.text(.1,3.3,"token",fontsize=18,color=TEXT);ax.text(12,3.3,"vector",fontsize=18,color=TEXT)
    _legend(ax)
    save(fig,path)


def vit_patches(path):
    fig,ax=canvas(4.7)
    flow(ax,[("Split\npatches","model"),("Flatten","model"),("Embed +\nposition","model"),("Encode","model"),("Pool","model"),("Score\nclasses","output")],w=1.7,gap=.43,labels=["[4,16,16,3]","[4,768]","[4,d]","[4,d]","[d]"],size=18)
    ax.text(.1,4.1,"Illustrative RGB image: 32×32",fontsize=21,color=INK)
    ax.text(.2,1.0,"16×16×3=768 values/patch; four patches.",fontsize=18,color=TEXT)
    _legend(ax,("model","output"))
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
    ax.set_xlabel("length n",fontsize=18)
    ax.set_ylabel("score-table GB",fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_title("Calculated score-table memory",loc="left",fontsize=20)
    fig.text(.11,.03,"One head; two bytes/score; excludes other tensors.",fontsize=17,color=TEXT)
    save(fig,path)


def bert_summary(path):
    """The head the lab's TODO 5 rule selects: largest it -> animal weight over all 144 heads."""
    result=json.loads((ASSETS/"bert_attention.json").read_text(encoding="utf-8"))
    weight=result["it_to_animal"]
    fig,ax=plt.subplots(figsize=(13,4.6))
    fig.subplots_adjust(left=.25,right=.95,bottom=.26,top=.78)
    ax.barh([1,0],[weight,1-weight],color=[BLUE,GREEN],height=.52)
    ax.set_yticks([1,0],["animal", "other positions"],fontsize=20)
    ax.set_xlim(0,1); ax.tick_params(axis="x",labelsize=18)
    ax.set_xlabel("Query it: weight",fontsize=20)
    for y,v in [(1,weight),(0,1-weight)]:
        ax.text(v+.025,y,f"{v:.3f}",va="center",fontsize=21,color=INK)
    ax.set_title(f"Archived selected BERT: layer {result['layer']}, head {result['head']}",
                 fontsize=21,loc="left")
    fig.text(.06,.06,"Largest it→animal weight; selection disclosed. Routing alone is not explanation.",
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
        ax.set_xticks([0,8,12],[0,8,12],fontsize=18)
        ax.set_yticks([0,8,12],[0,8,12],fontsize=18)
        ax.set_title(f"L{head['layer']} H{head['head']}",loc="left",fontsize=18)
        ax.set_xlabel("source",fontsize=18)
        ax.grid(False)
    axes[0].set_ylabel("query",fontsize=18)
    bar=fig.colorbar(image,ax=axes,fraction=.025,pad=.02)
    bar.set_label("weight",fontsize=18); bar.set_ticks([0,.5,1]); bar.ax.tick_params(labelsize=18)
    fig.text(.1,.025,f"L=layer; H=head. 8=it; 11=tired; 12=full-stop. Future max={data['max_future_weight']:.1f}; heads={data['layers']*data['heads_per_layer']}.",fontsize=18,color=TEXT)
    save(fig,path)


FIGURES={"cover":cover,"rnn_vs_attention":rnn_vs_attention,"qkv":qkv,"attn_worked":attn_worked,
         "causal_mask":causal_mask,"multihead":multihead,"positional":positional,"block":block,
         "three_archs":three_archs,"architecture_comparison":architecture_comparison,"kv_cache":kv_cache,
         "moe":moe,"vit_patches":vit_patches,"complexity":complexity,
         "bert_summary":bert_summary,"bert_matrix":bert_matrix,"gpt2_causal_summary":gpt2_causal_summary,
         **{name:asset(name) for name in ["chargpt_loss"]}}
