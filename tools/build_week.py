"""Build weekly teaching packs: slides (.pptx), teaching notes (.docx), lab notebooks (.ipynb).

Sources (one set per week, edit these):
  curriculum/section_7_3.yaml          revised descriptor row (topic, detail, tutorials)
  curriculum/weeks/week_XX.yaml        40 core slides + objectives + coverage terms
  curriculum/beginner/week_XX.json     vocabulary, technical diagrams and code walkthroughs
  curriculum/notes/week_XX.md          teaching notes (Markdown)
  curriculum/labs/week_XX_lab.py       lab notebook in percent format with solution markers
  curriculum/figures/week_XX.py        matplotlib figures used by slides and notes

Outputs:
  deliverables/Week_XX_<Slug>/Week_XX_Lecture_Slides.pptx
                              Week_XX_Teaching_Notes.docx
                              Week_XX_Lab.ipynb            (student version)
                              Week_XX_Lab_Solutions.ipynb  (instructor version)
  build/week_XX/spec.json, qa.json     intermediate spec and QA report

Usage: .venv/Scripts/python.exe tools/build_week.py --weeks 1,2 | all
"""
from __future__ import annotations

import argparse
import shutil
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml
import beginner

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT / "curriculum"
BUILD = ROOT / "build"
DELIV = ROOT / "deliverables"
TOOLS = ROOT / "tools"
INK = "#1E1B4B"

SLIDE_TYPES = {
    "title": [], "outcomes": [], "agenda": ["items"], "section": ["number", "title"],
    "bullets": ["title", "bullets"], "cards": ["title", "cards"], "compare": ["title", "columns"],
    "table": ["title", "header", "rows"], "flow": ["title", "steps"], "figure": ["title", "figure"],
    "equation": ["title", "eq"], "code": ["title", "code"], "quiz": ["question", "options", "answer"],
    "callout": ["statement"], "stat": ["title", "stats"], "lab": ["steps"], "summary": ["points"],
    "resources": ["items"], "discussion": ["prompt"], "technical": ["title", "diagram"],
}
REQUIRED_NOTE_SECTIONS = [
    "Lecture plan", "Lecture notes", "Common misconceptions", "Responsible AI", "Lab guide",
    "Practice questions", "Glossary", "Readings",
]
MIN_NOTE_WORDS = 45


# ---------------------------------------------------------------- loading
PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def fill_placeholders(obj, values: dict):
    """Replace {{KEY}} in every string of a parsed structure with measured values from the week's fill.json."""
    if isinstance(obj, str):
        return PLACEHOLDER_RE.sub(lambda m: str(values.get(m.group(1), m.group(0))), obj)
    if isinstance(obj, list):
        return [fill_placeholders(x, values) for x in obj]
    if isinstance(obj, dict):
        return {k: fill_placeholders(v, values) for k, v in obj.items()}
    return obj


def load_week(n: int) -> dict:
    rows = yaml.safe_load((CUR / "section_7_3.yaml").read_text(encoding="utf-8"))["weeks"]
    row = next(r for r in rows if r["week"] == n)
    fill_path = CUR / "assets" / f"week_{n:02d}" / "fill.json"  # written by tools/fill_results.py from measured results
    values = json.loads(fill_path.read_text(encoding="utf-8")) if fill_path.exists() else {}
    wk = fill_placeholders(yaml.safe_load((CUR / "weeks" / f"week_{n:02d}.yaml").read_text(encoding="utf-8")), values)
    wk["week"] = n
    wk["topic"] = row["topic"].rstrip(".")
    wk["slug"] = row["slug"]
    wk["detail"] = " ".join(row["detail"].split())
    wk["tutorials"] = [" ".join(t.split()) for t in row["tutorials"]]
    wk["notes_md"] = fill_placeholders((CUR / "notes" / f"week_{n:02d}.md").read_text(encoding="utf-8"), values)
    return beginner.enrich(wk)


def validate(wk: dict) -> list[str]:
    errs = []
    slides = wk.get("slides", [])
    expected = wk.get("expected_slides", 40)
    if len(slides) != expected:
        errs.append(f"expected {expected} slides, found {len(slides)}")
    for i, s in enumerate(slides, 1):
        t = s.get("type")
        if t not in SLIDE_TYPES:
            errs.append(f"slide {i}: unknown type {t!r}")
            continue
        for f in SLIDE_TYPES[t]:
            if f not in s:
                errs.append(f"slide {i} ({t}): missing field {f!r}")
        words = len(str(s.get("notes", "")).split())
        if words < MIN_NOTE_WORDS:
            errs.append(f"slide {i} ({t}: {s.get('title', '')[:40]}): speaker notes only {words} words")
    for k in ("subtitle", "mimlos", "objectives", "coverage"):
        if not wk.get(k):
            errs.append(f"missing top-level field {k!r}")
    heads = [h.strip() for h in re.findall(r"^# (.+)$", wk["notes_md"], flags=re.M)]
    for req in REQUIRED_NOTE_SECTIONS:
        if not any(req.lower() in h.lower() for h in heads):
            errs.append(f"teaching notes missing a '# {req}…' section")
    left = sorted(set(PLACEHOLDER_RE.findall(json.dumps(slides, ensure_ascii=False) + wk.get("notes_md", ""))))
    if left:
        errs.append(f"unfilled result placeholders (run tools/fill_results.py): {', '.join(left)}")
    return errs


# ---------------------------------------------------------------- figures & equations
def run_figures(n: int) -> dict:
    src = CUR / "figures" / f"week_{n:02d}.py"
    out = BUILD / "figures" / f"week_{n:02d}"
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return {}
    style = CUR / "figures" / "_style.py"
    h = hashlib.sha1(src.read_bytes() + style.read_bytes())
    assets_dir = CUR / "assets" / f"week_{n:02d}"
    if assets_dir.exists():  # real results produced by curriculum/assets/make_week_XX.py
        for f in sorted(assets_dir.iterdir()):
            h.update(f.name.encode() + f.read_bytes())
    digest = h.hexdigest()
    stamp = out / ".hash"
    sys.path.insert(0, str(CUR / "figures"))
    spec = importlib.util.spec_from_file_location(f"figs_week_{n:02d}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = stamp.exists() and stamp.read_text() == digest
    assets = {}
    for fid, fn in mod.FIGURES.items():
        path = out / f"{fid}.png"
        if not (fresh and path.exists()):
            fn(path)
        assets[fid] = path
    stamp.write_text(digest)
    return assets


def render_equation(latex: str, fontsize: int = 22) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latex = re.sub(r"(?<![A-Za-z])\\frac", r"\\dfrac", latex)  # display-size fractions: \frac renders tiny on a slide
    key = hashlib.sha1(f"{fontsize}|{latex}".encode()).hexdigest()[:16]
    path = BUILD / "eq" / f"{key}.png"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color=INK)
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def img_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def asset(path: Path) -> dict:
    w, h = img_size(path)
    return {"path": str(path), "w": w, "h": h}


# Whole LaTeX command names -> Unicode (matched as complete words, so \le never eats \left or \leq).
SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
    "lambda": "λ", "mu": "μ", "sigma": "σ", "phi": "φ", "pi": "π", "tau": "τ", "Sigma": "Σ", "Delta": "Δ",
    "nabla": "∇", "partial": "∂", "sum": "Σ", "prod": "Π", "int": "∫", "in": "∈", "sim": "~", "approx": "≈",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "times": "×", "cdot": "·", "pm": "±", "to": "→",
    "infty": "∞", "mid": "|", "top": "ᵀ", "ldots": "…", "dots": "…", "varnothing": "∅", "emptyset": "∅",
    "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos", "sup": "sup", "max": "max", "min": "min",
    "arg": "arg", "left": "", "right": "", "quad": " ", "qquad": "  ",
}
STRUCTURAL = {"mathrm", "text", "mathbf", "mathcal", "operatorname", "mathbb", "hat", "bar", "tilde", "frac", "tfrac", "dfrac", "sqrt"}
GREEK = {"\\" + k: v for k, v in SYMBOLS.items()}  # kept for tools that list the supported commands
BLACKBOARD = {"E": "𝔼", "R": "ℝ", "N": "ℕ", "Z": "ℤ", "P": "ℙ"}
ACCENT = {"hat": "\u0302", "bar": "\u0304", "tilde": "\u0303"}
SUB = str.maketrans("0123456789aehijklmnoprstuvx+-", "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋")
SUP = str.maketrans("0123456789+-nTi", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿᵀⁱ")
ARG = r"(?:\{([^{}]*)\}|(\\[A-Za-z]+)|([A-Za-z0-9]))"  # {group}, \command or a single character


def _arg(m, i):
    return next(g for g in m.groups()[i:i + 3] if g is not None)


def inline_math(expr: str) -> str:
    """Small LaTeX→Unicode conversion for inline $...$ in teaching notes (display math is rendered as images)."""
    s = expr
    s = re.sub(r"\\mathbb\{([A-Z])\}", lambda m: BLACKBOARD.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\(?:mathrm|text|mathbf|mathcal|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\[td]?frac(?:12|\{1\}\{2\})", "½", s)
    s = re.sub(r"\\[td]?frac\{([^{}]*)\}\{([^{}]*)\}",
               lambda m: "/".join(p if re.fullmatch(r"[\w\\|]+", p) else f"({p})" for p in (m.group(1), m.group(2))), s)
    s = re.sub(r"\\(hat|bar|tilde)\s*" + ARG, lambda m: inline_math(_arg(m, 1)) + ACCENT[m.group(1)], s)
    s = re.sub(r"\\sqrt\s*" + ARG, lambda m: "√" + (lambda a: a if len(a) == 1 else f"({a})")(inline_math(_arg(m, 0))), s)
    s = s.replace(r"\|", "‖").replace(r"\,", " ").replace(r"\;", " ").replace(r"\ ", " ").replace(r"\!", "")
    def symbol(m):
        out = SYMBOLS.get(m.group(1), m.group(1))
        glued = out[:1].isalpha() and m.start() > 0 and s[m.start() - 1].isalnum()
        return " " + out if glued else out  # a\log y -> "a log y", not "alog y"
    s = re.sub(r"\\([A-Za-z]+)", symbol, s).replace("^ᵀ", "ᵀ")
    s = re.sub(r"_\{([^}]*)\}", lambda m: m.group(1).translate(SUB) if all(c in "0123456789aehijklmnoprstuvx+-" for c in m.group(1)) else "_" + m.group(1), s)
    s = re.sub(r"_([0-9a-z])", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\^\{([^}]*)\}", lambda m: m.group(1).translate(SUP) if all(c in "0123456789+-nTi" for c in m.group(1))
               else "^" + (m.group(1) if len(m.group(1)) == 1 else f"({m.group(1)})"), s)
    s = re.sub(r"\^([0-9nTi])", lambda m: m.group(1).translate(SUP), s)
    return s.replace("{", "").replace("}", "").replace("\\", "")


def prepare_notes(md: str, eqs: dict) -> str:
    def block(m):
        latex = " ".join(m.group(1).split())
        p = render_equation(latex, fontsize=18)
        key = p.stem
        eqs[key] = asset(p)
        return f"\n\n![](eq:{key})\n\n"

    md = re.sub(r"\$\$(.+?)\$\$", block, md, flags=re.S)
    # Conditional-probability bars must not become Markdown table separators.
    md = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: "*" + inline_math(m.group(1)).replace("|", r"\|" ) + "*", md)
    return md


# ---------------------------------------------------------------- notebooks
CELL_RE = re.compile(r"^# %%(?P<md> \[markdown\])?(?P<meta>.*)$")


def parse_percent(text: str) -> list[dict]:
    cells, cur = [], None
    for line in text.splitlines():
        m = CELL_RE.match(line)
        if m:
            if cur:
                cells.append(cur)
            tags = []
            tm = re.search(r"tags=(\[.*?\])", m.group("meta"))
            if tm:
                tags = json.loads(tm.group(1))
            cur = {"type": "markdown" if m.group("md") else "code", "lines": [], "tags": tags}
            continue
        if cur is None:
            continue
        if cur["type"] == "markdown":
            line = line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    for c in cells:  # trim blank edges
        while c["lines"] and not c["lines"][-1].strip():
            c["lines"].pop()
        while c["lines"] and not c["lines"][0].strip():
            c["lines"].pop(0)
    return [c for c in cells if c["lines"]]


PLACEHOLDER = "pass  # TODO: write your code here"


def student_code(lines: list[str]) -> list[str]:
    """Replace solution blocks with stubs. A '### STUB code' line inside a block, or directly after its END marker,
    becomes runnable starter code in place of the block; blocks without a stub become a `pass` placeholder."""
    out, in_sol, stubs, indent = [], False, [], ""
    for ln in lines:
        s = ln.strip()
        if s == "### BEGIN SOLUTION":
            in_sol, stubs, indent = True, [], ln[: len(ln) - len(ln.lstrip())]
            continue
        if s == "### END SOLUTION":
            out.extend(stubs or [indent + PLACEHOLDER])
            in_sol = False
            continue
        if in_sol:
            if s.startswith("### STUB"):
                stubs.append(indent + s[len("### STUB"):].strip())
            continue
        if s.startswith("### STUB"):
            if out and out[-1].strip() == PLACEHOLDER:
                out.pop()
            out.append(ln[: len(ln) - len(ln.lstrip())] + s[len("### STUB"):].strip())
            continue
        out.append(ln)
    return out


def solution_code(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.strip() not in ("### BEGIN SOLUTION", "### END SOLUTION") and not ln.strip().startswith("### STUB")]


def student_md(lines: list[str]) -> list[str]:
    out, skip = [], False
    for ln in lines:
        if "<!-- BEGIN ANSWER -->" in ln:
            skip = True
            out.append("*✍️ Write your answer here.*")
            continue
        if "<!-- END ANSWER -->" in ln:
            skip = False
            continue
        if not skip:
            out.append(ln)
    return out


def solution_md(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if "<!-- BEGIN ANSWER -->" in ln:
            out.append("**✅ Model answer:**")
        elif "<!-- END ANSWER -->" in ln:
            continue
        else:
            out.append(ln)
    return out


RUN_BANNER = """> **How to run this notebook**
> - **Google Colab (recommended):** File ▸ Upload notebook, then Runtime ▸ Change runtime type ▸ **T4 GPU**. Run cells top to bottom with Shift+Enter.
> - **Local Jupyter / VS Code:** Python 3.10+; run the install cell once. A GPU is optional: every cell has a CPU-friendly setting.
> - **API keys (optional cells only):** store keys in Colab ▸ 🔑 Secrets or an environment variable. Never paste a key into a notebook you share.
> - Cells marked **TODO** are yours to complete. Questions marked ✍️ need a short written answer."""


def build_notebooks(n: int, wk: dict, outdir: Path) -> tuple[Path, Path]:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    src = CUR / "labs" / f"week_{n:02d}_lab.py"
    cells = parse_percent(src.read_text(encoding="utf-8"))
    figures = {k: Path(v["path"]) for k, v in wk.get("assets", {}).get("figures", {}).items() if k.startswith("beginner_")}
    if not figures and beginner.load(n):
        figures = beginner.diagram_assets(n)
    diagram_dir = outdir / "Diagrams"
    if figures:
        diagram_dir.mkdir(exist_ok=True)
        for fid, path in figures.items():
            shutil.copy2(path, diagram_dir / (fid + ".png"))
            shutil.copy2(path.with_suffix(".svg"), diagram_dir / (fid + ".svg"))
    paths = []
    for variant in ("student", "solution"):
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        }
        for i, c in enumerate(cells):
            if variant == "student" and "solution-only" in c["tags"]:
                continue  # instructor tooling (e.g. saving results for the slides)
            meta = {"tags": c["tags"]} if c["tags"] else {}
            if c["type"] == "markdown":
                lines = student_md(c["lines"]) if variant == "student" else solution_md(c["lines"])
                cell = new_markdown_cell("\n".join(lines), metadata=meta)
                beginner.attach_figures(cell, figures)
                nb.cells.append(cell)
                if i == 0:
                    banner = RUN_BANNER
                    if variant == "solution":
                        banner = "> **INSTRUCTOR VERSION — contains solutions. Do not distribute before the lab.**\n\n" + banner
                    nb.cells.append(new_markdown_cell(banner))
            else:
                lines = student_code(c["lines"]) if variant == "student" else solution_code(c["lines"])
                code = "\n".join(lines)
                try:  # both variants must at least be valid Python (magics such as %pip excluded)
                    compile("\n".join(ln for ln in lines if not ln.lstrip().startswith(("%", "!"))), f"{variant} cell {i}", "exec")
                except SyntaxError as e:
                    raise RuntimeError(f"week {n} {variant} notebook, cell {i}: {e}") from e
                if variant == "student" and "### STUB" in code:
                    raise RuntimeError(f"week {n} student notebook, cell {i}: unconverted ### STUB line")
                nb.cells.append(new_code_cell(code, metadata=meta))
        nbformat.validate(nb)
        name = f"Week_{n:02d}_Lab.ipynb" if variant == "student" else f"Week_{n:02d}_Lab_Solutions.ipynb"
        path = outdir / name
        nbformat.write(nb, str(path))
        beginner.python_export(nb, path.with_suffix(".py"))
        paths.append(path)
    stu = paths[0].read_text(encoding="utf-8")
    if "### BEGIN SOLUTION" in stu or "BEGIN ANSWER" in stu:
        raise RuntimeError("student notebook still contains solution markers")
    return paths[0], paths[1]


# ---------------------------------------------------------------- build
def build(n: int) -> dict:
    wk = load_week(n)
    errs = validate(wk)
    bdir = BUILD / f"week_{n:02d}"
    bdir.mkdir(parents=True, exist_ok=True)
    outdir = DELIV / f"Week_{n:02d}_{wk['slug']}"
    outdir.mkdir(parents=True, exist_ok=True)

    figs = {k: asset(p) for k, p in run_figures(n).items()}
    figs.update({k: asset(p) for k, p in beginner.diagram_assets(n).items()})
    eqs: dict = {}
    for s in wk["slides"]:
        for f in ("figure",):
            if s.get(f) and s[f] not in figs:
                errs.append(f"slide '{s.get('title', s['type'])}': unknown figure {s[f]!r}")
        if s["type"] == "equation":
            latex_list = s["eq"] if isinstance(s["eq"], list) else [s["eq"]]
            s["eq_assets"] = [asset(render_equation(l)) for l in latex_list]
    notes_md = prepare_notes(wk["notes_md"], eqs)
    for fid in re.findall(r"\]\(fig:([\w-]+)\)", notes_md):
        if fid not in figs:
            errs.append(f"notes: unknown figure {fid!r}")

    spec = {k: wk[k] for k in ("week", "topic", "slug", "subtitle", "detail", "tutorials", "mimlos", "objectives", "slides")}
    spec["notes_md"] = notes_md
    spec["assets"] = {"figures": figs, "eqs": eqs}
    wk["assets"] = spec["assets"]
    spec_path = bdir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")

    pptx = outdir / f"Week_{n:02d}_Lecture_Slides.pptx"
    docx = outdir / f"Week_{n:02d}_Teaching_Notes.docx"
    for script, out in (("slides.js", pptx), ("notes.js", docx)):
        r = subprocess.run(["node", str(TOOLS / script), str(spec_path), str(out)], capture_output=True, text=True, encoding="utf-8")
        print(r.stdout.strip())
        if r.returncode:
            errs.append(f"{script} failed: {r.stderr.strip()[-800:]}")
    try:
        build_notebooks(n, wk, outdir)
    except Exception as e:  # noqa: BLE001 - report every build problem in QA
        errs.append(f"notebook build failed: {e}")

    # coverage: every curated 7.3 term must appear in slides or notes
    text = (json.dumps(wk["slides"], ensure_ascii=False) + wk["notes_md"]).lower()
    missing = [t for t in wk.get("coverage", []) if t.lower() not in text]
    if missing:
        errs.append(f"7.3 coverage terms missing: {missing}")
    words = sum(len(str(s.get("notes", "")).split()) for s in wk["slides"])
    qa = {"week": n, "slides": len(wk["slides"]), "speaker_note_words": words,
          "notes_md_words": len(wk["notes_md"].split()), "errors": errs, "outdir": str(outdir)}
    (bdir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="all")
    ap.add_argument("--labs-only", action="store_true", help="only rebuild the notebooks (no slides, notes or figures)")
    a = ap.parse_args()
    weeks = sorted({w for part in a.weeks.split(",") for w in (range(1, 13) if part == "all" else range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
    bad = 0
    if a.labs_only:
        rows = {r["week"]: r for r in yaml.safe_load((CUR / "section_7_3.yaml").read_text(encoding="utf-8"))["weeks"]}
        for n in weeks:
            outdir = DELIV / f"Week_{n:02d}_{rows[n]['slug']}"
            outdir.mkdir(parents=True, exist_ok=True)
            print(f"week {n:02d}: notebooks ->", [p.name for p in build_notebooks(n, {}, outdir)])
        return 0
    for n in weeks:
        if not (CUR / "weeks" / f"week_{n:02d}.yaml").exists():
            print(f"week {n}: no source yet, skipped")
            continue
        qa = build(n)
        status = "OK" if not qa["errors"] else f"{len(qa['errors'])} issue(s)"
        print(f"week {n:02d}: {status} | {qa['slides']} slides | {qa['speaker_note_words']} note words | notes {qa['notes_md_words']} words")
        for e in qa["errors"]:
            print("   -", e)
        bad += bool(qa["errors"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
