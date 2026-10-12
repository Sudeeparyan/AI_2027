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
import base64
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
import lecture_flows
from output_files import atomic_text, atomic_copy
from speaker_notes import legacy_warning, validate_speaker_notes

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
    "course_map": ["title", "figure"], "infographic": ["title", "figure", "recap"],
}
REQUIRED_NOTE_SECTIONS = [
    "Lecture plan", "Lecture notes", "Common misconceptions", "Responsible AI", "Lab guide",
    "Practice questions", "Glossary", "Readings",
]
MIN_NOTE_WORDS = 45


# ---------------------------------------------------------------- loading
PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


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


def validate(wk: dict, *, strict_notes: bool | None = None,
             note_warnings: list[str] | None = None) -> list[str]:
    errs = []
    legacy_slides = []
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
        for f in ("bullets", "points"):
            if f in s and (not isinstance(s[f], list) or any(not isinstance(item, str) for item in s[f])):
                errs.append(f"slide {i} ({t}): {f} must be a list of strings; quote YAML text containing ': '")
        errs.extend(f"slide {i} ({t}): {error}" for error in lecture_flows.errors(s))
        note_check = validate_speaker_notes(str(s.get("notes", "")), strict_legacy=strict_notes,
                                            quiz_answer=s.get("answer") if t == "quiz" else None)
        errs.extend(f"slide {i} ({t}: {s.get('title', '')[:40]}): {error}" for error in note_check.errors)
        if note_check.warnings:
            legacy_slides.append(i)
    warning = legacy_warning(legacy_slides)
    if warning and note_warnings is not None:
        note_warnings.append(warning)
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
    h.update((ROOT / "tools" / "technical_diagrams.js").read_bytes())
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
    # Mathtext lets the integral sign's slanted tail run into the next symbol; a space plus a thin space clears it.
    latex = re.sub(r"\\int(?![A-Za-z])((?:\s*[_^](?:\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9]))*)\s*", r"\\int\1\\ \\, ", latex)
    key =hashlib.sha1(f"{fontsize}|{latex}".encode()).hexdigest()[:16]
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
    result = {"path": str(path), "w": w, "h": h}
    scene_path = path.with_suffix(".scene.json")
    if scene_path.exists():
        result["scene"] = json.loads(scene_path.read_text(encoding="utf-8"))
    return result


def lecture_flow_assets(n: int, wk: dict) -> dict[str, Path]:
    flows = lecture_flows.collect(wk)
    if not flows:
        return {}
    out = BUILD / "figures" / f"week_{n:02d}" / "lecture_flows"
    out.mkdir(parents=True, exist_ok=True)
    source = out / "flows.json"
    atomic_text(source, json.dumps({"flows": flows}, ensure_ascii=False, indent=1))
    subprocess.run(["node", str(TOOLS / "technical_diagrams.js"), "--flows", str(source), str(out)], check=True)
    return {entry["id"]: out / (entry["id"] + ".png") for entry in flows}


# Whole LaTeX command names -> Unicode (matched as complete words, so \le never eats \left or \leq).
SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
    "lambda": "λ", "mu": "μ", "sigma": "σ", "phi": "φ", "pi": "π", "tau": "τ", "eta": "η", "Sigma": "Σ", "Delta": "Δ",
    "nabla": "∇", "partial": "∂", "sum": "Σ", "prod": "Π", "int": "∫", "in": "∈", "sim": "∼", "approx": "≈",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "times": "×", "cdot": "·", "pm": "±", "to": "→",
    "infty": "∞", "mid": "|", "vert": "|", "Vert": "‖", "top": "ᵀ", "ldots": "…", "dots": "…", "varnothing": "∅", "emptyset": "∅",
    "log": "log", "ln": "ln", "exp": "exp", "sin": "sin", "cos": "cos", "sup": "sup", "max": "max", "min": "min",
    "arg": "arg", "left": "", "right": "", "quad": " ", "qquad": "  ",
}
STRUCTURAL = {"mathrm", "text", "mathbf", "mathcal", "operatorname", "mathbb", "hat", "bar", "tilde", "frac", "tfrac", "dfrac", "sqrt"}
GREEK = {"\\" + k: v for k, v in SYMBOLS.items()}  # kept for tools that list the supported commands
BLACKBOARD = {"E": "𝔼", "R": "ℝ", "N": "ℕ", "Z": "ℤ", "P": "ℙ"}
ACCENT = {"hat": "\u0302", "bar": "\u0304", "tilde": "\u0303"}
SUB_CHARS, SUP_CHARS = "0123456789aehijklmnoprstuvx+-=()", "0123456789+-=()nTi"
SUB = str.maketrans(SUB_CHARS, "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎")
SUP = str.maketrans(SUP_CHARS, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀⁱ")


def script(content: str, tag: str) -> str:
    """Unicode sub/superscript when every character has one; otherwise <sub>/<sup>,
    which tools/notes.js renders as real Word subscript or superscript text."""
    chars, table = (SUB_CHARS, SUB) if tag == "sub" else (SUP_CHARS, SUP)
    content = content.strip()
    if content and all(c in chars for c in content):
        return content.translate(table)
    return f"<{tag}>{content}</{tag}>"
ARG = r"(?:\{([^{}]*)\}|(\\[A-Za-z]+)|([A-Za-z0-9]))"  # {group}, \command or a single character


def _arg(m, i):
    return next(g for g in m.groups()[i:i + 3] if g is not None)


def inline_math(expr: str) -> str:
    """Small LaTeX→Unicode conversion for inline $...$ in teaching notes (display math is rendered as images)."""
    s = expr.replace(r"\{", "\x01").replace(r"\}", "\x02")  # literal braces, as in U\{1, \dots, T\}
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
    # Single characters first (p_g inside E_{p_g}), then braced groups.
    s = re.sub(r"_([^\s{}\\_^<])", lambda m: script(m.group(1), "sub"), s)
    s = re.sub(r"\^([^\s{}\\_^<])", lambda m: script(m.group(1), "sup"), s)
    while True:  # innermost groups first, so E_{p_{data}} also converts its outer group
        before = s
        s = re.sub(r"_\{([^{}]*)\}", lambda m: script(m.group(1), "sub"), s)
        s = re.sub(r"\^\{([^{}]*)\}", lambda m: script(m.group(1), "sup"), s)
        if s == before:
            break
    s = s.replace("{", "").replace("}", "").replace("\\", "").replace("\x01", "{").replace("\x02", "}")
    # A hyphen between two letters is a word hyphen (as in \text{x-axis}); any other "-" is a minus sign.
    return re.sub(r"(?<![A-Za-z])-|-(?![A-Za-z])", "−", s)


# Inline maths follows Pandoc's rule, so prices such as $0.10 / $0.40 stay text: the opening $ has a
# non-space on its right, the closing $ a non-space on its left and no digit after it.
INLINE_MATH = re.compile(r"(?<![\\$])\$(?=\S)([^$\n]+?)(?<=\S)\$(?!\d)")
# Projected text only: speaker notes, code, LaTeX equations and identifiers are left alone.
SLIDE_TEXT_SKIP = {"notes", "code", "eq", "figure", "icon", "url", "id", "image"}


def slide_math(node, key: str = ""):
    """Convert $...$ in projected slide text to Unicode; <sub>/<sup> where Unicode has no script letter."""
    if key in SLIDE_TEXT_SKIP:
        return node
    if isinstance(node, str):
        return INLINE_MATH.sub(lambda m: inline_math(m.group(1)), node)
    if isinstance(node, dict):
        return {k: slide_math(v, k) for k, v in node.items()}
    if isinstance(node, list):
        return [slide_math(v, key) for v in node]
    return node


def prepare_notes(md: str, eqs: dict) -> str:
    def block(m):
        latex = " ".join(m.group(1).split())
        p = render_equation(latex, fontsize=18)
        key = p.stem
        eqs[key] = asset(p)
        return f"\n\n![](eq:{key})\n\n"

    md = re.sub(r"\$\$(.+?)\$\$", block, md, flags=re.S)
    # Conditional-probability bars must not become Markdown table separators.
    # The converted maths is wrapped in <var>...</var>; tools/notes.js sets single-letter variables in italics and
    # keeps digits, operators and words such as log upright. Its own | and * (as in D*) are escaped for Markdown.
    md = INLINE_MATH.sub(lambda m: "<var>" + inline_math(m.group(1)).replace("|", r"\|").replace("*", r"\*") + "</var>", md)
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
> - **Google Colab:** File ▸ Upload notebook. Keep the CPU runtime for the core path. Run cells top to bottom with Shift+Enter. Free resources vary; there is no guaranteed GPU.
> - **Local Jupyter / VS Code:** select a Python 3.11+ kernel with the listed packages. Run the install cell once if needed.
> - **Optional routes:** leave `USE_GPU`, `USE_GEMINI` and `USE_OPENAI` false for the complete local CPU lab. To compare a hosted answer, enable exactly one API switch in shared setup.
> - **Gemini:** use synthetic prompts only. Store `GEMINI_API_KEY` in Colab Secrets or the environment. Account, age, region, quota and distribution terms affect eligibility; see the platform note. Never enable billing for this lab.
> - **OpenAI API:** an optional paid alternative, using `OPENAI_API_KEY` from Secrets or the environment. A ChatGPT subscription does not supply API credit. Check your account's API pricing and budget first. Local users can point `LOCAL_ENV_FILE` to a private `.env`; never upload it or paste keys into notebook cells.
> - Cells marked **TODO** are yours to complete before running dependent cells. Questions marked ✍️ need a short written answer.
> - Read each diagram by following its numbered blocks. The solid arrows carry data to the next block. A dashed arrow shows a step that repeats."""

LAB_PINS = {
    "torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.18.0", "diffusers": "0.40.0",
    "peft": "0.21.1", "sentence-transformers": "5.7.0", "ipywidgets": "7.7.1",
    "google-genai": "2.29.0", "trl": "1.15.0", "openai": "3.28.0", "python-dotenv": "1.2.4",
}


def install_cell(code: str, *, first: bool) -> str:
    """Use the dated platform inventory; never silently replace local test packages."""
    value_flags = {"-i", "--index-url", "--extra-index-url", "--trusted-host", "-f", "--find-links",
                   "--cache-dir", "-t", "--target", "--root", "--prefix", "--timeout", "--retries",
                   "--proxy", "--cert", "--client-cert", "-c", "--constraint", "-r", "--requirement",
                   "--only-binary", "--no-binary", "--platform", "--python-version", "--implementation", "--abi"}
    output = []
    for line in code.splitlines():
        if not re.match(r"^\s*[%!]pip\s+install\b", line):
            output.append(line)
            continue
        replacements = []
        skip_value = False
        for token in re.finditer(r'''"[^"]*"|'[^']*'|\S+''', line):
            raw = token[0]
            quote = raw[0] if raw[:1] in {"'", '"'} and raw[-1:] == raw[:1] else ""
            value = raw[1:-1] if quote else raw
            if value.startswith("#"):
                break
            if skip_value:
                skip_value = False
                continue
            if value in value_flags:
                skip_value = True
                continue
            requirement = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)(\[[^\]]+\])?(?:[<>=!~][^;]*)?(;.*)?", value)
            if requirement:
                name = requirement[1].lower().replace("_", "-").replace(".", "-")
                if name in LAB_PINS:
                    pinned = requirement[1] + (requirement[2] or "") + "==" + LAB_PINS[name] + (requirement[3] or "")
                    replacements.append((token.start(), token.end(), quote + pinned + quote))
        for start, end, value in reversed(replacements):
            line = line[:start] + value + line[end:]
        output.append(line)
    code = "\n".join(output)
    if first:
        code += "\n%pip install -q google-genai==2.29.0 ipywidgets==7.7.1 openai==3.28.0 python-dotenv==1.2.4"
    return code


SHARED_SETUP_INTRO = """### Shared setup: choose, reproduce, check

This cell chooses CPU computation, fixes the random seed and prints package versions.
Expect a runtime record and no API request. `check()` prints helpful feedback.

Leave `USE_GEMINI = False` and `USE_OPENAI = False` for the complete local lab.
For an optional comparison, enable exactly one switch and rerun this setup cell.
`ask_model()` selects Gemini (`gemini-3.5-flash-lite`) or the OpenAI API (`gpt-4.1-mini`).
Both use a bounded retry, a shared 40-attempt ceiling and a local response cache.
Free quotas are account-specific; a failed request returns `None` for the local fallback.
Use only invented examples. Students in the EEA, UK and Switzerland should check
the [Gemini terms](https://ai.google.dev/gemini-api/terms) before opting in:
API-client distribution has a separate paid-service condition. Local tasks need no key or billing.
OpenAI API usage may incur charges, separately from ChatGPT subscriptions; check
[API pricing](https://developers.openai.com/api/docs/pricing) before opting in.
Use `GEMINI_API_KEY` or `OPENAI_API_KEY` in Colab Secrets or your local environment.
For local `.env` use, set `LOCAL_ENV_FILE` to that file's path. Secrets are read only after opt-in.
Use invented prompts only. OpenAI requests set `store=False`; this is not a promise of zero retention.

**Common errors:** a missing optional key means use the local result; a package import
error means check the selected kernel and install cell. Change only one switch at a time."""


def build_notebooks(n: int, wk: dict, outdir: Path) -> tuple[Path, Path]:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    src = CUR / "labs" / f"week_{n:02d}_lab.py"
    if not wk.get("beginner"):
        wk["beginner"] = beginner.load(n) or {}
    cells = parse_percent(src.read_text(encoding="utf-8"))
    available = {k: Path(v["path"]) for k, v in wk.get("assets", {}).get("figures", {}).items()}
    figures = {k: path for k, path in available.items() if k.startswith(("beginner_", "lecture_flow_"))}
    if not any(k.startswith("beginner_") for k in figures) and beginner.load(n):
        figures.update(beginner.diagram_assets(n))
    if "assets" not in wk:
        # --labs-only still resolves explicitly authored lecture-flow images.
        figures.update(lecture_flow_assets(n, wk if wk.get("slides") else load_week(n)))
    named_figures = {identifier for cell in cells if cell["type"] == "markdown"
                     for identifier in re.findall(r"\(fig:([\w-]+)\)", "\n".join(cell["lines"]))}
    extra_figures = named_figures - set(figures)
    if extra_figures and "assets" not in wk:
        available.update(run_figures(n))  # cached generated lecture figures also work with --labs-only
    for identifier in extra_figures:
        if identifier not in available:
            raise ValueError(f"Notebook references unknown figure {identifier}")
        figures[identifier] = available[identifier]
    diagram_dir = outdir / "Diagrams"
    if figures:
        diagram_dir.mkdir(exist_ok=True)
        export_diagrams = BUILD / f"week_{n:02d}" / "Diagrams"
        export_diagrams.mkdir(parents=True, exist_ok=True)
        manifest = BUILD / f"week_{n:02d}" / "notebook_assets.json"
        previous = set(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.exists() else set()
        for directory in (diagram_dir, export_diagrams):
            for old in directory.iterdir():
                if re.fullmatch(r"lecture_flow_\d+\.(?:png|svg)", old.name) and old.stem not in figures:
                    old.unlink()  # only stale generator-owned lecture-flow exports
            for identifier in previous - set(figures):
                for extension in ("png", "svg"):
                    (directory / f"{identifier}.{extension}").unlink(missing_ok=True)
        for fid, path in figures.items():
            atomic_copy(path, diagram_dir / (fid + ".png"))
            if path.with_suffix(".svg").exists():
                atomic_copy(path.with_suffix(".svg"), diagram_dir / (fid + ".svg"))
            else:
                # A stored model output is a raster image, unlike the editable diagrams.
                # Embed its exact PNG bytes in an SVG container without changing the archive.
                from PIL import Image
                with Image.open(path) as picture:
                    width, height = picture.size
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                atomic_text(diagram_dir / (fid + ".svg"),
                            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                            f'viewBox="0 0 {width} {height}"><image width="{width}" height="{height}" '
                            f'href="data:image/png;base64,{encoded}"/></svg>\n')
            atomic_copy(path, export_diagrams / (fid + ".png"))
        atomic_text(manifest, json.dumps(sorted(figures), indent=2) + "\n")
    paths = []
    for variant in ("student", "solution"):
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        }
        setup_added = False
        install_added = False
        for i, c in enumerate(cells):
            if variant == "student" and "solution-only" in c["tags"]:
                continue  # instructor tooling (e.g. saving results for the slides)
            meta = {"tags": c["tags"]} if c["tags"] else {}
            if c["type"] == "markdown":
                lines = student_md(c["lines"]) if variant == "student" else solution_md(c["lines"])
                cell = new_markdown_cell("\n".join(lines), metadata=meta)
                beginner.attach_figures(cell, figures)
                nb.cells.append(cell)
                explicitly_placed = "fig:beginner_mechanism" in src.read_text(encoding="utf-8")
                if "attachment:beginner_overview.png" in cell.source and not explicitly_placed:
                    for diagram in wk.get("beginner", {}).get("diagrams", []):
                        if diagram["id"] in ("overview", "lab"):
                            continue
                        guide = new_markdown_cell(
                            f"### {diagram['title']}\n\n{diagram['subtitle']}\n\n"
                            f"![{diagram['title']}](fig:beginner_{diagram['id']})\n\n"
                            + "\n".join(f"{j}. {step}" for j, step in enumerate(diagram["trace"], 1))
                            + f"\n\n**Think before running:** {diagram['check_question']}"
                            + (f"\n\n**Instructor explanation:** {diagram['check_answer']}" if variant == "solution" else "")
                        )
                        beginner.attach_figures(guide, figures)
                        nb.cells.append(guide)
                if i == 0:
                    banner = RUN_BANNER
                    if variant == "solution":
                        banner = "> **INSTRUCTOR VERSION — contains solutions. Do not distribute before the lab.**\n\n" + banner
                    nb.cells.append(new_markdown_cell(banner))
            else:
                lines = student_code(c["lines"]) if variant == "student" else solution_code(c["lines"])
                code = "\n".join(lines)
                is_install = any(line.lstrip().startswith(("%pip ", "!pip ")) for line in lines)
                if is_install:
                    code = install_cell(code, first=not install_added)
                    install_added = True
                    meta = {"tags": sorted(set(c["tags"]) | {"colab-install"})}
                elif not setup_added:
                    intro = nb.cells.pop() if nb.cells and nb.cells[-1].cell_type == "markdown" else new_markdown_cell(
                        "Run this cell and inspect its printed output.")
                    nb.cells.append(new_markdown_cell(SHARED_SETUP_INTRO))
                    nb.cells.append(new_code_cell((CUR / "labs" / "_setup.py").read_text(encoding="utf-8"),
                                                  metadata={"tags": ["shared-setup"]}))
                    # Preserve the original adjacent introduction for this first lab code cell.
                    nb.cells.append(intro)
                    setup_added = True
                try:  # both variants must at least be valid Python (magics such as %pip excluded)
                    compile("\n".join(ln for ln in lines if not ln.lstrip().startswith(("%", "!"))), f"{variant} cell {i}", "exec")
                except SyntaxError as e:
                    raise RuntimeError(f"week {n} {variant} notebook, cell {i}: {e}") from e
                if variant == "student" and "### STUB" in code:
                    raise RuntimeError(f"week {n} student notebook, cell {i}: unconverted ### STUB line")
                nb.cells.append(new_code_cell(code, metadata=meta))
        # Recaps travel with the notebook and finish the lab as well as class.
        if "beginner_course_map" in figures:
            map_cell = new_markdown_cell("## This week's place in the course\n\n![Course map](fig:beginner_course_map)")
            beginner.attach_figures(map_cell, figures)
            nb.cells.insert(2, map_cell)
        recap = wk["beginner"].get("recap")
        if recap and "beginner_recap" in figures:
            recap_cell = new_markdown_cell(f"## {recap['title']}\n\n![Class recap](fig:beginner_recap)\n\n"
                + recap["example"] + "\n\n**Explain without looking:** " + recap["question"]
                + ("\n\n**Instructor answer:** " + recap["answer"] if variant == "solution" else ""))
            beginner.attach_figures(recap_cell, figures)
            nb.cells.append(recap_cell)
        # nbformat gives each cell a random id, so every rebuild changed every notebook's bytes. Numbered ids keep
        # an unchanged lab byte-identical, which keeps git diffs and test_labs.py notebook hashes meaningful.
        for k, cell in enumerate(nb.cells):
            cell["id"] = f"cell-{k:03d}"
        nbformat.validate(nb)
        name = f"Week_{n:02d}_Lab.ipynb" if variant == "student" else f"Week_{n:02d}_Lab_Solutions.ipynb"
        path = outdir / name
        atomic_text(path, nbformat.writes(nb) + "\n")
        export_dir = BUILD / f"week_{n:02d}"
        export_dir.mkdir(parents=True, exist_ok=True)
        beginner.python_export(nb, export_dir / path.with_suffix(".py").name)
        # These two generator-owned exports have moved to build/, as agreed in D3.
        path.with_suffix(".py").unlink(missing_ok=True)
        paths.append(path)
    stu = paths[0].read_text(encoding="utf-8")
    if "### BEGIN SOLUTION" in stu or "BEGIN ANSWER" in stu:
        raise RuntimeError("student notebook still contains solution markers")
    return paths[0], paths[1]


# ---------------------------------------------------------------- build
def build(n: int, *, strict_notes: bool | None = None) -> dict:
    wk = load_week(n)
    note_warnings = []
    errs = validate(wk, strict_notes=strict_notes, note_warnings=note_warnings)
    wk["slides"] = [slide_math(s) for s in wk["slides"]]
    bdir = BUILD / f"week_{n:02d}"
    bdir.mkdir(parents=True, exist_ok=True)
    outdir = DELIV / f"Week_{n:02d}_{wk['slug']}"
    outdir.mkdir(parents=True, exist_ok=True)

    figs = {k: asset(p) for k, p in run_figures(n).items()}
    figs.update({k: asset(p) for k, p in beginner.diagram_assets(n).items()})
    figs.update({k: asset(p) for k, p in lecture_flow_assets(n, wk).items()})
    eqs: dict = {}
    for s in wk["slides"]:
        for f in ("figure",):
            if s.get(f) and s[f] not in figs:
                errs.append(f"slide '{s.get('title', s['type'])}': unknown figure {s[f]!r}")
        if s["type"] == "equation":
            latex_list = s["eq"] if isinstance(s["eq"], list) else [s["eq"]]
            s["eq_assets"] = [asset(render_equation(l)) for l in latex_list]
    notes_md = prepare_notes(lecture_flows.notes_with_flows(wk["notes_md"], wk["slides"]), eqs)
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
          "notes_md_words": len(wk["notes_md"].split()), "errors": errs,
          "warnings": note_warnings, "outdir": str(outdir)}
    (bdir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="all")
    ap.add_argument("--labs-only", action="store_true", help="only rebuild the notebooks (no slides, notes or figures)")
    ap.add_argument("--strict-speaker-notes", action="store_true", default=None,
                    help="reject legacy speaker-note templates instead of warning during migration")
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
        qa = build(n, strict_notes=a.strict_speaker_notes)
        status = "OK" if not qa["errors"] else f"{len(qa['errors'])} issue(s)"
        print(f"week {n:02d}: {status} | {qa['slides']} slides | {qa['speaker_note_words']} note words | notes {qa['notes_md_words']} words")
        for warning in qa.get("warnings", []):
            print("   WARNING:", warning)
        for e in qa["errors"]:
            print("   -", e)
        bad += bool(qa["errors"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
