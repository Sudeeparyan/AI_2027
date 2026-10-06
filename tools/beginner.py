"""Shared beginner explanations for slides, notes, diagrams and lab notebooks.

The authored JSON in curriculum/beginner is the single source for the new
technical walkthroughs. Existing lecture material remains part of each pack.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(n: int) -> dict | None:
    path = ROOT / "curriculum" / "beginner" / f"week_{n:02d}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["week"] != n:
        raise ValueError(f"{path}: wrong week")
    if {d["id"] for d in data["diagrams"]} != {"overview", "mechanism", "lab"} or len(data["diagrams"]) != 3:
        raise ValueError(f"{path}: expected overview, mechanism and lab diagrams")
    for d in data["diagrams"]:
        if not 4 <= len(d["steps"]) <= 6 or len(d["arrows"]) != len(d["steps"]) - 1:
            raise ValueError(f"{path}: invalid steps or connectors in {d['id']}")
        if d.get("feedback") and not 0 <= d.get("feedback_to", 0) < len(d["steps"]):
            raise ValueError(f"{path}: invalid feedback target")
        for key in ("title", "subtitle", "explanation", "trace", "check_question", "check_answer"):
            if not d.get(key):
                raise ValueError(f"{path}: missing {key} in {d['id']}")
    return data


def diagram_assets(n: int) -> dict[str, Path]:
    source = ROOT / "curriculum" / "beginner" / f"week_{n:02d}.json"
    if not source.exists():
        return {}
    out = ROOT / "build" / "figures" / f"week_{n:02d}" / "beginner"
    subprocess.run(["node", str(ROOT / "tools" / "technical_diagrams.js"), str(source), str(out)], check=True)
    return {f"beginner_{d['id']}": out / f"beginner_{d['id']}.png" for d in load(n)["diagrams"]}


def remap_slide_references(md: str, mapping: dict[int, int]) -> str:
    """Remap plan-table slide cells and explicit 'slide(s) N' references."""
    def numbers(text: str) -> str:
        return re.sub(r"\b\d+\b", lambda m: str(mapping.get(int(m[0]), int(m[0]))), text)

    md = re.sub(r"(?i)\bslides?\s+\d+(?:[–\-]\d+)?", lambda m: numbers(m[0]), md)
    lines = md.splitlines()
    in_plan = False
    slide_column = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            in_plan = "lecture plan" in line.lower()
            slide_column = None
        if not in_plan or not line.startswith("|"):
            continue
        cells = line.split("|")
        if any(c.strip().lower() == "slides" for c in cells):
            slide_column = next(j for j, c in enumerate(cells) if c.strip().lower() == "slides")
        elif slide_column is not None and slide_column < len(cells):
            cells[slide_column] = numbers(cells[slide_column])
            lines[i] = "|".join(cells)
    return "\n".join(lines) + "\n"


def notes_section(data: dict) -> str:
    parts = ["# Beginner walkthrough", "",
             "Read the overall flow before the detailed theory. Each numbered block performs one operation. "
             "The text below a block names the data it passes on. A feedback line means the system repeats an operation. "
             "The code names match this week's lab so you can follow the same example in the slides and notebook.", "",
             "## Starting vocabulary", "", "| Term | Plain meaning | Check your understanding |", "|---|---|---|"]
    clean = lambda v: str(v).replace("|", "/").replace("\n", " ")
    for p in data["prerequisites"]:
        parts.append("| " + " | ".join(clean(p[k]) for k in ("term", "meaning", "check")) + " |")
    for d in data["diagrams"]:
        parts += ["", f"## {d['title']}", "", d["subtitle"], "",
                  f"![{d['title']}](fig:beginner_{d['id']})", "", d["explanation"], "", "### Worked example", ""]
        parts += [f"{i}. {t}" for i, t in enumerate(d["trace"], 1)]
        parts += ["", f"**Check your understanding.** {d['check_question']}", "",
                  f"**Instructor explanation.** {d['check_answer']}"]
    parts += ["", "## Where to look in the code", "", "| Code symbol | What it does | What to inspect |", "|---|---|---|"]
    for p in data["code_walkthrough"]:
        parts.append("| " + " | ".join(clean(p[k]) for k in ("symbol", "role", "inspect")) + " |")
    parts += ["", "## Common points of confusion", ""] + [f"- {p}" for p in data["pitfalls"]]
    parts += ["", "## Quick glossary", ""]
    parts += [f"**{p['term']}.** {p['meaning']}\n" for p in data["glossary"]]
    return "\n".join(parts) + "\n\n"


def enrich(wk: dict) -> dict:
    data = load(wk["week"])
    if not data:
        return wk
    original = wk["slides"]
    diagrams = {d["id"]: d for d in data["diagrams"]}
    def diagram(kind: str) -> dict:
        d = diagrams[kind]
        return {"type": "technical", "title": d["title"], "diagram": d, "beginner_support": True,
                "notes": d["explanation"] + "\n\nWorked example:\n" + "\n".join(d["trace"]) +
                "\n\nCheck: " + d["check_question"] + "\nAnswer: " + d["check_answer"]}
    prereq = {"type": "table", "title": "Starting vocabulary", "header": ["Term", "Plain meaning"],
              "rows": [[p["term"], p["meaning"]] for p in data["prerequisites"]], "max_font": 22,
              "beginner_support": True,
              "notes": "Introduce these terms before the main theory. " + " ".join(
                  p["term"] + ": " + p["meaning"] + " Ask: " + p["check"] for p in data["prerequisites"]) +
                  " Let students explain one example aloud, then connect their example to the next flow diagram. "
                  "Use the numbered blocks to name the inputs and outputs before introducing notation."}
    mechanism_idx = next((i for i, s in enumerate(original) if s.get("title") == diagrams["mechanism"].get("insert_before_title")),
                         next((i for i, s in enumerate(original) if s["type"] == "equation"),
                         next((i for i, s in enumerate(original) if s["type"] == "figure"), 9))
                         )
    mechanism_idx = max(7, mechanism_idx)
    lab_idx = next((i for i, s in enumerate(original) if s["type"] == "lab"), len(original) - 3)
    insert = {3: [prereq], 5: [diagram("overview")]}
    insert.setdefault(mechanism_idx, []).append(diagram("mechanism"))
    insert.setdefault(lab_idx, []).append(diagram("lab"))
    slides, mapping = [], {}
    for i, s in enumerate(original):
        slides.extend(insert.get(i, []))
        mapping[i + 1] = len(slides) + 1
        slides.append(s)
    wk["slides"] = slides
    wk["expected_slides"] = len(original) + 4
    wk["beginner"] = data
    wk["base_slide_map"] = mapping
    md = remap_slide_references(wk["notes_md"], mapping)
    # Include support slides in the corresponding plan-table ranges.
    for pos, supports in sorted(insert.items(), reverse=True):
        start = mapping[pos + 1]
        pattern = rf"(?m)^(\|[^\n]*?\|\s*){start}([–\-]\d+\s*\|)"
        md = re.sub(pattern, lambda m: m[1] + str(start - len(supports)) + m[2], md)
    plan_note = ("\nThe four supporting slides fit within the existing lecture segments. "
                 "Use the starting vocabulary and overall flow during the introduction, the mechanism diagram "
                 "beside the first detailed explanation, and the lab flow during the lab preview. "
                 "Pause for predictions before showing a worked example.\n\n")
    pos = md.find("<!-- pagebreak -->")
    if pos >= 0:
        md = md[:pos] + plan_note + md[pos:]
    wk["notes_md"] = md.replace("# Lecture notes", notes_section(data) + "# Lecture notes", 1)
    return wk


def attach_figures(cell, figures: dict[str, Path]) -> None:
    def image_ref(match):
        fid = match[1]
        if fid not in figures:
            raise ValueError(f"Notebook references unknown figure {fid}")
        name = fid + ".png"
        cell.setdefault("attachments", {})[name] = {"image/png": base64.b64encode(figures[fid].read_bytes()).decode("ascii")}
        return "(attachment:" + name + ")"
    cell.source = re.sub(r"\(fig:([\w-]+)\)", image_ref, cell.source)


def python_export(nb, path: Path) -> None:
    """Export real Python with percent cells, portable install commands, and TODOs."""
    import shlex
    parts = ["# -*- coding: utf-8 -*-", "# Open this file in VS Code with the Jupyter extension, or run cells in order.",
             "# The notebook contains embedded diagrams. Standalone PNGs are in Diagrams/."]
    for c in nb.cells:
        if c.cell_type == "markdown":
            source = re.sub(r"attachment:(beginner_[\w-]+\.png)", r"Diagrams/\1", c.source)
            parts += ["", "# %% [markdown]"] + ["# " + ln for ln in source.splitlines()]
        else:
            lines = []
            for ln in c.source.splitlines():
                if ln.startswith("%pip "):
                    args = shlex.split(ln[len("%pip "):])
                    lines += ["import subprocess as _install_process", "import sys as _install_sys",
                              "_install_process.check_call([_install_sys.executable, '-m', 'pip'] + " + repr(args) + ")"]
                elif ln.lstrip().startswith(("%", "!")):
                    raise ValueError(f"Unsupported notebook magic in Python export: {ln}")
                else:
                    lines.append(ln)
            parts += ["", "# %%"] + lines
    text = "\n".join(parts) + "\n"
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")
