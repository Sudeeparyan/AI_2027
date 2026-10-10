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
    glossary = ROOT / "curriculum" / "glossary.json"
    if glossary.exists():
        meanings = {p["term"].casefold(): p["meaning"] for p in json.loads(glossary.read_text(encoding="utf-8"))["terms"]}
        for item in data["prerequisites"] + data["glossary"]:
            item["meaning"] = meanings.get(item["term"].casefold(), item["meaning"])
    if data["week"] != n:
        raise ValueError(f"{path}: wrong week")
    ids = [d["id"] for d in data["diagrams"]]
    if len(ids) != len(set(ids)) or not {"overview", "mechanism", "lab"}.issubset(ids):
        raise ValueError(f"{path}: unique IDs and overview, mechanism and lab diagrams are required")
    for d in data["diagrams"]:
        if not 4 <= len(d["steps"]) <= 7 or len(d["arrows"]) != len(d["steps"]) - 1:
            raise ValueError(f"{path}: invalid steps or connectors in {d['id']}")
        for step in d["steps"]:
            if step.get("role", "model") not in {"input", "data", "model", "loss", "output", "human", "tool"}:
                raise ValueError(f"{path}: unknown block role in {d['id']}")
        for branch in d.get("branches", []):
            if (not all(isinstance(branch.get(k), int) and 0 <= branch[k] < len(d["steps"]) for k in ("from", "to"))
                    or branch["from"] == branch["to"] or not branch.get("label")
                    or branch.get("lane", "top") not in {"top", "bottom"}):
                raise ValueError(f"{path}: invalid branch in {d['id']}")
        if d.get("feedback") and not 0 <= d.get("feedback_to", 0) < len(d["steps"]):
            raise ValueError(f"{path}: invalid feedback target")
        if d.get("feedback") and not 0 <= d.get("feedback_from", len(d["steps"]) - 1) < len(d["steps"]):
            raise ValueError(f"{path}: invalid feedback source")
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
    subprocess.run(["node", str(ROOT / "tools" / "teaching_support.js"), str(source), str(out)], check=True)
    data = load(n)
    assets = {f"beginner_{d['id']}": out / f"beginner_{d['id']}.png" for d in data["diagrams"]}
    assets["beginner_course_map"] = out / "beginner_course_map.png"
    if data.get("recap"):
        assets["beginner_recap"] = out / "beginner_recap.png"
    return assets


def remap_slide_references(md: str, mapping: dict[int, int]) -> str:
    """Remap plan-table slide cells and explicit 'slide(s) N' references."""
    def numbers(text: str) -> str:
        def reference(m):
            first = int(m[1])
            if m[3] is None:
                return str(mapping.get(first, first))
            last = int(m[3])
            resolved = [mapping.get(n, n) for n in range(first, last + 1)]
            return f"{min(resolved)}{m[2]}{max(resolved)}"
        return re.sub(r"\b(\d+)(?:([–-])(\d+))?\b", reference, text)

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
    parts = ["# Beginner walkthrough", "", "## Where we are in the course", "",
             "![The twelve-week course map](fig:beginner_course_map)", "",
             "Read the overall flow before the detailed theory. Each block performs one operation. "
             "The text below a block names the data it passes on. A feedback line means the system repeats an operation. "
             "The code names match this week's lab so you can follow the same example in the slides and notebook.", "",
             "## Starting vocabulary", "", "| Term | Plain meaning | Check your understanding |", "|---|---|---|"]
    clean = lambda v: str(v).replace("|", "/").replace("\n", " ")
    for p in data["prerequisites"]:
        parts.append("| " + " | ".join(clean(p[k]) for k in ("term", "meaning", "check")) + " |")
    parts += ["", "**Answers to the checks.**", ""] + [f"- **{p['term']}:** {p['answer']}" for p in data["prerequisites"]]
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
    # Keep the authored summary's detailed teaching points in Word. The final
    # slide replaces that summary instead of increasing the deck length.
    recap_index = next((i for i, s in enumerate(original) if s["type"] == "summary"), None)
    recap_slide = None
    if data.get("recap") and recap_index is not None:
        old = original[recap_index]
        recap_slide = {"type": "infographic", "title": data["recap"]["title"],
                       "figure": "beginner_recap", "recap": data["recap"],
                       "notes": data["recap"]["notes"] + "\n\nExit question: " + data["recap"]["question"] +
                       "\nAnswer: " + data["recap"]["answer"] + "\n\n" + old.get("notes", "")}
        wk["notes_md"] += "\n# Class recap\n\n" + "\n".join("- " + p for p in old["points"]) + "\n\n"
        wk["notes_md"] += f"![{data['recap']['title']}](fig:beginner_recap)\n\n"
        # The recap figure already prints the worked example and the next-week line; only the exit question is new.
        wk["notes_md"] += "**Exit question.** " + data["recap"]["question"]
        wk["notes_md"] += "\n\n**Answer.** " + data["recap"]["answer"] + "\n"
    diagrams = {d["id"]: d for d in data["diagrams"]}
    def diagram(kind: str) -> dict:
        d = diagrams[kind]
        return {"type": "technical", "title": d["title"], "diagram": d, "beginner_support": True,
                "notes": d["explanation"] + "\n\nWorked example:\n" + "\n".join(d["trace"]) +
                "\n\nCheck: " + d["check_question"] + "\nAnswer: " + d["check_answer"]}
    prereq = {"type": "table", "title": "Starting vocabulary", "header": ["Term", "Plain meaning", "Quick check"],
              "rows": [[p["term"], p["meaning"], p["check"]] for p in data["prerequisites"]], "max_font": 20,
              "col_widths": [22, 39, 39], "beginner_support": True,
              "notes": "Introduce these terms before the main theory. Give pairs one minute to answer the quick "
                  "checks, then take answers. " + " ".join(
                  p["term"] + ": " + p["meaning"] + " Check: " + p["check"] + " Answer: " + p["answer"]
                  for p in data["prerequisites"]) +
                  " Let students explain one example aloud, then connect their example to the next flow diagram. "
                  "Use the blocks to name the inputs and outputs before introducing notation."}
    lab_idx = next((i for i, s in enumerate(original) if s["type"] == "lab"), len(original) - 3)
    insert = {3: [prereq]}
    if not diagrams["overview"].get("insert_before_title"):
        insert[5] = [diagram("overview")]  # default: open the first section with the whole system
    for kind, d in diagrams.items():
        if kind == "lab" or (kind == "overview" and not d.get("insert_before_title")):
            continue
        matches = [i for i, s in enumerate(original) if s.get("title") == d.get("insert_before_title")]
        if len(matches) != 1:
            raise ValueError(f"Week {wk['week']}: {kind} must name one exact insertion title")
        insert.setdefault(matches[0], []).append(diagram(kind))
    insert.setdefault(lab_idx, []).append(diagram("lab"))
    slides, mapping = [], {}
    for i, s in enumerate(original):
        slides.extend(insert.get(i, []))
        mapping[i + 1] = len(slides) + 1
        if i == recap_index and recap_slide:
            continue
        if s["type"] == "agenda":
            s = {**s, "type": "course_map", "title": "Where we are in the course", "figure": "beginner_course_map"}
            s["notes"] += "\n\nTrace the highlighted week. Earlier concepts feed later systems. " \
                "Ask which earlier idea today needs. Common mistake: treating each week as an unrelated tool."
        slides.append(s)
    if recap_slide:
        mapping[recap_index + 1] = len(slides) + 1
        slides.append(recap_slide)
    wk["slides"] = slides
    wk["expected_slides"] = len(slides)
    wk["beginner"] = data
    wk["base_slide_map"] = mapping
    md = remap_slide_references(wk["notes_md"], mapping)
    # Include support slides in the corresponding plan-table ranges.
    for pos, supports in sorted(insert.items(), reverse=True):
        start = mapping[pos + 1]
        pattern = rf"(?m)^(\|[^\n]*?\|\s*){start}([–\-]\d+\s*\|)"
        md = re.sub(pattern, lambda m: m[1] + str(start - len(supports)) + m[2], md)
    plan_note = (f"\nThe {len(slides) - len(original)} supporting slides fit within the existing lecture segments. "
                 "Use the starting vocabulary and overall flow during the introduction, the mechanism diagram "
                 "beside the detailed explanation, the build and use diagrams beside their topics, "
                 "and the lab flow during the lab preview. "
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
            parts += ["", "# %% [markdown]"] + ["# " + ln if ln else "#" for ln in source.splitlines()]
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
