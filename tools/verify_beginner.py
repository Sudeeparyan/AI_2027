"""Audit the built beginner teaching packs without executing their lab code.

Run ``python tools/verify_beginner.py`` after building all weeks. The report is
written to build/beginner_verification.json. ``--self-test`` checks the small
reference-remapping, attachment, and portable-export helpers independently.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import shlex
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import yaml

import beginner

ROOT = Path(__file__).resolve().parents[1]
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}
DIAGRAM_IDS = {"overview", "mechanism", "lab"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKERS = re.compile(r"BEGIN\s+(?:SOLUTION|ANSWER)|END\s+(?:SOLUTION|ANSWER)|###\s*STUB")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def normalise(text: str) -> str:
    return " ".join(text.split())


def expected_export_source(nb) -> str:
    """Derive portable code from notebook code cells, independently of export I/O."""
    lines = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for line in cell.source.splitlines():
            if line.startswith("%pip "):
                arguments = shlex.split(line[5:])
                lines.extend([
                    "import subprocess as _install_process",
                    "import sys as _install_sys",
                    "_install_process.check_call([_install_sys.executable, '-m', 'pip'] + "
                    + repr(arguments) + ")",
                ])
            else:
                require(not line.lstrip().startswith(("%", "!")),
                        f"unsupported notebook magic: {line}")
                lines.append(line)
        lines.append("")
    return "\n".join(lines)


def ast_dump(source: str) -> str:
    return ast.dump(ast.parse(source), include_attributes=False)


def check_excluded_source_cells(nb, lab_source: str) -> int:
    """Catch leaked instructor cells even if their solution-only tag was lost."""
    headers = list(re.finditer(r"(?m)^# %%[^\n]*\n", lab_source))
    notebook_statements = [ast.dump(statement, include_attributes=False)
                           for cell in nb.cells if cell.cell_type == "code"
                           for statement in ast.parse("\n".join(
                               line for line in cell.source.splitlines()
                               if not line.lstrip().startswith(("%", "!")))).body]
    excluded = 0
    for index, header in enumerate(headers):
        if "solution-only" not in header.group():
            continue
        excluded += 1
        end = headers[index + 1].start() if index + 1 < len(headers) else len(lab_source)
        body = lab_source[header.end():end]
        if "[markdown]" in header.group():
            markdown = "\n".join(line[2:] if line.startswith("# ") else line
                                 for line in body.splitlines()).strip()
            require(not any(normalise(markdown) in normalise(cell.source)
                            for cell in nb.cells if cell.cell_type == "markdown"),
                    "solution-only source markdown leaked without its tag")
        else:
            clean = "\n".join(line for line in body.splitlines()
                              if not line.strip().startswith(("### STUB", "%", "!")))
            statements = [ast.dump(node, include_attributes=False) for node in ast.parse(clean).body]
            if statements:
                require(not any(notebook_statements[start:start + len(statements)] == statements
                                for start in range(len(notebook_statements) - len(statements) + 1)),
                        "solution-only source code leaked without its tag")
    return excluded


def self_tests() -> list[str]:
    """Small regression cases protect numbers, image portability, and code fidelity."""
    passed = []
    # Builds map every original slide, so ranges resolve through each member.
    mapping = {n: n + 4 for n in range(1, 9)}
    source = (
        "# Lecture plan at a glance\n\n"
        "| Time | Slides | Segment |\n|---|---|---|\n"
        "| 0–5 min | 1–4 | 4 examples |\n"
        "| 5–8 min | 5-8 | 8 examples |\n\n"
        "See slide 1 and Slides 4–5; slide 99 is outside the map.\n"
        "Use T = 5, 8 epochs, and https://example.test/slides/4.\n\n"
        "# Lecture notes\n| Time | Slides |\n| 0–5 | 1–4 |\n"
    )
    result = beginner.remap_slide_references(source, mapping)
    require("| 0–5 min | 5–8 | 4 examples |" in result, "slide table range was not remapped")
    require("| 5–8 min | 9-12 | 8 examples |" in result, "hyphen range was not remapped")
    require("See slide 5 and Slides 8–9; slide 99" in result, "explicit slide references failed")
    require("T = 5, 8 epochs, and https://example.test/slides/4" in result,
            "unrelated numbers or URL were remapped")
    require("# Lecture notes\n| Time | Slides |\n| 0–5 | 1–4 |" in result,
            "non-plan table was remapped")
    passed.append("reference ranges remap; timings, ordinary numbers, URLs, and non-plan tables stay fixed")

    moved_column = "# Lecture plan\n| Segment | Time | Slides |\n| intro | 1–4 min | 1–4 |\n"
    require("| intro | 1–4 min | 5–8 |" in beginner.remap_slide_references(moved_column, mapping),
            "slide column position was assumed rather than read")
    passed.append("slide-column position is detected from the header")

    hidden_source = "# %% tags=[\"solution-only\"]\ninstructor_result = 123\n"
    clean_student = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("student_result = 0")])
    require(check_excluded_source_cells(clean_student, hidden_source) == 1, "excluded source cell was missed")
    leaked_student = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("instructor_result = 123")])
    try:
        check_excluded_source_cells(leaked_student, hidden_source)
    except AssertionError:
        pass
    else:
        raise AssertionError("tagless instructor code leak was not detected")
    passed.append("instructor-only source code is rejected even when its notebook tag was stripped")

    # A real minimal PNG makes attachment bytes independently checkable.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jJawAAAAASUVORK5CYII="
    )
    with tempfile.TemporaryDirectory(prefix="beginner-verification-") as temp:
        directory = Path(temp)
        figure = directory / "beginner_overview.png"
        figure.write_bytes(png)
        cell = nbformat.v4.new_markdown_cell("![flow](fig:beginner_overview)")
        beginner.attach_figures(cell, {"beginner_overview": figure})
        require(cell.source == "![flow](attachment:beginner_overview.png)", "attachment URL was not rewritten")
        require(base64.b64decode(cell.attachments[figure.name]["image/png"]) == png,
                "attachment bytes differ from source PNG")
        unknown = nbformat.v4.new_markdown_cell("![missing](fig:unknown)")
        try:
            beginner.attach_figures(unknown, {"beginner_overview": figure})
        except ValueError:
            pass
        else:
            raise AssertionError("unknown figure silently accepted")
        passed.append("PNG attachments resolve exactly; unknown figures raise an error")

        nb = nbformat.v4.new_notebook(cells=[cell, nbformat.v4.new_code_cell(
            "%pip install -q 'some-package[extra]>=1.0'\n"
            "values = [1, 2, 3]\n"
            "def total(xs):\n    return sum(xs)\n"
        )])
        export = directory / "lab.py"
        beginner.python_export(nb, export)
        exported = export.read_text(encoding="utf-8")
        compile(exported, str(export), "exec")
        require("Diagrams/beginner_overview.png" in exported, "export image path is not portable")
        require(ast_dump(exported) == ast_dump(expected_export_source(nb)), "export changed notebook code")
        tree = ast.parse(exported)
        pip_call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute) and n.func.attr == "check_call")
        require(isinstance(pip_call.args[0], ast.BinOp), "pip argument list was not preserved")
        require(ast.literal_eval(pip_call.args[0].right) == ["install", "-q", "some-package[extra]>=1.0"],
                "quoted package constraint was changed")
        require("shell=True" not in exported, "portable pip export unexpectedly invokes a shell")
        passed.append("portable Python compiles, preserves code AST and quoted pip arguments, and links diagrams")

        unsupported = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("%time print(1)")])
        try:
            beginner.python_export(unsupported, directory / "unsupported.py")
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported export magic silently accepted")
        passed.append("unsupported notebook magics fail explicitly")
    return passed


def verify_schema(week: int, data: dict, original: dict) -> dict:
    require(data.get("week") == week, "beginner JSON has the wrong week")
    for key in ("prerequisites", "glossary", "pitfalls", "code_walkthrough"):
        require(isinstance(data.get(key), list) and bool(data[key]), f"missing or empty {key}")
    for key, fields in (("prerequisites", ("term", "meaning", "check", "answer")),
                        ("glossary", ("term", "meaning")),
                        ("code_walkthrough", ("symbol", "role", "inspect"))):
        for item in data[key]:
            require(all(isinstance(item.get(f), str) and item[f].strip() for f in fields),
                    f"invalid {key} entry: {item}")
    diagrams = data.get("diagrams", [])
    ids = [d.get("id") for d in diagrams]
    require(len(set(ids)) == len(ids) and DIAGRAM_IDS.issubset(ids),
            "expected unique IDs including overview, mechanism, and lab diagrams")
    for diagram in diagrams:
        for key in ("title", "subtitle", "explanation", "check_question", "check_answer"):
            require(isinstance(diagram.get(key), str) and diagram[key].strip(), f"missing diagram {key}")
        steps = diagram.get("steps", [])
        require(4 <= len(steps) <= 7, f"{diagram['id']}: expected 4–7 steps")
        require(len(diagram.get("arrows", [])) == len(steps) - 1,
                f"{diagram['id']}: wrong connector count")
        # null = deliberately no connector (an independent input such as random noise)
        require(all(a is None or (isinstance(a, str) and a.strip()) for a in diagram["arrows"]), "empty arrow label")
        touched = {i + j for i, a in enumerate(diagram["arrows"]) if a for j in (0, 1)}
        touched |= {b[k] for b in diagram.get("branches", []) for k in ("from", "to")}
        require(touched == set(range(len(steps))), f"{diagram['id']}: a block has no incoming or outgoing connector")
        require(3 <= len(diagram.get("trace", [])) <= 5, f"{diagram['id']}: expected 3–5 worked steps")
        require(all(isinstance(t, str) and t.strip() for t in diagram["trace"]), "empty trace step")
        require(60 <= len(diagram["explanation"].split()) <= 180,
                f"{diagram['id']}: explanation outside 60–180 words")
        for step in steps:
            require(all(isinstance(step.get(k), str) for k in ("title", "detail", "code")), "invalid step fields")
            require(bool(step["title"].strip()) and bool(step["detail"].strip()), "empty step text")
            require(len(step["detail"].split()) <= 12, f"{diagram['id']}: detail exceeds 12 words")
            require(step.get("role", "model") in {"input", "data", "model", "loss", "output", "human", "tool"},
                    f"{diagram['id']}: unknown block role")
        for branch in diagram.get("branches", []):
            require(all(isinstance(branch.get(k), int) and 0 <= branch[k] < len(steps) for k in ("from", "to")),
                    f"{diagram['id']}: branch endpoint outside diagram")
            require(branch["from"] != branch["to"] and bool(branch.get("label")), "invalid branch")
            require(branch.get("lane", "top") in {"top", "bottom"}, "invalid branch lane")
        if diagram.get("feedback"):
            destination = diagram.get("feedback_to")
            require(isinstance(destination, int) and 0 <= destination < len(steps), "invalid feedback target")
        if diagram["id"] not in ("overview", "lab"):
            anchor = diagram.get("insert_before_title")
            require(bool(anchor), "mechanism insertion title missing")
            require(sum(s.get("title") == anchor for s in original["slides"]) == 1,
                    f"mechanism insertion title must match exactly one original slide: {anchor}")
    return {"diagrams": len(diagrams), "mechanism_anchor": next(d["insert_before_title"] for d in diagrams if d["id"] == "mechanism")}


def verify_spec(spec: dict, data: dict) -> dict:
    slides = spec["slides"]
    expected = 40 + 1 + len(data["diagrams"])
    require(len(slides) == expected, f"expected {expected} slides, found {len(slides)}")
    technical = [(i, s) for i, s in enumerate(slides) if s.get("type") == "technical"]
    require(len(technical) == len(data["diagrams"]), "technical slide count differs from diagram source")
    require({s.get("diagram", {}).get("id") for _, s in technical} == {d["id"] for d in data["diagrams"]},
            "technical diagram IDs differ from authored diagrams")
    vocabulary = [s for s in slides if s.get("beginner_support") and s.get("title") == "Starting vocabulary"]
    require(len(vocabulary) == 1 and vocabulary[0]["type"] == "table", "expected one starting-vocabulary table")
    require(sum(s["type"] == "course_map" for s in slides) == 1, "expected one course map")
    if data.get("recap"):
        require(slides[-1]["type"] == "infographic" and slides[-1].get("recap") == data["recap"],
                "class must end with its authored recap infographic")
    # Each "## Slide N: Title" heading in the notes' slide companion must name deck slide N (after the
    # support slides were inserted). Untitled or replaced slides may use their generic names.
    generic = {"title": {"Title"}, "outcomes": {"Outcomes"}, "course_map": {"Agenda"}, "infographic": {"Summary"},
               "resources": {"Resources"}, "quiz": {"Quiz", "Quick check"}}
    for m in re.finditer(r"(?m)^## Slide (\d+): (.+)$", spec["notes_md"]):
        number, heading = int(m[1]), m[2].strip()
        require(1 <= number <= len(slides), f"notes heading {m[0]!r} cites a slide outside the deck")
        slide = slides[number - 1]
        require(heading == (slide.get("title") or "").strip() or heading in generic.get(slide["type"], set()),
                f"notes heading {m[0]!r} does not match deck slide {number}: {slide.get('title')!r}")
    for index, slide in technical:
        authored = next(d for d in data["diagrams"] if d["id"] == slide["diagram"]["id"])
        require(slide["diagram"] == authored and slide["title"] == authored["title"], "built diagram differs from JSON")
        if authored.get("insert_before_title"):
            following = next((s for s in slides[index + 1:] if not s.get("beginner_support")), None)
            require(following and following.get("title") == authored["insert_before_title"],
                    "diagram slide does not precede its exact core anchor")
    return {"slides": expected, "technical_slides": [i + 1 for i, _ in technical], "vocabulary_slides": 1}


def verify_pptx(path: Path, spec: dict) -> dict:
    metrics = []
    editable_figures = []
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        expected = len(spec["slides"])
        require(len(names) == expected, f"PPTX contains {len(names)} slide parts, expected {expected}")
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        require(len(presentation.findall("p:sldIdLst/p:sldId", NS)) == expected, "PPTX slide order has wrong count")
        for number, slide in enumerate(spec["slides"], 1):
            figure_asset = spec.get("assets", {}).get("figures", {}).get(slide.get("figure"), {})
            if figure_asset.get("scene"):
                figure_root = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
                require(not figure_root.findall(".//p:pic", NS), f"slide {number}: editable scene was rasterised")
                visible_text = normalise(" ".join(t.text or "" for t in figure_root.findall(".//a:t", NS)))
                labels = [item["text"] for item in figure_asset["scene"]["items"] if item["kind"] == "text"]
                for label in labels:
                    require(normalise(label) in visible_text, f"slide {number}: scene text missing: {label!r}")
                editable_figures.append({"slide": number, "figure": slide["figure"], "scene_texts": len(labels), "pictures": 0})
            if slide.get("type") != "technical":
                continue
            root = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            diagram = slide["diagram"]
            geometry = root.findall(".//a:prstGeom", NS)
            rectangles = sum(g.get("prst") in {"rect", "roundRect"} for g in geometry)
            lines = sum(g.get("prst") == "line" for g in geometry) + len(root.findall(".//p:cxnSp", NS))
            text_shapes = len(root.findall(".//p:sp/p:txBody", NS))
            pictures = len(root.findall(".//p:pic", NS))
            require(rectangles >= len(diagram["steps"]), f"slide {number}: missing native block rectangles")
            require(lines >= len(diagram["steps"]) - 1, f"slide {number}: missing native connector lines")
            require(text_shapes >= 2 * len(diagram["steps"]), f"slide {number}: missing native text objects")
            require(pictures == 0, f"slide {number}: contains {pictures} pictures instead of a fully native technical diagram")
            text = normalise(" ".join(t.text or "" for t in root.findall(".//a:t", NS)))
            labels = [diagram["title"], diagram["subtitle"]]
            labels += [value for step in diagram["steps"] for value in (step["title"], step["detail"], step["code"]) if value]
            labels += [a for a in diagram["arrows"] if a]
            labels += [b["label"] for b in diagram.get("branches", [])]
            if diagram.get("feedback"):
                labels.append(diagram["feedback"])
            for label in labels:
                require(normalise(label) in text, f"slide {number}: missing label {label!r}")
            metrics.append({"slide": number, "native_rectangles": rectangles, "native_lines": lines,
                            "native_text_objects": text_shapes, "pictures": pictures})
    return {"slide_count": expected, "technical": metrics, "editable_figures": editable_figures}


def verify_docx(path: Path, data: dict, diagram_dir: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        text = normalise(" ".join(t.text or "" for t in root.findall(".//w:t", NS)))
        require("Beginner walkthrough" in text, "DOCX missing beginner section")
        for properties in root.findall(".//wp:docPr", NS):
            require(bool(properties.get("name")), "DOCX image missing required OOXML name; Word may reject the file")
            require(bool(properties.get("descr")), "DOCX image missing accessible description")
        for diagram in data["diagrams"]:
            require(normalise(diagram["title"]) in text, f"DOCX missing diagram title {diagram['title']!r}")
        media = [n for n in archive.namelist() if n.startswith("word/media/")]
        hashes = {hashlib.sha256(archive.read(n)).hexdigest() for n in media}
        for diagram in data["diagrams"]:
            image = diagram_dir / f"beginner_{diagram['id']}.png"
            require(hashlib.sha256(image.read_bytes()).hexdigest() in hashes,
                    f"DOCX missing exact embedded {image.name}")
        for image in diagram_dir.glob("beginner_*.png"):
            require(hashlib.sha256(image.read_bytes()).hexdigest() in hashes,
                    f"DOCX missing exact shared support image {image.name}")
    return {"beginner_titles": len(data["diagrams"]), "embedded_beginner_images": len(data["diagrams"]), "total_media": len(media)}


def verify_diagrams(directory: Path, data: dict) -> dict:
    pngs = sorted(directory.glob("*.png"))
    svgs = sorted(directory.glob("*.svg"))
    expected = {f"beginner_{d['id']}" for d in data["diagrams"]}
    expected.add("beginner_course_map")
    if data.get("recap"):
        expected.add("beginner_recap")
    require({p.stem for p in pngs} == expected, "Diagrams directory PNGs differ from authored diagrams")
    require({p.stem for p in svgs} == expected, "Diagrams directory SVGs differ from authored diagrams")
    for path in pngs:
        require(path.read_bytes().startswith(PNG_SIGNATURE), f"invalid PNG signature: {path.name}")
    for path in svgs:
        require(ET.parse(path).getroot().tag == "{http://www.w3.org/2000/svg}svg", f"invalid SVG: {path.name}")
    return {"png": len(pngs), "svg": len(svgs)}


def verify_notebook(path: Path, diagram_dir: Path, student: bool, lab_source: str) -> dict:
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    references = []
    for index, cell in enumerate(nb.cells):
        require("fig:" not in cell.source, f"cell {index}: unresolved fig: reference")
        if student:
            require(not MARKERS.search(cell.source), f"cell {index}: solution/answer/stub marker leaked")
            require("solution-only" not in cell.metadata.get("tags", []), f"cell {index}: solution-only cell leaked")
        if cell.cell_type == "code":
            require(index > 0 and nb.cells[index - 1].cell_type == "markdown",
                    f"cell {index}: code has no adjacent teaching introduction")
            code = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!")))
            compile(code, f"{path.name}: cell {index}", "exec")
        else:
            for reference in re.findall(r"attachment:([^\s)]+)", cell.source):
                require(reference in cell.get("attachments", {}), f"cell {index}: unresolved attachment {reference}")
                mime = cell.attachments[reference]
                require("image/png" in mime, f"cell {index}: attachment {reference} is not PNG")
                value = mime["image/png"]
                raw = base64.b64decode("".join(value) if isinstance(value, list) else value, validate=True)
                require(raw.startswith(PNG_SIGNATURE), f"cell {index}: attachment {reference} has invalid PNG bytes")
                require(raw == (diagram_dir / reference).read_bytes(), f"cell {index}: attachment differs from {reference}")
                references.append(reference)
    expected_images = {p.name for p in diagram_dir.glob("*.png")}
    require(len(references) == len(expected_images) and set(references) == expected_images,
            f"expected every authored diagram once, found {references}")
    export = path.with_suffix(".py")
    source = export.read_text(encoding="utf-8")
    compile(source, str(export), "exec")
    require(ast_dump(source) == ast_dump(expected_export_source(nb)), f"{export.name}: code AST differs from notebook")
    require("fig:" not in source and "attachment:" not in source, f"{export.name}: non-portable image reference")
    for reference in set(references):
        require(f"Diagrams/{reference}" in source, f"{export.name}: missing portable diagram link")
    if student:
        require(not MARKERS.search(source), f"{export.name}: solution/answer/stub marker leaked")
    excluded = check_excluded_source_cells(nb, lab_source) if student else 0
    return {"nbformat_valid": True, "cells": len(nb.cells), "resolved_png_references": len(references),
            "code_compiles": True, "python_export_compiles": True, "matching_code_ast": True,
            "student_markers_clean": student, "excluded_source_cells_absent": excluded}


def symbol_warnings(data: dict, lab: str) -> list[str]:
    """Flag uncertain symbol roots for review; do not reject valid prose labels."""
    missing = []
    for item in data["code_walkthrough"]:
        symbol = item["symbol"]
        roots = re.findall(r"(?:^|\s[/+]\s)([A-Za-z_]\w*)", symbol)
        if not roots or any(not re.search(r"\b" + re.escape(root) + r"\b", lab) for root in roots):
            missing.append(symbol)
    return missing


def audit_week(week: int) -> dict:
    result = {"week": week, "checks": {}, "failures": [], "review_symbols": []}

    def check(name, operation):
        try:
            result["checks"][name] = {"ok": True, **operation()}
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            result["checks"][name] = {"ok": False, "error": message}
            result["failures"].append(f"{name}: {message}")

    try:
        data = json.loads((ROOT / "curriculum" / "beginner" / f"week_{week:02d}.json").read_text(encoding="utf-8"))
        original = yaml.safe_load((ROOT / "curriculum" / "weeks" / f"week_{week:02d}.yaml").read_text(encoding="utf-8"))
        spec = json.loads((ROOT / "build" / f"week_{week:02d}" / "spec.json").read_text(encoding="utf-8"))
        lab = (ROOT / "curriculum" / "labs" / f"week_{week:02d}_lab.py").read_text(encoding="utf-8")
    except Exception as error:
        result["failures"].append(f"source/spec read: {type(error).__name__}: {error}")
        result["ok"] = False
        return result
    output = ROOT / "deliverables" / f"Week_{week:02d}_{spec['slug']}"
    diagrams = output / "Diagrams"
    check("schema", lambda: verify_schema(week, data, original))
    check("spec", lambda: verify_spec(spec, data))
    check("diagrams", lambda: verify_diagrams(diagrams, data))
    check("pptx", lambda: verify_pptx(output / f"Week_{week:02d}_Lecture_Slides.pptx", spec))
    check("docx", lambda: verify_docx(output / f"Week_{week:02d}_Teaching_Notes.docx", data, diagrams))
    check("student_notebook_and_python", lambda: verify_notebook(output / f"Week_{week:02d}_Lab.ipynb", diagrams, True, lab))
    check("solution_notebook_and_python", lambda: verify_notebook(output / f"Week_{week:02d}_Lab_Solutions.ipynb", diagrams, False, lab))
    result["review_symbols"] = symbol_warnings(data, lab)
    result["ok"] = not result["failures"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run helper regression cases only")
    args = parser.parse_args()
    try:
        tests = self_tests()
    except Exception as error:
        print(f"SELF-TEST FAILED: {type(error).__name__}: {error}")
        return 1
    print(f"Self-tests: {len(tests)} passed")
    if args.self_test:
        for test in tests:
            print(f"  PASS {test}")
        return 0
    weeks = [audit_week(week) for week in range(1, 13)]
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "self_tests": tests,
              "ok": all(week["ok"] for week in weeks), "weeks": weeks}
    path = ROOT / "build" / "beginner_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for week in weeks:
        print(f"Week {week['week']:02d}: {'PASS' if week['ok'] else 'FAIL'} ({len(week['checks'])} checks)")
        for failure in week["failures"]:
            print(f"  {failure}")
        if week["review_symbols"]:
            print("  REVIEW symbols: " + ", ".join(week["review_symbols"]))
    print(f"Result: {sum(week['ok'] for week in weeks)}/12 packs passed; report: {path.relative_to(ROOT)}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
