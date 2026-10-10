"""Verify additional Week 5 pilot artifact contracts without executing labs.

Run after rebuilding: python tools/verify_pilot_artifacts.py
Writes only build/revision_2026_10/pilot_artifacts.json. Reuses the existing
beginner artifact checks and adds protected-input, font, guidance and readability
evidence. Passing this tool is not a claim of visual inspection or lab execution.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import subprocess
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import nbformat
import yaml

import build_week
import verify_beginner

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "build" / "revision_2026_10"
PROTECTED = {"context", "sources", "legacy"}
PLACEHOLDER = re.compile(r"\{\{[A-Za-z0-9_]+\}\}")
NS = verify_beginner.NS
# Drawn diagrams that must reach the deck as editable shapes. causal_mask is a use/block grid drawn with imshow, so
# it is a picture like the measured attention heatmaps; the course map is built by the beginner pipeline.
REQUIRED_SCENES = {
    "rnn_vs_attention", "qkv", "multihead", "positional", "block",
    "three_archs", "architecture_comparison", "kv_cache", "moe", "vit_patches", "beginner_course_map",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))  # the protected-hash baseline was written with a BOM


def one(paths, label: str) -> Path:
    choices = sorted(paths)
    require(len(choices) == 1, f"Expected one {label}; found {len(choices)}")
    return choices[0]


def protected_inputs(baseline: Path) -> dict:
    records = read_json(baseline)
    expected, skipped = {}, []
    for record in records:
        path = Path(record.get("Path", record.get("path", "")))
        if not path.is_absolute():
            path = ROOT / path
        relative = path.resolve().relative_to(ROOT)
        if relative.parts[0].lower() not in PROTECTED:
            skipped.append(str(relative))
            continue
        expected[str(relative).casefold()] = (path, str(record.get("Hash", record.get("hash", ""))).lower())
    require(bool(expected), "Protected-input baseline has no context/sources/legacy records")
    changed, missing = [], []
    for key, (path, expected_hash) in expected.items():
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
        elif digest(path) != expected_hash:
            changed.append(str(path.relative_to(ROOT)))
    current = {str(path.relative_to(ROOT)).casefold(): path for name in PROTECTED
               for path in (ROOT / name).rglob("*") if path.is_file()}
    added = [str(current[key].relative_to(ROOT)) for key in sorted(set(current) - set(expected))]
    require(not changed and not missing and not added,
            f"Protected inputs changed: modified={changed}, missing={missing}, added={added}")
    return {"baseline": str(baseline.relative_to(ROOT)), "files_checked": len(expected),
            "modified": changed, "missing": missing, "added": added,
            "nonprotected_baseline_entries_skipped": skipped, "unchanged": True}


def syllabus(before: Path) -> dict:
    current_path = ROOT / "curriculum" / "section_7_3.yaml"
    current = yaml.safe_load(current_path.read_text(encoding="utf-8"))["weeks"]
    topics = [{"week": row["week"], "topic": row["topic"]} for row in current]
    require([row["week"] for row in current] == list(range(1, 13)), "Section 7.3 week order changed")
    snapshots = sorted((before / "curriculum").rglob("section_7_3.yaml"))
    result = {"source_sha256": digest(current_path), "ordered_topics_sha256": canonical_digest(topics),
              "ordered_topics": topics, "baseline_available": bool(snapshots)}
    if not snapshots:
        result["limitation"] = "No baseline Section 7.3 source was available for comparison."
        return result
    baseline = one(snapshots, "Section 7.3 baseline snapshot")
    old_rows = yaml.safe_load(baseline.read_text(encoding="utf-8"))["weeks"]
    old_topics = [{"week": row["week"], "topic": row["topic"]} for row in old_rows]
    result.update({"baseline": str(baseline.relative_to(ROOT)), "baseline_source_sha256": digest(baseline),
                   "baseline_ordered_topics_sha256": canonical_digest(old_topics),
                   "source_bytes_unchanged": digest(baseline) == digest(current_path),
                   "ordered_topics_unchanged": topics == old_topics})
    require(topics == old_topics, "Section 7.3 topics or their order differ from the baseline")
    return result


def string_items(value: object, location: str = "spec"):
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from string_items(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from string_items(child, f"{location}[{index}]")


def spec_contract(spec: dict) -> dict:
    require(spec.get("week") == 5, "The resolved specification is not Week 5")
    require(len(spec.get("slides", [])) == 46, "Week 5 must remain exactly 46 slides")
    unresolved = [{"location": location, "placeholders": PLACEHOLDER.findall(text)}
                  for location, text in string_items(spec) if PLACEHOLDER.search(text)]
    require(not unresolved, f"Unresolved fill placeholders: {unresolved}")
    authored = (ROOT / "curriculum" / "weeks" / "week_05.yaml").read_text(encoding="utf-8")
    authored += "\n" + (ROOT / "curriculum" / "notes" / "week_05.md").read_text(encoding="utf-8")
    keys = sorted({match[2:-2] for match in PLACEHOLDER.findall(authored)})
    fill = read_json(ROOT / "curriculum" / "assets" / "week_05" / "fill.json")
    require(all(key in fill for key in keys), f"Measured fill keys missing: {set(keys) - set(fill)}")
    figures = spec.get("assets", {}).get("figures", {})
    scene_ids = sorted(fid for fid, asset in figures.items() if asset.get("scene"))
    require(REQUIRED_SCENES.issubset(scene_ids), f"Editable scene metadata missing: {sorted(REQUIRED_SCENES - set(scene_ids))}")
    scene_counts = {}
    for fid in scene_ids:
        scene = figures[fid]["scene"]
        require(scene.get("width", 0) > 0 and scene.get("height", 0) > 0, f"Invalid scene dimensions: {fid}")
        require(bool(scene.get("items")), f"Empty scene: {fid}")
        sidecar = Path(figures[fid]["path"]).with_suffix(".scene.json")
        require(sidecar.is_file(), f"Scene sidecar missing: {fid}")
        require(read_json(sidecar) == scene, f"Built scene metadata differs from its figure sidecar: {fid}")
        scene_counts[fid] = {kind: sum(item["kind"] == kind for item in scene["items"])
                            for kind in ("text", "block", "route")}
    return {"slides": 46, "unresolved_fill_placeholders": unresolved, "authored_fill_keys": keys,
            "editable_scene_ids": scene_ids, "scene_item_counts": scene_counts}


def technical_scenes(spec: dict) -> dict[int, dict]:
    diagrams = [(number, slide["diagram"]) for number, slide in enumerate(spec["slides"], 1)
                if slide.get("type") == "technical"]
    script = "const fs=require('fs'),t=require('./tools/technical_diagrams.js');const ds=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(ds.map(d=>t.scene(d))));"
    completed = subprocess.run(["node", "-e", script], input=json.dumps([diagram for _, diagram in diagrams]),
                               capture_output=True, text=True, encoding="utf-8", cwd=ROOT, check=True)
    scenes = json.loads(completed.stdout)
    return {number: scene for (number, _), scene in zip(diagrams, scenes, strict=True)}


def shape_text(shape) -> str:
    return verify_beginner.normalise(" ".join(node.text or "" for node in shape.findall(".//a:t", NS)))


def shape_font_sizes(shape) -> list[float]:
    sizes = []
    autofit = shape.find("p:txBody/a:bodyPr/a:normAutofit", NS)
    scale = int(autofit.get("fontScale", "100000")) / 100000 if autofit is not None else 1.0
    for paragraph in shape.findall("p:txBody/a:p", NS):
        default = paragraph.find("a:pPr/a:defRPr", NS)
        for run in paragraph.findall("a:r", NS):
            if not (run.findtext("a:t", default="", namespaces=NS) or "").strip():
                continue
            properties = run.find("a:rPr", NS)
            size = properties.get("sz") if properties is not None else None
            size = size or (default.get("sz") if default is not None else None)
            require(size is not None, f"Font size is not explicit for scene text: {shape_text(shape)!r}")
            sizes.append(float(size) / 100 * scale)
    require(bool(sizes), f"No measurable text runs in scene shape: {shape_text(shape)!r}")
    return sizes


def flattened(scene: dict) -> dict:
    """The scene as tools/technical_diagrams.js flattenedScene places it: one text item per line."""
    items = []
    for item in scene["items"]:
        if item["kind"] == "text":
            items += [{**item, "text": row} for row in item["text"].splitlines()]
        else:
            items.append(item)
    return {**scene, "items": items}


def scene_font_contract(deck: Path, spec: dict) -> dict:
    scenes = technical_scenes(spec)
    for number, slide in enumerate(spec["slides"], 1):
        asset = spec.get("assets", {}).get("figures", {}).get(slide.get("figure"), {})
        if asset.get("scene"):
            scenes[number] = asset["scene"]
    records, failures = [], []
    with ZipFile(deck) as archive:
        for number, scene in sorted(scenes.items()):
            root = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            available = defaultdict(list)
            for shape in root.findall(".//p:sp", NS):
                if shape.find("p:txBody", NS) is not None:
                    available[shape_text(shape)].append(shape)
            item_records = []
            # The deck places each line of a multi-line label as its own text shape (flattenedScene).
            for item in flattened(scene)["items"]:
                if item["kind"] != "text" or not item["text"].strip():
                    continue
                label = verify_beginner.normalise(item["text"])
                candidates = available[label]
                require(bool(candidates), f"Slide {number}: no distinct text shape for scene item {label!r}")
                shape = candidates.pop(0)
                sizes = shape_font_sizes(shape)
                smallest = min(sizes)
                item_records.append({"text": label, "minimum_font_pt": round(smallest, 3)})
                if smallest < 14 - 0.001:
                    failures.append({"slide": number, "text": label, "minimum_font_pt": smallest})
            records.append({"slide": number, "scene_text_items": len(item_records),
                            "minimum_font_pt": min(row["minimum_font_pt"] for row in item_records),
                            "items": item_records})
    require(not failures, f"Scene text smaller than 14 pt: {failures}")
    return {"slides_checked": len(records), "scope": "Only text shapes matched to scene items; slide chrome/title excluded",
            "minimum_required_pt": 14, "scene_slides": records, "failures": failures}


def clean_code(source: str) -> str:
    return "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))


def notebook_contract(directory: Path) -> dict:
    source = (ROOT / "curriculum" / "labs" / "week_05_lab.py").read_text(encoding="utf-8")
    source_cells = build_week.parse_percent(source)
    results, attachment_records = {}, []
    for variant, filename, transform in (
        ("student", "Week_05_Lab.ipynb", build_week.student_code),
        ("instructor", "Week_05_Lab_Solutions.ipynb", build_week.solution_code),
    ):
        path = directory / filename
        result = verify_beginner.verify_notebook(path, directory / "Diagrams", variant == "student", source)
        nb = nbformat.read(path, as_version=4)
        actual = [cell for cell in nb.cells if cell.cell_type == "code"]
        expected = [cell for cell in source_cells if cell["type"] == "code"
                    and not (variant == "student" and "solution-only" in cell["tags"])]
        require(len(actual) == len(expected), f"{variant}: built code-cell count differs from source")
        for index, (built, authored) in enumerate(zip(actual, expected, strict=True), 1):
            target = clean_code("\n".join(transform(authored["lines"])))
            require(verify_beginner.ast_dump(clean_code(built.source)) == verify_beginner.ast_dump(target),
                    f"{variant}: code cell {index} differs from its intended source variant")
        guidance = []
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                require(index > 0 and nb.cells[index - 1].cell_type == "markdown",
                        f"{variant}: cell {index + 1} has no immediately preceding Markdown guide")
                guide = nb.cells[index - 1].source
                require(all(label in guide for label in ("**What/why:**", "**Predict:**", "**Expected output:**")),
                        f"{variant}: cell {index + 1} guide lacks what/why, prediction or expected output")
                guidance.append(index + 1)
            references = set(re.findall(r"attachment:([^\s)]+)", cell.source))
            require(set(cell.get("attachments", {})) == references,
                    f"{variant}: cell {index + 1} has unused or unresolved attachments")
            for name, representations in cell.get("attachments", {}).items():
                encoded = representations["image/png"]
                encoded = "".join(encoded) if isinstance(encoded, list) else encoded
                raw = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
                standalone = directory / "Diagrams" / name
                require(raw == standalone.read_bytes(), f"{variant}: attachment {name} differs from standalone")
                attachment_records.append({"variant": variant, "cell_one_based": index + 1, "image": name,
                                           "sha256": hashlib.sha256(raw).hexdigest(), "referenced": True,
                                           "identical_to_standalone": True})
        if variant == "instructor":
            require(len(actual) == 17, f"Expected 17 instructor code cells, found {len(actual)}")
        else:
            todo_lines = [line.strip() for cell in actual for line in cell.source.splitlines() if "TODO" in line]
            require(len(todo_lines) >= 5, "Student implementation TODOs are missing")
            require("**Model answer:**" not in "\n".join(cell.source for cell in nb.cells), "Instructor answer heading leaked")
            result["implementation_todo_lines"] = todo_lines
        result.update({"code_cells": len(actual), "guided_code_cell_locations": guidance,
                       "code_ast_matches_intended_source_variant": True})
        results[variant] = result
    return {"variants": results, "attachments": attachment_records,
            "student_instructor_separation": "Exact code ASTs compared with source marker transformations; instructor-only content checked by existing helper"}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text, re.UNICODE))


def readability(deck: Path, spec: dict) -> dict:
    counts, sentence_warnings, skipped = [], [], []
    with ZipFile(deck) as archive:
        for number, slide in enumerate(spec["slides"], 1):
            scene = spec.get("assets", {}).get("figures", {}).get(slide.get("figure"), {}).get("scene")
            if slide.get("type") in {"title", "section", "technical"} or scene:
                skipped.append({"slide": number, "type": slide.get("type"),
                                "reason": "Title/section or diagram; ordinary-body threshold does not describe this layout"})
            else:
                root = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
                texts = []
                for shape in root.findall(".//p:sp", NS):
                    offset = shape.find("p:spPr/a:xfrm/a:off", NS)
                    if offset is None:
                        continue
                    top_in = int(offset.get("y", "0")) / 914400
                    if 1.60 <= top_in < 6.95:
                        text = shape_text(shape)
                        if text:
                            texts.append(text)
                body = " ".join(texts)
                count = word_count(body)
                counts.append({"slide": number, "title": slide.get("title", slide.get("type")),
                               "body_word_count": count, "above_40_words": count > 40, "body_text": body})
            notes = slide.get("notes", "")
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", notes):
                sentence = sentence.strip()
                count = word_count(sentence)
                if count > 25:
                    sentence_warnings.append({"slide": number, "word_count": count, "sentence": sentence})
    return {"method": "Body words counted from ordinary PowerPoint text shapes beginning 1.60-6.95 inches down; title/chrome and diagram layouts excluded. Sentence boundaries are a punctuation/newline heuristic.",
            "ordinary_slide_counts": counts, "above_40_word_slides": [row["slide"] for row in counts if row["above_40_words"]],
            "skipped_layouts": skipped, "script_sentences_above_25_words": sentence_warnings,
            "warning_only": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=REVISION / "before")
    parser.add_argument("--protected-baseline", type=Path, default=REVISION / "protected_hashes.json")
    parser.add_argument("--output", type=Path, default=REVISION / "pilot_artifacts.json")
    args = parser.parse_args()
    report = {"week": 5, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "ok": True,
              "visual_inspection_claimed": False, "labs_executed_by_this_tool": False,
              "checks": {}, "failures": [], "warnings": []}
    directory = None
    spec = None
    checks = [
        ("protected_inputs", lambda: protected_inputs(args.protected_baseline)),
        ("section_7_3", lambda: syllabus(args.before)),
    ]
    try:
        directory = one((ROOT / "deliverables").glob("Week_05_*"), "Week 5 deliverables folder")
        spec = read_json(ROOT / "build" / "week_05" / "spec.json")
    except Exception as error:
        report["ok"] = False
        report["failures"].append({"check": "input_artifacts", "error": f"{type(error).__name__}: {error}"})
    if directory and spec:
        deck = directory / "Week_05_Lecture_Slides.pptx"
        checks.extend([
            ("spec", lambda: spec_contract(spec)),
            ("native_pptx", lambda: verify_beginner.verify_pptx(deck, spec)),
            ("scene_fonts", lambda: scene_font_contract(deck, spec)),
            ("notebooks", lambda: notebook_contract(directory)),
            ("readability", lambda: readability(deck, spec)),
        ])
    for name, check in checks:
        try:
            result = check()
            report["checks"][name] = {"ok": True, **result}
            print(f"PASS {name}")
        except Exception as error:
            report["ok"] = False
            report["checks"][name] = {"ok": False}
            report["failures"].append({"check": name, "error": f"{type(error).__name__}: {error}",
                                       "traceback": traceback.format_exc()})
            print(f"FAIL {name}: {error}")
    section = report["checks"].get("section_7_3", {})
    if not section.get("baseline_available"):
        report["warnings"].append("Section 7.3 baseline source/topics could not be compared.")
    elif not section.get("source_bytes_unchanged"):
        report["warnings"].append("Section 7.3 source bytes changed; topic/order is preserved. Review Detail/Tutorial edits separately.")
    reading = report["checks"].get("readability", {})
    if reading.get("above_40_word_slides"):
        report["warnings"].append(f"Ordinary slides above 40 body words: {reading['above_40_word_slides']}")
    if reading.get("script_sentences_above_25_words"):
        report["warnings"].append(f"Speaker-script sentences above 25 words: {len(reading['script_sentences_above_25_words'])}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
