"""Build the Week 5 approval gallery from existing, actually rendered artifacts.

Run after the final build and Office render:
    python tools/build_pilot_review.py

The HTML embeds every image and download, so it can be shared as one file.
Before figures are recovered from the snapshotted PPTX, never from its stale
absolute paths. No deliverable, source specification or rendering is changed.
Use --snapshot-only to extract baseline figures/notebook images before review.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "build" / "revision_2026_10"
TECHNICAL = ("overview", "mechanism", "training", "inference", "lab")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_one(paths, description: str) -> Path:
    paths = sorted(paths)
    if len(paths) != 1:
        raise ValueError(f"Expected one {description}; found {len(paths)}")
    return paths[0]


def slide_title(slide: dict, number: int, spec: dict) -> str:
    return str(slide.get("title") or {
        "title": spec.get("topic", "Week 5"), "outcomes": "Learning outcomes",
        "agenda": "Today's route", "quiz": "Check your understanding",
        "summary": "Recap", "resources": "Further reading",
    }.get(slide.get("type"), f"Slide {number}"))


def figure_id(slide: dict) -> str | None:
    if slide.get("figure"):
        return str(slide["figure"])
    diagram = slide.get("diagram", {})
    return f"beginner_{diagram['id']}" if diagram.get("id") else None


def normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def rendered_pages(directory: Path, prefix: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.png$")
    result = {}
    for path in directory.glob(f"{prefix}-*.png"):
        match = pattern.fullmatch(path.name)
        if match:
            result[int(match.group(1))] = path
    return dict(sorted(result.items()))


def extract_before_figures(before: Path, spec: dict) -> tuple[dict[str, Path], list[dict]]:
    """Read exact embedded figure bytes, using the baseline slide-to-figure map."""
    deck = require_one(before.glob("Week_05_*/Week_05_Lecture_Slides.pptx"), "baseline deck")
    out = before / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures, provenance = {}, []
    with ZipFile(deck) as archive:
        for index, slide in enumerate(spec["slides"], 1):
            fid = slide.get("figure")
            if not fid:
                continue
            xml_name = f"ppt/slides/slide{index}.xml"
            xml = ET.fromstring(archive.read(xml_name))
            rels = ET.fromstring(archive.read(f"ppt/slides/_rels/slide{index}.xml.rels"))
            targets = {rel.get("Id"): rel.get("Target") for rel in rels}
            embeds = [node.get(f"{{{NS['r']}}}embed") for node in xml.findall(".//a:blip", NS)]
            if len(embeds) != 1:
                raise ValueError(f"Baseline slide {index}, {fid}: expected one figure, found {len(embeds)}")
            member = posixpath.normpath(posixpath.join("ppt/slides", targets[embeds[0]]))
            data = archive.read(member)
            target = out / f"{fid}{Path(member).suffix}"
            target.write_bytes(data)
            figures[fid] = target
            provenance.append({"figure": fid, "slide": index, "archive": str(deck.relative_to(ROOT)),
                               "member": member, "sha256": sha256(data)})
    diagram_dir = require_one(before.glob("Week_05_*/Diagrams"), "baseline diagram folder")
    for path in sorted(diagram_dir.glob("*.png")):
        figures[path.stem] = path
        provenance.append({"figure": path.stem, "file": str(path.relative_to(ROOT)),
                           "sha256": sha256(path.read_bytes())})
    missing = set(spec.get("assets", {}).get("figures", {})) - set(figures)
    if missing:
        raise ValueError(f"Cannot recover baseline figure(s): {', '.join(sorted(missing))}")
    write_json(out / "manifest.json", provenance)
    return figures, provenance


def extract_notebook_images(folder: Path, out: Path, standalone: dict[str, Path]) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    matches = {}
    for fid, path in standalone.items():
        matches.setdefault(sha256(path.read_bytes()), []).append(fid)
    records = []
    for notebook in sorted(folder.glob("Week_05_*.ipynb")):
        nb = read_json(notebook)
        variant = "solutions" if "Solutions" in notebook.name else "student"
        for cell_index, cell in enumerate(nb.get("cells", []), 1):
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            images = []
            for name, representations in cell.get("attachments", {}).items():
                if "image/png" in representations:
                    images.append((name, representations["image/png"], "attachment", f"attachment:{name}" in source))
            for output_index, output in enumerate(cell.get("outputs", []), 1):
                encoded = output.get("data", {}).get("image/png")
                if encoded:
                    images.append((f"output_{output_index}.png", encoded, "output", True))
            for name, encoded, kind, referenced in images:
                encoded = "".join(encoded) if isinstance(encoded, list) else encoded
                data = base64.b64decode(re.sub(r"\s+", "", encoded), validate=True)
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
                destination = out / f"{variant}_cell_{cell_index:03d}_{safe_name}"
                destination.write_bytes(data)
                digest = sha256(data)
                records.append({"notebook": notebook.name, "cell_one_based": cell_index,
                                "image": name, "kind": kind, "referenced": referenced,
                                "sha256": digest, "standalone_matches": sorted(matches.get(digest, [])),
                                "path": str(destination.relative_to(ROOT))})
    write_json(out / "manifest.json", records)
    return records


def after_figures(spec: dict) -> dict[str, Path]:
    result = {}
    for fid, asset in spec.get("assets", {}).get("figures", {}).items():
        path = Path(asset["path"] if isinstance(asset, dict) else asset)
        path = path if path.is_absolute() else ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"After figure is missing: {fid}: {path}")
        result[fid] = path
    return result


def image_markup(path: Path, alt: str, caption: str = "") -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img loading="lazy" src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(alt, quote=True)}">'
            f'<figcaption>{html.escape(caption or alt)}</figcaption></figure>')


def download_link(path: Path, label: str) -> str:
    if not path.is_file():
        return f'<span class="missing">{html.escape(label)}: file not present</span>'
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<a download="{html.escape(path.name, quote=True)}" '
            f'href="data:{mime};base64,{encoded}">{html.escape(label)}</a>')


def paired_markup(before: Path | None, after: Path, name: str, old_caption: str, new_caption: str) -> str:
    left = image_markup(before, f"Before: {name}", old_caption) if before else (
        '<div class="new"><p>No corresponding baseline figure or slide.</p><p>Added in the pilot.</p></div>')
    return f'<div class="pair"><div><h4>Before</h4>{left}</div><div><h4>After</h4>{image_markup(after, f"After: {name}", new_caption)}</div></div>'


def find_slide(spec: dict, ids: tuple[str, ...] = (), title_words: tuple[str, ...] = ()) -> tuple[int, dict] | None:
    for number, slide in enumerate(spec["slides"], 1):
        if figure_id(slide) in ids:
            return number, slide
    if title_words:
        for number, slide in enumerate(spec["slides"], 1):
            title = normalise(slide_title(slide, number, spec))
            if all(word in title for word in title_words):
                return number, slide
    return None


def match_before(slide: dict, spec: dict) -> tuple[int, dict] | None:
    fid = figure_id(slide)
    if fid:
        match = find_slide(spec, (fid,))
        if match:
            return match
    target = normalise(str(slide.get("title", "")))
    if target:
        for number, candidate in enumerate(spec["slides"], 1):
            if normalise(str(candidate.get("title", ""))) == target:
                return number, candidate
    return None


def comparison_png(before: Path | None, after: Path, destination: Path, old_label: str, new_label: str) -> None:
    """A labelled contact sheet of unmodified render images, plus a new-item placeholder."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default(size=22)
    small = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default(size=18)
    panel_w, panel_h, margin, header = 960, 620, 24, 86
    sheet = Image.new("RGB", (2 * panel_w + 3 * margin, panel_h + header + 2 * margin), "#f3f6fa")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(((before, old_label), (after, new_label))):
        x, y = margin + index * (panel_w + margin), margin
        draw.text((x + 12, y + 10), "BEFORE" if index == 0 else "AFTER", font=font, fill="#172337")
        words, rows, line = label.split(), [], ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textbbox((0, 0), candidate, font=small)[2] > panel_w - 24:
                rows.append(line)
                line = word
            else:
                line = candidate
        rows.append(line)
        draw.multiline_text((x + 12, y + 40), "\n".join(rows), font=small, fill="#172337", spacing=3)
        draw.rectangle((x, y + header, x + panel_w, y + header + panel_h), fill="white", outline="#cbd5e1")
        if path:
            with Image.open(path) as original:
                im = original.convert("RGB")
                im.thumbnail((panel_w - 16, panel_h - 16), Image.Resampling.LANCZOS)
                sheet.paste(im, (x + (panel_w - im.width) // 2, y + header + (panel_h - im.height) // 2))
        else:
            draw.text((x + 80, y + header + panel_h // 2), "Added in the pilot: no baseline counterpart", font=font, fill="#455468")
    sheet.save(destination)


def audit_html(source: str) -> str:
    tables, paragraphs, current = [], [], []
    for line in source.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []
            if line.strip():
                paragraphs.append(line)
    if current:
        tables.append(current)
    output = "".join(f"<p>{html.escape(line.lstrip('# ').strip())}</p>" for line in paragraphs)
    for table in tables:
        rows = [[cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))] for line in table]
        output += '<div class="table-scroll"><table><thead><tr>'
        output += "".join(f"<th scope=\"col\">{html.escape(cell)}</th>" for cell in rows[0]) + "</tr></thead><tbody>"
        for row in rows[1:]:
            if all(re.fullmatch(r":?-+:?", cell) for cell in row):
                continue
            output += "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        output += "</tbody></table></div>"
    return output


def attachment_report_html(records: list[dict], label: str) -> str:
    output = f'<h3>{html.escape(label)}</h3><div class="table-scroll"><table><thead><tr><th scope="col">Notebook / cell</th><th scope="col">Image</th><th scope="col">Matches standalone bytes</th><th scope="col">Referenced</th><th scope="col">SHA-256</th></tr></thead><tbody>'
    for record in records:
        values = [f"{record['notebook']} / cell {record['cell_one_based']}", record["image"],
                  ", ".join(record["standalone_matches"]) or "No standalone match (inspect separately)",
                  str(record["referenced"]), record["sha256"]]
        output += "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
    output += "</tbody></table></div>"
    unique = {}
    for record in records:
        unique.setdefault(record["sha256"], record)
    for record in unique.values():
        output += image_markup(ROOT / record["path"], f"{label}: {record['image']}",
                               f"{record['notebook']}, cell {record['cell_one_based']}; identical copies listed above")
    return output


CSS = """
:root{font-family:Arial,Helvetica,sans-serif;color:#172337;background:#f3f6fa;line-height:1.5}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:28px}
h1,h2,h3,h4{line-height:1.25}h1{font-size:2.1rem}h2{margin-top:2rem}h3{margin-top:1.5rem}
a{color:#005b9a;text-decoration:underline;overflow-wrap:anywhere}a:focus-visible,summary:focus-visible{outline:3px solid #d55e00;outline-offset:4px}
nav,.downloads{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}.downloads a{background:white;border:1px solid #b7c8d8;padding:8px 12px;border-radius:6px}
section,details{background:white;border:1px solid #cbd5e1;border-radius:8px;padding:20px;margin:20px 0}
summary{font-weight:bold;cursor:pointer;font-size:1.15rem}details[open]>summary{margin-bottom:20px}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;align-items:start}
figure{margin:12px 0 24px}img{display:block;width:100%;height:auto;border:1px solid #cbd5e1;background:white}
figcaption{font-size:.93rem;color:#3b4d63;padding-top:7px}.new{background:#f3f6fa;border:1px dashed #8da2b8;padding:32px;min-height:220px}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid #cbd5e1;padding:9px;text-align:left;vertical-align:top}th{background:#e4f2fb}td{overflow-wrap:anywhere;min-width:90px}
.note{border-left:4px solid #0072b2;padding:12px;background:#e4f2fb}.missing{color:#9d3300}.gallery-item{padding-top:10px;border-top:1px solid #e2e8f0}
@media(max-width:850px){main{padding:14px}.pair{grid-template-columns:1fr}section,details{padding:14px}h1{font-size:1.65rem}}
@media print{details{display:block}summary{display:none}img{max-height:80vh;object-fit:contain}section{break-before:page}.downloads,nav{display:none}}
"""


def build(args) -> dict:
    before = args.before.resolve()
    out = args.output.resolve()
    old_spec = read_json(before / "spec_week_05.json")
    old_figures, figure_provenance = extract_before_figures(before, old_spec)
    old_deliverables = require_one(before.glob("Week_05_*"), "baseline deliverables folder")
    old_attachments = extract_notebook_images(old_deliverables, before / "notebook_images", old_figures)
    if args.snapshot_only:
        return {"baseline_figures": len(old_figures), "baseline_notebook_images": len(old_attachments),
                "baseline_attachment_hash_matches": sum(bool(r["standalone_matches"]) for r in old_attachments),
                "manifest": str(before / "notebook_images" / "manifest.json")}

    new_spec = read_json(args.after_spec)
    new_figures = after_figures(new_spec)
    old_pages = rendered_pages(args.before_render, "slides")
    new_pages = rendered_pages(args.after_render, "slides")
    notes_pages = rendered_pages(args.after_render, "notes")
    if len(old_pages) != len(old_spec["slides"]) or len(new_pages) != len(new_spec["slides"]):
        raise ValueError("Rendered slide count does not match its specification; render before building the review")
    if not notes_pages:
        raise ValueError("No rendered Word pages; render notes before building the review")
    out.mkdir(parents=True, exist_ok=True)
    new_deliverables = require_one((ROOT / "deliverables").glob("Week_05_*"), "current deliverables folder")
    new_attachments = extract_notebook_images(new_deliverables, out / "notebook_images", new_figures)
    source = args.audit.read_text(encoding="utf-8") if args.audit.is_file() else "Audit file not present."
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             f'<title>Week 5 pilot: before and after review</title><style>{CSS}</style></head><body><main>',
             '<h1>Week 5: Transformers &amp; Attention</h1><p>October 2026 pilot approval review.</p>',
             f'<p class="note">{len(new_spec["slides"])} rendered lecture slides, {len(notes_pages)} rendered Word pages and {len(new_figures)} standalone figure images. Images and downloads are embedded in this file. This gallery presents review evidence; it does not itself certify visual quality or successful lab execution.</p>',
             '<nav aria-label="Review sections"><a href="#key">Key slide changes</a><a href="#technical">Five technical diagrams</a><a href="#figures">Every standalone figure</a><a href="#audit">Audit</a><a href="#slides">All slides</a><a href="#notes">All Word pages</a><a href="#notebooks">Notebook images</a></nav>',
             '<div class="downloads">']
    for name, label in (("Week_05_Lecture_Slides.pptx", "Download PowerPoint"),
                        ("Week_05_Teaching_Notes.docx", "Download Word notes"),
                        ("Week_05_Lab.ipynb", "Download student notebook"),
                        ("Week_05_Lab_Solutions.ipynb", "Download instructor notebook")):
        parts.append(download_link(new_deliverables / name, label))
    parts.append(download_link(args.audit, "Download audit"))
    parts.append('</div><section id="key"><h2>Key slide changes</h2>')
    comparisons = []
    keys = [("overview", ("beginner_overview",), ()),
            ("mechanism", ("beginner_mechanism",), ()),
            ("training", ("beginner_training",), ()),
            ("cache", ("kv_cache",), ()),
            ("course_map", ("course_map", "course_big_map", "big_map", "beginner_course_map"), ("course", "map"))]
    for key, ids, words in keys:
        found = find_slide(new_spec, ids, words)
        if not found:
            raise ValueError(f"Expected key after slide was not found: {key}")
        new_number, slide = found
        old_match = match_before(slide, old_spec)
        old_number = old_match[0] if old_match else None
        before_image = old_pages[old_number] if old_number else None
        title = slide_title(slide, new_number, new_spec)
        old_label = f"Slide {old_number}: {slide_title(old_match[1], old_number, old_spec)}" if old_match else "Added in the pilot"
        new_label = f"Slide {new_number}: {title}"
        parts.append(f'<h3>{html.escape(key.replace("_", " ").title())}: after slide {new_number}</h3>')
        parts.append(paired_markup(before_image, new_pages[new_number], title, old_label, new_label))
        destination = out / f"before_after_{key}.png"
        comparison_png(before_image, new_pages[new_number], destination, old_label, new_label)
        comparisons.append({"key": key, "before_slide": old_number, "after_slide": new_number,
                            "png": str(destination.relative_to(ROOT))})
    parts.append('</section><section id="technical"><h2>Five technical diagrams: before and after</h2>')
    for key in TECHNICAL:
        fid = f"beginner_{key}"
        if fid not in old_figures or fid not in new_figures:
            raise ValueError(f"Required technical diagram missing: {fid}")
        found = find_slide(new_spec, (fid,))
        title = slide_title(found[1], found[0], new_spec) if found else key.title()
        parts.append(f'<h3>{html.escape(title)} ({html.escape(fid)})</h3>')
        parts.append(paired_markup(old_figures[fid], new_figures[fid], title, "Baseline standalone PNG", "Revised standalone PNG"))
    parts.append('</section><section id="figures"><h2>Every standalone figure</h2><p>Includes the conceptual diagrams, measured plots, course map and comparison infographic. Before images are extracted from the baseline PowerPoint or its diagram folder.</p>')
    for fid, path in new_figures.items():
        found = find_slide(new_spec, (fid,))
        title = slide_title(found[1], found[0], new_spec) if found else fid.replace("_", " ")
        label = f"After slide {found[0]}" if found else "Word/notebook figure or supporting asset"
        parts.append(f'<details><summary>{html.escape(fid)}: {html.escape(title)}</summary>')
        parts.append(paired_markup(old_figures.get(fid), path, title, "Baseline figure", label))
        parts.append('</details>')
    parts.append('</section><section id="audit"><h2>Week 5 audit</h2>')
    parts.append(audit_html(source))
    parts.append('</section><details id="slides"><summary>Inspect every rendered after slide</summary>')
    for number, path in new_pages.items():
        title = slide_title(new_spec["slides"][number - 1], number, new_spec)
        parts.append(f'<div class="gallery-item"><h3>Slide {number}: {html.escape(title)}</h3>')
        parts.append(image_markup(path, f"Week 5 slide {number}: {title}"))
        parts.append('</div>')
    parts.append('</details><details id="notes"><summary>Inspect every rendered after Word page</summary>')
    for number, path in notes_pages.items():
        parts.append(f'<div class="gallery-item"><h3>Word page {number}</h3>')
        parts.append(image_markup(path, f"Week 5 teaching notes, page {number}"))
        parts.append('</div>')
    parts.append('</details><details id="notebooks"><summary>Inspect notebook image placement and exact-byte matches</summary>')
    parts.append(attachment_report_html(old_attachments, "Before notebook images"))
    parts.append(attachment_report_html(new_attachments, "After notebook images"))
    parts.append('</details></main></body></html>')
    review = out / "week_05_review.html"
    review.write_text("\n".join(parts), encoding="utf-8")
    manifest = {"review_html": str(review.relative_to(ROOT)), "before_slide_count": len(old_pages),
                "after_slide_count": len(new_pages), "after_word_page_count": len(notes_pages),
                "after_figure_count": len(new_figures), "comparison_pngs": comparisons,
                "baseline_figure_provenance": figure_provenance,
                "before_notebook_images": old_attachments, "after_notebook_images": new_attachments,
                "all_review_images_embedded": True, "all_downloads_embedded": True}
    write_json(out / "review_manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key not in ("baseline_figure_provenance", "before_notebook_images", "after_notebook_images")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=REVISION / "before")
    parser.add_argument("--before-render", type=Path, default=REVISION / "before" / "render_week_05")
    parser.add_argument("--after-render", type=Path, default=ROOT / "build" / "render" / "week_05")
    parser.add_argument("--after-spec", type=Path, default=ROOT / "build" / "week_05" / "spec.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "research" / "AUDIT_2026-10.md")
    parser.add_argument("--output", type=Path, default=REVISION / "review")
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
