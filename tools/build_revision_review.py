"""Build a local visual review gallery from actual rendered course artifacts."""
from pathlib import Path
from urllib.parse import quote
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/revision_review.html"


def href(path):
    return quote(Path(__import__("os").path.relpath(path, OUT.parent)).as_posix())


def image(path, caption):
    if not path.exists():
        return f"<p>{html.escape(caption)}: no saved image available.</p>"
    url = href(path)
    return f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Generative AI teaching pack: visual review</title><style>
body{font:17px/1.5 system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#172033;background:#f7f9fc}h1,h2,h3{line-height:1.2}a{color:#075985}nav{display:flex;flex-wrap:wrap;gap:12px;position:sticky;top:0;background:#f7f9fc;padding:12px 0}section{background:white;padding:24px;border-radius:12px;margin:24px 0}figure{margin:0}img{width:100%;border:1px solid #dbe3ec;border-radius:6px}figcaption{padding:6px 0 16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}.slides{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}summary{cursor:pointer;font-weight:650;padding:12px 0}.notice{padding:16px;background:#eaf4fb;border-left:4px solid #0369a1}@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style><h1>Generative AI: Weeks 1–12</h1><p>Open any image at full size. Each class ends with an editable recap and an exit question. The five technical diagrams, course map and recap are shared by the slides, teaching notes and notebooks.</p><p class="notice">This gallery shows generated artifacts, not an automatic claim that they were reviewed. The audit and revision report record actual checks and any remaining verification limits. Starting images come from the saved 8 October snapshot; only Weeks 1–4 had fresh baseline renders in that run.</p><p><a href="AUDIT_2026-10.md">Audit</a> · <a href="REVISION_2026-10.md">Revision and verification report</a></p><nav>''']
    parts += [f'<a href="#week-{w}">Week {w}</a>' for w in range(1, 13)]
    parts.append('</nav>')
    for week in range(1, 13):
        spec = json.loads((ROOT / f"build/week_{week:02d}/spec.json").read_text(encoding="utf-8"))
        delivery = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*"))
        rendered = ROOT / f"build/render/week_{week:02d}"
        old = ROOT / f"build/revision_before_20261008/render/week_{week:02d}"
        parts.append(f'<section id="week-{week}"><h2>Week {week}: {html.escape(spec["topic"].rstrip("."))}</h2>')
        links = [(delivery/f"Week_{week:02d}_Lecture_Slides.pptx", "Editable slides"),
                 (delivery/f"Week_{week:02d}_Teaching_Notes.docx", "Teaching notes"),
                 (delivery/f"Week_{week:02d}_Lab.ipynb", "Student lab"),
                 (rendered/"notes.pdf", "Rendered notes PDF")]
        parts.append('<p>' + ' · '.join(f'<a href="{href(p)}">{label}</a>' for p, label in links) + '</p>')
        parts.append('<h3>End-of-class recap</h3>')
        parts.append(image(delivery / "Diagrams/beginner_recap.png", f"Week {week} recap"))
        parts.append('<details><summary>Starting and revised slides</summary><div class="grid">')
        for before, after, label in [(2,2,"Learning outcomes"),(3,3,"Course connections"),(45,46,"End-of-class recap")]:
            parts.append(image(old/f"slides-{before:02d}.png", f"Starting snapshot: {label}"))
            parts.append(image(rendered/f"slides-{after:02d}.png", f"Revised: {label}"))
        parts.append('</div></details><details><summary>All seven shared diagrams</summary><div class="grid">')
        for name in ("course_map", "overview", "mechanism", "training", "inference", "lab", "recap"):
            parts.append(image(delivery/f"Diagrams/beginner_{name}.png", name.replace("_", " ").title()))
        parts.append('</div></details><details><summary>All 46 rendered slides</summary><div class="slides">')
        for index in range(1,47):
            parts.append(image(rendered/f"slides-{index:02d}.png", f"Slide {index}: {spec['slides'][index-1].get('title','')}"))
        parts.append('</div></details></section>')
    parts.append('</html>')
    OUT.write_text('\n'.join(parts), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
