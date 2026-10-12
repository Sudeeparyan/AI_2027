"""Build the course-level documents in deliverables/00_Course/.

  Instructor_Guide.docx          how to use the pack, course map, MIMLO matrix, labs, models and licences, maintenance
  Section_7.3_Rationale.docx     what changed in descriptor Section 7.3 and why, week by week, with the original text

Authored text lives in curriculum/course/; tables are computed from curriculum/section_7_3.yaml, the weekly
YAML files, the lab sources, curriculum/sources.yaml and the original descriptor, so they cannot drift.

Run: .venv/Scripts/python.exe tools/build_course.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_descriptor as bd  # noqa: E402
from build_week import prepare_notes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT / "curriculum"
OUT = ROOT / "deliverables" / "00_Course"
BUILD = ROOT / "build" / "course"

# Archived 9 October 2026 experiment summary, retained for interpreting earlier stored figures.
# These settings/times are not the current CPU core and never supply LAB_TABLE.
# The original measurements remain in research/lab_runs_2026-10.json under runs.
LAB_INFO = {
    1: ("CPU is enough", "≈ 1 min, laptop CPU"), 2: ("GPU recommended; CPU works", "≈ 9 min, laptop CPU"),
    3: ("GPU recommended; CPU works", "≈ 19 min, laptop CPU"),
    4: ("GPU for Stable Diffusion; Part A on CPU", "≈ 17 min, laptop CPU (SD-Turbo path)"),
    5: ("GPU recommended; CPU works", "≈ 10 min, GTX 1650"), 6: ("CPU works; GPU faster", "≈ 8 min, laptop CPU"),
    7: ("GPU recommended", "≈ 58 min, GTX 1650"),
    8: ("GPU recommended; full CPU run is slow", "≈ 84 min, laptop CPU (training 53 min)"),
    9: ("GPU recommended; CPU works", "≈ 1 min, GTX 1650"),
    10: ("GPU recommended; CPU uses smaller models", "≈ 7 min, laptop CPU (Qwen3-VL-2B)"),
    11: ("GPU recommended; CPU works", "≈ 3 min, GTX 1650"),
    12: ("GPU recommended; CPU uses a smaller model", "≈ 5 min, GTX 1650"),
}
MODELS = [  # (model or dataset, weeks, licence label; check the exact release's full terms)
    ("SmolLM2-135M-Instruct (CPU core); 360M-Instruct (archive)", "1, 6, 11, 12; archive 1", "Apache-2.0"),
    ("GPT-2, BERT base (Hugging Face Hub)", "5, 6", "MIT; Apache-2.0"),
    ("Flan-T5 small / base (Google)", "6", "Apache-2.0"),
    ("Qwen2.5 0.5B / 1.5B Instruct, Qwen2.5 0.5B base, Qwen3 0.6B (Alibaba)", "6, 7, 8, 11, 12", "Apache-2.0"),
    ("Stable Diffusion 1.5", "4", "CreativeML OpenRAIL-M (use restrictions)"),
    ("SD-Turbo (Stability AI)", "4, 10", "Stability AI Community License; conditional commercial use, registration and revenue limits"),
    ("CLIP ViT-B/32 (OpenAI)", "4, 9, 10", "MIT (OpenAI CLIP repository)"),
    ("Whisper tiny / base (OpenAI)", "9, 10", "Apache-2.0 (model card)"),
    ("SmolVLM-256M-Instruct (CPU core); Qwen3-VL-2B-Instruct (archive/stretch)", "10", "Apache-2.0"),
    ("MMS-TTS English (Meta)", "10", "CC BY-NC 4.0 (non-commercial)"),
    ("all-MiniLM-L6-v2 (Sentence Transformers; CPU core)", "12", "Apache-2.0"),
    ("BGE-small-en v1.5 (BAAI); ms-marco MiniLM cross-encoder (archive/stretch)", "12", "MIT; Apache-2.0"),
    ("MNIST, Fashion-MNIST (torchvision)", "2, 3, 4", "Research and teaching use; Fashion-MNIST MIT"),
    ("Tiny Shakespeare; makemore names (Karpathy)", "5; 1", "Public-domain text; MIT"),
    ("Banking77 (PolyAI)", "8", "CC BY 4.0 (original release)"),
    ("CIFAR-10; LibriSpeech sample clips", "9", "Research use; CC BY 4.0"),
]


def weeks_spec() -> list[dict]:
    rows = yaml.safe_load((CUR / "section_7_3.yaml").read_text(encoding="utf-8"))["weeks"]
    for r in rows:
        wk = yaml.safe_load((CUR / "weeks" / f"week_{r['week']:02d}.yaml").read_text(encoding="utf-8"))
        r["mimlos"] = wk["mimlos"]
        lab = (CUR / "labs" / f"week_{r['week']:02d}_lab.py").read_text(encoding="utf-8")
        heading = re.search(r"^# # Week (\d+) Lab: (.+)$", lab, flags=re.M | re.I)
        if heading is None or int(heading.group(1)) != r["week"]:
            raise ValueError(f"Week {r['week']}: missing or mismatched lab heading")
        r["lab_title"] = heading.group(2).strip()
    return rows


def table(header: list[str], rows: list[list]) -> str:
    esc = lambda c: str(c).replace("|", "/").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows])


def latest_cpu_core_runs(week_numbers: list[int], ledger: dict | None = None) -> dict[int, dict]:
    """Select measured current cores; never substitute archived or smoke timings."""
    if ledger is None:
        ledger = json.loads((ROOT / "research" / "lab_runs_2026-10.json").read_text(encoding="utf-8"))
    latest: dict[int, tuple[datetime, int, dict]] = {}
    for index, run in enumerate(ledger.get("iteration_3_full_runs", [])):
        runtime = run.get("runtime") or {}
        if (run.get("week") not in week_numbers or run.get("ok") is not True
                or run.get("mode") != "full" or run.get("path") != "core"
                or run.get("requested_device") != "cpu" or runtime.get("device") != "cpu"):
            continue
        try:
            started = datetime.fromisoformat(run["started_at_utc"].replace("Z", "+00:00"))
            seconds = float(run["seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Week {run.get('week')}: invalid CPU core run record") from exc
        if started.tzinfo is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Week {run['week']}: runtime needs positive seconds and a timezone-aware date")
        if not run.get("notebook_sha256") or not run.get("executed_notebook_sha256"):
            raise ValueError(f"Week {run['week']}: CPU core run lacks notebook evidence hashes")
        candidate = (started.astimezone(timezone.utc), index, run)
        if run["week"] not in latest or candidate[:2] > latest[run["week"]][:2]:
            latest[run["week"]] = candidate
    missing = sorted(set(week_numbers) - set(latest))
    if missing:
        raise ValueError(f"Missing successful iteration 3 CPU core measurements for weeks {missing}")
    return {week: latest[week][2] for week in week_numbers}


def cpu_lab_table(weeks: list[dict], runs: dict[int, dict]) -> str:
    rows = []
    for week in weeks:
        run = runs[week["week"]]
        threads = run["runtime"].get("cpu_threads")
        hardware = run.get("hardware", "CPU")
        if threads is not None:
            hardware += f"; {threads} threads"
        started = datetime.fromisoformat(run["started_at_utc"].replace("Z", "+00:00"))
        seconds = float(run["seconds"])
        rows.append([week["week"], week["lab_title"], hardware,
                     f"{seconds:.1f} s ({seconds / 60:.2f} min)", started.astimezone(timezone.utc).date().isoformat()])
    return table(["Week", "Lab", "Measured CPU setup", "Cached core runtime", "Date (UTC)"], rows)


def cpu_lab_summary(runs: dict[int, dict]) -> str:
    """Keep the course-level budget claim tied to the selected records."""
    slowest = max(runs.values(), key=lambda run: float(run["seconds"]))
    within = sum(float(run["seconds"]) <= 900 for run in runs.values())
    return (f"{within} of {len(runs)} latest recorded CPU cores meet the 15-minute waiting budget. "
            f"The longest is Week {slowest['week']}: {float(slowest['seconds']):.1f} seconds "
            f"({float(slowest['seconds']) / 60:.2f} minutes) on its recorded CPU setup.")


def original_rows() -> dict[int, dict]:
    """Topic, Detail and Tutorials of the head professor's original Section 7.3 table."""
    with zipfile.ZipFile(bd.SOURCE) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    start, end = bd.locate_table(xml)
    tbl = bd.parse_fragment(xml[start:end], re.search(r"<w:document\b[^>]*>", xml).group(0))
    out = {}
    for tr in tbl.findall(bd.w("tr"))[1:]:
        cells = tr.findall(bd.w("tc"))
        paras = lambda tc: [" ".join("".join(t.text or "" for t in p.iter(bd.w("t"))).split()) for p in tc.iter(bd.w("p"))]  # noqa: E731
        week = int(bd.cell_text(cells[1]))
        out[week] = {"topic": bd.cell_text(cells[0]), "detail": " ".join(x for x in paras(cells[2]) if x),
                     "tutorials": [x for x in paras(cells[3]) if x]}
    return out


def instructor_guide(weeks: list[dict]) -> str:
    md = (CUR / "course" / "instructor_guide.md").read_text(encoding="utf-8")
    week_map = table(["Week", "Lecture topic (descriptor 7.3)", "Lab: what students build", "MIMLOs"],
                     [[w["week"], w["topic"].rstrip("."), w["lab_title"], ", ".join(map(str, w["mimlos"]))] for w in weeks])
    matrix = table(["MIMLO"] + [str(w["week"]) for w in weeks],
                   [[f"MIMLO {m}"] + ["✓" if m in w["mimlos"] else "" for w in weeks] for m in range(1, 6)])
    runs = latest_cpu_core_runs([w["week"] for w in weeks])
    labs = cpu_lab_table(weeks, runs)
    archive = table(["Week", "Earlier experiment settings", "Archived full runtime (9 October 2026)"],
                    [[w["week"], *LAB_INFO[w["week"]]] for w in weeks])
    models = table(["Model or dataset", "Weeks", "Licence (see the card)"], [list(m) for m in MODELS])
    for k, v in {"WEEK_MAP": week_map, "MIMLO_MATRIX": matrix, "LAB_TABLE": labs,
                 "LAB_SUMMARY": cpu_lab_summary(runs), "ARCHIVED_LAB_TABLE": archive, "MODEL_TABLE": models}.items():
        md = md.replace("{{" + k + "}}", v)
    return md


def rationale(weeks: list[dict]) -> str:
    orig = original_rows()
    rat = {r["week"]: r for r in yaml.safe_load((CUR / "course" / "rationale_7_3.yaml").read_text(encoding="utf-8"))["weeks"]}
    sources = {s["id"]: s for s in yaml.safe_load((CUR / "sources.yaml").read_text(encoding="utf-8"))["sources"]}
    for w in weeks:  # the professor's lecture topics and week order must be unchanged
        assert " ".join(orig[w["week"]]["topic"].split()) == " ".join(w["topic"].split()), w["week"]
    md = [(CUR / "course" / "rationale_intro.md").read_text(encoding="utf-8").strip(), "", "# Summary of changes by week", ""]
    md.append(table(["Week", "Topic (unchanged)", "Main additions to Detail", "Why"],
                    [[w["week"], w["topic"].rstrip("."), rat[w["week"]]["added"], rat[w["week"]]["why"]] for w in weeks]))
    md += ["", "<!-- pagebreak -->", "", "# Week-by-week comparison", "",
           "For each week: the original text, the proposed text, and the reasons. Topics and week order are unchanged in every row.", ""]
    for w in weeks:
        n, o, r = w["week"], orig[w["week"]], rat[w["week"]]
        md += [f"## Week {n}: {w['topic'].rstrip('.')}", ""]
        md.append(table(["", "Original", "Proposed"], [
            ["Detail", o["detail"], " ".join(w["detail"].split())],
            ["Tutorials", " / ".join(o["tutorials"]), " / ".join(" ".join(t.split()) for t in w["tutorials"])],
        ]))
        refs = "; ".join(f"[{sources[s]['label']}]({sources[s]['url']})" for s in r["sources"])
        md += ["", f"**Change to Tutorials.** {r['tutorials']}", "", f"**Why.** {r['why']}", "", f"**Key sources.** {refs}", ""]
    return "\n".join(md)


def render(name: str, meta: dict, md: str) -> Path:
    BUILD.mkdir(parents=True, exist_ok=True)
    eqs: dict = {}
    spec = {"kind": "course", **meta, "notes_md": prepare_notes(md, eqs), "assets": {"figures": {}, "eqs": eqs}}
    spec_path = BUILD / f"{name}.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    out = OUT / f"{name}.docx"
    r = subprocess.run(["node", str(ROOT / "tools" / "notes.js"), str(spec_path), str(out)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr)
    print(r.stdout.strip())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-only", action="store_true", help="rebuild the instructor guide without rewriting the descriptor rationale")
    args = parser.parse_args()
    weeks = weeks_spec()
    render("Instructor_Guide", {"kicker": "COURSE GUIDE", "title": "Instructor Guide",
                                "compact_tables": True,
                                "subtitle": "Generative AI: how to teach the 12-week module with this pack",
                                "meta": "MSc in Artificial Intelligence  ·  12 weeks  ·  2-hour lecture + 2-hour lab per week"},
           instructor_guide(weeks))
    glossary = json.loads((CUR / "glossary.json").read_text(encoding="utf-8"))
    glossary_md = "# One vocabulary across the twelve weeks\n\nThese definitions also supply the starting vocabulary and quick glossaries in the weekly materials. Specific examples show how each term is used in that week's model.\n\n"
    glossary_md += table(["Term", "Plain meaning", "Weeks"],
                         [[p["term"], p["meaning"], ", ".join(map(str, p["weeks"]))] for p in glossary["terms"]])
    render("Course_Glossary", {"kicker": "COURSE REFERENCE", "title": glossary["title"],
                              "compact_tables": True,
                              "subtitle": "Plain-language definitions used throughout Weeks 1–12",
                              "meta": "MSc in Artificial Intelligence · October 2026"}, glossary_md)
    if args.guide_only:
        return 0
    render("Section_7.3_Rationale", {"kicker": "MODULE DESCRIPTOR", "title": "Section 7.3: proposed revision and rationale",
                                     "compact_tables": True,
                                     "subtitle": "Generative AI: what changes in the weekly content, what stays the same, and why",
                                     "meta": "For the module leader and programme team  ·  September 2026"},
           rationale(weeks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
