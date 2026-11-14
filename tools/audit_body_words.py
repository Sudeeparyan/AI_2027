"""Enforce the 40-word limit on ordinary slide bodies in all built decks.

Count projected body text, including tables, labels and callouts. Titles and
running chrome are excluded. Diagram scenes retain their separate readability
and visual checks; this text-shape audit does not count words inside images.
"""
import json
from pathlib import Path

from verify_pilot_artifacts import readability

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = []
    for week in range(1, 13):
        spec = json.loads((ROOT / f"build/week_{week:02d}/spec.json").read_text(encoding="utf-8"))
        deck = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*/Week_{week:02d}_Lecture_Slides.pptx"))
        result = readability(deck, spec)
        flags = [row for row in result["ordinary_slide_counts"] if row["above_40_words"]]
        report.append({"week": week, "flags": flags, "readability": result})
        print(f"week {week:02d}: {len(flags)} ordinary slide bodies above 40 words")
        for row in flags:
            print(f"  slide {row['slide']}: {row['body_word_count']} words — {row['title']}")
    (ROOT / "build/body_word_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(row["flags"] for row in report))


if __name__ == "__main__":
    raise SystemExit(main())
