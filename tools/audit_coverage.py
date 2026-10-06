"""Check that every item in each week's Section 7.3 Detail and Tutorials is taught in that week's slides or notes.

Each Detail cell is split into items at punctuation; an item counts as covered when all of its content words
(minus a small stop list) appear in the week's slide text, speaker notes or teaching notes. Prints the items
that are not found, for a human to review (a word-level check cannot prove good teaching, only presence).

Usage: .venv/Scripts/python.exe tools/audit_coverage.py
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CUR = ROOT / "curriculum"
STOP = set("""a an the and or of for to in on with by as vs e g eg etc from into its their this that these those
be is are how what why use using used including such other key concepts overview applications motivation example examples
understand usage paradigm""".split())
ALIASES = {"regularization": "regularisation", "visualize": "visualise", "muti": "multi", "parallelisation": "parallel",
           "stabilisation": "stabil", "summarisation": "summar"}


def words(text: str) -> list[str]:
    ws = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
    ws += [part for w in ws if "-" in w for part in w.split("-")]  # "few-shot" also counts as "few" and "shot"
    return [ALIASES.get(w, w) for w in ws if w not in STOP and len(w) > 2]


def stem(w: str) -> str:
    """A crude stem: plural and verb endings removed, then the first five letters ("choosing" ~ "choose")."""
    w = re.sub(r"ies$", "y", w)
    w = re.sub(r"(ss|sh|ch|x)es$", r"\1", w)
    w = re.sub(r"(?<!s)s$", "", w)
    return re.sub(r"(ing|ed)$", "", w)[:5]


def main() -> int:
    rows = yaml.safe_load((CUR / "section_7_3.yaml").read_text(encoding="utf-8"))["weeks"]
    missing_total = 0
    for r in rows:
        n = r["week"]
        wk = yaml.safe_load((CUR / "weeks" / f"week_{n:02d}.yaml").read_text(encoding="utf-8"))
        text = json.dumps(wk, ensure_ascii=False) + (CUR / "notes" / f"week_{n:02d}.md").read_text(encoding="utf-8")
        have = {stem(w) for w in words(text)}
        items = [i.strip() for i in re.split(r"[.;:()]|, (?![^()]*\))", " ".join(r["detail"].split())) if len(i.strip()) > 3]
        missing = [i for i in items if not all(stem(w) in have for w in words(i))]
        missing_total += len(missing)
        print(f"week {n:2d}: {len(items) - len(missing)}/{len(items)} Detail items found" + (f"; check: {missing}" if missing else ""))
    print("items to review:", missing_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
