"""Show whether any executable lab code changed after the last full run of each week.

Compares the code cells of build/week_NN/executed_solutions_full.ipynb (written by test_labs.py --full) with
the built solutions notebook in deliverables/. Comment-only and blank lines are ignored, so editing Markdown,
comments or the runtime header does not count as a change. The runner skips %pip install cells and the
Week 11 demo.launch cell, so those appear only in the built notebook.

A stored result (results.json) or a quoted runtime stays valid only while this reports the same code.

Usage: .venv/Scripts/python.exe tools/compare_lab_code.py [--weeks 1-12] [-v]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED = ("%pip install", "demo.launch(")  # cells the runner does not execute


def code_cells(path):
    cells = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        lines = [l.rstrip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
        code = "\n".join(lines)
        if code and not code.startswith(SKIPPED):
            cells.append(code)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1-12")
    ap.add_argument("-v", action="store_true", help="print the differing cells")
    args = ap.parse_args()
    first, _, last = args.weeks.partition("-")
    status = 0
    for week in range(int(first), int(last or first) + 1):
        full = ROOT / f"build/week_{week:02d}/executed_solutions_full.ipynb"
        built = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*/Week_{week:02d}_Lab_Solutions.ipynb"), None)
        if not full.exists() or built is None:
            print(f"week {week:2d}: no full run or no built notebook")
            continue
        a, b = code_cells(full), code_cells(built)
        only_full = [c for c in a if c not in set(b)]
        only_built = [c for c in b if c not in set(a)]
        same = not only_full and not only_built
        status |= not same
        print(f"week {week:2d}: {'same code' if same else 'CODE CHANGED'} since the full run "
              f"({len(a)} executed code cells)")
        if args.v:
            for c in only_full:
                print("   full run only:", c[:300].replace("\n", " | "))
            for c in only_built:
                print("   built only:   ", c[:300].replace("\n", " | "))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
