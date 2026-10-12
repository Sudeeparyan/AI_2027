"""Show whether any executable lab code changed after the last full run of each week.

Compares the code cells of build/week_NN/executed_solutions_full.ipynb (written by test_labs.py --full) with
the built solutions notebook in deliverables/. Comment-only and blank lines are ignored, so editing Markdown,
comments or the runtime header does not count as a change. The runner skips %pip install cells and the
Week 11 demo.launch cell, so those appear only in the built notebook.

A stored result (results.json) or a quoted runtime stays valid only while this reports the same code.

Usage: .venv/Scripts/python.exe tools/compare_lab_code.py [--weeks 1-12] [-v]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED = ("%pip install", "demo.launch(")  # cells the runner does not execute
SKIP_TAGS = {"colab-install", "skip-test"}


def normalise_code(source):
    """Ignore comments and formatting, retaining literals and statement order."""
    try:
        return ast.dump(ast.parse(source), include_attributes=False)
    except SyntaxError:
        # Preserve notebook magics verbatim rather than guessing their meaning.
        return source.strip()


def code_cells(path):
    cells = []
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        if SKIP_TAGS & set(cell.get("metadata", {}).get("tags", [])):
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if src.strip() and not src.lstrip().startswith(SKIPPED):
            code = normalise_code(src)
            if code != "Module(body=[], type_ignores=[])":
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
        record = ROOT / f"build/week_{week:02d}/lab_test_full.json"
        if not full.exists() or built is None or not record.exists():
            print(f"week {week:2d}: no full run or no built notebook")
            status = 1
            continue
        result = json.loads(record.read_text(encoding="utf-8"))
        if result.get("ok") is not True or result.get("mode") != "full" or result.get("week") != week:
            print(f"week {week:2d}: no successful full-run record")
            status = 1
            continue
        expected_hash = result.get("executed_notebook_sha256")
        if expected_hash and hashlib.sha256(full.read_bytes()).hexdigest() != expected_hash:
            print(f"week {week:2d}: executed notebook does not match its run record")
            status = 1
            continue
        a, b = code_cells(full), code_cells(built)
        same = a == b
        status |= not same
        print(f"week {week:2d}: {'same code' if same else 'CODE CHANGED'} since the full run "
              f"({len(a)} executed code cells)")
        if args.v:
            for i in range(max(len(a), len(b))):
                old = a[i] if i < len(a) else "<missing>"
                new = b[i] if i < len(b) else "<missing>"
                if old != new:
                    print(f"   executed cell {i + 1}, full: {old[:300]}")
                    print(f"   executed cell {i + 1}, built: {new[:300]}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
