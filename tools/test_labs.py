"""Execute each week's SOLUTION notebook top to bottom and report failures.

Runs with the lab environment (.venv-labs), which has PyTorch, Transformers, Diffusers, PEFT, TRL, etc.
Cells tagged "colab-install" or "skip-test" are skipped (packages are pre-installed locally).
GENAI_LAB_SMOKE=1 makes notebooks use tiny settings (fewer epochs / smaller models) so the check is quick;
students on Colab run the full settings.

Run: .venv-labs/Scripts/python.exe tools/test_labs.py --weeks 1,2 [--full]
Writes build/week_XX/lab_test.json and the executed notebook build/week_XX/executed_solutions.ipynb
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
SKIP_TAGS = {"colab-install", "skip-test"}


def run(week: int, full: bool, timeout: int) -> dict:
    src = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*")) / f"Week_{week:02d}_Lab_Solutions.ipynb"
    nb = nbformat.read(str(src), as_version=4)
    nb.cells = [c for c in nb.cells if not SKIP_TAGS & set(c.get("metadata", {}).get("tags", []))]
    if not full:
        os.environ["GENAI_LAB_SMOKE"] = "1"
    else:
        os.environ.pop("GENAI_LAB_SMOKE", None)
    os.environ.pop("MPLBACKEND", None)  # keep the inline backend so figures are saved in the executed notebook
    workdir = ROOT / "build" / f"week_{week:02d}" / "lab_run"
    workdir.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
    t0 = time.time()
    result = {"week": week, "mode": "full" if full else "smoke", "ok": True, "error": None}
    try:
        client.execute()
    except CellExecutionError as e:
        result["ok"] = False
        result["error"] = str(e)[-3000:]
    except Exception as e:  # noqa: BLE001 - kernel death, timeouts
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"[-3000:]
    result["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "build" / f"week_{week:02d}"
    nbformat.write(nb, str(out / "executed_solutions.ipynb"))
    (out / "lab_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    a = ap.parse_args()
    weeks = sorted({w for part in a.weeks.split(",") for w in (range(1, 13) if part == "all" else range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
    bad = 0
    for w in weeks:
        r = run(w, a.full, a.timeout)
        print(f"week {w:02d} lab ({r['mode']}): {'PASS' if r['ok'] else 'FAIL'} in {r['seconds']} s")
        if not r["ok"]:
            bad += 1
            print(r["error"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
