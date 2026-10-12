"""Execute each week's SOLUTION notebook top to bottom and report failures.

Runs with the lab environment (.venv-labs), which has PyTorch, Transformers, Diffusers, PEFT, TRL, etc.
Cells tagged "colab-install" or "skip-test" are skipped (packages are pre-installed locally).
GENAI_LAB_SMOKE=1 makes notebooks use tiny settings (fewer epochs / smaller models) so the check is quick;
students run the measured CPU core settings; longer training is an optional stretch.

Run: .venv-labs/Scripts/python.exe tools/test_labs.py --weeks 1,2 [--full]
Writes build/week_XX/lab_test.json and the executed notebook build/week_XX/executed_solutions.ipynb
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
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


@contextmanager
def runtime_environment(full: bool, device: str):
    """Keep one notebook run's settings out of later runs in the same process."""
    keys = ("GENAI_LAB_SMOKE", "GENAI_FORCE_CPU", "GENAI_RUNNER_GPU", "CUDA_VISIBLE_DEVICES", "MPLBACKEND")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if full:
            os.environ.pop("GENAI_LAB_SMOKE", None)
        else:
            os.environ["GENAI_LAB_SMOKE"] = "1"
        os.environ["GENAI_FORCE_CPU"] = "1" if device == "cpu" else "0"
        os.environ["GENAI_RUNNER_GPU"] = "1" if device == "cuda" else "0"
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.pop("MPLBACKEND", None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(week: int, full: bool, timeout: int, device: str = "cpu", progress: bool = False) -> dict:
    with runtime_environment(full, device):
        return execute_lab(week, full, timeout, device, progress)


def execute_lab(week: int, full: bool, timeout: int, device: str, progress: bool) -> dict:
    src = next((ROOT / "deliverables").glob(f"Week_{week:02d}_*")) / f"Week_{week:02d}_Lab_Solutions.ipynb"
    nb = nbformat.read(str(src), as_version=4)
    nb.cells = [c for c in nb.cells if not SKIP_TAGS & set(c.get("metadata", {}).get("tags", []))]
    workdir = ROOT / "build" / f"week_{week:02d}" / "lab_run"
    workdir.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(workdir)}})
    if progress:
        def report_cell(cell, cell_index):
            if cell.cell_type == "code":
                print(f"week {week:02d}: running code cell {cell_index + 1}/{len(nb.cells)}", flush=True)
        client.on_cell_start = report_cell
    t0 = time.time()
    mode = "full" if full else "smoke"
    result = {"week": week, "mode": mode, "ok": True, "error": None,
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "timeout_per_cell": timeout, "notebook_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
              "runner_python": sys.executable, "requested_device": device, "path": "core" if full else "smoke"}
    try:
        client.execute()
    except CellExecutionError as e:
        result["ok"] = False
        result["error"] = str(e)[-3000:]
    except Exception as e:  # noqa: BLE001 - kernel death, timeouts
        result["ok"] = False
        result["error"] = f"{type(e).__name__}: {e}"[-3000:]
    result["seconds"] = round(time.time() - t0, 1)
    if full and device == "cpu":
        result["cpu_core_budget_seconds"] = 900
        result["within_cpu_core_budget"] = result["seconds"] <= 900
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            for line in output.get("text", "").splitlines():
                if line.startswith("Runtime: "):
                    try:
                        result["runtime"] = json.loads(line[len("Runtime: "):])
                    except ValueError:
                        pass
    out = ROOT / "build" / f"week_{week:02d}"
    nbformat.write(nb, str(out / "executed_solutions.ipynb"))
    executed_path = out / f"executed_solutions_{mode}.ipynb"
    nbformat.write(nb, str(executed_path))
    result["executed_notebook_sha256"] = hashlib.sha256(executed_path.read_bytes()).hexdigest()
    (out / "lab_test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"lab_test_{mode}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="1")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--progress", action="store_true", help="show code-cell progress without printing source or secrets")
    a = ap.parse_args()
    weeks = sorted({w for part in a.weeks.split(",") for w in (range(1, 13) if part == "all" else range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
    bad = 0
    busy = ROOT / "build" / ".busy"
    try:
        with busy.open("x", encoding="utf-8") as handle:
            handle.write(f"lab-test {a.weeks} {a.device} {'full' if a.full else 'smoke'}")
    except FileExistsError:
        print(f"Machine busy: {busy.read_text(encoding='utf-8')}. Retry after that task releases it.")
        return 2
    try:
        for w in weeks:
            r = run(w, a.full, a.timeout, a.device, a.progress)
            print(f"week {w:02d} lab ({r['mode']}): {'PASS' if r['ok'] else 'FAIL'} in {r['seconds']} s", flush=True)
            if not r["ok"]:
                bad += 1
                print(r["error"], flush=True)
            elif r.get("within_cpu_core_budget") is False:
                bad += 1
                print("CPU core exceeds 15 minutes: reduce its waiting time before release.", flush=True)
    finally:
        busy.unlink(missing_ok=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
