"""Copy figures produced by an executed lab notebook into curriculum/assets/week_XX/ for use in slides and notes.

Figures are located by a phrase that appears in the source of the cell that produced them, so re-ordering cells
does not break the mapping.

Only a successful mode-specific full run is accepted. A later smoke run cannot
replace its figures. All requested figures are validated before any asset changes.

Usage: .venv/Scripts/python.exe tools/harvest_figures.py --week 9 [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import nbformat
from PIL import Image
from compare_lab_code import code_cells

ROOT = Path(__file__).resolve().parents[1]

# week -> {asset name: (phrase in cell source, index of the image output within that cell)}
MAPS = {
    9: {"lab_similarity": ("sim = img_emb[show_idx] @ txt_emb.T", 0), "lab_confusion": ("Zero-shot CLIP on CIFAR-10", 0),
        "lab_retrieval": ("top-6 images for", 0), "lab_modality_gap": ("modality gap", 0),
        "lab_spectrogram": ("plt.specgram(", 0)},
    10: {"lab_caption": ("CAPTION_FIGURE", 0), "lab_t2i": ("T2I_FIGURE", 0), "lab_i2i": ("I2I_FIGURE", 0),
         "lab_chart": ("CHART_FIGURE", 0)},
    11: {"lab_latency": ("LAT_FIGURE", 0), "lab_quant": ("QUANT_FIGURE", 0), "lab_batch": ("BATCH_FIGURE", 0),
         "lab_monitor": ("MONITOR_FIGURE", 0)},
    12: {"lab_retrieval": ("RETRIEVAL_FIGURE", 0), "lab_rag": ("RAG_FIGURE", 0)},
}


def full_notebook(week: int, root: Path = ROOT):
    """Reject failed, stale or incomplete execution before touching source assets."""
    if week not in MAPS:
        raise ValueError("Figure harvesting is supported for weeks 9–12.")
    run_dir = root / "build" / f"week_{week:02d}"
    nb_path = run_dir / "executed_solutions_full.ipynb"
    record_path = run_dir / "lab_test_full.json"
    if not nb_path.is_file() or not record_path.is_file():
        raise ValueError(f"Week {week}: run tools/test_labs.py --weeks {week} --full first.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("ok") is not True or record.get("mode") != "full" or record.get("week") != week:
        raise ValueError(f"Week {week}: no successful full-run record; figures were not changed.")
    expected_hash = record.get("executed_notebook_sha256")
    if expected_hash and hashlib.sha256(nb_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Week {week}: executed notebook does not match its run record.")
    built = list((root / "deliverables").glob(f"Week_{week:02d}_*/Week_{week:02d}_Lab_Solutions.ipynb"))
    if len(built) != 1 or code_cells(nb_path) != code_cells(built[0]):
        raise ValueError(f"Week {week}: rebuild and run the current lab in full; executed code differs.")
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip():
            if cell.get("execution_count") is None or any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                raise ValueError(f"Week {week}: full notebook has an unexecuted or failed cell.")
    return nb, nb_path, record


def harvest(week: int, *, dry_run: bool = False, root: Path = ROOT) -> list[str]:
    nb, nb_path, record = full_notebook(week, root)
    figures = {}
    for name, (phrase, idx) in MAPS[week].items():
        matches = [c for c in nb.cells if c.cell_type == "code" and phrase in c.source]
        if len(matches) != 1:
            raise ValueError(f"Week {week}: {name} must match exactly one code cell; found {len(matches)}.")
        pngs = [o["data"]["image/png"] for o in matches[0].get("outputs", []) if "data" in o and "image/png" in o["data"]]
        if len(pngs) > idx:
            data = base64.b64decode("".join(pngs[idx].split()), validate=True)
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Week {week}: {name} is not a PNG image.")
            with Image.open(BytesIO(data)) as image:
                image.verify()
            figures[name] = data
    missing = sorted(set(MAPS[week]) - set(figures))
    if missing:
        print(f"week {week}: missing {missing}; no assets changed")
        return missing
    if dry_run:
        print(json.dumps({"week": week, "source": str(nb_path), "figures": sorted(figures),
                          "full_run_ok": True, "assets_changed": False,
                          "executed_hash_in_record": bool(record.get("executed_notebook_sha256"))}))
    else:
        out_dir = root / "curriculum" / "assets" / f"week_{week:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in figures.items():
            (out_dir / f"{name}.png").write_bytes(data)
        print(f"week {week}: saved {sorted(figures)} from the successful full run")
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate without changing source assets")
    args = ap.parse_args()
    try:
        sys.exit(1 if harvest(args.week, dry_run=args.dry_run) else 0)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
