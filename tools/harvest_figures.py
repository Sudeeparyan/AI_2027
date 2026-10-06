"""Copy figures produced by an executed lab notebook into curriculum/assets/week_XX/ for use in slides and notes.

Figures are located by a phrase that appears in the source of the cell that produced them, so re-ordering cells
does not break the mapping.

Usage: .venv/Scripts/python.exe tools/harvest_figures.py --week 9
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]

# week -> {asset name: (phrase in cell source, index of the image output within that cell)}
MAPS = {
    9: {"lab_similarity": ("cosine similarity", 0), "lab_confusion": ("Zero-shot CLIP on CIFAR-10", 0),
        "lab_retrieval": ("top-6 images for", 0), "lab_modality_gap": ("modality gap", 0),
        "lab_spectrogram": ("plt.specgram(", 0)},
    10: {"lab_caption": ("CAPTION_FIGURE", 0), "lab_t2i": ("T2I_FIGURE", 0), "lab_i2i": ("I2I_FIGURE", 0),
         "lab_chart": ("CHART_FIGURE", 0)},
    11: {"lab_latency": ("LAT_FIGURE", 0), "lab_quant": ("QUANT_FIGURE", 0), "lab_batch": ("BATCH_FIGURE", 0),
         "lab_monitor": ("MONITOR_FIGURE", 0)},
    12: {"lab_retrieval": ("RETRIEVAL_FIGURE", 0), "lab_rag": ("RAG_FIGURE", 0)},
}


def harvest(week: int) -> list[str]:
    nb_path = ROOT / "build" / f"week_{week:02d}" / "executed_solutions.ipynb"
    nb = nbformat.read(str(nb_path), as_version=4)
    out_dir = ROOT / "curriculum" / "assets" / f"week_{week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, (phrase, idx) in MAPS[week].items():
        for cell in nb.cells:
            if cell.cell_type != "code" or phrase not in cell.source:
                continue
            pngs = [o["data"]["image/png"] for o in cell.get("outputs", []) if "data" in o and "image/png" in o["data"]]
            if len(pngs) > idx:
                (out_dir / f"{name}.png").write_bytes(base64.b64decode(pngs[idx]))
                saved.append(name)
            break
    missing = sorted(set(MAPS[week]) - set(saved))
    print(f"week {week}: saved {saved}; missing {missing}")
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    sys.exit(1 if harvest(ap.parse_args().week) else 0)
