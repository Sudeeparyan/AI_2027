"""REAL BERT attention numbers for the week 5 teaching notes.

The original make_week_05.py stores the selected BERT head only as a picture
with small labels. This script measures the same sentence again, finds the head
with the largest weight from "it" to "animal" (as the lab's TODO 5 does) and
saves that head's full table, so the notes can draw it at a readable size.

Run:  .venv-labs/Scripts/python.exe curriculum/assets/make_week_05_bert.py
Output: curriculum/assets/week_05/bert_attention.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

OUT = Path(__file__).resolve().parent / "week_05" / "bert_attention.json"
SENTENCE = "The animal didn't cross the street because it was too tired."

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").eval()
enc = tok(SENTENCE, return_tensors="pt")
with torch.no_grad():
    att = torch.stack(model(**enc, output_attentions=True).attentions)[:, 0]  # layers, heads, query, key
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
i_it, i_animal = tokens.index("it"), tokens.index("animal")
scores = att[:, :, i_it, i_animal]
layer, head = divmod(int(scores.argmax()), att.shape[1])
result = {
    "model": "bert-base-uncased",
    "sentence": SENTENCE,
    "tokens": tokens,
    "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "layer": layer + 1,
    "head": head + 1,
    "it_to_animal": round(float(scores.max()), 4),
    "heads": int(scores.numel()),
    "median_it_to_animal": round(float(scores.flatten().median()), 4),
    "heads_above_half": int((scores > 0.5).sum()),
    "weights": [[round(float(v), 4) for v in row] for row in att[layer, head]],
}
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(f"wrote {OUT}: layer {layer + 1}, head {head + 1}, it -> animal {result['it_to_animal']}")
