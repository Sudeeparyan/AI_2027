"""REAL GPT-2 attention numbers for the week 5 lecture summary.

The original make_week_05.py stores GPT-2 attention only as a picture with
small labels. This script measures the same sentence and heads again and saves
the numbers, so the lecture figure can be drawn at a readable size.

Run:  .venv-labs/Scripts/python.exe curriculum/assets/make_week_05_gpt2.py
Output: curriculum/assets/week_05/gpt2_attention.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

OUT = Path(__file__).resolve().parent / "week_05" / "gpt2_attention.json"
SENTENCE = "The animal didn't cross the street because it was too tired."
HEADS = [(0, 0), (4, 11), (9, 6)]  # zero-based (layer, head), as in gpt2_attention.png

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", attn_implementation="eager").eval()
enc = tok(SENTENCE, return_tensors="pt")
with torch.no_grad():
    out = model(**enc, output_attentions=True)
att = torch.stack(out.attentions)[:, 0]  # layers, heads, query, key
n = att.shape[-1]
future = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
tokens = [t.replace("Ġ", "·") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
result = {
    "model": "gpt2",
    "sentence": SENTENCE,
    "tokens": tokens,
    "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "layers": int(att.shape[0]),
    "heads_per_layer": int(att.shape[1]),
    "max_future_weight": float(att[..., future].max()),
    "heads": [{"layer": l + 1, "head": h + 1, "weights": [[round(float(v), 4) for v in row] for row in att[l, h]]}
              for l, h in HEADS],
}
OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(f"wrote {OUT}: {len(tokens)} tokens, max future weight {result['max_future_weight']}")
