"""Insert the measured week 5 results (curriculum/assets/week_05/metrics.json) into the slides and notes placeholders."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "curriculum/assets/week_05/metrics.json").read_text(encoding="utf-8"))
c = m["chargpt"]


def clean(t, n=95):
    t = t.replace("ROMEO:\n", "", 1)
    t = re.sub(r"\n+", " / ", re.sub(r'[*`"\\]', "", t))  # blank lines between speakers become one separator
    return (t[:n] + "…") if len(t) > n else t


desc = (f"{m['chargpt_params'] / 1e6:.1f} M-parameter model, {c['steps']} steps ({c['minutes']:.0f} min on a laptop GPU): "
        f"validation loss {c['val_loss_start']:.2f} → {c['val_loss_end']:.2f} nats/char, perplexity {c['val_ppl_end']:.1f} (uniform: {c['vocab']}).")
slides = {"{{CHARGPT_DESC}}": desc, "{{SAMPLE_0}}": clean(m["chargpt_samples"]["0"], 60),
          "{{SAMPLE_300}}": clean(m["chargpt_samples"]["300"]), "{{SAMPLE_END}}": clean(m["chargpt_samples"][str(c["steps"])])}
b = m["bert_it_animal_head"]
final = clean(m["chargpt_samples"][str(c["steps"])], 140)
notes = {"{{BERT_LAYER}}": str(b["layer"]), "{{BERT_HEAD}}": str(b["head"]), "{{BERT_WEIGHT}}": f"{b['weight']:.2f}",
         "{{CHARGPT_NOTES}}": (f"A {m['chargpt_params'] / 1e6:.1f} M-parameter character-level transformer (4 layers, 6 heads, d = 192, context 128) "
                               f"trained for {c['steps']} steps on Tiny Shakespeare ({c['minutes']:.0f} minutes on a GTX 1650 laptop GPU) reduced the validation "
                               f"loss from {c['val_loss_start']:.2f} to {c['val_loss_end']:.2f} nats per character (perplexity {c['val_ppl_end']:.1f}; uniform "
                               f"guessing over {c['vocab']} characters would give {c['vocab']}). Samples went from random characters (step 0) to word-like "
                               f"fragments with line structure (step 300) to Shakespeare-like dialogue with speaker names (final): \"{final}\" "
                               "It has learned spelling, format and style, not meaning.")}

for rel, rep in (("curriculum/weeks/week_05.yaml", slides), ("curriculum/notes/week_05.md", notes)):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    for k, v in rep.items():
        if k in s:
            s = s.replace(k, v)
    p.write_text(s, encoding="utf-8")
print(desc)
