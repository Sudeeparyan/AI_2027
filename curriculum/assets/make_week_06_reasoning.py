"""REAL results for week 6, part 5: a small reasoning model (Qwen3-0.6B) with thinking on vs off.

Each problem is run three times per mode with the sampling settings from the Qwen3 model card
(thinking: temperature 0.6, top-p 0.95, top-k 20; non-thinking: temperature 0.7, top-p 0.8, top-k 20).
Greedy decoding is NOT used: in thinking mode it makes Qwen3 repeat itself until the token budget runs out,
which is what happened in the first version of this script (4,096 tokens, no answer).

Adds a "reasoning" entry to curriculum/assets/week_06/metrics.json (run make_week_06.py first).
Run:  .venv-gpu/Scripts/python.exe curriculum/assets/make_week_06_reasoning.py   (≈ 15 min on a small GPU)
"""
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parent / "week_06"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen3-0.6B"
SEEDS = (0, 1, 2)
BUDGET = {True: 2048, False: 300}  # max new tokens with / without thinking
SAMPLING = {True: dict(temperature=0.6, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}
PROBLEMS = [  # (name, question, correct answer)
    ("word problem", "Tom has 3 boxes with 4 apples each. He gives away 5 apples, then buys twice as many apples as he has left. "
                     "How many apples does he have now? Answer with a number.", 21),
    ("scheduling puzzle", "A lab has 3 GPUs. Each training run needs 2 GPUs for 4 hours. How many hours are needed to finish 5 runs "
                          "if runs can execute in parallel whenever GPUs are free? Answer with a number.", 20),
]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEV).eval()


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def run(question, think, seed):
    torch.manual_seed(seed)
    inputs = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True, enable_thinking=think).to(DEV)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=BUDGET[think], do_sample=True, pad_token_id=tok.eos_token_id, **SAMPLING[think])
    new = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    finished = len(new) < BUDGET[think]
    answer = text.split("</think>")[-1] if think else text  # the part after the reasoning block
    return {"seed": seed, "tokens": len(new), "seconds": round(time.time() - t0, 1), "finished": finished,
            "answer": last_number(answer) if finished else None, "answer_tail": answer.strip()[-200:]}


results = {}
for name, question, gold in PROBLEMS:
    results[name] = {"question": question, "correct_answer": gold}
    for think in (False, True):
        runs = [run(question, think, s) for s in SEEDS]
        for r in runs:
            r["correct"] = r["answer"] is not None and abs(r["answer"] - gold) < 1e-6
            print(name, "thinking" if think else "no thinking", {k: r[k] for k in ("seed", "tokens", "seconds", "answer", "correct")}, flush=True)
        results[name]["thinking" if think else "no_thinking"] = {
            "runs": runs, "correct": sum(r["correct"] for r in runs), "n": len(runs),
            "mean_tokens": round(sum(r["tokens"] for r in runs) / len(runs)), "mean_seconds": round(sum(r["seconds"] for r in runs) / len(runs), 1),
            "unfinished": sum(not r["finished"] for r in runs), "budget": BUDGET[think]}

metrics_path = OUT / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
metrics["reasoning"] = {"model": MODEL_ID, "device": DEV, "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU",
                        "sampling": {"thinking": SAMPLING[True], "no_thinking": SAMPLING[False]}, "problems": results}
metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print("updated", metrics_path)
