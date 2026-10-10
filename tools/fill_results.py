"""Turn measured results into the text that fills {{PLACEHOLDERS}} in the slides and teaching notes.

Reads curriculum/assets/week_XX/metrics.json (asset scripts) and results.json (instructor lab runs with
GENAI_RESULTS_PATH) and writes curriculum/assets/week_XX/fill.json. tools/build_week.py substitutes the
values at build time and refuses to build while any placeholder is unfilled.

Usage: .venv/Scripts/python.exe tools/fill_results.py --weeks 1-12
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "curriculum" / "assets"


def load(week: int, name: str) -> dict:
    p = ASSETS / f"week_{week:02d}" / name
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}%"


def short(model_id: str) -> str:
    return model_id.split("/")[-1]


def hardware(device: str, gpu: str | None = None) -> str:
    if device.startswith("cuda"):
        name = gpu or (device.split("(")[-1].rstrip(")") if "(" in device else "")
        if not name:
            return "a GPU"
        return f"an {name.replace('NVIDIA GeForce ', 'NVIDIA ')} laptop GPU" if "1650" in name else f"a {name} GPU"
    return "a laptop CPU"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def clean(text: str, n: int = 110) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip().replace("|", "/").replace("$", "")
    if len(t) <= n:
        return t
    cut = t[: n - 1]
    if " " in cut[n // 2:]:  # end on a whole word
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


def stem(question: str) -> str:
    """A question without its answer-format suffix; the week 10 lab stores only the first 55 characters, so mark a cut."""
    q = re.sub(r" *Answer with.*$", "", question).strip()
    return q if q.endswith("?") or len(question) < 55 else q[: q.rfind(" ")].rstrip(" ,;:") + "…"


def last_number(text: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return float(nums[-1]) if nums else None


# ---------------------------------------------------------------- week 1
def week1() -> dict:
    """Format the recorded full lab run (results.json written with GENAI_RESULTS_PATH)."""
    r = load(1, "results.json")
    if r.get("run_mode") != "full":
        raise ValueError("week 1 notes quote a full run; a smoke run must not replace it")
    when = dt.datetime.fromisoformat(r["measured_utc"])
    mat = {T: next(p for t, p in rows if t.strip() == "mat") for T, rows in r["top_tokens"].items()}
    others = [t.strip() for t, _ in r["top_tokens"]["1.0"] if t.strip() != "mat"]
    nll = {w: f"{v:.2f} nats (perplexity {math.exp(v):.1f})" for w, v in r["name_nll"].items()}
    return {
        "W01_RUN": (f"a full run on {when.day} {when:%B %Y} on {hardware(r['device'])} "
                    f"with {short(r['model_id'])} ({r['model_params'] / 1e6:.0f} million parameters)"),
        "W01_NAMES": f"{r['names']:,}",
        "W01_NLL_ANNA": nll["anna"], "W01_NLL_EMMA": nll["emma"], "W01_NLL_XQZZ": nll["xqzz"],
        "W01_UNIFORM": f"{r['dataset_nll']['uniform']:.2f}", "W01_UNIGRAM": f"{r['dataset_nll']['unigram']:.2f}",
        "W01_BIGRAM": f"{r['dataset_nll']['bigram']:.2f}",
        "W01_ACC_LR": f"{r['accuracy']['logistic_regression']:.3f}", "W01_ACC_QDA": f"{r['accuracy']['qda']:.3f}",
        "W01_TOKENS": (f"{r['token_counts']['English']} tokens in English, {r['token_counts']['Irish']} in Irish "
                       f"and {r['token_counts']['Hindi']} in Hindi"),
        "W01_MAT": f"{mat['0.5']:.2f} at T = 0.5, {mat['1.0']:.2f} at T = 1.0 and {mat['1.5']:.2f} at T = 1.5",
        "W01_RUNNERS_UP": ", ".join(f"*{t}*" for t in others[:-1]) + f" and *{others[-1]}*",
        # Quoted inside double quotes in the notes, so inner quotes become single quotes.
        "W01_FICTION_PLAIN": clean(r["fictional_paper"]["plain"], 150).replace('"', "'"),
        "W01_FICTION_HONEST": clean(r["fictional_paper"]["honesty_instruction"], 150).replace('"', "'"),
    }


# ---------------------------------------------------------------- week 2
def week2() -> dict:
    """Format the archived asset run (make_week_02.py): 2-D model, β models and the anomaly test."""
    m = load(2, "metrics.json")
    b = m["beta"]
    out = {"W02_AUC": f"{m['anomaly_auc']:.4f}", "W02_AUC_PCT": pct(m["anomaly_auc"], 2),
           "W02_REC2": f"{m['vae2_final']['rec']:.0f}", "W02_KL2": f"{m['vae2_final']['kl']:.1f}"}
    for key, name in [("0.1", "B01"), ("1.0", "B1"), ("4.0", "B4")]:
        out[f"W02_REC_{name}"] = f"{b[key]['rec']:.1f}"
        out[f"W02_KL_{name}"] = f"{b[key]['kl']:.1f}"
    return out


# ---------------------------------------------------------------- week 4
def week4() -> dict:
    m = load(4, "metrics.json")
    d = m["mnist_ddpm"]
    t = m.get("sd_times_seconds", {})
    parts = []
    for key, label in [("sd15_2steps", "SD 1.5, 2 steps"), ("sd15_10steps", "10 steps"), ("sd15_25steps", "25 steps"),
                       ("sdturbo_1steps", "SD-Turbo, 1 step"), ("sdturbo_4steps", "SD-Turbo, 4 steps")]:
        if key in t:
            parts.append(f"{label}: {t[key]:.0f} s")
    secs = {f"W04_{name}": f"{t[key]:.0f}" for name, key in
            [("T2", "sd15_2steps"), ("T5", "sd15_5steps"), ("T10", "sd15_10steps"), ("T25", "sd15_25steps"),
             ("G1", "sd15_25steps_g1.0"), ("G75", "sd15_25steps_g7.5"), ("TURBO1", "sdturbo_1steps"),
             ("TURBO4", "sdturbo_4steps")]}
    return {**secs, "UNET_PARAMS": f"{d['unet_params']:,}", "MNIST_EPOCHS": str(d["epochs"]), "MNIST_DEVICE": hardware(d["device"]),
            "MNIST_MIN": f"{d['train_minutes']:.0f}",
            "SD_DEVICE": ("a laptop CPU, float32" if str(m.get("sd_device", "cpu")).startswith("cpu") else hardware(m["sd_device"])),
            "SD_TIMES": "; ".join(parts) + " per 512 × 512 image"}


# ---------------------------------------------------------------- week 5
def week5() -> dict:
    """Format the archived full experiment; a smoke run must never replace it."""
    m = load(5, "metrics.json")
    bert = load(5, "bert_attention.json")
    gpt2 = load(5, "gpt2_attention.json")
    c = m["chargpt"]
    samples = m["chargpt_samples"]
    # A later full run of the lab notebook itself (GENAI_RESULTS_PATH export), shown beside the archived experiment.
    lab = load(5, "results.json")
    assert lab["mode"] == "full", "only a full lab run may be quoted"
    lc = lab["chargpt"]
    run_day = dt.datetime.fromisoformat(lab["recorded_at_utc"])
    return {
        "w05_lab_device": hardware(lab["device"], lab.get("device_name")),
        "w05_lab_date": f"{run_day.day} {run_day:%B %Y}",
        "w05_lab_val": f"{lc['final_validation_loss']:.2f}",
        "w05_lab_ppl": f"{lc['perplexity']:.2f}",
        "w05_lab_minutes": f"{lc['seconds_including_validation'] / 60:.1f}",
        "w05_lab_batches": str(lc["validation_batches_per_check"]),
        "w05_val_start": f"{c['val_loss_start']:.2f}",
        "w05_val_end": f"{c['val_loss_end']:.2f}",
        "w05_ppl_end": f"{c['val_ppl_end']:.2f}",
        "w05_steps": f"{c['steps']:,}",
        "w05_minutes": f"{c['minutes']:.1f}",
        "w05_params": f"{m['chargpt_params']:,}",
        "w05_vocab": str(c["vocab"]),
        "w05_device": hardware(c["device"]),
        # The BERT head is chosen with the lab's TODO 5 rule over all 144 heads (make_week_05_bert.py).
        "w05_attention_weight": f"{bert['it_to_animal']:.3f}",
        "w05_bert_layer": str(bert["layer"]),
        "w05_bert_head": str(bert["head"]),
        "w05_bert_median": f"{bert['median_it_to_animal']:.2f}",
        "w05_bert_above_half": str(bert["heads_above_half"]),
        # Smallest weight that the third GPT-2 head (layer 10, head 7) puts on the first token, rows 2 onwards.
        "w05_gpt2_first_min": f"{min(row[0] for row in gpt2['heads'][2]['weights'][1:]):.2f}",
        "w05_sample_start": clean(samples["0"], 180),
        "w05_sample_end": clean(samples[str(c["steps"])], 240),
    }


# ---------------------------------------------------------------- week 6
def week6() -> dict:
    m = load(6, "metrics.json")
    tk = m["tokenizers"]
    texts = list(next(iter(tk["counts"].values())))
    tok_table = table(["Tokenizer", "Vocabulary"] + texts,
                      [[k, f"{tk['vocab'][k]:,}"] + [tk["counts"][k][t] for t in texts] for k in tk["counts"]])
    nt = m["next_token"]
    nt_parts = []
    for prompt, top in nt.items():
        nt_parts.append(f"after “{prompt}” the top tokens are " + ", ".join(f"{tok} ({p:.2f})" for tok, p in top[:4]))
    fam = m["families"]
    bert = ", ".join(f"“{w}” ({p:.2f})" for w, p in fam["bert_fill_mask"][:3])
    rows = [[clean(q, 60), clean(fam["t5"][q_t5], 60), clean(fam["qwen"][q], 70)]
            for q_t5, q in zip(fam["t5"], fam["qwen"])]
    fam_md = (f"**BERT (encoder, fill-mask)** on “The capital of Ireland is [MASK].”: {bert}. BERT fills gaps; it does not write answers.\n\n"
              + table(["Prompt", "Flan-T5-base (encoder–decoder)", "Qwen2.5-0.5B-Instruct (decoder)"], rows))
    r = m["reasoning"]
    dev = hardware(r["device"], r.get("gpu"))

    def mode(x, long=False):
        text = f"{x['correct']}/{x['n']} correct, ≈{x['mean_tokens']:,} tokens"
        if x["unfinished"]:
            text += f", {x['unfinished']} ran out of the {x['budget']:,}-token budget"
        return text + (f", ≈{x['mean_seconds']:.0f} s each" if long else "")

    short_parts, long_parts, rows = [], [], []
    for name, pr in r["problems"].items():
        off, on = pr["no_thinking"], pr["thinking"]
        short_parts.append(f"{name}: {off['correct']}/{off['n']} vs {on['correct']}/{on['n']} correct, ≈{off['mean_tokens']:,} vs ≈{on['mean_tokens']:,} tokens"
                           + (f" (thinking hit its {on['budget']:,}-token limit {'every time' if on['unfinished'] == on['n'] else str(on['unfinished']) + ' times'})"
                              if on["unfinished"] else ""))
        long_parts.append(f"**{name}** (answer {pr['correct_answer']:g}): without thinking {mode(off, True)}; with thinking {mode(on, True)}")
        rows += [[name, "off", f"{off['correct']}/{off['n']}", f"{off['mean_tokens']:,}", f"{off['mean_seconds']:.0f}", off["unfinished"]],
                 [name, "on", f"{on['correct']}/{on['n']}", f"{on['mean_tokens']:,}", f"{on['mean_seconds']:.0f}", on["unfinished"]]]
    notes = (f"Qwen3-0.6B on {dev}, three sampled runs per mode (the model card's settings; greedy decoding makes thinking mode loop). "
             + "; ".join(long_parts) + ".\n\n"
             + table(["Problem", "Thinking", "Correct", "Mean tokens", "Mean seconds", "Out of budget"], rows))
    nt_text = "; ".join(nt_parts) + "."
    return {"TOKENIZER_NOTES": tok_table, "NEXT_TOKEN_NOTES": nt_text[0].upper() + nt_text[1:], "FAMILY_OUTPUTS": fam_md,
            "REASONING_NOTES": notes,
            "REASONING_RESULT": "Measured (Qwen3-0.6B, 3 runs per mode, thinking off vs on): " + "; ".join(short_parts) + "."}


# ---------------------------------------------------------------- week 7
def week7() -> dict:
    r = load(7, "results.json")
    cl = {s["prompt"]: s for s in r["classification"]}
    few = ", ".join(f"{k} {pct(v['accuracy'], 1)}" for k, v in cl.items())
    inv = ", ".join(f"{k} {v.get('invalid', 0)}" for k, v in cl.items())
    n = next(iter(cl.values()))["n"]
    # figures/week_07.py draws the invalid-reply panel only when some reply was not a recognised label.
    invalid_note = ("Every reply in this run contained a recognised label, so the chart shows only accuracy and the "
                    "differences come from wrong labels, not from format errors."
                    if not any(v.get("invalid", 0) for v in cl.values()) else
                    f"The right-hand chart counts unparsed or ambiguous replies ({inv}); examples often fix this "
                    "format problem.")
    rs = {s["prompt"]: s for s in r["reasoning"]}
    cot = "; ".join(f"{k} {pct(v['accuracy'], 1)} ({v['avg_tokens']:.0f} tokens)" for k, v in rs.items())
    ex = r["extraction"]
    inj = {k: r["injection"][k] for k in sorted(r["injection"], key=lambda k: k != "naive")}  # naive first, then defended
    n_inj = 3  # injected documents in the lab's DOCS list
    j = r["judge"]
    # One decimal, as on the slides: 29/40 is 72.5%, which a whole-number table would print as 72% (and 35/40 as 88%).
    rows = [[k, v["n"], pct(v["accuracy"], 1), v.get("invalid", "–"), f"{v['avg_tokens']:.0f}"] for k, v in cl.items()]
    rows += [[k, v["n"], pct(v["accuracy"], 1), "–", f"{v['avg_tokens']:.0f}"] for k, v in rs.items()]
    res_table = (table(["Prompt", "n", "Accuracy", "Invalid", "Avg output tokens"], rows) + "\n\n"
                 + f"Structured extraction: {pct(ex['valid_rate'], 1)} valid JSON, {pct(ex['correct_rate'], 1)} with all fields correct. "
                 + "Keyword-flag rate in injected replies: " + ", ".join(f"{k} prompt {pct(v)}" for k, v in inj.items()) + ". "
                 + f"LLM-as-judge: consistent across both orders {pct(j['consistent_rate'])}, correct in both orders {pct(j['correct_both_rate'])}.")
    return {"MODEL": short(r["model"]),
            "FEWSHOT_SUMMARY": f"Accuracy on {n} messages: {few}.",
            "FEWSHOT_NOTES": f"**Measured ({short(r['model'])}, {n} messages, temperature 0):** accuracy {few}; invalid labels {inv}.",
            "FEWSHOT_INVALID": invalid_note,
            "COT_SUMMARY": cot[:1].upper() + cot[1:] + ".",  # a bullet: "Direct 50.0% (3 tokens); ..."
            "COT_NOTES": f"**Measured ({short(r['model'])}):** {cot}. Report accuracy together with token cost.",
            "EXTRACTION_NOTES": f"{pct(ex['valid_rate'], 1)} of outputs were valid JSON after at most one retry, but only {pct(ex['correct_rate'], 1)} had every field correct.",
            "INJECTION_SUMMARY": f"Measured keyword flags on {n_inj} injected-document replies: " + ", ".join(f"{k} prompt {pct(v)} ({round(v * n_inj)} of {n_inj})" for k, v in inj.items()) + ".",
            "INJECTION_NOTES": "the keyword heuristic flagged " + " and ".join(f"{round(v * n_inj)} of {n_inj} injected-document replies with the {k} prompt" for k, v in inj.items()) + ". Inspect the actual replies before claiming instruction-following.",
            "JUDGE_NOTES": f"In the lab, the small judge gave valid A/B verdicts selecting the same underlying answer in both orders for {pct(j['consistent_rate'])} of pairs and was correct in both orders for {pct(j['correct_both_rate'])}.",
            "RESULTS_TABLE": res_table}


# ---------------------------------------------------------------- week 8
def week8() -> dict:
    r = load(8, "results.json")
    comp = {c["model"]: c for c in r["comparison"]}
    acc = ", ".join(f"{k} {pct(v['accuracy'], 1)}" for k, v in comp.items())
    lora = f"LoRA (r = 16, attention projections) trains {r['trainable']:,} of {r['total']:,} parameters ({100 * r['trainable'] / r['total']:.2f}%)."
    train = f"{r['n_train']:,} training examples, 1 epoch, {r['train_minutes']:.0f} min on {hardware(r['device'], r.get('gpu'))}"
    rows = [[k, pct(v["accuracy"], 1), f"{v['invalid']:.1%} ({round(v['invalid'] * r['n_test'])} of {r['n_test']})"] for k, v in comp.items()]
    ft = comp["LoRA fine-tuned"]
    n_bad = round(ft["invalid"] * r["n_test"])
    run = (f"The recorded run fine-tuned {short(r['model'])} on {r['n_train']:,} examples for one epoch in about "
           f"{r['train_minutes']:.0f} minutes on {hardware(r['device'], r.get('gpu'))}. "
           + (f"The LoRA model still gave {n_bad} unparsed {'reply' if n_bad == 1 else 'replies'} ({pct(ft['invalid'], 1)})."
              if n_bad else "Every LoRA reply contained a recognised intent label."))
    return {"LORA_PARAMS": lora,
            "RUN_NOTE": run,
            "RESULTS_SUMMARY": f"Accuracy on {r['n_test']} held-out messages: {acc}.",
            "ADAPTER_SIZE": f"Saved adapter: {r['adapter_mb']:.1f} MB; base weights: {r['model_mb']:,.0f} MB.",
            "RESULTS_NOTES": table(["Model", "Accuracy", "Invalid answers"], rows)
            + f"\n\nTest set: {r['n_test']} held-out Banking77 messages (10 intents). Training: {train}. Adapter: {r['adapter_mb']:.1f} MB."}


# ---------------------------------------------------------------- week 9
def template_name(key: str) -> str:
    """Readable name for a stored CLIP prompt template: "label only: '{}'" becomes the class name alone, "cat"."""
    if "{}" not in key:
        return key
    shown = key.split(":", 1)[1].strip() if key.startswith("label only") else key
    text = shown.strip("'").format("cat")
    return f"class name only (“{text}”)" if key.startswith("label only") else f"“{text}”"


def week9() -> dict:
    r = load(9, "results.json")
    n_batch = round(math.exp(r["loss_random"]))
    loss = (f"batch of N = {n_batch} image–caption pairs: correct pairs {r['loss_correct']:.2f}; captions shuffled {r['loss_shuffled']:.2f}; "
            f"uniform-choice baseline ln N = {r['loss_random']:.2f}.")
    zs = "; ".join(f"{template_name(k)} {pct(v, 1)}" for k, v in r["zero_shot"].items())
    prec = ", ".join(f"@{k} {v:.2f}" for k, v in r["precision_at_k"].items())
    ex = r["asr_examples"][0] if r.get("asr_examples") else None
    wer = f"Whisper on LibriSpeech sample clips: mean WER {pct(r['mean_wer'], 1)}."
    if ex:
        hyp = ex.get("whisper", ex.get("hypothesis", ""))
        wer += f" Example: “{clean(ex['reference'], 48)}” → “{clean(hyp, 48)}”"
        wer += " (“mister” vs “Mr.” counts as an error unless text is normalised)." if "mister" in ex["reference"] and "Mr." in hyp else "."
    rows = [[template_name(k), pct(v, 1)] for k, v in r["zero_shot"].items()]
    return {"LOSS_NOTES": loss[0].upper() + loss[1:], "ZS_NOTES": f"Zero-shot accuracy on {r['n_images']:,} CIFAR-10 test images: {zs}.",
            "PREC_NOTES": f"Text→image retrieval precision {prec} (fraction of top-k images of the queried class).", "WER_NOTES": wer,
            "RESULTS_NOTES": table(["Prompt template", "Zero-shot accuracy"], rows)
            + f"\n\nRetrieval precision {prec}. Contrastive loss: {loss} {wer}"}


# ---------------------------------------------------------------- week 10
def week10() -> dict:
    r = load(10, "results.json")
    vlm = short(r["vlm"])
    qa = r["qa"]
    wrong = [q for q in qa if not q["correct"]]
    qa_note = f"{vlm} answered {sum(q['correct'] for q in qa)} of {len(qa)} questions correctly"
    qa_note += (": missed " + "; ".join(f"“{stem(q['question'])}” (said “{clean(q['answer'], 20)}”, gold {q['gold']})" for q in wrong[:2]) + ".") if wrong else "."
    pope = r["pope"]
    absent = [p for p in pope if p["gold"] == "no"]
    yes_absent = sum(1 for p in absent if str(p["answer"]).lower().startswith("yes"))
    pope_note = (f"{vlm}: {sum(p['correct'] for p in pope)} of {len(pope)} yes/no probes correct; it said “yes” to {yes_absent} of {len(absent)} absent objects or fields.")
    t2i = r["t2i"]
    own = [x["CLIP score (own prompt)"] for x in t2i]
    other = [x["CLIP score (other prompts, mean)"] for x in t2i]
    t2i_note = f"CLIP score against its own prompt {min(own):.1f}–{max(own):.1f} vs other prompts {min(other):.1f}–{max(other):.1f}: each own-prompt score exceeded the mean score for the other prompts in this run." \
        if all(a > b for a, b in zip(own, other)) else f"CLIP score own prompt {min(own):.1f}–{max(own):.1f} vs other prompts {min(other):.1f}–{max(other):.1f}."
    sp = r["speech"]
    mean_wer = sum(s["WER"] for s in sp) / len(sp)
    speech_note = f"mean WER {pct(mean_wer, 1)} over {len(sp)} sentences"
    worst = max(sp, key=lambda s: s["WER"])
    if worst["WER"] > 0:
        speech_note += f" (e.g. “{clean(worst['text'], 60)}” → “{clean(worst['transcript'], 60)}”)"
    caps = "; ".join(f"{k}: “{clean(v, 110)}”" for k, v in r["captions"].items())
    rows = [[clean(q["question"], 50), q["gold"], clean(q["answer"], 30), "✓" if q["correct"] else "✗"] for q in qa]
    return {"CAPTIONS": caps, "QA_NOTES": qa_note, "POPE_NOTES": pope_note, "T2I_NOTES": t2i_note, "SPEECH_NOTES": speech_note + ".",
            "RESULTS_NOTES": f"Vision-language model: {vlm} on {hardware(r['device'], r.get('gpu'))}.\n\n" + table(["Question", "Gold", "Answer", ""], rows)
            + f"\n\nHallucination probe: {pope_note} Text-to-image: {t2i_note} Speech round trip: {speech_note}."}


# ---------------------------------------------------------------- week 11
def week11() -> dict:
    r = load(11, "results.json")
    hw = hardware(r["device"], r.get("gpu"))
    lat = r["latency"]
    ttft = [x["TTFT (s)"] for x in lat]
    tot = [x["total (s)"] for x in lat]
    tps = [x["tokens/s"] for x in lat]
    def span(values, fmt):
        lo, hi = fmt.format(min(values)), fmt.format(max(values))
        return lo if lo == hi else f"{lo}–{hi}"
    lat_note = (f"{short(r['model'])} on {hw}: TTFT {span(ttft, '{:.2f}')} s, full answers {span(tot, '{:.1f}')} s "
                f"at {span(tps, '{:.0f}')} tokens/s.")
    q = {x["scheme"]: x for x in r["quant"]}
    base = next(iter(q))
    quant_note = "Perplexity: " + "; ".join(f"{k.replace(',', '')} {v['perplexity']:.1f} (≈ {v['approx. size (MB)']:,} MB)" for k, v in q.items()) + "."
    b = r["batch"]
    # One decimal, as on the chart's bar labels, so the printed values give the printed ratio (86.2 / 13.2 = 6.5).
    batch_note = (f"{b[0]['tokens/s']:.1f} tokens/s at batch 1 → {b[-1]['tokens/s']:.1f} tokens/s at batch {b[-1]['batch size']} "
                  f"({b[-1]['tokens/s'] / b[0]['tokens/s']:.1f}× on {hw}).")
    mon = r["monitor"]
    mon_note = (f"{mon['requests']} requests: {mon['blocked']} blocked (injection), {mon['redacted']} with personal data redacted; "
                f"p50 {mon['p50 latency (s)']:.2f} s, p95 {mon['p95 latency (s)']:.2f} s.")
    rows_l = [[x["question"], x["TTFT (s)"], x["total (s)"], x["output tokens"], x["tokens/s"]] for x in lat]
    rows_q = [[k, v["perplexity"], v["weight error (layer 0)"], f"{v['approx. size (MB)']:,}"] for k, v in q.items()]
    rows_b = [[x["batch size"], x["seconds"], x["tokens/s"]] for x in b]
    notes = (f"Model {short(r['model'])} on {hw} ({base} weights).\n\n"
             + table(["Question", "TTFT (s)", "Total (s)", "Output tokens", "Tokens/s"], rows_l) + "\n\n"
             + table(["Quantisation", "Perplexity", "Weight error (layer 0)", "Approx. size (MB)"], rows_q) + "\n\n"
             + table(["Batch size", "Seconds for 8 requests", "Tokens/s"], rows_b) + "\n\n" + "Monitoring summary: " + mon_note)
    return {"LAT_NOTES": lat_note, "QUANT_NOTES": quant_note, "BATCH_NOTES": batch_note, "MONITOR_NOTES": mon_note, "RESULTS_NOTES": notes}


# ---------------------------------------------------------------- week 12
def week12() -> dict:
    r = load(12, "results.json")
    ret = r["retrieval"]
    r1 = ", ".join(f"{k.split(' (')[0]} {v['recall@1']:.2f}" for k, v in ret.items())
    mrr = ", ".join(f"{k.split(' (')[0]} {v['MRR']:.2f}" for k, v in ret.items())
    ret_note = f"Gold-document hit@1 (the lab's recall@1): {r1}. MRR: {mrr}."  # the chart's name for the metric
    s = r["summary"]
    no, rag = s["no retrieval"], s["RAG (hybrid + re-rank, top 3)"]
    abst = rag.get("abstains when answer missing")
    rag_note = (f"{short(r['llm'])}: accuracy {pct(no['accuracy (answerable)'])} without retrieval → {pct(rag['accuracy (answerable)'])} with RAG; "
                f"numbers supported by sources {pct(rag['numbers supported by sources'])}; cited the right document {pct(rag['cites the right document'])}; "
                f"{'abstained' if abst == 1 else 'did not abstain'} on the unanswerable question.")
    tools = [t["content"].split("(")[0] for t in r["agent_trace"] if t["type"] == "tool"]
    final = r["agent_answer"]
    got = last_number(final)
    agent_note = (f"{short(r['llm'])} called {' → '.join(tools) if tools else 'no tools'}; final answer "
                  f"{'68.8 (correct)' if got is not None and abs(got - 68.8) < 0.05 else '“' + clean(final, 60) + '” (expected 68.8)'}.")
    audit = r.get("audit", [])
    if audit:
        a = audit[0]
        inj_note = (f"the agent tried send_email to {a['arguments'].get('to', '?')}: the approval gate "
                    f"{'allowed' if a['approved'] else 'blocked'} it.")
    else:
        inj_note = "the agent did not attempt to send e-mail; even if it had, the approval gate would have blocked the external address."
    rows = [[k, v["recall@1"], v["recall@3"], v["MRR"]] for k, v in ret.items()]
    rag_rows = [[clean(x["question"], 55), "✓" if x["correct"] else "✗", clean(x["answer"], 45)] for x in r["rag"]]
    notes = (f"Models: {short(r['llm'])} (generation), {short(r['embedder'])} (embeddings), {short(r['reranker'])} (re-ranking) on "
             f"{hardware(r['device'], r.get('gpu'))}; {r['n_chunks']} chunks.\n\n" + table(["Retriever", "Recall@1", "Recall@3", "MRR"], rows) + "\n\n"
             + table(["Question", "RAG correct", "Answer"], rag_rows) + f"\n\nAgent: {agent_note} Injection test: {inj_note}")
    return {"RETRIEVAL_NOTES": ret_note, "RAG_NOTES": rag_note, "AGENT_NOTES": agent_note, "INJECTION_NOTES": inj_note[0].upper() + inj_note[1:],
            "RESULTS_NOTES": notes}


WEEKS = {1: week1, 2: week2, 4: week4, 5: week5, 6: week6, 7: week7, 8: week8, 9: week9, 10: week10, 11: week11, 12: week12}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default=",".join(map(str, WEEKS)))
    args = ap.parse_args()
    status = 0
    for w in sorted({w for part in args.weeks.split(",") for w in (range(1, 13) if part == "all" else range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))}):
        if w not in WEEKS:
            continue  # week 3 has no result placeholders
        try:
            values = WEEKS[w]()
        except FileNotFoundError as e:
            print(f"week {w}: missing {e}")
            status = 1
            continue
        out = ASSETS / f"week_{w:02d}" / "fill.json"
        out.write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"week {w}: {len(values)} values -> {out.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
