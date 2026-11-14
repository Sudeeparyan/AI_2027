# %% [markdown]
# # Week 8 Lab: Fine-tune a small LLM with LoRA for a domain task
#
# **Module:** Generative AI (MSc in Artificial Intelligence) · **Time:** 2 hours · **Learning outcomes:** MIMLO 1, 3, 4
#
# **Scenario.** A bank's support team wants incoming customer messages routed to the right intent. Last week you
# prompted **Qwen2.5-0.5B-Instruct** with fixed weights. This week you **fine-tune the same model** with a LoRA adapter
# and **measure** whether training beats the best prompt on the same held-out messages.
#
# 1. Prepare real Banking77 messages in **chat format**, with separate training, validation and test sets.
# 2. Explore **prompting baselines** (zero-shot, few-shot) on validation messages, keeping the test untouched.
# 3. Add **LoRA** adapters with Hugging Face **PEFT** and check the trainable-parameter count.
# 4. Run **supervised fine-tuning** (SFT) with Hugging Face **TRL** on the CPU.
# 5. Compare **base vs fine-tuned** on the same test set and read the errors.
# 6. Switch the adapter **off and on**, save it, and compare its size with the base model.
#
# **CPU core:** five intents, 200 training messages, one epoch of rank-8 LoRA, then 50 held-out test messages.
# The teaching notes give the measured CPU time on the course laptop; the final cell prints your own.
# **Optional stretch:** `RUN_FULL = True` runs a larger experiment like the lecture's stored run (ten intents, about
# 1,200 training messages, rank 16, 200 test messages). Measure it separately on your chosen hardware outside the core.
#
# Use the injected setup's `DEVICE`, `SMOKE`, `SEED`, `check()` and `ask_model()`. Keep `USE_GEMINI = False` and
# `USE_OPENAI = False` for the local core. The hosted comparison sends invented demonstrations and queries only; OpenAI API
# use is billed separately from a ChatGPT subscription.
#
# | Part | Activity | Suggested class time |
# |---|---|---|
# | 0 | Recall vocabulary; predict the baselines | 10 min |
# | 1 | Data splits and chat format | 15 min |
# | 2 | Prompting baselines | 15 min |
# | 3 | LoRA configuration and counts | 15 min |
# | 4 | Fine-tuning (runs while you discuss) | 25 min |
# | 5 | Fair comparison and errors | 20 min |
# | 6 | Adapter off/on, saving, self-checks | 20 min |
#
# Class time includes discussion and coding; it is not a measured compute time.

# %% [markdown]
# ## Start here: what changes during fine-tuning?
#
# Customer messages and known intent labels become training pairs. The pretrained model already supplies general
# language patterns. LoRA freezes its base weights and adds small trainable correction matrices. Supervised
# fine-tuning uses the known label tokens to train those corrections. The test set checks whether the adapted model
# beats the prompting baselines on messages it never saw.
#
# ![Adapt a banking classifier](fig:beginner_overview)
#
# ## Your map from the workflow to code
#
# | Step | Find this code | Observe this |
# |---|---|---|
# | Filter and split | `INTENTS`, `train_ds`, `val_ds`, `test_ds` | Which messages serve each purpose |
# | Format and baseline | `to_chat`, `evaluate` | Prompt/completion boundary and base predictions |
# | Count trainables | `lora_config`, `trainable`, `total` | Which parameters can change |
# | Train and monitor | `trainer`, `hist` | Training and validation loss |
# | Compare and read errors | `comparison`, `errors` | Valid labels, wrong decisions and unparsed replies |
# | Check and save | `disable_adapter`, `save_pretrained` | Adapter-off behaviour and the saved files |
#
# **Read:** an **intent** is the kind of help a customer wants. A **baseline** is the result before training. An
# **adapter** is a small set of added weights. **SFT** names the training objective; **LoRA** names which weights it
# updates. **Predict → run → change → check:** predict each table before running it, change one setting at a time,
# and check every TODO with its `check_…()` cell. Use the training set to learn, the validation set to watch
# training, and the test set once for the final comparison.

# %% [markdown]
# **Purpose:** Install the model, data, adapter, training and widget libraries in Colab.
# **Why now:** Later cells import Transformers, Datasets, PEFT, TRL, pandas and ipywidgets.
# **Expected output:** Installation messages; a prepared local environment can skip this cell.

# %% tags=["colab-install"]
%pip install -q transformers accelerate datasets peft trl pandas matplotlib ipywidgets

# %% [markdown]
# **Purpose:** Import tools, choose the core settings and run three readiness checks.
# **Why now:** The shared setup has already chosen the device and seed; this cell fixes the experiment sizes.
# **Expected output:** Three ✅ lines and the chosen settings. Smoke mode uses tiny sizes for automated testing.

# %%
import hashlib
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
RUN_FULL = False            # True: a larger run like the stored one (ten intents, rank 16); measure separately
if RUN_FULL:
    INTENTS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed", "pending_transfer",
               "activate_my_card", "cash_withdrawal_charge", "verify_my_identity", "age_limit", "Refund_not_showing_up"]
    N_TRAIN_PER, N_VAL_PER, N_TEST_PER, RANK = None, None, 20, 16
else:
    INTENTS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed", "age_limit"]
    N_TRAIN_PER, N_VAL_PER, N_TEST_PER, RANK = 40, 6, 10, 8
if SMOKE:
    N_TRAIN_PER, N_VAL_PER, N_TEST_PER = 4, 2, 2
core_started = time.perf_counter()

ready = check(16 / 20 == 0.8, "16 correct out of 20 is accuracy 0.8 (80%).",
              "Accuracy divides correct items by all scored items.")
ready &= check(8 * (896 + 896) == 14336, "Rank 8 on an 896 × 896 projection adds 8 × (896 + 896) = 14,336 numbers.",
               "LoRA adds r × (inputs + outputs) numbers for each adapted projection.")
ready &= check(len(set(INTENTS)) == len(INTENTS), "Every intent label is distinct.",
               "Each label must appear once in INTENTS.")
assert ready, "Review these three ideas before continuing."
print("device:", DEVICE, "| model:", MODEL_ID, "| intents:", len(INTENTS), "| rank:", RANK,
      "| full run:", RUN_FULL, "| smoke:", SMOKE)

# %% [markdown]
# ## Part 1 · Domain data in chat format
#
# Banking77 contains real customer-support messages labelled with 77 intents (licence CC-BY-4.0). We keep a few
# intents and take **separate** training, validation and test messages for each one.
#
# > **Read:** training teaches the model; validation supports choices; test messages stay untouched until Part 5.
# > **Predict:** which part of a training pair should contain the known intent?
# > **Core:** split the data and write `to_chat`.
# > **Change one thing:** print one training pair, then one validation message, and note which one contains the answer.
# > **Check:** run `check_to_chat`; confirm the three sets contain different messages.
# > **Optional stretch:** add a sixth intent to `INTENTS` and predict how the trainable count changes (it does not).
# > **Common errors:** `KeyError: 'label_text'` means a different dataset column name; a prompt that already
# > contains the label lets the model copy the answer, so evaluation would be meaningless.

# %% [markdown]
# **Purpose:** Filter the chosen intents and create separate training, validation and test sets.
# **Why now:** The test messages must stay out of training and out of every choice we make.
# **Expected output:** Set sizes and the number of messages per intent.

# %%
raw = load_dataset("mteb/banking77")


def take(split, n_per, exclude=frozenset()):
    """The first n_per messages of each intent after a fixed shuffle (all of them when n_per is None)."""
    rows = []
    for label in INTENTS:
        part = split.filter(lambda ex: ex["label_text"] == label).shuffle(seed=0)
        texts = [t for t in part["text"] if t not in exclude]   # never train on a test message
        texts = texts if n_per is None else texts[:n_per]
        rows += [{"text": t, "label_text": label} for t in texts]
    return rows


test_ds = Dataset.from_list(take(raw["test"], N_TEST_PER))   # Banking77's own test split: never used for learning
test_texts = frozenset(test_ds["text"])
if RUN_FULL:
    pool = Dataset.from_list(take(raw["train"], None, test_texts)).shuffle(seed=0).train_test_split(test_size=0.1, seed=0)
    train_ds, val_ds = pool["train"], pool["test"]
else:
    pool = take(raw["train"], N_TRAIN_PER + N_VAL_PER, test_texts)
    # Per intent: the first N_TRAIN_PER messages train, the next N_VAL_PER validate; no message is in both.
    train_ds = Dataset.from_list([r for i, r in enumerate(pool) if i % (N_TRAIN_PER + N_VAL_PER) < N_TRAIN_PER]).shuffle(seed=0)
    val_ds = Dataset.from_list([r for i, r in enumerate(pool) if i % (N_TRAIN_PER + N_VAL_PER) >= N_TRAIN_PER])
assert not set(train_ds["text"]) & test_texts, "A test message also appears in training."
assert not set(val_ds["text"]) & test_texts, "A test message also appears in validation."
assert not set(train_ds["text"]) & set(val_ds["text"]), "A validation message also appears in training."
# One validation item per intent keeps this demonstration short; five items cannot establish a reliable ranking.
dev_ds = Dataset.from_list([next(ex for ex in val_ds if ex["label_text"] == lab) for lab in INTENTS])
print(f"train {len(train_ds)} | validation {len(val_ds)} | test {len(test_ds)}")
pd.Series(train_ds["label_text"]).value_counts()

# %% [markdown]
# **TODO 1:** write `to_chat(example)` returning a **prompt–completion** pair in conversational form:
#
# * `prompt`: a list with one user message: `f"{INSTRUCTION}\nMessage: {example['text']}\nIntent:"`;
# * `completion`: a list with one assistant message whose content is the intent label.
#
# TRL trains only on the **completion** tokens; the prompt is read but masked out of the loss.
# **Hint:** `{"prompt": [{"role": "user", "content": ...}], "completion": [{"role": "assistant", "content": ...}]}`.
#
# **Purpose:** Format each message and its known answer as a chat pair, then check the format.
# **Expected output:** One ✅ line and one formatted training pair.

# %%
INSTRUCTION = "Classify the banking customer message into one intent.\nIntents: " + ", ".join(INTENTS)


def to_chat(example):
    ### BEGIN SOLUTION
    return {"prompt": [{"role": "user", "content": f"{INSTRUCTION}\nMessage: {example['text']}\nIntent:"}],
            "completion": [{"role": "assistant", "content": example["label_text"]}]}
    ### END SOLUTION
    ### STUB return {"prompt": [], "completion": []}  # TODO 1: build the user prompt and the assistant answer


def check_to_chat(fn):
    probe = {"text": "PROBE MESSAGE", "label_text": INTENTS[0]}
    try:
        pair = fn(probe)
        prompt, completion = pair["prompt"], pair["completion"]
        ok = (len(prompt) == 1 and prompt[0]["role"] == "user" and INSTRUCTION in prompt[0]["content"]
              and prompt[0]["content"].rstrip().endswith("Message: PROBE MESSAGE\nIntent:")
              and INTENTS[0] not in prompt[0]["content"].split("Message:")[-1]
              and completion == [{"role": "assistant", "content": INTENTS[0]}])
    except (KeyError, TypeError, IndexError):
        ok = False
    return check(ok, "TODO 1: one user prompt ending in 'Intent:' and one assistant answer holding the label.",
                 "The prompt needs INSTRUCTION, the message and 'Intent:'; the completion holds only the label.")


chat_ready = check_to_chat(to_chat)
assert chat_ready, "Complete TODO 1 before training."
sft_train = train_ds.map(to_chat, remove_columns=train_ds.column_names)
sft_val = val_ds.map(to_chat, remove_columns=val_ds.column_names)
print(sft_train[0])

# %% [markdown]
# ## Part 2 · Baselines: prompting the base model
#
# We reuse last week's harness idea: fixed model, greedy decoding and parsed-label accuracy. Here we explore
# one validation message per intent, so scores are teaching observations, not reliable rankings. The test stays closed.
# The few-shot demonstrations come from training messages and are used only by the local model.
#
# > **Read:** development examples let you compare choices without looking at the final test.
# > **Predict:** will few-shot change a label or only the reply format?
# > **Core:** run both baselines on `dev_ds` and read their raw replies.
# > **Change one thing:** select a different training example per intent, rebuild `SHOTS` and rerun on `dev_ds`.
# > **Check:** confirm every scored message comes from validation; inspect invalid replies as well as accuracy.
# > **Optional stretch:** score the complete `val_ds` before freezing choices; report its size with each metric.
# > **Common errors:** a reply such as "The intent is age limit" still counts, because the parser accepts spaces
# > for underscores; a reply naming two intents counts the first one mentioned; a reply cut off by the
# > token cap before any intent counts as invalid.
#
# **Predict before running:** will few-shot beat zero-shot, and by how much?
#
# ### One test, three models
#
# ![One test, three models](fig:beginner_lab)
#
# The same harness first scores development messages. After choices are frozen, the final table uses
# the same test messages for all three rows, the same parser and the same comparison with the reference label.

# %% [markdown]
# **Purpose:** Load the base model, define a lenient label parser and explore both validation baselines.
# **Why now:** We inspect prompting before training while reserving the test for the final comparison.
# **Expected output:** Two lines with accuracy, invalid rate and seconds.

# %%
tok = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE).eval()


@torch.no_grad()
def predict(model, messages, max_new_tokens=32):
    # 32 tokens: zero-shot replies often quote the message first, so a shorter cap cuts them before the label.
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# A lenient parser, so the baselines are not punished for wrapping the label in a sentence ("Intent: age_limit",
# "The intent is age limit"): the prediction is the first intent label mentioned, with spaces or underscores.
LABEL_RE = {lab: re.compile(r"(?<![a-z])" + lab.lower().replace("_", "[ _]") + r"(?![a-z])") for lab in INTENTS}


def normalise(text):
    hits = [(m.start(), lab) for lab, pattern in LABEL_RE.items() if (m := pattern.search(text.lower()))]
    return min(hits)[1] if hits else None


def evaluate(model, make_messages, ds, name=""):
    # Always name the scored split explicitly; development work must never silently use the final test.
    rows, started = [], time.perf_counter()
    for ex in ds:
        reply = predict(model, make_messages(ex))
        pred = normalise(reply)
        rows.append({"text": ex["text"], "gold": ex["label_text"], "raw": reply, "pred": pred,
                     "correct": pred == ex["label_text"]})
    df = pd.DataFrame(rows)
    print(f"{name:18s} accuracy {df.correct.mean():.3f} | invalid {df.pred.isna().mean():.3f} | "
          f"{time.perf_counter() - started:.0f} s")
    return df


zero = lambda ex: to_chat(ex)["prompt"]  # noqa: E731
shots = {}
for ex in train_ds:
    shots.setdefault(ex["label_text"], ex["text"])   # one training example per intent
SHOTS = "\n".join(f"Message: {t}\nIntent: {lab}" for lab, t in shots.items())
few = lambda ex: [{"role": "user", "content": f"{INSTRUCTION}\n\nExamples:\n{SHOTS}\n\nMessage: {ex['text']}\nIntent:"}]  # noqa: E731

df_zero_dev = evaluate(base, zero, dev_ds, name="dev, zero-shot")
df_few_dev = evaluate(base, few, dev_ds, name="dev, few-shot")

# %% [markdown]
# **Purpose:** Read what the base model actually wrote.
# **Why now:** A missing label (pred None) is a format failure; a wrong label is a decision failure.
# **Expected output:** Up to four rows per prompt with the reference, the raw reply and the parsed label.

# %%
pd.concat([df_zero_dev.assign(prompt="zero-shot"), df_few_dev.assign(prompt="few-shot")]).groupby("prompt").head(4)[["prompt", "gold", "raw", "pred"]]

# %% [markdown]
# ## Part 3 · Add LoRA adapters
#
# > **Read:** the base weights stay fixed; the optimiser learns only the added matrices.
# > **Predict:** will doubling rank double the adapter count?
# > **Core:** complete TODO 2, then count trainable weights.
# > **Change one thing:** move the rank slider without training.
# > **Check:** run `check_lora` and compare the count with the hand calculation.
# > **Optional stretch:** calculate the adapter count for one feed-forward projection using its input/output sizes.
# > **Common errors:** target module names must match actual layers; `qproj` is different from `q_proj`.
#
# LoRA freezes the pretrained weight $W$ and learns a low-rank update $\Delta W = \frac{\alpha}{r} B A$ with
# $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. $W$ maps $k$ input features to $d$ outputs;
# $A$ maps $k$ features to $r$, then $B$ maps those $r$ features to $d$. The adapter learns $r(k + d)$ numbers
# instead of $d k$. Here $\alpha / r = 2$.
#
# ### A LoRA training update
#
# ![A LoRA training update](fig:beginner_mechanism)
#
# 1. For an illustrative 8-input, 8-output projection, the frozen W has 64 entries.
# 2. Rank r = 2 uses A with shape [2, 8] and B with shape [8, 2]: 32 trainable entries in total.
# 3. W x and B(Ax) are parallel contributions; the adapter is not a layer after W.
# 4. Their summed output feeds the rest of the network and the answer-token loss.
# 5. The optimiser updates A and B, while W keeps its pretrained values.
#
# **Predict before running:** does the adapter take W x as its input, or the same x as the base projection?
#
# <!-- BEGIN ANSWER -->
# **Instructor explanation:** The same x. The base and adapter paths run in parallel and their outputs are added.
# <!-- END ANSWER -->
#
# **TODO 2:** create a `LoraConfig` with `r=RANK`, `lora_alpha=2 * RANK`, `lora_dropout=0.05`, applied to the
# attention projections `q_proj, k_proj, v_proj, o_proj`, with `bias="none"` and `task_type="CAUSAL_LM"`.
# **Hint:** `LoraConfig(r=..., lora_alpha=..., lora_dropout=..., target_modules=[...], bias=..., task_type=...)`.
#
# > **Common errors:** a misspelled module name such as `"qproj"` raises a ValueError listing the real module names.

# %% [markdown]
# **Purpose:** Configure LoRA, count trainable versus total weights, and check the count by hand.
# **Why now:** The count shows exactly which parameters the optimiser can change.
# **Expected output:** The trainable count and percentage, then one ✅ line.

# %%
from peft import LoraConfig, get_peft_model

### BEGIN SOLUTION
lora_config = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none", task_type="CAUSAL_LM")
### END SOLUTION
### STUB lora_config = LoraConfig(task_type="CAUSAL_LM")  # TODO 2: set the rank, scale, dropout and target modules

probe = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32), lora_config)
trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
total = sum(p.numel() for p in probe.parameters())
print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
del probe

# Each adapted projection adds r × (inputs + outputs); sum that over the base model's attention projections.
ADAPTED = ("q_proj", "k_proj", "v_proj", "o_proj")
expected = sum(RANK * (m.in_features + m.out_features) for name, m in base.named_modules()
               if name.endswith(ADAPTED) and hasattr(m, "in_features"))


def check_lora(config, count):
    ok = (getattr(config, "r", None) == RANK and getattr(config, "lora_alpha", None) == 2 * RANK
          and set(getattr(config, "target_modules", None) or []) == set(ADAPTED) and count == expected)
    return check(ok, f"TODO 2: rank {RANK} on q, k, v and o trains {expected:,} numbers, matching r × (inputs + outputs).",
                 "Set r=RANK, lora_alpha=2*RANK and target_modules=['q_proj','k_proj','v_proj','o_proj'].")


lora_ready = check_lora(lora_config, trainable)

# %% [markdown]
# ### Change one setting without training
#
# **Purpose:** See how the rank changes the trainable count before spending any training time.
# **Why now:** Rank is the key LoRA setting; the count grows linearly with it.
# **Expected output:** A slider; moving it prints the count and the percentage of the base model. Automated runs
# skip this cell.

# %% tags=["skip-test"]
per_rank = expected // RANK          # numbers added per unit of rank across all adapted projections
base_count = sum(p.numel() for p in base.parameters())


def show_rank(r=RANK):
    print(f"rank {r}: {r * per_rank:,} trainable numbers ({100 * r * per_rank / base_count:.2f}% of the base model)")


try:
    import ipywidgets as widgets
    widgets.interact(show_rank, r=widgets.IntSlider(min=1, max=64, step=1, value=RANK, continuous_update=False))
except ImportError:
    print("ipywidgets is not available; call show_rank(4), show_rank(16) … instead.")

# %% [markdown]
# ## Part 4 · Supervised fine-tuning with TRL
#
# `SFTTrainer` applies the chat template, masks the prompt and trains only the LoRA parameters. `step` counts optimiser
# updates, not examples: with batch size 8, 200 messages give 25 steps in one epoch. Training and validation losses
# appear on different rows because they are measured at different times.
#
# > **Read:** answer-token loss teaches the label; validation loss checks separate messages.
# > **Predict:** what might a rising validation loss mean when training loss falls?
# > **Core:** train for one epoch, plot both losses and score the small development subset.
# > **Change one thing:** set the learning rate to `2e-3` and rerun Part 4 only; compare validation loss and replies.
# > **Check:** confirm the trainer uses the shared device and compare `df_tuned_dev` with the development baselines.
# > **Optional stretch:** set `RUN_FULL = True` at the top and rerun the notebook before class.
# > **Common errors:** an `SFTConfig` argument error usually means a different TRL version; compare with the
# > installed version's documentation.
#
# ### From examples to a saved adapter
#
# ![From examples to a saved adapter](fig:beginner_training)
#
# 1. A training message about a missing card has a known intent label.
# 2. `to_chat` pairs the message prompt with that label as the completion.
# 3. The trainer changes adapter weights to reduce the answer-token loss.
# 4. Validation loss on separate messages shows whether learning transfers.
# 5. The adapter is saved with its base-model identity and settings.
#
# **Predict before running:** does a small adapter file contain the whole model?
#
# <!-- BEGIN ANSWER -->
# **Instructor explanation:** No. It contains the added weights and configuration; load the compatible base model too.
# <!-- END ANSWER -->

# %% [markdown]
# **Purpose:** Train the adapter on the answer tokens and monitor validation loss.
# **Why now:** This is the real fine-tuning job; it takes several minutes on a CPU.
# **Expected output:** The training time in minutes and a plot of training and validation loss.
# The default TRL loss needs an optional kernel unavailable on this Windows CPU installation.
# Our trainer uses ordinary PyTorch cross-entropy on the same answer tokens, with the same adapter gradients.
# It projects only scored positions into vocabulary scores, keeping memory small; all prompt positions still
# pass through the model as context. The next-token targets are shifted by one position before masking.
#
# **Worked shape example:** one five-token sequence gives four next-token positions. Suppose its labels are
# `[-100, -100, -100, 4, 5]`: the prompt positions use the ignore marker `-100`. Shifting gives
# `[-100, -100, 4, 5]`, so only two positions contribute to loss. Their hidden vectors have shape `[2, d]`,
# where `d` is the model's feature width. The vocabulary head produces `[2, V]` scores, where `V` is the
# vocabulary size. Prompt and padding labels are ignored, while prompt tokens still supply context.
# This trainer class is supplied support code, not a TODO. Your experiment changes the LoRA and SFT arguments below.

# %%
from trl import SFTConfig, SFTTrainer
from transformers.modeling_outputs import CausalLMOutput


class PortableSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Keep TRL's chat preparation and masks, while using a loss available on every CPU platform.
        adapted = self.accelerator.unwrap_model(model)
        causal = adapted.get_base_model() if hasattr(adapted, "get_base_model") else adapted
        context = {key: value for key, value in inputs.items() if key != "labels"}
        context.update(use_cache=False, return_dict=True)
        hidden = causal.base_model(**context).last_hidden_state[:, :-1]
        targets = inputs["labels"][:, 1:]       # Position t predicts the token at t + 1.
        scored = targets != -100               # Ignore prompt and padding labels supplied by TRL.
        logits = causal.get_output_embeddings()(hidden[scored]).float()
        token_count = scored.sum() if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(token_count, device=logits.device).clamp(min=1)
        loss = torch.nn.functional.cross_entropy(logits, targets[scored], reduction="sum") / denominator
        outputs = CausalLMOutput(loss=loss, logits=logits)
        return (loss, outputs) if return_outputs else loss


assert lora_ready, "Complete TODO 2 before training."
steps_per_epoch = -(-len(sft_train) // (2 if SMOKE else 8))
args = SFTConfig(
    output_dir="banking-lora",
    use_cpu=(DEVICE == "cpu"),  # Trainer must obey the shared CPU default even when a GPU is available.
    num_train_epochs=1,
    max_steps=2 if SMOKE else -1,
    per_device_train_batch_size=2 if SMOKE else 8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2,
    logging_steps=1 if SMOKE else 5,
    eval_strategy="no" if SMOKE else "steps",
    eval_steps=max(1, steps_per_epoch // 2),
    max_length=128,
    completion_only_loss=True,  # learn the answer tokens while reading the prompt
    eos_token="<|im_end|>",     # Qwen chat turn-end token
    save_strategy="no",
    report_to="none",
    fp16=False,
    bf16=False,
    seed=SEED,
)
ft_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
trainer = PortableSFTTrainer(model=ft_model, args=args, train_dataset=sft_train, eval_dataset=None if SMOKE else sft_val,
                     peft_config=lora_config, processing_class=tok)
assert next(trainer.model.parameters()).device.type == DEVICE, "Trainer did not use the selected shared device."
started = time.perf_counter()
trainer.train()
train_minutes = (time.perf_counter() - started) / 60
print(f"training took {train_minutes:.1f} min for {trainer.state.global_step} optimiser steps")

hist = pd.DataFrame(trainer.state.log_history)
plt.plot(hist.dropna(subset=["loss"]).step, hist.dropna(subset=["loss"]).loss, marker="o", label="training loss")
if "eval_loss" in hist:
    ev = hist.dropna(subset=["eval_loss"])
    plt.plot(ev.step, ev.eval_loss, marker="s", label="validation loss")
plt.xlabel("optimiser step"); plt.ylabel("loss (answer tokens)"); plt.legend(); plt.title("LoRA SFT on the CPU core")
plt.show()

# %% [markdown]
# **Purpose:** Inspect adapted replies on the same small validation subset used by the baselines.
# **Why now:** Any learning-rate or prompt changes belong here, before viewing the test.
# **Expected output:** A validation accuracy line and up to five raw replies; these few items do not establish superiority.

# %%
df_tuned_dev = evaluate(trainer.model.eval(), zero, dev_ds, name="dev, LoRA")
display(df_tuned_dev[["gold", "raw", "pred"]].head())

# %% [markdown]
# ## Part 5 · Freeze choices, then compare on the final test
#
# > **Read:** a final test estimates new-message performance after prompt and training choices are frozen.
# > **Predict:** why would choosing a learning rate from the test make its score optimistic?
# > **Core:** freeze choices, score all three models once and inspect their errors.
# > **Change one thing:** tighten a reporting claim after inspecting an error; keep model and prompt settings frozen.
# > **Check:** run `check_comparison`; all three rows must use the same test messages and parser.
# > **Optional stretch:** collect a genuinely new test set for another period or channel after further development.
# > **Common errors:** do not rerun this test to choose a setting; return to validation before the final comparison.
#
# **TODO 3:** after choosing settings on validation, evaluate both base prompts and the fine-tuned model on `test_ds`.
# The fine-tuned model uses the **zero-shot** format it learned. Build `comparison` with `model`, `accuracy` and `invalid`.
# **Hint:** use `evaluate(base, zero, test_ds)`, `evaluate(base, few, test_ds)` and `evaluate(tuned, zero, test_ds)`,
# then `pd.DataFrame([{"model": "base, zero-shot", "accuracy": df_zero.correct.mean(), "invalid": df_zero.pred.isna().mean()}, ...])`.
# Freeze settings before running. Saved checksums identify the actual adapter weights, training data and configuration.
# An unchanged model reuses saved predictions; a changed model or training setup requires a new final test.
#
# ### Use and check the adapted classifier
#
# ![Use the adapted classifier](fig:beginner_inference)
#
# 1. Choose a held-out message about a card that has not arrived.
# 2. Format it with the same chat template used in training.
# 3. Run the base model with the adapter; no weights change.
# 4. Parse the reply into one intent and compare it with the reference.
#
# **Predict before running:** why use the same parser for base and adapted answers?
#
# <!-- BEGIN ANSWER -->
# **Instructor explanation:** Otherwise a score difference could come from different checking rules rather than the model.
# <!-- END ANSWER -->

# %% [markdown]
# **Purpose:** Evaluate the adapter on the same test messages and build the comparison table.
# **Why now:** The same messages, parser and decoding make the three rows comparable.
# **Expected output:** Three accuracy lines (or cached predictions), a three-row table and one ✅ line.

# %%
tuned = trainer.model.eval()


def adapter_fingerprint(model):
    # A fingerprint is a short checksum: changing any trained adapter value changes this record.
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:  # Read only the small adapter; do not copy the frozen base model.
            description = (name, str(parameter.dtype), tuple(parameter.shape))
            digest.update(json.dumps(description).encode("utf-8"))
            values = parameter.detach().cpu().contiguous().view(torch.uint8)
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


# Store choices before revealing any final score. The synthetic probe and local demonstrations stay local.
# The second checksum also records training messages, seed and configuration; it never leaves the notebook.
training_fingerprint = hashlib.sha256(json.dumps(
    {"examples": list(train_ds), "configuration": args.to_dict()},
    sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
final_settings = {"model": MODEL_ID, "intents": tuple(INTENTS), "rank": RANK,
                  "learning_rate": args.learning_rate, "seed": args.seed, "steps": trainer.state.global_step,
                  "adapter_fingerprint": adapter_fingerprint(tuned), "training_fingerprint": training_fingerprint,
                  "zero_template": zero({"text": "FROZEN_TEMPLATE", "label_text": INTENTS[0]}),
                  "few_template": few({"text": "FROZEN_TEMPLATE"}),
                  "parser": {lab: pattern.pattern for lab, pattern in LABEL_RE.items()}}
### BEGIN SOLUTION
if "_W08_FINAL_TEST" in globals():
    if _W08_FINAL_TEST["settings"] != final_settings:
        raise RuntimeError("Final test already viewed. Develop on validation and reserve a new final test after changing settings.")
    print("Reusing the saved final-test predictions; no test messages are scored again.")
    df_zero, df_few, df_tuned = (_W08_FINAL_TEST[k] for k in ("zero", "few", "tuned"))
else:
    df_zero = evaluate(base, zero, test_ds, name="base, zero-shot")
    df_few = evaluate(base, few, test_ds, name="base, few-shot")
    df_tuned = evaluate(tuned, zero, test_ds, name="LoRA fine-tuned")
    _W08_FINAL_TEST = {"settings": final_settings, "zero": df_zero, "few": df_few, "tuned": df_tuned}
comparison = pd.DataFrame([
    {"model": "base, zero-shot", "accuracy": df_zero.correct.mean(), "invalid": df_zero.pred.isna().mean()},
    {"model": "base, few-shot", "accuracy": df_few.correct.mean(), "invalid": df_few.pred.isna().mean()},
    {"model": "LoRA fine-tuned", "accuracy": df_tuned.correct.mean(), "invalid": df_tuned.pred.isna().mean()},
])
### END SOLUTION
### STUB df_zero, df_few, df_tuned, comparison = df_zero_dev, df_few_dev, df_zero_dev, pd.DataFrame()  # TODO 3: freeze settings, score test_ds once and build the table


def check_comparison(table, tuned_df):
    try:
        frames = (df_zero, df_few, tuned_df)
        ok = (list(table["model"]) == ["base, zero-shot", "base, few-shot", "LoRA fine-tuned"]
              and tuned_df is not df_zero
              and all(list(frame["text"]) == list(test_ds["text"]) for frame in frames)
              and all(abs(float(table["accuracy"].iloc[i]) - float(frame.correct.mean())) < 1e-9
                      and abs(float(table["invalid"].iloc[i]) - float(frame.pred.isna().mean())) < 1e-9
                      for i, frame in enumerate(frames)))
    except (KeyError, IndexError, AttributeError, TypeError):
        ok = False
    return check(ok, "TODO 3: three rows on the same test messages, with the adapter's own accuracy.",
                 "Score all three models on test_ds once, then compute each row from its own accuracy and invalid rate.")


assert check_comparison(comparison, df_tuned), "Complete TODO 3, then rerun this cell before reading errors or reporting results."
comparison

# %% [markdown]
# **Purpose:** Read the remaining mistakes and the confusion table.
# **Why now:** An average hides which intents are confused with each other.
# **Expected output:** The error count, up to ten wrong replies and a table of reference versus predicted intent.

# %%
errors = df_tuned[~df_tuned.correct]
print(f"{len(errors)} errors")
display(errors[["text", "gold", "raw"]].head(10))
pd.crosstab(df_tuned.gold, df_tuned.pred.fillna("INVALID"))

# %% [markdown]
# ✍️ **Question 1.** Compare the three rows. Did fine-tuning help, and by how much? Look at the remaining errors: are
# they model errors or ambiguous messages? Would the conclusion change with a larger or a different test set?
#
# <!-- BEGIN ANSWER -->
# Report this run's accuracies and invalid rates; fine-tuning is not guaranteed to win. Few-shot examples can clarify the labels without training. SFT can teach the label format and the differences between similar intents, without sending examples in every future prompt. Inspect mistakes such as card_arrival versus lost_or_stolen_card to separate wrong decisions from ambiguous wording. The parser is lenient, so a sentence containing a recognised label counts. With 50 test messages, one changed prediction is 2 percentage points, so small gaps need cautious wording; the stored full run used 200 messages (0.5 points each). A new period, channel or customer group may change the result.
# <!-- END ANSWER -->

# %% [markdown]
# ## Part 6 · Forgetting, modularity and size
#
# LoRA keeps the base weights frozen, so the adapter can be **switched off**. One general question illustrates
# possible specialisation; it is not a broad test of earlier abilities.
#
# > **Read:** disabling LoRA removes the added correction and restores the frozen base path.
# > **Predict:** does disabling the adapter change the base weights?
# > **Core:** ask the same question with the adapter on and off; save the adapter and measure it.
# > **Change one thing:** ask a banking question with the adapter off and compare the reply.
# > **Check:** compare file sizes with parameter counts and explain both answers.
# > **Optional stretch:** design a separate general-question set to measure forgetting; label its sources and limitations.
# > **Common errors:** an adapter loaded onto a different base model fails or behaves unpredictably.

# %% [markdown]
# **Purpose:** Ask one general question with the adapter on and off.
# **Why now:** Fine-tuning on one narrow task can change other behaviour.
# **Expected output:** Two replies from the same base model.

# %%
q = [{"role": "user", "content": "In one sentence, what is the capital of France and why is it famous?"}]
print("adapter on: ", predict(tuned, q, max_new_tokens=40))
with tuned.disable_adapter():
    print("adapter off:", predict(tuned, q, max_new_tokens=40))

# %% [markdown]
# **TODO 4:** save the adapter, then compute `adapter_mb` (total size of the files in the adapter folder) and
# `model_mb` (base parameters × bytes per parameter).
# **Hint:** `sum(f.stat().st_size for f in Path("banking-lora-adapter").iterdir()) / 1e6` and
# `sum(p.numel() * p.element_size() for p in base.parameters()) / 1e6`.
#
# **Purpose:** Save the added weights, measure the files and compare with the base model.
# **Why now:** A small adapter is what makes swapping one base model between tasks practical.
# **Expected output:** The two sizes in MB and one ✅ line.

# %%
tuned.save_pretrained("banking-lora-adapter")
### BEGIN SOLUTION
adapter_mb = sum(f.stat().st_size for f in Path("banking-lora-adapter").iterdir()) / 1e6
model_mb = sum(p.numel() * p.element_size() for p in base.parameters()) / 1e6
### END SOLUTION
### STUB adapter_mb, model_mb = 0.0, 0.0  # TODO 4: measure the adapter files and the base parameters


def check_sizes(adapter_size, base_size):
    expected_adapter = trainable * 4 / 1e6  # 4 bytes per trainable fp32 number
    ok = (abs(adapter_size - expected_adapter) < 0.15 * expected_adapter + 0.05
          and abs(base_size - (total - trainable) * 4 / 1e6) < 1)
    return check(ok, "TODO 4: the adapter holds about 4 bytes per trainable number; the base holds 4 bytes per weight.",
                 "Add up the file sizes in the adapter folder; multiply each base parameter count by its element size.")


assert check_sizes(adapter_mb, model_mb), "Complete TODO 4, then rerun this cell before comparing or reporting sizes."
print(f"adapter: {adapter_mb:.1f} MB | base model weights (fp32): {model_mb:,.0f} MB")

# %% [markdown]
# ✍️ **Question 2.** What did the model do with the general question, with the adapter on and off, and why? How does
# LoRA's modularity help when one base model serves several departments?
#
# <!-- BEGIN ANSWER -->
# Report the two replies from your run. With the adapter on, the reply can change in style or accuracy, for example by adding confident but unsupported details, because the adapter was trained only on short banking labels; with the adapter off, the frozen base answers as before, because LoRA never changed the original weights. Check every factual claim in both replies. One question is an illustration, not a measurement: use a varied general-question set to measure forgetting. In practice one base model can be loaded once and several small adapters swapped per request or department, which saves memory and makes rollback simple. To reduce forgetting, mix general instruction data into training, train for fewer steps or a lower learning rate, and evaluate general ability alongside the task metric.
# <!-- END ANSWER -->

# %% [markdown]
# ### Optional: compare with a hosted model
#
# **Purpose:** Ask one hosted model to classify invented queries using separate invented demonstrations (five calls at most).
# **Why now:** A large hosted model is a realistic alternative to fine-tuning; compare it on the same invented items.
# **Expected output:** With both switches False (default), a message that the local results stand. With one switch
# enabled and a key available, a table of hosted and local predictions. Both demonstrations and queries are invented;
# no Banking77 text or local `SHOTS` is sent to either provider.

# %%
# Hosted demonstrations are written for this exercise; never reuse the dataset-backed local SHOTS.
HOSTED_EXAMPLES = [("I ordered a replacement card last week and it has not reached me.", "card_arrival"),
                   ("My debit card was stolen from my bag yesterday.", "lost_or_stolen_card"),
                   ("How many pounds will I receive for twenty euros?", "exchange_rate"),
                   ("My attempt to load funds onto the account failed.", "top_up_failed"),
                   ("Can a sixteen-year-old open this type of account?", "age_limit")]


def hosted_prompt(text):
    examples = "\n".join(f"Message: {message}\nIntent: {label}"
                         for message, label in HOSTED_EXAMPLES if label in INTENTS)
    return f"{INSTRUCTION}\n\nInvented examples:\n{examples}\n\nMessage: {text}\nIntent:"


INVENTED = [("My new card still hasn't turned up after ten days.", "card_arrival"),
            ("Someone took my wallet with my card in it.", "lost_or_stolen_card"),
            ("What rate do you use to convert euro to dollars?", "exchange_rate"),
            ("I tried to add money and it was declined.", "top_up_failed"),
            ("Is there a minimum age to open an account?", "age_limit")]
hosted_rows = []
for text, gold in INVENTED[: 1 if SMOKE else None]:
    if gold not in INTENTS:
        continue
    reply = ask_model(hosted_prompt(text), temperature=0.0, max_output_tokens=16)
    if reply is None:
        break
    hosted_rows.append({"text": text, "gold": gold, "hosted": normalise(reply),
                        "local LoRA": normalise(predict(tuned, zero({"text": text, "label_text": gold})))})
if hosted_rows:
    display(pd.DataFrame(hosted_rows))
else:
    print("No hosted predictions (switches off, no key or no response). The local comparison table above is the result.")

# %% [markdown]
# ## Optional extensions
#
# **QLoRA (GPU only).** Load the base model in 4-bit and train LoRA on top:
#
# ```python
# from transformers import BitsAndBytesConfig
# bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", quantization_config=bnb, device_map="auto")
# ```
#
# **DPO (preference tuning).** With a dataset of `prompt`, `chosen`, `rejected` examples:
#
# ```python
# from trl import DPOConfig, DPOTrainer
# trainer = DPOTrainer(model=model, args=DPOConfig(output_dir="dpo", beta=0.1), train_dataset=prefs, processing_class=tok, peft_config=lora_config)
# ```
#
# ✍️ **Question 3.** Your team is choosing between (a) few-shot prompting a large API model and (b) this LoRA-tuned
# 0.5B model for routing 50,000 messages per day. Compare them on accuracy, cost, latency, privacy and maintenance.
#
# <!-- BEGIN ANSWER -->
# (a) A large API model with few-shot prompts may reach high accuracy without training, and updates are easy (edit the prompt), but it costs per token for 50,000 long prompts per day (the label list and examples are resent each time), adds network latency, sends customer data to a third party (GDPR and contract review needed) and depends on the provider's model versions. (b) The LoRA-tuned small model needs labelled data and a training and evaluation pipeline, but inference is cheap and fast on modest hardware, data stays in-house, the version is fixed, and the adapter is small and easy to roll back; it must be retrained when intents change and monitored for drift. For high-volume, stable, privacy-sensitive classification, (b) is usually preferable if its measured accuracy meets the bar; (a) suits prototyping or rapidly changing categories.
# <!-- END ANSWER -->

# %% [markdown]
# ## Evidence for your report
#
# **Purpose:** Collect this run's results with their settings and print the elapsed time.
# **Why now:** Reported numbers need the model, data sizes, rank and run mode beside them.
# **Expected output:** A summary line and the elapsed seconds; a JSON file only when the teacher sets GENAI_RESULTS_PATH.

# %%
elapsed_seconds = time.perf_counter() - core_started
loss_hist = hist.dropna(subset=["loss"])[["step", "loss"]].to_dict("list")
eval_hist = hist.dropna(subset=["eval_loss"])[["step", "eval_loss"]].to_dict("list") if "eval_loss" in hist else {}
run_results = {
    "run": {"model": MODEL_ID, "device": str(DEVICE), "smoke": bool(SMOKE), "full": bool(RUN_FULL), "seed": SEED,
            "intents": INTENTS, "rank": RANK, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "train_minutes": train_minutes, "steps": trainer.state.global_step, "elapsed_seconds": elapsed_seconds},
    "trainable": trainable, "total": total, "comparison": comparison.to_dict("records"),
    "adapter_mb": adapter_mb, "model_mb": model_mb, "loss": loss_hist, "eval": eval_hist,
}
best = comparison.sort_values("accuracy").iloc[-1]
print(f"Best: {best['model']} ({best['accuracy']:.0%}) | training {train_minutes:.1f} min | elapsed {elapsed_seconds:.0f} s")
results_target = os.environ.get("GENAI_RESULTS_PATH")
if results_target:
    target = Path(results_target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("Results exported to:", target)

# %% [markdown]
# ## Five self-checks
#
# 1. Which tokens contribute to the SFT loss in this lab: the prompt, the answer, or both?
# 2. Rank 8 on an 896 × 896 projection adds how many trainable numbers?
# 3. Why must the test messages never appear in training or in choosing settings?
# 4. The adapter scores 4 points higher than few-shot on 50 messages. How many messages is that?
# 5. Why can the adapter be switched off without reloading the model?
#
# <!-- BEGIN ANSWER -->
# 1. The answer (completion) tokens; the prompt is read as context but masked out of the loss.
# 2. 8 × (896 + 896) = 14,336.
# 3. Otherwise the test rewards memorisation or tuning to those messages and overstates accuracy on new messages.
# 4. Two messages, because each message is 2 percentage points out of 50.
# 5. LoRA adds separate A and B matrices beside the frozen weights, so disabling them restores the original W x path.
# <!-- END ANSWER -->
