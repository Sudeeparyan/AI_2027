# W08 research: real, small fine-tuning

Checked **2026-10-10**. R3 scope: built slide titles and lab headings only; no lab execution.

**Teaching explanations (our analogies, grounded in sources):**
- Prompting is a task brief; retrieval opens reference pages; fine-tuning changes a learned habit. Use the same banking query in all three. Limit: neither training nor retrieval guarantees truth ([HF SFT](https://huggingface.co/docs/trl/main/en/sft_trainer)).
- LoRA is a removable correction sheet over a fixed machine: `W` stays fixed; two smaller matrices produce its update ([PEFT explanation](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)). Limit: the correction is a matrix operation, not stored replacement answers.
- Preference pairs resemble choosing the better of two drafts. DPO learns from these comparisons without a separately fitted reward model ([original paper](https://arxiv.org/abs/2305.18290)). Limit: preference is not objective truth or proof of safety.

**Five misconceptions:** (1) SFT always updates every parameter; (2) fine-tuning is a reliable database update; (3) a smaller training loss proves generalisation; (4) LoRA removes all training memory/compute; (5) DPO needs no reference policy or trustworthy preference data. Give one counterexample and a self-check per misconception.

**Current practice:** PEFT `LoraConfig` with TRL `SFTTrainer` remains documented. Distinguish prompt-completion loss from full-sequence loss; `assistant_only_loss` needs compatible chat-template generation markers ([TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)). Keep plain float32 CPU LoRA in the core; QLoRA and larger alignment jobs are stretch topics. [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) is an available compact teaching checkpoint; [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) illustrates current chat-template thinking switches, with non-thinking mode for short labels. Neither is labelled latest/best. R0 lists Colab package snapshots, not a tested lock; TRL is absent from those inventories, so E2 must install/test it explicitly.

**Redraw:** labels -> train/test split -> chat formatting -> tokenisation -> frozen model + trainable adapter -> held-out evaluation -> saved adapter. LoRA shape inset: `W[6,6]`, `A[2,6]`, `B[6,2]`, so `BA[6,6]`; these are illustrative dimensions. Solid arrows carry tokens/loss/weights; dashed arrows show optimiser feedback. Reuse figures across artifacts.

**Beginner barriers:** chat templates, answer masking, gradients, rank, quantisation and alignment arrive too quickly. Show one string becoming token IDs, with ignored versus supervised positions, before the loss equation. Show one actual adapter update before RLHF/DPO comparisons. Explain why held-out examples never enter training.

**Proposed CPU core, target <=15 minutes (unmeasured):** retain LO3's real causal-LM LoRA job: SmolLM2-135M, 32 short synthetic banking examples, three intents, sequence cap 64, rank 2 and an initial cap of 16 optimiser steps. Compare base/adapted outputs on the same separate test set; check gradients, frozen weights and save/reload. No promised accuracy gain. Slider changes rank's parameter count without retraining. Keep the existing broader Banking77 job optional with measured stored results. Measure cold/warm waiting and cut steps if necessary. Optional Gemini provides a prompt-only baseline, not free fine-tuning; different-model scores cannot isolate adapter effects. Preserve `USE_GEMINI=False`, synthetic inputs, local completion and R0's EEA/UK regional terms caveat.

**Further reading (5):** [LoRA paper](https://arxiv.org/abs/2106.09685); [PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora); [TRL SFT guide](https://huggingface.co/docs/trl/main/en/sft_trainer); [DPO paper](https://arxiv.org/abs/2305.18290); [video: LoRA explained by its inventor](https://www.youtube.com/watch?v=DhRoTONcyZE) (linked by the [authors' repository](https://github.com/microsoft/LoRA)).

Handoff: source research complete; training configuration is a proposal. CPU timing, package compatibility, Gemini access, video playback and widget rendering unverified.
