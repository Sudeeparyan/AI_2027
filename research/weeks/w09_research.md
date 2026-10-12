# W09 research: connecting images, words and audio

Checked **2026-10-10**. R3 scope: built slide titles and lab headings only; no lab execution.

**Teaching explanations (our analogies, grounded in sources):**
- Images and captions receive coordinates on a shared map; compare directions to find a match ([CLIP paper](https://arxiv.org/abs/2103.00020)). Limit: a vector is learned numerical evidence, not a complete picture of meaning.
- Contrastive training is a matching game: reward the correct image-caption pair among distractors ([CLIP code](https://github.com/openai/CLIP)). Start with a three-by-three score matrix. Limit: an unmatched caption can still describe another image; negatives are not always semantically false.
- Speech-to-gallery search is a relay: Whisper transcribes, CLIP encodes the text, similarity ranks images ([Whisper](https://github.com/openai/whisper)). Limit: this pipeline does not directly align audio and image embeddings; errors can propagate.

**Five misconceptions:** (1) CLIP generates images; (2) cosine similarity is a calibrated probability; (3) shared dimensions mean image/text distributions are identical; (4) every off-diagonal pair is truly wrong; (5) speech transcription is a perfect transcript. Ask students to identify a failure example for each.

**Current practice:** [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) remains a historical teaching checkpoint. Compare [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224): the 2025 family adds multilingual and localisation/dense-feature objectives to SigLIP. Verify multilingual performance. [Whisper](https://github.com/openai/whisper) lists `tiny`/`tiny.en`; `turbo` is not trained for translation. Never transfer its A100 speed table to CPU slides. Transformers model/processor APIs are documented; E2 must test versions, using R0's snapshots for planning.

**Redraw:** image and text encoder paths merge at normalisation/similarity; label embeddings `[B,D]` and scores `[B,B]`, then ranking. Three pairs reveal the diagonal, temperature and row/column softmax. Distinguish SigLIP's pairwise sigmoid from CLIP's batch softmax. Separate speech -> transcript -> text encoder -> gallery ranking prevents claims of direct audio-image alignment. Label arrows and role-colour legends.

**Beginner barriers:** matrix transpose, batch axes, normalisation, temperature, softmax versus sigmoid and precision@k arrive together. Print the shape after each operation. Calculate one dot product and one row's loss before the general equation. Explain false negatives. Audio brings file paths, sample rates and `ffmpeg`; isolate it as an optional stretch with a stored measured transcript.

**Proposed CPU core, target <=15 minutes (unmeasured):** encode 12 supplied non-personal images and four captions once using CLIP ViT-B/32; cache features; classify, retrieve and calculate one three-pair loss locally. Slider changes temperature using cached scores, not repeated model calls. Retain a genuine encoder run and report cold-download versus warm execution. Larger galleries, SigLIP 2 and one short Whisper-tiny clip are optional; use stored measured outputs for discussion. Optional Gemini makes <=3 synthetic-image description calls through R0's text-output multimodal model, then compare descriptions with retrieval failures. It provides no local CLIP training or guaranteed truth. Keep `USE_GEMINI=False`, synthetic/public non-personal data, a complete local fallback and R0's EEA/UK free-API terms caveat; do not promise eligibility.

**Further reading (5):** [CLIP paper](https://arxiv.org/abs/2103.00020); [CLIP implementation](https://github.com/openai/CLIP); [SigLIP 2 paper](https://arxiv.org/abs/2502.14786); [Whisper implementation](https://github.com/openai/whisper); [Stanford Online video: Multimodal Deep Learning, Douwe Kiela](https://www.youtube.com/watch?v=5vfIT5LOkR0) (2023; foundational).

Handoff: source research complete; video metadata verified in search, direct playback untested. CPU/download timing, model outputs, Gemini access and widget rendering unmeasured.
