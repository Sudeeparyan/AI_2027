# Lecture plan at a glance

This week turns last week's multimodal foundations into applications (MIMLO 2, 3, 4). Students should leave able to explain the two designs of multimodal LLMs, describe how such models are trained and adapted, use models for image, speech and cross-modal tasks, choose evaluation methods for each modality (including hallucination probes), and propose a layered provenance and consent strategy for synthetic media. The archived measured images and results come from `curriculum/assets/week_10/`; the revised CPU core records its own observations separately.

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | Receipt photo: reading, reasoning and privacy risks. |
| 5–30 min | 5–12 | 1 · Multimodal LLMs | Two designs; capabilities; families; real captions; visual tokens; real chart/document QA; quiz. |
| 30–43 min | 13–16 | 2 · Training and fine-tuning | Training pipeline; data; adapting with LoRA. |
| 43–68 min | 17–24 | 3 · Generation across modalities | Map; real text→image and image→image; editing and control; speech; music and video; quiz. |
| 68–75 min | – | Break | |
| 75–92 min | 25–28 | 4 · Evaluation | Metrics by task; multimodal hallucination (real probe); human evaluation. |
| 92–114 min | 29–37 | 5 · Responsible use, applications, trends | Harms; provenance layers; labelling discussion; consent and rights; applications; trends; design pattern; checklist. |
| 114–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Use the fictional invoice supplied by the lab, and ask a local multimodal model to read a field. Then ask it to extract the line items as JSON and total them yourself. Students see the "perceive with the model, compute with code" pattern before you name it.

<!-- pagebreak -->

# Lecture notes

Start with the vocabulary and overall flow. For each section below, read the simple explanation first, trace the worked example aloud, and ask students to name the input and output. Introduce the equation only after the operation makes sense. The detailed text retains the full syllabus, while the lab and checks show what each method can and cannot establish.

## 1. Multimodal LLMs

**Start here.** A vision-language model uses an image and text together to generate an answer. It can read the wrong value and still write fluently. Trace pixels → visual features → language input → answer text. Modular designs use a learned bridge; native designs train with mixed data types. Supported features depend on the exact model.

**Small worked example.** For a chart with bars at 10, 20 and 30, ask which bar is largest and check against the known values. Then ask for a sum and compute 10 + 20 + 30 = 60 in code. Reading and arithmetic are separate error sources.

A **multimodal large language model (MLLM)** processes text alongside other supported data types. When it accepts images with text, it is often called a **vision-language model (VLM)**. Some systems also accept audio or video or generate media. The exact model and endpoint determine the available inputs and outputs.

**Two designs.**

* **Modular (bridge) models** connect separately pretrained parts: a vision (or audio) encoder such as CLIP/SigLIP, a small **projector** that maps its features into the LLM's embedding space, and the LLM (the LLaVA recipe from week 9). Cheap to build and common among open models; usually text output.
* **Natively multimodal ("omni") models** are trained from the start on interleaved sequences of text, image and audio tokens (early fusion). One model can read and often generate several modalities, which enables real-time voice conversations. The GPT and Gemini families and open families such as Qwen-Omni follow this direction.

![Modular vs natively multimodal](fig:mllm_types)

**Capabilities.** Captioning and visual question answering (VQA), alt text for accessibility; **document understanding** (invoices, forms, PDFs, handwriting); **chart and table reading**; video understanding by sampling frames (and audio); speech understanding and spoken answers; and **screen understanding** for computer-use agents (week 12).

| Family | Access | Inputs | Notes |
|---|---|---|---|
| GPT (OpenAI) | Proprietary | Text, images, audio | Native voice and image generation in the app |
| Gemini (Google) | Proprietary | Text, images, audio, video, long PDFs | Natively multimodal; long video context |
| Claude (Anthropic) | Proprietary | Text, images, PDFs | Strong document and chart reading |
| Qwen-VL / Qwen-Omni | Open-weight | Images, video; audio in Omni | Many sizes; strong on documents |
| LLaVA | Open research | Images | Reference modular design |
| SmolVLM, Gemma, Llama vision | Open-weight | Images | Small models for laptops and devices |

**Archived captions (Qwen3-VL-2B, CPU; asset date not recorded):** {{CAPTIONS}}

**Visual tokens and cost.** In a simple patch design, 336/14 = 24 patches per side, giving 24 × 24 = 576 patch vectors. Other models add, combine or tile visual features. Higher resolution can preserve small text but needs more computation. Video samples frames and may miss events between them. Providers use different media billing rules; estimate a small sample with the current endpoint pricing before processing a large collection.

**Archived cross-modal reasoning, measured.** The lab generates a bar chart and an invoice image, so the correct answers are known exactly (the busiest day, a single value, a total that requires adding five bars, the invoice number, the total due, the customer). **Lab result:** {{QA_NOTES}}

![Generated chart and invoice used for evaluation](fig:lab_chart)

> **Key idea:** Let the model **perceive** (extract fields into a schema validated in code, as in week 7) and let **code compute** totals and cross-check them against printed totals. Do not trust a VLM's arithmetic across many images, or its self-reported confidence.

## 2. Training and fine-tuning multimodal models

**Start here.** A training example pairs an image with a caption or a question and desired answer. A bridge can first learn alignment; instruction examples then teach useful visual responses. The lab uses pretrained models. A project considering adaptation needs enough representative images, data permission and a separate final test.

**Small worked example.** An image-caption example says "a red cup". A visual instruction example asks "What colour is the cup?" and answers "red". A new photo of a red cup should appear only in evaluation if it is meant to test generalisation.

![A typical VLM training pipeline](fig:vlm_training)

1. **Pre-train the parts** separately: vision encoder (CLIP/SigLIP) and LLM. This is where most compute goes.
2. **Alignment pre-training:** train the projector (sometimes more) on image–caption pairs so visual features land in the LLM's input space.
3. **Visual instruction tuning** (Liu et al., 2023): fine-tune on image-based conversations, now including charts, documents, OCR, screenshots and grounding; much of this data is synthetic, generated with stronger models or rendered with known answers.
4. **Preference tuning / RL** (week 8) to reduce hallucination and improve helpfulness.
5. **Adapt (optional):** freeze the vision encoder and apply **LoRA** to the LLM (and optionally the projector) on your own image–question–answer examples; TRL supports vision-language fine-tuning.

**Training data:** images with captions, mixed image/text documents, speech with transcripts, and labelled tasks such as charts, document fields or object locations. Some examples are generated by a model or drawn from known values. Review their correctness, remove duplicates and check licences and consent (including the LAION issues from week 9).

**When to fine-tune:** prompting plus structured extraction first; fine-tune when evaluation on **your own images** shows persistent reading or format errors. Regulated domains (e.g. medical imaging) need clinical validation and human oversight.

## 3. Generation across modalities

**Start here.** The requested output determines the model route. Image generation produces pixels, text-to-speech produces a waveform, and a vision-language model produces answer text. Image editing starts from an existing picture and controls how much it may change. These routes may use different models and processors.

**Small worked example.** Compare editing strength 0.3 and 0.9 with the same image and prompt. The higher setting adds more starting noise and usually permits more change, but does not guarantee a sharper or better picture.

![Generation tasks, methods and example families](fig:generation_map)

| Task | How it works | Example families |
|---|---|---|
| Text → image | Latent diffusion or flow transformers conditioned on text (week 4) | Stable Diffusion, FLUX, Imagen, GPT image models |
| Image → image | Add partial noise to an image and denoise with a new prompt; inpainting; control | SDEdit, ControlNet, instruction editors |
| Text → speech | Acoustic model + vocoder (VITS/MMS) or language models over audio-codec tokens | MMS-TTS, Bark, Kokoro, voice cloning systems |
| Speech → speech | Speech LLMs that listen and answer in voice with low latency | Voice modes of assistants |
| Text → music/audio | Diffusion or token LMs over audio codecs | MusicGen, Stable Audio |
| Text → video | Diffusion transformers over spacetime patches, some with synchronised audio | Sora, Veo families, open video models |

**Archived text → image.** SD-Turbo is a distilled model: the stored experiment used two steps, zero guidance and 512 × 512 outputs. This figure has no per-image runtime claim. **Lab result:** {{T2I_NOTES}}

![SD-Turbo text-to-image outputs](fig:lab_t2i)

**Archived image-to-image.** Strength controls how much starting noise the pipeline adds before denoising. A lower value tends to retain more original structure; a higher one allows greater change. This scheduler derives effective steps from steps times strength, with rounding. Inspect the outputs rather than assuming a setting guarantees a particular result.

![Image-to-image at three strengths](fig:lab_i2i)

**Editing, control and personalisation.** Inpainting/outpainting (regenerate a masked region or extend the canvas); control (ControlNet and adapters following edges, depth, pose or a reference image); instruction-based editing ("make it winter"), now conversational in natively multimodal models; personalisation with LoRA or DreamBooth on a few images of a subject or style, which raises likeness and style-rights questions.

**Speech.** **TTS** means text-to-speech. A model predicts sound features or tokens and creates a waveform; VITS-based MMS combines these stages. Audio-codec token models represent short sound segments as IDs. Voice cloning can imitate a speaker and requires permission and safeguards against impersonation. Speech-to-speech systems recognise or directly process sound and reply aloud; delay varies by system. **Archived round trip (MMS-TTS → Whisper):** {{SPEECH_NOTES}}

**Music, sound and video.** MusicGen is a language model over audio-codec tokens; Stable Audio is a latent diffusion model. Rights holders have sued several music-generation companies over training data. Text-to-video models extend diffusion transformers to space and time; since 2025 some generate synchronised sound. Open problems: temporal consistency, physics, long durations and cost.

## 4. Evaluating multimodal generative AI

**Start here.** Choose a check for the actual task. Known chart values check factual answers. CLIP similarity checks some prompt alignment, rather than truth or complete image quality. WER measures recovered words, not whether generated speech sounds natural. Human review checks aspects these small automatic measures miss.

**Small worked example.** A generated image can score well for "two red cups" while showing three cups. Count the cups as well as reading the similarity score. A speech round trip can recover all words yet sound unnatural; listen to it.

| Task | Automatic metrics | Human evaluation |
|---|---|---|
| Text → image | FID (realism), CLIP score (alignment), GenEval (composition) | Side-by-side human preference; rubric for alignment and artefacts |
| Image → text (captions) | CIDEr, BLEU, CLIPScore | Correctness, completeness, invented details |
| VQA, charts, documents | Accuracy against ground truth | Error analysis by question type |
| Text → speech | WER via ASR, speaker similarity | Mean Opinion Score (naturalness), intelligibility |
| Text → video | FVD, CLIP-based consistency | Motion quality, physics, prompt adherence |
| MLLM hallucination | POPE (yes/no object probes), CHAIR (invented objects in captions) | Audit of invented objects, numbers, text |

**CLIP score** is 100 × the cosine similarity between CLIP's image and text embeddings. A useful sanity check is that the lab compares each image’s own-prompt score with its mean score against the other prompts. That comparison does not establish that the own prompt ranks first.

**Multimodal hallucination** means describing absent objects, misreading numbers, inventing text in documents, or agreeing with leading questions. **POPE** (Li et al., 2023) asks balanced "Is there a {object} in the image?" questions about present and absent objects and reveals a **yes-bias**. **Archived classroom probe:** {{POPE_NOTES}} Mitigations: higher resolution; "answer only from what is visible"; extraction with verification; preference training against hallucination.

**Human evaluation:** show two outputs side by side and randomise their positions. Give reviewers a rubric: prompt match, defects and realism for images; naturalness and clarity for sound. Use varied content, accents, languages and raters. Report agreement and disagreements. Automatic scores check different pieces of quality and support this review.

## 5. Responsible use, applications and trends

**Start here.** Realistic generated media can impersonate people or mislead viewers. Obtain permission for likeness and voice use, protect personal data and disclose generated or materially edited content. Provenance records how media was made or changed; a signature or watermark supports inspection but does not prove that a scene is true.

**Small worked example.** A PNG text label disappears when saved as an ordinary JPEG. This demonstrates fragile metadata, not a failure test of signed C2PA. Ask what information survives a screenshot or upload and what viewers can actually verify.

**Harms of realistic synthetic media.**

* **Fraud and impersonation:** in early 2024 an employee of the engineering firm Arup in Hong Kong transferred about US\$25 million after a video call in which the other participants were deepfakes; cloned voices are used in "family emergency" and CEO scams.
* **Non-consensual intimate imagery**, including of minors: among the most damaging uses; criminalised in many jurisdictions.
* **Disinformation** about public figures and events, and the **liar's dividend** (real evidence dismissed as fake).
* **Harassment and scams** using synthetic personas.

**Provenance and labelling work in layers; none is sufficient alone.**

![Provenance layers and their weaknesses](fig:provenance)

| Layer | What it does | Weakness |
|---|---|---|
| Visible label / disclosure | Tells viewers content is AI-generated or edited | Easy to crop or omit |
| C2PA content credentials | Cryptographically signed history of an asset (tool, edits, author) | Can be stripped; needs platform support |
| Invisible watermark (e.g. SynthID) | Signal embedded in pixels, audio or text | Only if the generator adds it; heavy edits weaken it |
| Detection classifiers | Predict whether media is synthetic | Arms race; false positives on real media |

**Law.** Article 50 applies from 2 August 2026. Providers generally disclose direct AI interaction unless obvious, and must machine-mark synthetic outputs under Article 50(2), subject to exceptions including standard editing. Deployers disclose deepfakes and certain public-interest text; the text rule includes a human-review/editorial exception. [Official Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50). Systems marketed before 2 August 2026 have until 2 December 2026 for the Article 50(2) marking/detection duty only. [Commission FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act).

**Consent, likeness and rights.** Obtain explicit permission before imitating a real person's face or voice in this project. Check applicable likeness, copyright and model/data licence rules. Identifiable people and document details may be personal data under GDPR. Minimise collection, use an appropriate lawful basis and check third-party processing terms.

**Applications.** Accessibility (alt text, captions, audio description); document processing (invoices, forms, contracts); multilingual voice assistants; education (explaining diagrams, feedback on handwritten work with teacher review); design and marketing (concepts, mock-ups, localisation, with labelling); inspection and robotics (including vision-language-action models).

**Trends (2026–27).** Any-to-any models; real-time voice and video agents; video generation with sound and more control; computer-use agents (week 12); world models for simulation and robotics; on-device multimodal small models.

**A design pattern for multimodal applications:** *Input* (check quality and consent) → *Perceive* (VLM/ASR extracts into a schema) → *Verify* (code checks totals, formats, cross-references) → *Act or generate* → *Review and label* (human review for risky cases; provenance on outputs). Slogan: **models perceive, code computes and checks, people decide when stakes are high.**

# Common misconceptions

| Misconception | Correction |
|---|---|
| "If a VLM reads a chart correctly, it can also total it." | Reading and arithmetic are different skills; aggregation errors are common. Extract, then compute in code. |
| "Multimodal LLMs see the whole image at full detail." | Images are resized or tiled into a limited number of visual tokens; small text can be lost at low resolution. |
| "A fluent caption is an accurate caption." | Captions can include invented objects, places or times (hallucination); check them before using as alt text. |
| "Image-to-image 'strength' is a quality setting." | It controls how much of the original is replaced by noise before denoising. |
| "A low WER means the synthetic voice is good." | The round trip measures intelligibility for one ASR model, not naturalness, prosody or speaker similarity. |
| "Watermarks solve deepfakes." | Watermarks exist only if the generator adds them and can be weakened; combine credentials, labels, detection and platform policy. |
| "Metadata in the file proves provenance." | Ordinary re-saving or screenshots strip it, as the lab shows; C2PA signs it but can still be removed. |
| "Open models can be used for anything." | Several popular multimodal models restrict commercial use: MMS-TTS is CC BY-NC; SD-Turbo uses the Stability AI Community License with conditional commercial permissions. |

# Responsible AI lens: synthetic media and personal data

* **Consent for likeness and voice:** never generate or clone an identifiable real person without explicit consent; this includes classmates in project demos.
* **Personal data in images and audio:** faces, voices, number plates, documents in the background and location metadata; minimise, blur or anonymise before sending to external APIs.
* **Disclosure:** label generated or materially edited media for users. Article 50 has separate provider/deployer duties and exceptions; see the law paragraph.
* **Fairness:** image generators and speech systems show documented disparities across skin tones, genders, accents and languages; evaluate by group.
* **Accessibility is a benefit and a responsibility:** automated alt text and captions must be accurate enough for people who rely on them.
* **Licences:** record the model licence (research-only and non-commercial licences are common) and the terms for generated outputs.

# Lab guide and answers

**Before the lab:** open the notebook in free Google Colab with a Python runtime, or in local VS Code with the course kernel. Run the pinned installation and single injected setup once. CPU is the default; leave `USE_GPU`, `USE_GEMINI` and `USE_OPENAI` disabled for the complete local core.

**Current CPU core:** SmolVLM-256M-Instruct; float32; maximum input side 384 pixels; at most 32 generated tokens per answer; two captions, three reading questions and two yes/no probes. It also computes known totals, runs semantic TODO probes, executes the real word-edit WER algorithm and resaves honest metadata. The design target is at most 15 minutes of cached CPU waiting. Initial installs and checkpoint downloads are additional.

**Measured local core:** {{W10_CORE_TIME}} on {{W10_CORE_DATE}}, using {{W10_CORE_THREADS}} CPU threads. Installation, downloads, widgets and optional stretch/API calls are excluded. This is a local run, not a timed Colab run. {{W10_CORE_RESULTS}}


**Optional extensions:** `RUN_STRETCH = False` keeps SD-Turbo generation/editing, CLIP inference and MMS-TTS/Whisper inference disabled. Their original implementations remain available behind that switch. The core inspects stored image strips and transcript examples. `GENAI_VLM` is an instructor override: changing it changes the experiment and requires separate recording and timing. A GPU may help extensions, but is optional; the current core does not choose it automatically.

**Optional text comparison:** choose exactly one shared `USE_GEMINI` or `USE_OPENAI` switch only if wanted. `ask_model` uses Colab Secrets or environment variables; local users may choose a private `.env` through `LOCAL_ENV_FILE`. Keys are never displayed or placed in notebooks. Availability follows the account and provider terms; OpenAI API charges are separate from ChatGPT subscriptions. The one request contains fictional amounts, and a complete local calculation remains if the API is disabled or unavailable. No provider or image-upload service is required.

**Historical runtime record:** the ledger records 416.0 seconds on a laptop CPU for the earlier full notebook, starting 9 October 2026 at 21:26:31 UTC, with Qwen3-VL-2B selected through `GENAI_VLM` and cached models. Its notebook SHA-256 was `4c77bd1c6d11624eead9ea4470f222fc51119be28cdaebe25c28d1ef1ad57d5e`. This record describes that earlier notebook, not the revised SmolVLM core. The results JSON identifies Qwen3-VL-2B and CPU but does not record an asset date; the asset date is not inferred from this timing record.

**Instructor export:** set `GENAI_RESULTS_PATH` to a distinct `results_cpu_core.json` when measuring the current core. Its existing result keys remain, with empty image/speech extension tables and an explicit `run_kind=core`. The original `results.json`, `fill.json` and image bytes remain archived; do not harvest a core run as full extension evidence.

**Hand-in:** predictions, full replies with their known references, reading/arithmetic/hallucination error analysis, verified code totals, metadata observations and five answered self-checks. If an extension runs, submit its fresh configuration and outputs separately from archived evidence.


## Worked examples: representations, scores and verification

**1. Trace image shapes before trusting a caption.** A single RGB image resized to width 384 and height 256 can be represented as `[1, 3, 256, 384]`: one image, three colour channels, rows and columns. This is an illustrative tensor layout before model-specific tiling and batching. The actual processor prints its returned tensor shapes in `ask`; inspect those instead of assuming every VLM shares this shape. In a simple patch design, $N=(H/P)(W/P)$, where $N$ is patch count, $H$ image height, $W$ width and $P$ square patch size. At 336 × 336 pixels with 14-pixel patches, $N=24×24=576$. Cropping, additional summary tokens and tiling change this calculation.

**2. Decode the continuation.** Autoregressive generation follows $p(a\mid I,q)=\prod_{t=1}^{T}p(a_t\mid I,q,a_{1:t-1})$. Here $p$ is the model probability, $a$ the complete answer, $I$ the image, $q$ the question, $t$ a token position, $T$ the answer length, $a_t$ the next answer token and $a_{1:t-1}$ the earlier answer tokens; the product multiplies conditional probabilities across positions. `generate` returns a sequence containing the input prefix and generated continuation. If the input has 30 token IDs and the sequence has 42, decode positions 30 onward: 12 new answer tokens. A token may be a word fragment rather than a whole word. This calculation does not establish factual accuracy.

**3. Separate perception from calculation.** The synthetic chart source is `[12, 30, 18, 25, 9]`, totalling 94. Thursday is 25, and Tuesday is the largest value. The fictional invoice has 40 × EUR 0.50 and 2 × EUR 7.50, giving EUR 20 + EUR 15 = EUR 35. A model can correctly read one field yet fail aggregation. Code can also correctly sum EUR 200 and EUR 15 if the first field was misread. Inspect source values before summing. A schema-valid number remains a candidate value until checked against the image.

**4. Understand image-text alignment.** The extension computes $s=100(e_I\cdot e_T)/(\Vert e_I\Vert\Vert e_T\Vert)$. Here $s$ is the reported CLIP cosine score, $e_I$ the image embedding vector, $e_T$ the text embedding vector, the centred dot is their dot product and each norm is vector length. The factor 100 rescales similarity; it does not turn the score into a probability. For illustrative vectors `[1,0]` and `[1,1]`, the cosine is $1/\sqrt{2}$, giving approximately 70.7. This is a worked calculation, not a lab measurement. A high score for “two red cups” can coexist with three visible cups; count them and inspect defects separately.

**5. Understand speech errors.** Word error rate is $WER=(S+D+I)/N$. $S$ counts substituted reference words, $D$ deleted words, $I$ inserted words and $N$ reference word count. The dynamic-programming table considers the minimum edit path. For reference “the red cup” and hypothesis “the blue cup”, one substitution across three reference words gives $1/3$. Changing “nine” to “9” may preserve meaning but creates a literal token disagreement unless the normaliser treats them equivalently. The recogniser may also correct an expected mispronunciation. Listen for naturalness, pronunciation and timing separately from WER.

**6. Test operations without heavy downloads.** `check_edits()` passes a probe pipeline to `make_edits`: three calls must use strengths 0.3, 0.6 and 0.9, four steps, zero guidance and equal seeds. Each returned image must match its distinct pipeline output. Calling the pipeline and then returning repeated source images fails. `check_speech()` supplies a probe synthesiser with a deliberately non-default sampling rate and known sample values. The recogniser must receive that waveform and rate, and the helper must return stripped text. These probes check implementation contracts; only the optional real inference establishes generated media quality.

**7. Label the actual source honestly.** The core saves a chart fixture with `ai_generated=false`, `model=matplotlib/synthetic-fixture` and its rendering description. The real diffusion extension uses `ai_generated=true`, its actual pipeline identifier and prompt. Reopening the PNG checks exact fields; the JPEG experiment checks their disappearance. The same text fields can be edited or forged, so this exercise establishes metadata behaviour, not signed provenance or truth.

## Core/extension checklist by part

| Part | Core task | Change one thing | Optional stretch | Common error |
|---|---|---|---|---|
| Images | Trace captions and tensor shapes. | Shorten the requested description. | Compare another checkpoint with separate evidence. | Treating fluent details as visible facts. |
| Chart/invoice | Check three fields and compute source totals. | Change one source value. | Evaluate all six original questions. | Summing misread values correctly. |
| Images/editing | Inspect archived prompt and strength strips. | Predict a strength effect before revealing it. | Execute SD-Turbo, edits and CLIP. | Changing seed and strength together. |
| Speech | Run word-edit examples and the semantic probe. | Substitute one reference word. | Execute MMS-TTS and Whisper, then listen. | Hardcoding a sampling rate. |
| Provenance | Resave honestly labelled fixture metadata. | Edit one unsigned field. | Label a real extension-generated image. | Calling an editable label C2PA or proof. |

## Five self-checks with answers

1. **What enters and leaves `ask`?** Image pixels and question token IDs enter; the continuation is decoded into answer text. Exact tensor shapes depend on the processor.
2. **Does a correct EUR 215 sum fix an extracted EUR 200 instead of EUR 20?** No. The arithmetic is correct for wrong inputs; compare fields with the image first.
3. **Why inspect absent-object replies and full answers?** They expose unsupported claims and yes bias; the substring/first-number heuristic can accept a misleading reply.
4. **Does high CLIP similarity or zero WER establish complete quality?** No. Global image-text alignment and recovered words are partial measures; count objects, inspect defects and listen separately.
5. **What does C2PA establish that unsigned PNG fields do not?** A signed provenance assertion can support verified origin/edit history and integrity. It does not prove that the depicted event occurred, and credentials can be unavailable after resharing.

## FAQ: choosing and checking a media route

**Does a multimodal model necessarily generate pictures?** No. Many VLMs generate text from images and questions. Name the accepted inputs and generated outputs for the exact checkpoint.

**Should I increase resolution until every invoice answer is correct?** Larger images may preserve text, but also cost computation and may still fail. Check the processor's resize/tiling rules and measure representative fields.

**Can the model judge whether its own answer is correct?** A second answer is another model output. Known values, independent calculation and source review supply stronger evidence for this fixture.

**Do these two probes reproduce POPE?** No. They are a classroom demonstration of present/absent checking. Only a complete leading `yes` or `no` word is scored; uncertainty and words such as `Nobody` are non-answers. The full benchmark uses its specified data and procedure.

**Why use an API for a calculation already supplied locally?** The optional comparison examines wording and the separation of extraction from arithmetic. It is an extension, with account costs and data handling considered explicitly.

![Checked multimodal application route](fig:lecture_flow_41)

## TODOs

* **TODO 1** `score(answer, gold)`: compare the first signed parsed number for numeric references, otherwise match a lower-cased substring. `check_score()` covers decimal equivalence, positive and negative signs, comma removal, text matching and mismatches. The heuristic remains deliberately limited.
* **TODO 2** implement `make_edits` using the supplied image-to-image pipeline at strengths 0.3, 0.6 and 0.9 with four steps, zero guidance and the same source/prompt/seed. `check_edits()` checks exact calls and the corresponding returned images with a lightweight probe; real generation remains optional.
* **TODO 3** implement `synthesise_and_transcribe`: tokenise the sentence, obtain `.waveform[0]`, preserve its configured sampling rate, pass both to ASR and return stripped text. `check_speech()` checks data flow with probes; `check_wer()` checks real word edits.
* **TODO 4** add honest `ai_generated`, `model` and `prompt` fields to `PngInfo`, save and reload. `check_metadata()` checks the selected source labels and their loss on JPEG conversion. Core fixture and real generated-image labels differ intentionally.

## Archived measured results (instructor run; unchanged asset bytes)

{{RESULTS_NOTES}}

## Model answers to the written questions

1. **VLM accuracy on charts and documents.** Judge the current core from its own full replies and references. The archived Qwen3-VL experiment shows that correct field reading can coexist with an aggregation error; another checkpoint needs its own evaluation and may fail different questions. Models can also answer "yes" to absent objects or unmentioned fields. Failures cluster into reading/OCR errors, arithmetic, and hallucination under leading questions. Before processing invoices: extract fields into a validated schema and compute totals in code, cross-check against printed totals, use higher resolution or a document-specialised model, measure field-level accuracy on a labelled set of real invoices, and keep human review for low-confidence or high-value cases.
2. **TTS → ASR round trip.** It measures intelligibility as judged by one ASR model. It does not measure naturalness, prosody, emotion, speaker similarity, pronunciation of names, or latency, and the ASR can "correct" expected words. Proper evaluation adds human listening tests (MOS, intelligibility), hard content (numbers, dates, names, accents, languages), latency measurements, and consent and disclosure checks for synthetic voices.
3. **Why metadata is insufficient.** Ordinary re-saving can strip a simple metadata label. C2PA signs provenance, while watermarks embed signals; each has limits. Neither proves the depicted event is true. Article 50 distinguishes provider machine-marking from deployer disclosure, with exceptions and a limited pre-August-system marking transition described above. Use several layers and check what survives resharing.

## Troubleshooting

* *Out of memory on Colab:* restart the runtime and run parts separately, or delete the VLM (`del vlm; torch.cuda.empty_cache()`) before Part 3.
* *Slow on CPU:* confirm the SmolVLM checkpoint, image cap and short answer cap. Keep `RUN_STRETCH` disabled for the core. The heavier models have separate download and inference costs; no current extension runtime is promised.
* *Different answers from the slides:* generation is deterministic with `do_sample=False` and fixed seeds, but GPU type and library versions can change results slightly.
* *Audio does not play in the notebook:* the WAV file `tts_example.wav` is saved in the working directory; download and play it locally.

# Practice questions with model answers

## Multiple choice

1. In a modular (LLaVA-style) MLLM, the projector: **(a)** generates images from the language model's output; **(b)** compresses audio into tokens for the language model; **(c)** computes a CLIP score between the image and the answer; **(d)** maps vision-encoder features into the LLM's embedding space. *Answer: (d).*
2. A 336 × 336 image with 14 × 14 patches produces how many visual tokens? **(a)** 24; **(b)** 196; **(c)** 576; **(d)** 1,024. *Answer: (c).*
3. POPE measures: **(a)** object hallucination with yes/no questions; **(b)** how realistic generated images look; **(c)** the naturalness of synthesised speech; **(d)** frame-to-frame consistency of generated video. *Answer: (a).*
4. In image-to-image generation, increasing strength from 0.3 to 0.9: **(a)** sharpens the output image while keeping its original composition; **(b)** keeps more of the original image's structure and colours; **(c)** adds more starting noise, so the output can move further from the original; **(d)** increases the guidance scale applied at each step. *Answer: (c).*

## Short answer

1. **Compare modular and natively multimodal LLMs.** (4 marks) *Model answer:* modular models join pretrained encoders to an LLM through a projector; cheap to build, usually text output, common in open models. Native models are trained on interleaved multimodal tokens from the start; they can take and produce several modalities (e.g. real-time voice) but need far more training compute.
2. **Describe the stages of training a vision-language model.** (4 marks) *Model answer:* pre-train encoder and LLM separately; alignment pre-training of the projector on image–caption pairs; visual instruction tuning on image conversations, charts and documents; preference tuning/RL to reduce hallucination; optional LoRA adaptation to a domain.
3. **Propose metrics to evaluate a text-to-speech system for a bank's phone assistant.** (4 marks) *Model answer:* WER via ASR for intelligibility, including numbers, dates and names; human MOS for naturalness; latency; evaluation across accents and languages of customers; a check that the voice is disclosed as synthetic and not cloned without consent.
4. **Explain why metadata-based labelling of AI images is fragile and what else should be used.** (3 marks) *Model answer:* re-saving, screenshots and platform uploads strip metadata; combine C2PA credentials, invisible watermarks, visible labels and platform policies.

## Exam-style question

**"A city council wants to use generative AI to (a) read scanned planning applications and extract key fields, and (b) generate illustrative images of proposed buildings for public consultation. Design the system, the evaluation, and the safeguards."** (20 marks)

*Marking guide:* choice of models and pipeline, including the extract-then-verify pattern and resolution considerations (5); evaluation: field-level accuracy on labelled applications, hallucination checks, human evaluation of images for fidelity (5); responsible AI: personal data in applications (GDPR), disclosure and provenance of generated images (Art. 50, C2PA/watermarks), risk that images mislead the public, accessibility (6); human oversight and monitoring (4).

# Glossary

| Term | Meaning |
|---|---|
| Multimodal LLM (MLLM) | LLM that accepts (and possibly generates) images, audio, video or documents |
| Vision-language model (VLM) | Model processing images and text jointly |
| Natively multimodal ("omni") | Model trained from the start on interleaved multimodal tokens |
| Projector | Small network mapping encoder features into the LLM's embedding space |
| Visual tokens | Image patches passed to the LLM as embeddings |
| Visual instruction tuning | Fine-tuning a VLM on image-based conversations and tasks |
| Visual question answering (VQA) | Answering questions about an image |
| Image-to-image (SDEdit) | Editing by noising an image partially and denoising with a new prompt |
| Strength | Fraction of the noise schedule applied in image-to-image |
| Inpainting / outpainting | Regenerating a masked region / extending the canvas |
| ControlNet | Adapter that conditions diffusion on edges, depth, pose, etc. |
| Text-to-speech (TTS) | Generating speech audio from text |
| Voice cloning | TTS imitating a specific speaker from sample audio |
| CLIP score | 100 × cosine similarity of CLIP image and text embeddings |
| POPE / CHAIR | Benchmarks for object hallucination in MLLMs |
| Mean Opinion Score (MOS) | Average human rating (e.g. 1–5) of speech naturalness |
| Deepfake | Realistic synthetic media depicting real people or events falsely |
| C2PA | Standard for cryptographically signed content credentials |
| Watermark (SynthID) | Invisible signal embedded in generated content for detection |
| Liar's dividend | Dismissing genuine evidence as fake because fakes exist |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 13, "Multimodal Models"; chapter 11, "Music Generation".

**Papers**

* Liu et al. (2023) [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485); Li et al. (2023) [Evaluating Object Hallucination in Large Vision-Language Models (POPE)](https://arxiv.org/abs/2305.10355); Meng et al. (2021) [SDEdit](https://arxiv.org/abs/2108.01073); Sauer et al. (2023) [Adversarial Diffusion Distillation (SD-Turbo)](https://arxiv.org/abs/2311.17042)

**Courses, guides and explainers**

* [MIT 6.S191 2026: Generative AI lecture](https://www.youtube.com/watch?v=R8V8CbuxryI)
* [SmolVLM-256M model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
* [CLIP original paper](https://arxiv.org/abs/2103.00020)
* [C2PA 2.3 specification](https://spec.c2pa.org/specifications/specifications/2.3/index.html)

* [Hugging Face blog: Vision language models explained](https://huggingface.co/blog/vlms)
* [Google Gemini API: Image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
* [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index)
* [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction)

**Provenance and law**

* [C2PA: content credentials](https://c2pa.org/) · [Google DeepMind: SynthID](https://deepmind.google/models/synthid/)
* [EU AI Act, Article 50](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50) · [European Commission: transparency obligations FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)

# Link to the group project

Apply perceive, verify, act, review and label. Report task accuracy, hallucination checks and human evaluation on your data. Record licences and consent for identifiable people. Label generated media visibly and with embedded metadata or credentials.

# Detailed explanations behind the short slides

The expanded explanations below retain the full topic coverage. References to measured diffusion, editing, speech and Qwen3-VL results describe archived experiments; the current core/extension split is defined in the lab guide above.

The slides show one idea at a time. Use the detail below for follow-up questions, technical terms and further study.

## Slide 1: Title

Welcome to week 10, multimodal generative AI applications. Last week we learned how models align modalities in a shared space. This week we look at the systems built on those foundations: multimodal LLMs that read photos, documents, charts and audio and answer in text or speech; generators that turn text into images, speech, music and video; and how to evaluate them and use them responsibly. Everything in the lab uses open models, and the slides show real outputs produced for the lecture, including where the models fail.

## Slide 2: Outcomes

Six outcomes matching the descriptor's week 10: multimodal LLMs; training and fine-tuning multimodal models; evaluation of multimodal generative AI; and applications, extended with generation across modalities and responsible use, which the revised 7.3 adds explicitly because deepfakes and content provenance are now regulated. The practical outcome is designing an application that combines a multimodal model with verification and labelling, which feeds directly into projects that process images, documents or speech. This week practises MIMLO 2, 3 and 4.

## Lecture plan

The plan follows the revised descriptor. Sections one and three contain the real outputs from the lab run: captions, chart and document question answering, text-to-image, image editing and speech round trips. Take the break after section three. Section five covers deepfakes, provenance and applications, and ends with a design pattern that students can reuse in their projects.

## Slide 4: Warm-up

Two minutes in pairs. Students usually identify reading errors first, then arithmetic errors, and some notice privacy: receipts can contain partial card numbers, locations and times. Add a fourth risk: the model may confidently report numbers that are not on the receipt, which is multimodal hallucination. This warm-up maps onto the lecture: section one shows that current models read and reason well but not perfectly; section four measures hallucination; section five covers privacy. The design lesson, extract then verify with code, appears at the end.

- You photograph a restaurant receipt and ask a chat assistant to split the bill between four people. What could go wrong?

- Reading: blurry numbers, unusual layout, handwriting, currency symbols.

- Reasoning: adding line items, tips, rounding, who ordered what.

- Privacy: what else is on the receipt (card digits, location, names)?

## Slide 5: Multimodal LLMs

Section one covers multimodal large language models, also called vision-language models or MLLMs. We compare the two main designs, list their capabilities, survey the model families, and look at real outputs from open models on photos, a chart and a document, including errors. About twenty-five minutes.

- How do chat models see, read and listen?

## Slide 6: Two designs: modular bridge vs. natively multimodal

There are two broad designs. Modular models connect separately pretrained encoders to a language model through a projector, as in LLaVA from last week; they are cheap to build and common among open models, and usually output text. Natively multimodal, sometimes called omni, models are trained from the start on interleaved sequences of text, image and audio tokens, so they can process and often generate several modalities in one model, which enables low-latency voice conversations and reasoning over audio and video. Several frontier model families, such as GPT and Gemini, and open families such as Qwen's Omni models, follow this direction.

- **Modular:** an image or audio encoder sends features through a bridge into an LLM; LLaVA is an example.

- **Native multimodal:** training mixes data types within one system. Supported inputs and outputs depend on the model.

- Follow the actual path from input to output; "multimodal" does not mean every model supports every data type.

## Slide 7: What multimodal LLMs can do

Multimodal LLMs have a wide range of capabilities. Describing and answering questions about images supports accessibility, for example generating alt text. Document understanding, reading invoices, forms and PDFs, is one of the most valuable business uses. Chart and table reading supports analytics. Video understanding samples frames and sometimes audio. Speech models understand and produce voice. Screen understanding lets agents operate graphical interfaces, previewed in week 12. For every capability, accuracy varies with image quality, resolution and domain, so evaluation is essential.

- **Describe & answer:** Describe a photo, compare images or answer a question about visible objects.

- **Documents & OCR:** **OCR** means recognising printed or written text. Extract document fields, then validate them.

- **Charts & tables:** Read a plotted value or table entry. Check the axes, units and source numbers.

- **Video:** Sample frames and audio to describe a clip. Events between sampled frames may be missed.

- **Speech:** Recognise spoken questions and, with a speech generator, reply aloud.

- **Screens (GUI):** **GUI** means graphical user interface. A screen-reading agent can propose interface actions.

## Slide 8: Multimodal model families (examples, 2026)

A snapshot of families, without version numbers, which change monthly. The proprietary families from OpenAI, Google and Anthropic all accept images and documents; OpenAI and Google also handle audio natively, and Gemini is notable for very long video and document contexts. Among open-weight models, the Qwen vision-language and omni families are strong on documents and charts; LLaVA is the research reference; small models such as SmolVLM run on laptops. The current CPU core uses SmolVLM-256M-Instruct. Archived Qwen3-VL-2B GPU results remain a separately labelled comparison.

| Family | Access | Inputs | Notes |
| --- | --- | --- | --- |
| GPT (OpenAI) | Proprietary | Text, images, audio (and more) | Native voice and image generation in the app |
| Gemini (Google) | Proprietary | Text, images, audio, video, long PDFs | Natively multimodal; long video context |
| Claude (Anthropic) | Proprietary | Text, images, PDFs | Strong document and chart reading |
| Qwen-VL / Qwen-Omni | Open-weight | Images, video; audio in Omni | Small to large sizes; strong on documents |
| LLaVA | Open research | Images | The reference modular design |
| SmolVLM, Gemma, Llama vision | Open-weight | Images | Small models for laptops and devices |

Examples describe model families, not a guaranteed feature list. Check the exact model, endpoint, licence and current documentation.

## Slide 9: Real captions from an open VLM

These are the two sample photographs from scikit-learn's standard image set, a temple in China and a flower, captioned by the open vision-language model in the lab run; the captions are quoted in the bullet. Generally such captions are accurate at the level of objects and scenes, but can add details that are not there, such as a time of day or a location. For accessibility uses such as alt text, a human should check captions, or at least the model should be asked to describe only what is clearly visible.

- {{CAPTIONS}}

- Captions are fluent; check them for invented details before using them as alt text.

## Slide 10: How images become tokens, and why resolution matters

Connect to weeks 5 and 9. A vision encoder cuts the image into patches; a three-hundred-and-thirty-six-pixel image with fourteen-pixel patches gives twenty-four by twenty-four, which is 576 visual tokens, each passed to the LLM. Reading small text in documents requires high resolution, so models tile large images or process them at their native resolution, which multiplies the token count. Video is sampled at a few frames per second, so long videos consume enormous numbers of tokens. The practical consequence is cost: providers price images and video in tokens, so processing thousands of scanned pages needs a budget estimate.

- A 336 × 336 image with 14 × 14 patches gives (336/14)² = 24² = 576 patch tokens in this example.

- More pixels can preserve small text, but usually increase computation and cost.

- Some models split a large image into **tiles** or choose its resolution dynamically.

- Video uses patches from many frames; sampling fewer frames may miss an event.

- Billing depends on the provider and endpoint. Estimate a small sample before processing a large collection.

## Slide 11: Cross-modal reasoning over a chart and a document (real)

This is the lab's evaluation design: we generated a bar chart and an invoice-style document, so the correct answers are known exactly: which day had the most tickets, a single value, a total that requires adding five numbers, the invoice number, the total due and the customer name. The bullet reports the measured accuracy of the open VLM. Typically single values and text fields are read correctly, while aggregation, adding up the bars, is less reliable because the model estimates rather than computes. This is the evidence behind the design rule on the next slide.

- {{QA_NOTES}}

- The chart and invoice are generated from known values, giving us reference answers to check.

## Slide 12: Quiz

**Question.** How should 200 invoice line items be totalled?

- **A.** Ask the model once for the grand total of all items
- **B.** Increase the temperature so the model reads more carefully
- **C.** Extract checked fields, sum in code, compare printed totals
- **D.** Trust the total if the model states high confidence

Answer: C. Separate perception from computation: use the model for what it is good at, reading fields into a schema validated in code, as in week 7, then compute totals with code and cross-check them against the totals printed on the invoices to detect reading errors automatically. A relies on the model's arithmetic across many images, which is error-prone and unverifiable. B adds randomness. D is unreliable: models' self-reported confidence is poorly calibrated. Add human review for invoices where the checks disagree.

## Slide 13: Training and fine-tuning multimodal models

Section two briefly explains how multimodal LLMs are trained, which the descriptor lists as training and fine-tuning multimodal models, and how you can adapt one to your own domain with the parameter-efficient methods from week 8. Keep it brief: students need the pipeline and the vocabulary, not the details of each stage. About thirteen minutes.

- How are these models built, and how can you adapt one?

## Slide 14: A typical training pipeline for a vision-language model

Walk through the stages. The vision encoder, typically CLIP or SigLIP, and the LLM are pretrained separately at great cost. Alignment trains the projector on image-caption pairs so that visual features land in the LLM's input space. Visual instruction tuning, introduced by LLaVA, fine-tunes on conversations about images, now including charts, documents, OCR and grounding tasks, often generated synthetically with the help of stronger models. Preference tuning or reinforcement learning, as in week 8, reduces hallucination and improves helpfulness. Finally, users can adapt the model to their domain with LoRA.

- Begin with an existing image encoder and language model; train the connection between them.

- **Visual instruction tuning** uses image questions with desired answers: captions, text reading, charts and documents.

- Costs depend on which components are trained, the data size and the image resolution.

## Slide 15: Data for multimodal training

Multimodal training data comes in several forms. Web image-caption pairs are plentiful but noisy, so many teams re-caption images with a strong model to get more detailed descriptions. Interleaved documents teach models to handle mixed sequences. Task-specific data, for OCR, charts, screenshots and video, is often generated synthetically, for example by rendering charts with known values, exactly what we do in the lab for evaluation. Audio data pairs speech with transcripts. Last week's LAION case showed why curation, licensing and consent are critical for images and voices.

- **Image-caption pairs:** an image beside a description of its content.

- **Interleaved documents:** text and images kept in their original order, as on a web page.

- **Task examples:** image questions and answers, document fields, charts, screens and marked object locations.

- **Audio examples:** recordings with transcripts or sound descriptions.

- Review labels, remove duplicates, and check permission, licensing and consent.

## Slide 16: Adapting a multimodal model to your domain

The week 8 approach carries over directly. For a domain such as a company's own forms, industrial inspection images or satellite imagery, you collect image-question-answer examples, freeze the vision encoder and apply LoRA to the language model and possibly the projector, using libraries such as TRL, which supports vision-language fine-tuning. The same decision rule applies: start with prompting and structured extraction; fine-tune when evaluation on your own images shows persistent reading or format errors. In regulated domains such as medical imaging, fine-tuned models need clinical validation and oversight.

- Start with a specific task, such as reading a company form or identifying a product defect.

- Try a clear prompt and a checked extraction format first.

- Further training needs labelled image-question-answer examples and a separate test using your image types.

- A common setup keeps the vision encoder fixed and adds LoRA to the LLM; the bridge may also be trained.

## Slide 17: Generation across modalities

Section three surveys generation across modalities, the descriptor's list of text to image, image to image, text to audio, text to video and so on. We look at real text-to-image and image-editing outputs from the lab, a speech round trip, and the state of music and video generation. About twenty-five minutes; take the break afterwards.

- From text to images, speech, music and video

## Slide 18: Generation across modalities: a map

This map summarises the main generation tasks, how they work and example model families. Images use latent diffusion or flow transformers from week 4. Editing uses partial noising, inpainting and control signals. Speech synthesis uses neural text-to-speech, increasingly language models over audio-codec tokens, which also enable voice cloning from short samples. Speech-to-speech models power real-time voice assistants. Music and sound use diffusion or token language models over audio codecs. Video uses spatio-temporal diffusion transformers. The common thread: diffusion or autoregressive models over tokens or latents, conditioned on text encoders.

## Slide 19: Text → image with a distilled model (real)

Three prompts rendered by SD-Turbo with two steps in the lab run. Inspect prompt match and defects in these stored images; no per-image runtime was measured for this claim. The bullet reports the CLIP scores: each image scores higher against its own prompt than the mean score against the other prompts, a simple sanity check of prompt alignment that the lab automates. Remind students that SD-Turbo uses the Stability AI Community License, with conditional commercial use, registration and revenue limits, which matters for projects.

- SD-Turbo is a distilled image generator. This run uses 2 denoising steps and no guidance.

- {{T2I_NOTES}}

- A short generation time does not establish image quality; inspect the requested details.

## Slide 20: Image → image: editing with partial noise (real)

Image-to-image editing, from week 4's SDEdit idea, adds noise to an existing image and denoises it with a new prompt. The strength parameter sets how far into the noise schedule we go: in the recorded run, 0.3 and 0.6 left the lighthouse painting almost unchanged, while 0.9 replaced most of it with a new coastal scene. None of the three looks like a pencil sketch: strength controls how much may change, not whether the style request is followed, so compare each result with what was asked. Inpainting restricts the change to a mask, and control methods such as ControlNet guide the preservation of edges or pose. These are the building blocks of editing features in design tools.

- **Editing strength** controls how much noise is added to the starting image.

- Low strength tends to preserve more of the input; high strength allows larger changes.

- **Inpainting** changes a marked region. Style transfer changes the visual appearance.

## Slide 21: Editing, control and personalisation

Beyond simple image-to-image, four editing families matter in practice. Inpainting and outpainting regenerate or extend regions. Control methods follow structural guides such as edges, depth maps or human pose. Instruction-based editing lets users describe changes in words, and natively multimodal models now support conversational editing. Personalisation with LoRA or DreamBooth teaches a model a specific person, product or artistic style from a handful of images, which is powerful for product photography and dangerous for impersonation and style imitation.

- **Inpainting:** edit a marked area. **Outpainting:** generate content beyond the original image boundary.

- **ControlNet/adapters:** supply an extra guide, such as edges, pose or depth.

- **Instruction editing:** describe a change in words, such as "make the scene snowy".

- **Personalisation:** train a subject or style adapter, for example with LoRA or DreamBooth. Check permission to use a person's likeness.

## Slide 22: Speech: text-to-speech, cloning and speech-to-speech

Speech generation has changed quickly. Classic neural text-to-speech converts text into acoustic features and then a waveform with a vocoder; VITS-based models such as Meta's MMS, used in the lab, do this end to end. Newer systems treat audio as codec tokens and use language models, enabling voice cloning from a few seconds of speech, valuable for accessibility and dubbing, but also used in fraud. Speech-to-speech assistants respond aloud; delay depends on the model and application. The bullet reports our lab's round trip: synthesise a sentence, transcribe it with Whisper, and measure word error rate as a cheap intelligibility check.

- **Text-to-speech (TTS)** turns written words into sound. An acoustic model predicts sound features; a **vocoder** converts them to a waveform.

- Other speech generators predict **audio-codec tokens**: IDs representing short sound segments.

- **Voice cloning** imitates a speaker from examples. Consent and protection against impersonation matter.

- Lab round trip: MMS-TTS generates speech; Whisper transcribes it. {{SPEECH_NOTES}}

## Slide 23: Music, sound and video

Music generation uses the same toolkit: Meta's MusicGen is a language model over audio-codec tokens; Stable Audio is a latent diffusion model. The music industry has sued several music-generation companies over training data, a live copyright issue. Text-to-video models such as OpenAI's Sora and Google's Veo families extend diffusion transformers to space and time, and since 2025 some also generate synchronised sound. The open problems are temporal consistency, realistic physics, long durations and cost. Realistic video with sound makes deepfakes far more convincing, which is why provenance matters.

- Music and sound generators can use diffusion or models predicting audio tokens, such as MusicGen.

- Video generators model image patches across both space and time; some also generate sound.

- Watch for changing faces or objects, implausible motion, long-run drift and high computation cost.

- Rights over training music and films, and deceptive media uses, remain important concerns.

## Slide 24: Quiz

**Question.** In image-to-image editing, what does increasing the 'strength' from 0.3 to 0.9 do?

- **A.** Sharper output with the same composition
- **B.** Identical output, but more memory is used
- **C.** A higher guidance scale for the prompt
- **D.** More starting noise; potentially larger changes

Answer: D. Strength controls how far along the noise schedule the original image is pushed before the model denoises it with the new prompt. Low strength keeps structure and colours; high strength keeps little of the original. It is related to the number of effective denoising steps, which is why the lab multiplies steps by strength. A, B and C confuse strength with other settings. This is the same forward process from week 4, applied to an existing image instead of pure noise.

## Slide 25: Evaluating multimodal generative AI

Welcome back. Section four is about evaluation, which the descriptor lists explicitly for multimodal models. We summarise metrics by task, focus on multimodal hallucination with a real probe from the lab, and discuss how to design human evaluation for images and audio. About seventeen minutes.

- How do you measure quality, alignment and truthfulness across modalities?

## Slide 26: Metrics by task

This table maps each task to its usual metrics. Images reuse FID and CLIP score from week 4, plus compositional benchmarks. Captions use CIDEr and CLIPScore. Question answering uses accuracy, which requires ground truth, which is why generating our own chart and invoice was useful. Speech uses word error rate through a recogniser and human Mean Opinion Scores for naturalness. Video uses Fréchet Video Distance and human rating. For multimodal LLMs, hallucination has its own benchmarks: POPE asks yes/no questions about objects, and CHAIR counts objects mentioned in captions that are not in the image.

| Task | Automatic metrics | Human evaluation |
| --- | --- | --- |
| Text → image | FID: dataset-level comparison; CLIP score: text-image similarity; GenEval: object/layout checks | Prompt match, visual errors, style |
| Image → text | CIDEr/BLEU: caption overlap; CLIPScore: image-caption similarity | Correct facts and omitted/invented details |
| Image questions | Accuracy against known answers | Separate reading, reasoning and calculation errors |
| Text → speech | WER after transcription; speaker similarity | Naturalness and how easily speech is understood |
| Text → video | FVD: dataset-level comparison; CLIP-based scores | Motion, continuity, physics, prompt match |
| Invented visual claims | POPE: object yes/no tests; CHAIR: invented caption objects | Check each claim against the image |

## Slide 27: Multimodal hallucination

Hallucination in multimodal models takes specific forms: naming objects that are absent, reading numbers wrongly, inventing text in documents, and answering yes to leading questions. POPE, from Li and colleagues in 2023, measures object hallucination with balanced yes/no questions and reveals a tendency to say yes. Our lab probe asked about present and absent objects in the photos, chart and invoice; the measured result is in the bullet. Mitigations include higher resolution, instructions to rely only on visible content, extraction with verification as in the quiz, and training methods that penalise hallucination.

- **Multimodal hallucination** means claiming details the input does not support: an absent object, wrong number or invented word.

- **POPE** asks about both present and absent objects. Check errors and whether the model tends to answer "yes".

- **Real lab probe.** {{POPE_NOTES}}

- Use sufficient resolution, checked extraction and verification. A prompt asking for visible facts alone is not a guarantee.

## Slide 28: Designing human evaluation for images and audio

Automated metrics for images and audio are only partial, so human evaluation is usually required, and week 7's principles apply. Pairwise comparisons are more reliable than absolute ratings, and randomising order prevents position bias. Rubrics make criteria explicit. Rater diversity and content diversity matter particularly for multimodal outputs: image generators and speech recognisers have documented disparities across skin tones, accents and languages. Report inter-rater agreement. The project's evaluation of any multimodal component must include such a small human study.

- Show answers or media side by side; randomise which is first.

- A **rubric** is a scoring guide: prompt match, visual defects and realism; for speech, naturalness and clarity.

- Include varied images, skin tones, accents, languages and raters.

- Report how often reviewers agree. Use automatic scores alongside this review.

## Slide 29: Responsible use, applications and trends

The final section covers the harms of realistic synthetic media, the layered approach to provenance and labelling, consent and rights, the most valuable applications, emerging trends for 2026 and 2027, a design pattern for multimodal applications, and a responsible-use checklist. Leave time for the labelling discussion. About twenty-two minutes.

- What can go wrong, how do we label synthetic media, and where is this useful?

## Slide 30: Harms of realistic synthetic media

These harms are real and current. In early 2024 an employee of the engineering firm Arup in Hong Kong transferred about twenty-five million US dollars after a video call in which the other participants were deepfakes. Non-consensual sexual deepfakes, including of minors, are one of the most damaging uses, and many jurisdictions have criminalised them. Synthetic media spreads disinformation, and the existence of fakes lets people dismiss genuine evidence, the liar's dividend. Scammers use synthetic personas. Students building multimodal applications should consider these misuse scenarios in their risk assessment.

- **Fraud and impersonation:** A fake face or voice can impersonate someone. A reported 2024 video-call scam cost a firm about US$25 million.

- **Non-consensual imagery:** A fabricated sexual image can severely harm the person depicted, especially when made or shared without consent.

- **Disinformation:** Fake media can mislead viewers; it can also make people dismiss real evidence as fake (the "liar's dividend").

- **Harassment and scams:** Synthetic identities and targeted fake content can support harassment and financial scams.

## Slide 31: Provenance and labelling: no single layer is enough

Provenance works in layers, each with weaknesses. Visible labels are simple but easy to remove. C2PA content credentials, backed by major technology and media companies, cryptographically sign the history of an asset; some cameras and platforms support them, but credentials can be stripped. Invisible watermarks, such as Google DeepMind's SynthID for images, audio, video and text, survive common edits but exist only if the generator adds them. Detectors predict whether media is synthetic but are in an arms race. The EU AI Act's Article 50 now requires machine-readable marking and disclosure of deepfakes. The lab shows how quickly simple metadata disappears.


Article 50 applies from 2 August 2026; provider marking and deployer disclosure have different scope and exceptions. For systems placed on the market before that date, the transition ends 2 December 2026 only for Article 50(2). [Official scope](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50), [Commission timing guidance](https://digital-strategy.ec.europa.eu/en/factpages/quick-facts-transparency-rules-ai-systems).

- **Provenance** records where media came from and how it was changed. It does not prove the scene is true.

- Use several layers: visible labels, **C2PA** signed content credentials, watermarks and detection. Each can fail.

- EU AI Act Article 50 separates provider marking duties from deployer disclosure duties; details and exceptions are in the notes.

## Slide 32: Discuss: where should the label go?

Three minutes in pairs, then take two or three views. There is no single right answer, which is the point: labelling rules depend on context and harm. Most students agree that a family photo needs no visible label, but a news photograph or an insurance claim does, because people rely on it as evidence. Note that the EU AI Act's deepfake disclosure focuses on content that falsely appears authentic, with lighter duties for evidently artistic or satirical work, and that C2PA credentials can record edits without a visible label, leaving the decision to the platform showing the image.

- A phone's camera app uses generative AI to remove a stranger from a holiday photo and brighten the sky. Should the photo carry an 'AI-edited' label?

- Where is the line between editing, enhancement and generation?

- Who is harmed if it is unlabelled: a family album vs. a news photo vs. an insurance claim?

- Which layer would you use: visible label, C2PA credential, watermark?

## Slide 33: Consent, likeness and rights

Rights issues are central to multimodal work. Faces and voices identify people, so using them requires consent, and laws protecting voice and likeness are spreading. Training on and imitating living artists' work is ethically and legally contested. Music and film rights holders have brought lawsuits against generative audio and video companies. Under GDPR, images of people and scanned documents are personal data, so processing them through third-party APIs needs a lawful basis and data-protection checks. These points go into the project's responsible-AI section.

- Get explicit permission before imitating a real person's face or voice in the project. Check applicable rights.

- Training data and imitation of artists' work raise copyright and licensing questions.

- Music and film rights holders have challenged generative-media training in court.

- Faces, voices, locations and document details may be personal data. Use a lawful basis and minimise what you collect.

## Slide 34: Valuable applications

Multimodal AI has many beneficial applications. Accessibility is perhaps the clearest: image descriptions and captions help blind and deaf users. Document processing, reading invoices, forms and contracts, is a major enterprise use. Voice assistants handle customer service in many languages. In education, models explain diagrams and give feedback on handwritten work, with teacher review. Design and marketing use image generation for concepts and localisation. Industrial inspection and robotics use vision-language models, including vision-language-action models that output robot actions. Each of these needs the evaluation and safeguards covered today.

- **Accessibility:** Alt text, live captions, audio description, reading documents aloud.

- **Document processing:** Invoices, forms, contracts, receipts: extract, validate, route.

- **Voice assistants:** Customer service and booking by phone; multilingual support.

- **Education:** Explaining diagrams, marking handwritten work with review, language practice.

- **Design and marketing:** Concept art, product mock-ups, localised visuals, with labelling.

- **Robotics and inspection:** Visual inspection, vision-language-action models for robots.

## Slide 35: Emerging trends (2026–27)

Looking ahead to the period when students graduate: models are converging on any-to-any designs; voice and video agents are becoming real time; video generation is adding sound, length and control; agents that see and operate computer screens are moving into products, which connects to week 12; world models that simulate environments are being developed for robotics and games; and small multimodal models run on phones. Remind students to verify any product claims when they encounter them: this area changes monthly.

- **Any-to-any systems:** combine several supported input and output types; verify the exact model's capabilities.

- **Live media assistants:** process speech or video while the user interacts.

- **Video with sound:** research seeks longer clips and better control over motion.

- **Computer-use agents:** read screens and propose actions (week 12).

- **World models:** learn how scenes change; **on-device models** run on the user's hardware. These are active research directions.

## Slide 36: A design pattern for multimodal applications

This design pattern summarises the lecture and suits most projects. Check input quality and consent. Use a multimodal model to perceive and extract content into a structured schema. Verify with code: totals, formats, cross-references and business rules. Then act, answer or generate. Finally, route risky or uncertain cases to human review and label any generated media with provenance. The callout is the slogan: models perceive, code computes and checks, people decide when the stakes are high.

- **Input:** image, document, audio or video; check quality and consent.

- **Perceive:** a VLM or speech recogniser extracts the content into a schema.

- **Verify:** code checks totals, formats and cross-references.

- **Use output:** answer, route the case or generate media.

- **Review / label:** human review for risky cases; provenance labels on outputs.

- Use the model to read input, code to calculate and check, and a person to review consequential decisions.

## Slide 37: Responsible-use checklist for multimodal projects

This adapts the five-step checklist from week 1 to multimodal work. Purpose includes asking whether people shown or heard could be harmed. Consent and data protection are central because faces, voices and documents are personal data. Verification must include hallucination and group-wise checks. Labelling synthetic media is now a legal duty in the EU. Recording licences matters because several popular multimodal models, including two in today's lab, restrict commercial use.

- **Purpose:** state the task and the people an error could affect.

- **Data:** obtain permission and an appropriate lawful basis; collect only needed images, voices and documents.

- **Check:** test reading errors and invented details across representative inputs and groups.

- **Label:** disclose generated or materially edited media; use provenance tools where useful.

- **Record:** save the model version, settings, scores and permitted uses under its licence.

## Slide 38: Lab 10: multimodal generative tasks, measured

The lab covers the descriptor's list of multimodal generative tasks: image to text, cross-modal reasoning, text to image, image to image, text to audio and speech to text, with text to video discussed rather than run because of cost. Every part has a measurement, so students practise evaluation as well as generation. The provenance exercise makes the fragility of metadata concrete. Remind students of the licence restrictions of SD-Turbo (conditional Community License) and MMS-TTS (non-commercial).

- Use a **VLM** to describe photos and answer image questions.

- Check answers about a chart and invoice against known values; test present and absent objects.

- Generate with SD-Turbo; edit an image at several strengths.

- Compare the generated images with the prompt using CLIP similarity and visual review.

- Generate speech with MMS-TTS; transcribe with Whisper and calculate WER.

- Add illustrative metadata to an image and observe what survives re-saving; this is not signed C2PA.

- **Current runtime:** {{W10_CORE_TIME}} for the cached local CPU core on {{W10_CORE_DATE}}, with {{W10_CORE_THREADS}} threads. Downloads and optional stretches are excluded. The earlier full-notebook record has different settings, described in the lab guide.

- **Hand in:** current core evidence and five self-check explanations; original heavy experiments and three written questions remain available as optional extensions.

## Slide 39: Summary

The six cards on the slide condense these eight summary points. Students should be able to explain the two MLLM designs, list generation tasks and their methods, choose evaluation metrics for each, and propose a layered provenance strategy. Next week we move from models to systems: deploying generative AI with frameworks and APIs, local and cloud serving, cost and latency, monitoring and guardrails, which every project will need.

- A multimodal system connects input encoders and output generators; supported media types vary.

- Reading detail depends on resolution and model ability; more tokens usually cost more.

- Alignment and visual instruction examples teach the connection; LoRA can adapt a model.

- Generation includes images, edits, speech, music and video.

- Use known answers, similarity scores, speech error rates and human review together.

- Invented details and realistic fake media can cause harm; protect consent and rights.

- Provenance and labels support checking but do not guarantee truth or legal compliance.

- Trace the input → model → checks → reviewed output before using an application.

## Slide 40: Resources

The Hugging Face blog on vision language models is the best technical overview. Google's image understanding guide shows how a frontier API handles images and documents. The Diffusers documentation covers the generation pipelines used in the lab. The Hugging Face Audio course covers speech synthesis and recognition. For provenance, the C2PA site, Google DeepMind's SynthID page and the text of Article 50 are the primary sources. The LLaVA paper is the key reference for multimodal instruction tuning. Foster chapter 13 and chapter 11 on music generation are the textbook readings. See the source-verification report for the latest checks and access limitations.

- **Reference:** [Hugging Face blog: Vision language models explained](https://huggingface.co/blog/vlms). Architectures, training, evaluation.
- **Reference:** [Google Gemini API: Image understanding](https://ai.google.dev/gemini-api/docs/image-understanding). Official guide to multimodal prompting.
- **Reference:** [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index). Text-to-image, img2img, inpainting.
- **Course:** [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction). TTS and speech models.
- **Reference:** [C2PA: Content credentials](https://c2pa.org/). Provenance standard.
- **Reference:** [Google DeepMind: SynthID](https://deepmind.google/models/synthid/). Watermarking for AI content.
- **Reference:** [EU AI Act, Article 50](https://artificialintelligenceact.eu/article/50/). Transparency obligations.
- **Paper:** [Liu et al. (2023) Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485). The modular MLLM recipe.


**Licence checked 9 October 2026.** The [official SD-Turbo licence](https://huggingface.co/stabilityai/sd-turbo/blob/main/LICENSE.md) permits limited commercial use subject to registration, revenue limits and other terms. It is not a blanket non-commercial-only licence.
