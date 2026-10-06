# Lecture plan at a glance

This week turns last week's multimodal foundations into applications (MIMLO 2, 3, 4). Students should leave able to explain the two designs of multimodal LLMs, describe how such models are trained and adapted, use models for image, speech and cross-modal tasks, choose evaluation methods for each modality (including hallucination probes), and propose a layered provenance and consent strategy for synthetic media. All images and numbers on the "real" slides come from running this week's lab notebook in advance (`curriculum/assets/week_10/`).

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

> **Teaching tip:** Bring a real receipt or open a scanned invoice, and ask a multimodal assistant live to total it. Then ask it to extract the line items as JSON and total them yourself. Students see the "perceive with the model, compute with code" pattern before you name it.

<!-- pagebreak -->

# Lecture notes

## 1. Multimodal LLMs

A **multimodal large language model (MLLM)**, often called a **vision-language model (VLM)** when the extra modality is images, accepts images, documents, audio or video together with text and answers in text (sometimes also in speech or images).

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

**Real captions (lab run):** {{CAPTIONS}}

**Visual tokens and cost.** A vision encoder cuts an image into patches: a 336 × 336 image with 14 × 14 patches gives 24 × 24 = **576 visual tokens**. Reading small text needs high resolution, so models **tile** large images or use **dynamic resolution**, multiplying the token count. Video is sampled at about one frame per second; an hour of video can be hundreds of thousands of tokens. Providers price images and video in tokens, so estimate the cost before processing thousands of files.

**Cross-modal reasoning, measured.** The lab generates a bar chart and an invoice image, so the correct answers are known exactly (the busiest day, a single value, a total that requires adding five bars, the invoice number, the total due, the customer). **Lab result:** {{QA_NOTES}}

![Generated chart and invoice used for evaluation](fig:lab_chart)

> **Key idea:** Let the model **perceive** (extract fields into a schema validated in code, as in week 7) and let **code compute** totals and cross-check them against printed totals. Do not trust a VLM's arithmetic across many images, or its self-reported confidence.

## 2. Training and fine-tuning multimodal models

![A typical VLM training pipeline](fig:vlm_training)

1. **Pre-train the parts** separately: vision encoder (CLIP/SigLIP) and LLM. This is where most compute goes.
2. **Alignment pre-training:** train the projector (sometimes more) on image–caption pairs so visual features land in the LLM's input space.
3. **Visual instruction tuning** (Liu et al., 2023): fine-tune on image-based conversations, now including charts, documents, OCR, screenshots and grounding; much of this data is synthetic, generated with stronger models or rendered with known answers.
4. **Preference tuning / RL** (week 8) to reduce hallucination and improve helpfulness.
5. **Adapt (optional):** freeze the vision encoder and apply **LoRA** to the LLM (and optionally the projector) on your own image–question–answer examples; TRL supports vision-language fine-tuning.

**Training data:** web image–caption pairs (often re-captioned by models), interleaved web documents, task data (OCR, charts, tables, screenshots, grounding boxes, video QA), and speech with transcripts. Curation, deduplication, licensing and consent matter even more than for text (week 9's LAION lesson).

**When to fine-tune:** prompting plus structured extraction first; fine-tune when evaluation on **your own images** shows persistent reading or format errors. Regulated domains (e.g. medical imaging) need clinical validation and human oversight.

## 3. Generation across modalities

![Generation tasks, methods and example families](fig:generation_map)

| Task | How it works | Example families |
|---|---|---|
| Text → image | Latent diffusion or flow transformers conditioned on text (week 4) | Stable Diffusion, FLUX, Imagen, GPT image models |
| Image → image | Add partial noise to an image and denoise with a new prompt; inpainting; control | SDEdit, ControlNet, instruction editors |
| Text → speech | Acoustic model + vocoder (VITS/MMS) or language models over audio-codec tokens | MMS-TTS, Bark, Kokoro, voice cloning systems |
| Speech → speech | Speech LLMs that listen and answer in voice with low latency | Voice modes of assistants |
| Text → music/audio | Diffusion or token LMs over audio codecs | MusicGen, Stable Audio |
| Text → video | Diffusion transformers over spacetime patches, some with synchronised audio | Sora, Veo families, open video models |

**Text → image (real).** SD-Turbo is a distilled model: two steps, no classifier-free guidance, a second or two per 512 × 512 image on a GPU. **Lab result:** {{T2I_NOTES}}

![SD-Turbo text-to-image outputs](fig:lab_t2i)

**Image → image (real).** The **strength** parameter sets how far along the noise schedule the original image is pushed before denoising with the new prompt: 0.3 keeps structure and colours; 0.9 keeps little of the original. Effective steps = steps × strength.

![Image-to-image at three strengths](fig:lab_i2i)

**Editing, control and personalisation.** Inpainting/outpainting (regenerate a masked region or extend the canvas); control (ControlNet and adapters following edges, depth, pose or a reference image); instruction-based editing ("make it winter"), now conversational in natively multimodal models; personalisation with LoRA or DreamBooth on a few images of a subject or style, which raises likeness and style-rights questions.

**Speech.** Neural TTS turns text into acoustic features and a waveform (VITS-based MMS does this end to end). Codec-token language models enable **voice cloning** from seconds of audio: valuable for accessibility and dubbing, but used in fraud. **Speech-to-speech** assistants answer in voice with sub-second latency. **Lab round trip (MMS-TTS → Whisper):** {{SPEECH_NOTES}}

**Music, sound and video.** MusicGen is a language model over audio-codec tokens; Stable Audio is a latent diffusion model. Rights holders have sued several music-generation companies over training data. Text-to-video models extend diffusion transformers to space and time; since 2025 some generate synchronised sound. Open problems: temporal consistency, physics, long durations and cost.

## 4. Evaluating multimodal generative AI

| Task | Automatic metrics | Human evaluation |
|---|---|---|
| Text → image | FID (realism), CLIP score (alignment), GenEval (composition) | Side-by-side human preference; rubric for alignment and artefacts |
| Image → text (captions) | CIDEr, BLEU, CLIPScore | Correctness, completeness, invented details |
| VQA, charts, documents | Accuracy against ground truth | Error analysis by question type |
| Text → speech | WER via ASR, speaker similarity | Mean Opinion Score (naturalness), intelligibility |
| Text → video | FVD, CLIP-based consistency | Motion quality, physics, prompt adherence |
| MLLM hallucination | POPE (yes/no object probes), CHAIR (invented objects in captions) | Audit of invented objects, numbers, text |

**CLIP score** is 100 × the cosine similarity between CLIP's image and text embeddings. A useful sanity check is that each image should score higher against its own prompt than against the other prompts (the lab automates this).

**Multimodal hallucination** means describing absent objects, misreading numbers, inventing text in documents, or agreeing with leading questions. **POPE** (Li et al., 2023) asks balanced "Is there a {object} in the image?" questions about present and absent objects and reveals a **yes-bias**. **Lab probe:** {{POPE_NOTES}} Mitigations: higher resolution; "answer only from what is visible"; extraction with verification; preference training against hallucination.

**Human evaluation for images and audio:** compare side by side (A/B) with randomised order; use a rubric (prompt fidelity, artefacts, realism; for speech, naturalness and intelligibility); include diverse raters and diverse content (skin tones, accents, languages); report inter-rater agreement; treat automatic metrics as a supplement.

## 5. Responsible use, applications and trends

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

**Law.** Article 50 of the EU AI Act applies from **2 August 2026**. Providers of systems that generate synthetic audio, images, video or text must mark outputs in a **machine-readable** way that is detectable as AI-generated; deployers must **disclose deepfakes** and AI-generated text published to inform the public on matters of public interest (lighter duties for evidently artistic, satirical or fictional work). Under the Digital Omnibus changes, systems already on the market before 2 August 2026 have until **2 December 2026** to meet the marking obligation. The European Commission's guidelines and code of practice on transparency give practical detail.

**Consent, likeness and rights.** Use a real person's face or voice only with explicit consent; laws protecting likeness and voice are spreading. Imitating living artists' styles and training on their work without consent is contested. Rights holders have sued generative audio and video companies. Under GDPR, images of people and scanned documents are personal data: processing them through third-party APIs needs a lawful basis and data-protection checks.

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
| "Open models can be used for anything." | Several popular multimodal models restrict commercial use: MMS-TTS is CC BY-NC; SD-Turbo needs Stability AI's licence for commercial use. |

# Responsible AI lens: synthetic media and personal data

* **Consent for likeness and voice:** never generate or clone an identifiable real person without explicit consent; this includes classmates in project demos.
* **Personal data in images and audio:** faces, voices, number plates, documents in the background and location metadata; minimise, blur or anonymise before sending to external APIs.
* **Disclosure duties:** label AI-generated or manipulated media, especially anything that could be mistaken for real evidence (EU AI Act Art. 50).
* **Fairness:** image generators and speech systems show documented disparities across skin tones, genders, accents and languages; evaluate by group.
* **Accessibility is a benefit and a responsibility:** automated alt text and captions must be accurate enough for people who rely on them.
* **Licences:** record the model licence (research-only and non-commercial licences are common) and the terms for generated outputs.

# Lab guide and answers

**Runtime:** Colab T4 GPU (≈ 15 min for all parts) with Qwen3-VL-2B-Instruct, SD-Turbo, CLIP ViT-B/32, MMS-TTS and Whisper base. On CPU the notebook automatically uses SmolVLM-256M-Instruct and runs more slowly. Models are downloaded from the Hugging Face Hub; no API key is needed. **Hand-in:** QA accuracy table, hallucination probe, generated images with CLIP scores, speech WER table, three written answers.

## TODOs

* **TODO 1** `score(answer, gold)`: for numeric gold answers, compare the first number found in the answer; otherwise check that the normalised gold string appears in the answer.
* **TODO 2** `AutoPipelineForImage2Image.from_pipe(t2i)` and call it at strengths 0.3, 0.6, 0.9 with `num_inference_steps=4`, `guidance_scale=0.0`.
* **TODO 3** `tts(**tts_tok(s, return_tensors="pt")).waveform`, then `asr({"raw": wav, "sampling_rate": sr})["text"]`, then WER.
* **TODO 4** add `ai_generated`, `model` and `prompt` text fields to `PngInfo`, save, reload, and observe that re-saving as JPEG drops them.

## Measured results (instructor run)

{{RESULTS_NOTES}}

## Model answers to the written questions

1. **VLM accuracy on charts and documents.** Strong small VLMs usually read single values and text fields correctly but are less reliable at aggregation (summing the chart), and can answer "yes" to absent objects or unmentioned fields; tiny models often read labels but miss numbers. Failures cluster into reading/OCR errors, arithmetic, and hallucination under leading questions. Before processing invoices: extract fields into a validated schema and compute totals in code, cross-check against printed totals, use higher resolution or a document-specialised model, measure field-level accuracy on a labelled set of real invoices, and keep human review for low-confidence or high-value cases.
2. **TTS → ASR round trip.** It measures intelligibility as judged by one ASR model. It does not measure naturalness, prosody, emotion, speaker similarity, pronunciation of names, or latency, and the ASR can "correct" expected words. Proper evaluation adds human listening tests (MOS, intelligibility), hard content (numbers, dates, names, accents, languages), latency measurements, and consent and disclosure checks for synthetic voices.
3. **Why metadata is insufficient.** Metadata is stripped by ordinary processing and can be forged. C2PA signs provenance so tampering is detectable but can still be removed and is not shown by every platform; invisible watermarks (SynthID) survive common edits but only exist if the generator adds them and can be weakened; detectors are an arms race. Article 50 requires machine-readable marking by providers (from 2 August 2026; 2 December 2026 for systems already on the market) and disclosure of deepfakes by deployers. Robust practice combines these layers.

## Troubleshooting

* *Out of memory on Colab:* restart the runtime and run parts separately, or delete the VLM (`del vlm; torch.cuda.empty_cache()`) before Part 3.
* *Slow on CPU:* the notebook switches to SmolVLM-256M; expect lower accuracy. SD-Turbo on CPU takes about a minute per image; reduce to one prompt.
* *Different answers from the slides:* generation is deterministic with `do_sample=False` and fixed seeds, but GPU type and library versions can change results slightly.
* *Audio does not play in the notebook:* the WAV file `tts_example.wav` is saved in the working directory; download and play it locally.

# Practice questions with model answers

## Multiple choice

1. In a modular (LLaVA-style) MLLM, the projector: **(a)** generates images; **(b)** maps vision-encoder features into the LLM's embedding space; **(c)** computes CLIP score; **(d)** compresses audio. *Answer: (b).*
2. A 336 × 336 image with 14 × 14 patches produces how many visual tokens? **(a)** 24; **(b)** 196; **(c)** 576; **(d)** 1,024. *Answer: (c).*
3. POPE measures: **(a)** image realism; **(b)** object hallucination with yes/no questions; **(c)** speech naturalness; **(d)** video consistency. *Answer: (b).*
4. In image-to-image generation, increasing strength from 0.3 to 0.9: **(a)** sharpens the image; **(b)** keeps more of the original; **(c)** adds more noise so the output follows the prompt more and the original less; **(d)** increases guidance. *Answer: (c).*

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

* [Hugging Face blog: Vision language models explained](https://huggingface.co/blog/vlms)
* [Google Gemini API: Image understanding](https://ai.google.dev/gemini-api/docs/image-understanding)
* [Hugging Face Diffusers documentation](https://huggingface.co/docs/diffusers/index)
* [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction)

**Provenance and law**

* [C2PA: content credentials](https://c2pa.org/) · [Google DeepMind: SynthID](https://deepmind.google/models/synthid/)
* [EU AI Act, Article 50](https://artificialintelligenceact.eu/article/50/) · [European Commission: transparency obligations FAQ](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)

# Link to the group project

Apply perceive, verify, act, review and label. Report task accuracy, hallucination checks and human evaluation on your data. Record licences and consent for identifiable people. Label generated media visibly and with embedded metadata or credentials.
