# Lecture plan at a glance

This week opens the multimodal part of the module (MIMLO 2). Students should leave able to define the key terms, explain how each modality becomes tokens or vectors, write and compute the contrastive loss, use CLIP for zero-shot classification and retrieval, compare multimodal architectures, and evaluate speech recognition with WER. Results and several figures come from running this week's lab notebook in advance (`curriculum/assets/week_09/`).

| Time | Slides | Segment | What you do |
|---|---|---|---|
| 0–5 min | 1–4 | Warm-up | Searching uncaptioned photos: why a shared space is needed. |
| 5–20 min | 5–10 | 1 · Modalities and terminology | Modalities; key terms; understanding vs generation; fusion; quiz. |
| 20–37 min | 11–15 | 2 · Representation learning | Representations; tokens for every modality; Whisper pipeline; shared spaces. |
| 37–70 min | 16–27 | 3 · Contrastive learning | CLIP; matrix; loss (write it); loss numbers; zero-shot; real results; confusion; retrieval; SigLIP; modality gap; quiz. |
| 70–77 min | – | Break | |
| 77–97 min | 28–32 | 4 · Architectures | Four families; history of vision-language transformers; LLaVA recipe; model map. |
| 97–117 min | 33–37 | 5 · Audio, evaluation, responsible AI | Whisper results; WER; evaluation table; responsible AI. |
| 117–120 min | 38–40 | Lab preview, summary, resources | |

> **Teaching tip:** Write the N × N similarity matrix on the board and ask students to "circle the positives": the diagonal. Then ask what a random model's loss would be (ln N). This makes the InfoNCE equation intuitive.

<!-- pagebreak -->

# Lecture notes

## 1. Modalities and terminology

A **modality** is a type of data with its own structure: text (discrete token sequences), images (2-D pixel grids), audio (1-D waveforms, usually converted to spectrograms), video (image sequences plus time, often with audio), and others such as depth, thermal and sensor data.

| Term | Meaning |
|---|---|
| Unimodal | Uses one modality |
| Cross-modal | Input in one modality, output or query in another (text→image search, captioning, speech→text) |
| Multimodal | Processes or combines several modalities |
| Alignment | Mapping modalities so corresponding content matches (photo of a dog ↔ "a dog") |
| Fusion | Combining information from modalities inside a model |
| Grounding | Linking words to regions, times or objects |

**Understanding vs generation.** Understanding maps images/audio/video to text, labels or vectors (zero-shot classification, retrieval, captioning, VQA, speech recognition). Generation maps text to other modalities (text-to-image, text-to-speech, text-to-video). They are linked: Stable Diffusion's text encoder is CLIP's, and CLIP score evaluates text-to-image alignment.

**Fusion strategies.**

* **Early fusion:** combine inputs/tokens first in one joint model; rich interactions, higher cost (natively multimodal models).
* **Late fusion:** separate encoders combined at the end (CLIP's similarity); efficient, ideal for retrieval.
* **Cross-attention fusion:** one modality attends to the other inside the network (Flamingo; diffusion U-Nets attending to prompts).

![Fusion strategies](fig:fusion)

## 2. Representation learning across modalities

A **representation** is a vector capturing content useful for many tasks. **Unimodal pre-training** learns representations within one modality (BERT, ViT, Whisper's encoder); **multimodal alignment** learns representations that correspond across modalities from naturally paired data (image–alt-text, video–subtitles, audio–descriptions). Web-scale pairs are noisy weak supervision, compensated by scale.

**Every modality as tokens:**

* **Images:** 16 × 16 patches → patch embeddings + positions (Vision Transformer).
* **Audio:** waveform → log-mel spectrogram → frames as tokens (Whisper); or discrete codec tokens for generation.
* **Video:** frames × patches (+ time): very many tokens; frame sampling or spatio-temporal "tubelets".
* **Discrete tokenizers:** VQ-style codebooks (week 2) let LLMs read and write images and audio as token IDs.

**Whisper** (Radford et al., 2022) is an encoder–decoder transformer: a log-mel spectrogram encoder and a text decoder with cross-attention, trained on 680,000 hours of weakly supervised web audio. It supports multilingual recognition, translation into English and timestamps.

![Whisper pipeline](fig:whisper_pipeline)

**Shared embedding spaces** place corresponding items close together so any modality can be compared with any other by cosine similarity: CLIP and SigLIP (image–text), CLAP (audio–text), ImageBind (six modalities bound through images).

![Modalities become vectors](fig:modalities)

## 3. Contrastive learning: CLIP and SigLIP

**CLIP** (Radford et al., 2021): 400 million web image–text pairs; image encoder (ViT or ResNet) and text transformer, each followed by a linear projection to a shared space (512-d for ViT-B/32); contrastive training with batches of 32,768; learned temperature. Zero-shot ImageNet accuracy matched a supervised ResNet-50 without ImageNet labels.

![The contrastive matrix](fig:clip_matrix)

### The contrastive loss

With normalised embeddings $u_i$ (images) and $v_j$ (texts) and temperature $\tau$:

$$s_{ij} = \frac{u_i \cdot v_j}{\tau}, \qquad \mathcal{L}_{\mathrm{img}\to\mathrm{txt}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s_{ii})}{\sum_{j} \exp(s_{ij})}$$

$$\mathcal{L}_{\mathrm{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{\mathrm{img}\to\mathrm{txt}} + \mathcal{L}_{\mathrm{txt}\to\mathrm{img}}\right)$$

Each row (and column) is a softmax classification whose correct class is the diagonal. A model that has learned nothing scores $\ln N$. Other items in the batch are the negatives, so larger batches give more and harder negatives.

**Worked example.** Suppose for one image the scaled similarities to three captions are [5, 1, 0] with the correct caption first. Softmax: $e^5 = 148.4$, $e^1 = 2.72$, $e^0 = 1$; probability of the correct caption $148.4 / 152.1 = 0.976$; loss contribution $-\ln 0.976 = 0.025$. If the correct caption had scored 1 instead, $p = 2.72/152.1 = 0.018$ and the loss would be 4.0.

**Lab measurement:** {{LOSS_NOTES}}

### Zero-shot classification and retrieval

Write each class into a template ("a photo of a {class}."), embed the texts and the image, and predict the most similar class. Retrieval ranks stored image embeddings by similarity to a query embedding; this scales to millions of images with a vector index (week 12).

![Zero-shot pipeline](fig:zeroshot_pipeline)

**Lab results (CLIP ViT-B/32, CIFAR-10):** {{ZS_NOTES}} {{PREC_NOTES}}

The lab's historical `recall_at_k` function computes **precision@k**: the fraction of returned pictures belonging to the query class, averaged over class queries. This differs from recall, which measures how many of all relevant items were retrieved. The benchmark recall@k metrics below concern a separate retrieval protocol.

![Zero-shot accuracy by prompt template](fig:zeroshot_results)

![Confusion matrix](fig:lab_confusion)

![Retrieval example](fig:lab_retrieval)

### SigLIP

$$\mathcal{L}_{\mathrm{SigLIP}} = -\frac{1}{N}\sum_{i,j} \log \sigma\left(z_{ij}\,(t\, u_i \cdot v_j + b)\right), \qquad z_{ij} = +1 \text{ if } i = j, \text{ else } -1$$

Each pair is an independent binary decision, so no batch-wide normalisation is required; SigLIP works well with smaller batches and its encoders are used in many recent VLMs (Zhai et al., 2023).

### The modality gap

Image and text embeddings occupy different regions of the shared space (Liang et al., 2022). Rankings remain meaningful, but image–image and image–text similarities are on different scales, so thresholds must be set per comparison type.

![Modality gap in real CLIP embeddings](fig:lab_modality_gap)

## 4. Multimodal architectures and vision-language transformers

| Family | Examples | Strengths | Limits |
|---|---|---|---|
| Dual encoder | CLIP, SigLIP, ALIGN, CLAP | Fast retrieval, zero-shot | No detailed reasoning about interactions |
| Fusion encoder | ViLBERT, LXMERT, VisualBERT | Joint understanding (VQA) | Slower; less flexible |
| Encoder–decoder | BLIP captioning, Whisper | Generate text from another modality | Task-specific |
| Vision encoder + LLM | LLaVA, BLIP-2, Qwen-VL, most MLLMs | Chat and reasoning about images and documents | Resolution limits; hallucination |

**History:** 2019 ViLBERT/LXMERT/VisualBERT (region features + BERT); 2021 CLIP, ALIGN (contrastive at web scale); 2022 Flamingo (frozen LLM + gated cross-attention), BLIP; 2023 BLIP-2 (Q-Former), LLaVA (projector + visual instruction tuning); 2024–26 natively multimodal models on interleaved text, images, audio and video.

**LLaVA recipe:** pretrained CLIP/SigLIP vision encoder → small MLP projector → visual tokens + text tokens → LLM. Stage 1 trains the projector on captions (alignment); stage 2 fine-tunes on image-based conversations (visual instruction tuning).

![Architecture families](fig:architectures)

## 5. Audio, evaluation and responsible AI

### Word error rate

$$\mathrm{WER} = \frac{S + D + I}{N}$$

Align hypothesis and reference with word-level edit distance: substitutions, deletions, insertions over reference words. Example: reference "the cat sat on the mat", hypothesis "the cat sat on mat": one deletion, WER = 1/6 ≈ 17%. WER can exceed 100%; normalisation (case, punctuation) must be reported.

**Lab measurement:** {{WER_NOTES}}

![Illustrative spectrogram of a LibriSpeech clip with linear frequency bins](fig:lab_spectrogram)

The plotted spectrogram illustrates frequency content over time. Whisper internally uses log-mel features, so this display is not its exact model input.

### Evaluation

| Capability | Metric | Benchmarks |
|---|---|---|
| Zero-shot classification | Top-1/top-5 accuracy | ImageNet variants, CIFAR |
| Retrieval | Recall@k both directions | MS-COCO, Flickr30k |
| Captioning | CIDEr, BLEU, CLIPScore, human | COCO Captions, NoCaps |
| VQA | Accuracy | VQAv2, TextVQA, ChartQA, DocVQA |
| Speech recognition | WER | LibriSpeech, Common Voice, FLEURS |
| Robustness and fairness | Performance by group/domain/accent | Custom audits |

# Common misconceptions

| Misconception | Correction |
|---|---|
| "Multimodal means the model generates images." | Multimodal covers understanding and generation; this week is mostly understanding. |
| "CLIP was trained to classify ImageNet." | It was trained contrastively on captions; classification is zero-shot via text prompts. |
| "Cosine similarities are comparable across modalities." | The modality gap puts image–image and image–text similarities on different scales. |
| "Bigger batches only speed training up." | In contrastive learning they supply more negatives and change what is learned. |
| "WER is the percentage of wrong words." | It counts substitutions, deletions and insertions over reference words; it can exceed 100%. |
| "Zero-shot models need no evaluation." | They must be tested on the target domain and across groups. |

# Responsible AI lens: multimodal data and models

* **Consent and copyright:** images, voices and videos scraped at scale, mostly without consent.
* **Harmful content:** a December 2023 Stanford Internet Observatory audit found child sexual abuse material in LAION-5B; LAION withdrew it and released a cleaned version (Re-LAION-5B) in 2024.
* **Bias:** CLIP's paper and audits reported misclassification of people into crime-related and non-human categories at different rates by race and gender.
* **Attacks:** typographic attacks (text written on objects changes CLIP's prediction); adversarial audio.
* **Surveillance and biometrics:** face and voice matching; EU AI Act restrictions on certain biometric uses.
* **Accessibility benefits:** captioning, speech recognition and image description help people with disabilities: evaluate for their contexts too.

# Lab guide and answers

**Runtime:** Colab T4 (≈ 10 min) or CPU (≈ 20–30 min). Data: CIFAR-10 test images (`uoft-cs/cifar10`), LibriSpeech sample clips (`hf-internal-testing/librispeech_asr_dummy`, decoded with `soundfile`). **Hand-in:** plots, accuracy/retrieval/loss/WER tables, three answers.

## TODOs

* **TODO 1** `F.normalize(feats, dim=-1)` for both encoders.
* **TODO 2** for each class, embed all templates, average, re-normalise; predict with `argmax` of `img_emb @ class_emb.T`.
* **TODO 3** `topk` of the query–image similarity; mean fraction of correct labels in the top k.
* **TODO 4** `logits = (u @ v.T) / tau`; symmetric cross-entropy with targets `arange(N)`.
* **TODO 5** word-level Levenshtein distance divided by reference length.

## Measured results (instructor run)

{{RESULTS_NOTES}}

## Troubleshooting

* *Audio decoding error mentioning `torchcodec`:* the notebook reads raw bytes and decodes with `soundfile` to avoid this dependency.
* *Slow CLIP on CPU:* reduce `N` to 300 images.
* *Different accuracies from the slides:* hardware and library versions differ slightly; the ranking of templates should be the same.

# Practice questions with model answers

## Multiple choice

1. CLIP combines modalities by: **(a)** early fusion; **(b)** late fusion of separately encoded embeddings; **(c)** cross-attention; **(d)** concatenating pixels and text. *Answer: (b).*
2. The loss of a random CLIP-style model on a batch of N pairs is about: **(a)** 0; **(b)** 1; **(c)** ln N; **(d)** N. *Answer: (c).*
3. Zero-shot classification with CLIP requires: **(a)** fine-tuning on labelled images; **(b)** text descriptions of the classes; **(c)** a GAN; **(d)** audio. *Answer: (b).*
4. In LLaVA, the projector: **(a)** generates images; **(b)** maps vision-encoder features into the LLM's token-embedding space; **(c)** computes WER; **(d)** replaces the LLM. *Answer: (b).*

## Short answer

1. **Compute the image-to-text loss for one image whose scaled similarities to three captions are [4, 2, 0], correct caption first.** (3 marks) *Model answer:* exp: 54.6, 7.39, 1.00; sum 62.99; p = 0.867; loss = −ln 0.867 ≈ 0.143.
2. **Compute WER: reference "open the lab report now" (5 words), hypothesis "open a lab report now please".** (3 marks) *Model answer:* 1 substitution (the→a) + 1 insertion (please) = 2; WER = 2/5 = 40%.
3. **Compare dual encoders with vision-encoder-plus-LLM architectures.** (4 marks) *Model answer:* dual encoders embed modalities separately for fast similarity-based retrieval and zero-shot classification but cannot reason about detailed interactions; LLM-based models feed visual tokens into an LLM for flexible question answering and reasoning at higher cost, with hallucination risk.
4. **Why does the choice of prompt template affect CLIP's zero-shot accuracy?** (3 marks) *Model answer:* the text encoder was trained on natural captions, so templates resembling captions produce embeddings closer to matching images; ensembles reduce variance from one phrasing.

## Exam-style question

**"Explain how CLIP learns a shared image–text embedding space and how that space enables zero-shot classification and retrieval. Discuss two limitations and the responsible-AI issues of training on web data."** (20 marks)

*Marking guide:* architecture and data (4); contrastive loss with explanation of positives/negatives and temperature (5); zero-shot classification and retrieval procedures (4); limitations such as the modality gap, fine-grained reasoning, distribution shift, typographic attacks (3); responsible AI: consent, harmful content, bias, surveillance (4).

# Glossary

| Term | Meaning |
|---|---|
| Modality | A type of data (text, image, audio, video, …) |
| Cross-modal | Relating one modality to another |
| Alignment | Making corresponding content across modalities match |
| Fusion (early/late/cross-attention) | Where and how modalities are combined in a model |
| Shared embedding space | Vector space in which several modalities can be compared |
| Contrastive learning | Learning by pulling positives together and pushing negatives apart |
| InfoNCE | Softmax cross-entropy contrastive loss over a batch |
| Zero-shot classification | Classifying with class descriptions, without task-specific training |
| Dual encoder | Separate encoders per modality compared by similarity |
| Vision-language model (VLM) | Model processing images and text jointly |
| Visual instruction tuning | Fine-tuning a VLM on image-based conversations |
| Modality gap | Separation between image and text embedding regions |
| Spectrogram | Time–frequency representation of audio |
| Word error rate | (S + D + I) / N for speech recognition |

# Readings, videos and further practice

**Core reading (descriptor 7.9)**

* Foster, D. (2023) *Generative Deep Learning*, 2nd ed.: chapter 13, "Multimodal Models".

**Papers**

* Radford et al. (2021) [CLIP](https://arxiv.org/abs/2103.00020); Zhai et al. (2023) [SigLIP](https://arxiv.org/abs/2303.15343); Radford et al. (2022) [Whisper](https://arxiv.org/abs/2212.04356); Dosovitskiy et al. (2020) [ViT](https://arxiv.org/abs/2010.11929); Liu et al. (2023) [LLaVA](https://arxiv.org/abs/2304.08485)

**Courses and explainers**

* [Hugging Face blog: Vision language models explained](https://huggingface.co/blog/vlms)
* [Hugging Face Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome) (multimodal units)
* [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction)
* [Google ML Crash Course: Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings)
* [3Blue1Brown / Welch Labs video on AI images and videos](https://www.youtube.com/watch?v=iv-5mZ_9CPY)

# Link to the group project

Projects involving images, audio or documents should decide which multimodal building blocks they need: a dual encoder for search or zero-shot tagging, Whisper for speech input, or a vision-language LLM for question answering (week 10). The evaluation plan must include a modality-appropriate metric (recall@k, accuracy, WER) and a check across relevant groups or conditions (accents, image types, languages).
