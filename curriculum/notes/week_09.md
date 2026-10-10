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

> **Teaching tip:** Write the N × N similarity matrix on the board and ask students to "circle the positives": the diagonal. Then ask what the uniform-choice loss would be (ln N). This makes the InfoNCE equation intuitive.

<!-- pagebreak -->

# Lecture notes

Start with the vocabulary and overall flow. For each section below, read the simple explanation first, trace the worked example aloud, and ask students to name the input and output. Introduce the equation only after the operation makes sense. The detailed text retains the full syllabus, while the lab and checks show what each method can and cannot establish.

## 1. Modalities and terminology

**Start here.** A modality is a data type, such as text, an image or sound. An encoder turns that input into useful numbers. Alignment learns which content corresponds across types. Fusion combines information from different inputs. Grounding links a word to a particular object, region or time.

**Small worked example.** The sentence "a dog on a beach" and a dog photo are a matching pair. Searching a photo gallery with that sentence is cross-modal: words are the query and pictures are the results.

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

**Start here.** A vector is an ordered list of numbers. Each model prepares its own input: text tokens, image patches or sound features. Projection layers make comparable vector sizes. Normalising a vector sets its length to one, so a dot product measures direction similarity rather than raw length.

**Small worked example.** Stored image scores for a query are 0.2, 0.7 and 0.4. The image at 0.7 ranks first. This is a model score, not a 70% probability that every query detail is present.

A **representation** is a vector describing useful input features. Pre-training within one type can use BERT for text, ViT for images or Whisper's audio encoder. Alignment across types uses matching examples, such as image/caption, video/subtitle or audio/description. Web pairs give **weak supervision**: a label-like signal that may be noisy.

**Every modality as tokens:**

* **Images:** 16 × 16-pixel patches (32 × 32 in the lab's ViT-B/32) → patch embeddings + positions (Vision Transformer).
* **Audio:** waveform → log-mel spectrogram → frames as tokens (Whisper); or discrete codec tokens for generation.
* **Video:** frames × patches (+ time): very many tokens; frame sampling or spatio-temporal "tubelets".
* **Discrete tokenizers:** VQ-style codebooks (week 2) let LLMs read and write images and audio as token IDs.

**Whisper** (Radford et al., 2022) has an audio encoder and a text decoder. It reads log-mel sound features; the text decoder uses cross-attention to those features when predicting words. The paper trained on 680,000 hours of weakly labelled web audio. The model supports multilingual recognition, English translation and timestamps, with differing quality by language and input.

![Whisper pipeline](fig:whisper_pipeline)

**Shared embedding spaces** place corresponding items close together so any modality can be compared with any other by cosine similarity: CLIP and SigLIP (image–text), CLAP (audio–text), ImageBind (six modalities bound through images).

![Modalities become vectors](fig:modalities)

## 3. Contrastive learning: CLIP and SigLIP

**Start here.** Contrastive learning compares labelled matching and non-matching pairs. CLIP makes a score for every image-caption combination in a batch, then asks which caption matches each image and which image matches each caption. The lab calculates this objective with fixed pretrained encoders; it does not train them.

**Small worked example.** Three pairs make a 3 × 3 score matrix. Correct pairs are the diagonal when pairing order is kept. Equal choice probabilities give loss ln 3, about 1.10. Four relevant pictures among five returned gives precision@5 = 4/5 = 0.8.

**CLIP** (Radford et al., 2021) learns from 400 million web image-caption pairs. Separate image and text encoders feed projection layers into a shared vector space; ViT-B/32 uses 512 dimensions. Contrastive batches had 32,768 pairs. A learned temperature scales their scores. On the paper's ImageNet evaluation, zero-shot accuracy matched a supervised ResNet-50 without using ImageNet labels for training.

![The contrastive matrix](fig:clip_matrix)

### The contrastive loss

With normalised embeddings $u_i$ (images) and $v_j$ (texts) and temperature $\tau$:

$$s_{ij} = \frac{u_i \cdot v_j}{\tau}, \qquad \mathcal{L}_{\mathrm{img}\to\mathrm{txt}} = -\frac{1}{N}\sum_{i=1}^{N} \, \log \frac{\exp(s_{ii})}{\sum_{j} \exp(s_{ij})}$$

$$\mathcal{L}_{\mathrm{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{\mathrm{img}\to\mathrm{txt}} + \mathcal{L}_{\mathrm{txt}\to\mathrm{img}}\right)$$

Each row turns image-caption scores into probabilities of which caption matches; each column asks the reverse question. Cross-entropy penalises low probability for the labelled match. Equal probabilities over $N$ choices give loss $\ln N$. Other items supply negative pairs, so larger batches provide more comparisons; not every comparison is necessarily a true mismatch.

**Worked example.** Suppose for one image the scaled similarities to three captions are [5, 1, 0] with the correct caption first. Softmax: $e^5 = 148.4$, $e^1 = 2.72$, $e^0 = 1$; probability of the correct caption $148.4 / 152.1 = 0.976$; loss contribution $-\ln 0.976 = 0.025$. If the correct caption had scored 1 instead, $p = 2.72/152.1 = 0.018$ and the loss would be 4.0.

**Lab measurement:** {{LOSS_NOTES}}

### Zero-shot classification and retrieval

For classification, write a description of each class, such as "a photo of a dog", and encode it. Compare an image vector with all class vectors and choose the largest score. For retrieval, encode one query and rank stored image vectors. Larger collections may use a vector index to make the search faster (week 12).

![Zero-shot pipeline](fig:zeroshot_pipeline)

**Lab results (CLIP ViT-B/32, CIFAR-10):** {{ZS_NOTES}} {{PREC_NOTES}}

The lab's historical `recall_at_k` function computes **precision@k**: the fraction of returned pictures belonging to the query class, averaged over class queries. This differs from recall, which measures how many of all relevant items were retrieved. The benchmark recall@k metrics below concern a separate retrieval protocol.

![Zero-shot accuracy by prompt template](fig:zeroshot_results)

![Confusion matrix](fig:lab_confusion)

![Retrieval example](fig:lab_retrieval)

### SigLIP

$$\mathcal{L}_{\mathrm{SigLIP}} = -\frac{1}{N}\sum_{i,j} \, \log \sigma\left(z_{ij}\,(t\, u_i \cdot v_j + b)\right), \qquad z_{ij} = +1 \text{ if } i = j, \text{ else } -1$$

Each pair is an independent binary decision, so no batch-wide normalisation is required; SigLIP works well with smaller batches and its encoders are used in many recent VLMs (Zhai et al., 2023).

### The modality gap

Image and text embeddings occupy different regions of the shared space (Liang et al., 2022). Rankings remain meaningful, but image–image and image–text similarities are on different scales, so thresholds must be set per comparison type.

![Modality gap in real CLIP embeddings](fig:lab_modality_gap)

## 4. Multimodal architectures and vision-language transformers

**Start here.** Separate encoders can support fast search. A model that answers image questions also needs a way for its language component to use visual features. A projector is a small bridge mapping those features to the language model's input space; cross-attention is another way to share information.

**Small worked example.** For "What colour is the car?", the image encoder supplies visual features and the question supplies text tokens. The bridge lets the language model use both to generate "red". The colour still needs checking against the image.

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

**Start here.** Speech recognition turns a sound recording into words. It can substitute a word, omit a word or add one. Word error rate divides all those errors by the number of reference words. An image-search score, caption metric and speech error rate check different tasks and should not be treated as interchangeable.

**Small worked example.** Reference: "the cat sat on the mat". Transcript: "the cat sat on mat". One word is missing from six reference words: WER = 1/6, about 17%. A perfect transcript alone does not establish fairness across accents.

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

**Runtime:** Colab T4 GPU or CPU. On the 4 GB GTX 1650 laptop GPU used to check this pack (9 October 2026), the full notebook ran in under a minute with the models and data already downloaded; the first run also downloads CLIP, Whisper, CIFAR-10 and the audio clips. The CPU path was not timed in full. Data: CIFAR-10 test images (`uoft-cs/cifar10`), LibriSpeech sample clips (`hf-internal-testing/librispeech_asr_dummy`, decoded with `soundfile`). **Hand-in:** plots, accuracy/retrieval/loss/WER tables, three answers.

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

1. CLIP combines modalities by: **(a)** late fusion of separately encoded embeddings; **(b)** early fusion of image patches and text tokens; **(c)** cross-attention between image and text layers; **(d)** concatenating raw pixels and text into one input. *Answer: (a).*
2. With equal choice probabilities, the CLIP-style loss on N pairs is: **(a)** 0; **(b)** 1; **(c)** ln N; **(d)** N. *Answer: (c).*
3. Zero-shot classification with CLIP requires: **(a)** fine-tuning on labelled images; **(b)** audio; **(c)** a GAN; **(d)** text descriptions of the classes. *Answer: (d).*
4. In LLaVA, the projector: **(a)** generates images from the LLM's text output; **(b)** maps vision-encoder features into the LLM's token-embedding space; **(c)** computes the word error rate of the generated caption; **(d)** replaces the LLM's final layers for vision questions. *Answer: (b).*

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

# Detailed explanations behind the short slides

The slides show one idea at a time. Use the detail below for follow-up questions, technical terms and further study.

## Slide 1: Title

Welcome to week 9, the start of the multimodal part of the module. So far our models have handled one kind of data at a time: images in weeks 2 to 4, text in weeks 5 to 8. Real-world information mixes text, images, audio and video, and the most capable systems of 2026 handle them together. This week covers the foundations: what modalities are, how each becomes vectors or tokens, how different modalities are aligned in a shared space, with CLIP as the central example, and the architectures that combine them. Next week builds on this with multimodal LLMs and generation across modalities.

## Slide 2: Outcomes

Six outcomes, matching the descriptor's week 9 and MIMLO 2, which asks students to analyse and apply multimodal techniques including alignment and cross-modal reasoning. The first two are vocabulary and representations. The third, contrastive learning with CLIP, is the technical core and includes a loss students should be able to write and compute. The fourth surveys architectures and vision-language transformers. The fifth adds audio with Whisper and evaluation metrics, touching MIMLO 4. The sixth is the responsible-AI lens, which is particularly important for web-scale image and audio data.

## Slide 3: Agenda

The plan follows the descriptor: definition of modalities; key concepts and terminology; representation learning across modalities; key multimodal architectures and models; contrastive learning with CLIP; and vision-language transformers. Section three is the longest because it contains the maths and the real results from the lab, run in advance on CIFAR-10. Take the break after section three.

## Slide 4: Warm-up

Give two minutes. Students usually propose training a classifier, then realise that they would need a class for every possible query and labelled data for each. The key insight to draw out is the last point: if text and images can be turned into vectors in the same space, where a matching image and description land close together, then any query can be compared with any image by a similarity score, with no task-specific training. That is exactly what CLIP does, and by the end of section three students will have done it in the lab.

- You have 10,000 holiday photos with no captions or tags. How could a computer find 'a dog on a beach at sunset'?

- Could you train a classifier? Which classes would you need, and where would labels come from?

- What if the next search is 'a red umbrella in the rain'?

- What would a system need so that **any** text query can be compared with **any** image?

## Slide 5: Modalities and terminology

Section one establishes the vocabulary: modalities and how each is represented, the precise meanings of cross-modal, multimodal, alignment and fusion, the difference between multimodal understanding and generation, and the three ways of fusing modalities. These terms appear in the exam and in every multimodal paper. About fifteen minutes.

- What exactly does 'multimodal' mean?

## Slide 6: Four main modalities, one idea: turn everything into vectors

A modality is a type of data with its own structure: text is a sequence of discrete symbols; images are two-dimensional grids of pixels; audio is a one-dimensional waveform, usually converted to a time-frequency spectrogram; video is a sequence of images with time and often audio. Each modality is split into units: sub-word tokens, image patches, spectrogram frames or codec tokens, and video frames with patches. A modality-specific encoder turns those units into vectors. Multimodal models then either place these vectors in a shared space or feed them into a shared model, which is the rest of today's lecture.

## Slide 7: Key terms

These definitions come straight from the descriptor's list: modality, cross-modal, multimodal, alignment and fusion; grounding is added because it appears in many vision-language papers. The distinction between cross-modal and multimodal is subtle: cross-modal emphasises going from one modality to another, for example text-to-image search, while multimodal means processing several together. Alignment is about correspondence between modalities; fusion is about how information is combined inside a model. Encourage students to use these terms precisely in their reports.

| Term | Meaning | Example |
| --- | --- | --- |
| Modality | A type of data with its own structure | Text, image, audio, video, depth, sensor data |
| Unimodal | Uses one modality | A text-only LLM; an image classifier |
| Cross-modal | Input in one modality, output or query in another | Text-to-image search; image captioning; speech-to-text |
| Multimodal | Processes or combines several modalities | A model answering questions about an image and text |
| Alignment | Mapping modalities so corresponding content matches | The photo of a dog and 'a dog' get similar vectors |
| Fusion | Combining information from modalities inside a model | Early, late or cross-attention fusion |
| Grounding | Linking words to specific regions, times or objects | 'the red car' ↔ its bounding box |

## Slide 8: Multimodal understanding vs. multimodal generation

Separate the two directions. Understanding maps non-text inputs to text, labels or vectors; generation maps text to non-text outputs. They are closely linked: the text encoder inside Stable Diffusion from week 4 is CLIP's text transformer, and we measured prompt alignment with CLIP score in the week 4 lab. This week focuses on understanding and alignment; next week covers multimodal LLMs and generation across modalities, including audio and video.

| Aspect | Understanding (this week) | Generation (weeks 4 and 10) |
|---|---|---|
| Input and output | Image/audio/video in → text, labels or vectors out | Text in → image, audio or video out |
| Tasks | Zero-shot classification, retrieval, captioning, visual question answering, speech recognition | Text-to-image, text-to-speech, text-to-video, image editing |
| Models | Models: CLIP, SigLIP, BLIP, Whisper, vision-language LLMs | Models: diffusion/flow models conditioned on text encoders; speech and video generators |

Text-guided generators need a text representation; CLIP-based scores are one way to check image-text alignment.

## Slide 9: Three ways to fuse modalities

Fusion is about where information is combined. Early fusion puts all modalities into one model from the start, allowing rich interactions, at higher cost; natively multimodal models such as those in the Gemini family process interleaved text and image tokens. Late fusion keeps separate encoders and combines only their outputs, as CLIP does with a similarity score; it is efficient because each item is encoded once, which is ideal for search over millions of images. Cross-attention fusion, from week 5, lets one modality query the other inside the network, as in DeepMind's Flamingo or the diffusion U-Net attending to the prompt.

- **Early fusion** combines inputs or tokens before a joint model processes them.

- **Late fusion** processes each input separately, then combines the results; CLIP compares image and text vectors.

- **Cross-attention** lets features from one input select useful features from another, as in Flamingo.

## Slide 10: Quiz

**Question.** CLIP encodes images and texts separately and compares them with cosine similarity. Which fusion strategy is this?

- **A.** Late fusion
- **B.** Early fusion
- **C.** Cross-attention fusion
- **D.** No fusion: CLIP is unimodal

Answer: A, late fusion. CLIP's image and text encoders never see each other's inputs; the modalities meet only at the end, in the similarity score between two embeddings. This is what makes CLIP so efficient for retrieval: you can encode a million images once, store the vectors, and compare any new text query against them. B and C describe designs where modalities interact inside the network. D is wrong: CLIP is multimodal because it aligns two modalities in one space.

## Slide 11: Representation learning across modalities

Section two covers representation learning: how images, audio and video are turned into sequences of vectors that transformers can process, and the idea of a shared embedding space in which different modalities can be compared directly. It links back to tokens and embeddings from weeks 5 and 6. About seventeen minutes.

- How does each modality become vectors that can be compared?

## Slide 12: Representation learning: vectors that capture meaning

Representation learning is the thread connecting this whole module: VAE latents, transformer embeddings and now multimodal embeddings are all vectors that capture meaning. Unimodal models learn good representations within one modality by self-supervision. Multimodal alignment needs paired data, where two modalities describe the same thing, and the web provides it at enormous scale: images with alt-text captions, videos with subtitles, podcasts with transcripts. This data is noisy, since captions often do not describe the image well, and scale does not remove misleading or missing labels.

- A **representation** is a list of numbers describing useful features of an input.

- A model can first learn one data type: BERT for words, ViT for pictures, Whisper's encoder for sound.

- **Alignment** trains matching content across types, such as a photo paired with its caption.

- Web captions are a noisy teaching signal: some describe the image poorly. More data does not remove this problem.

## Slide 13: Every modality as a sequence of tokens

The unifying trick of modern multimodal AI is to turn every modality into a sequence of tokens, so the same transformer machinery applies. Images become patches, as in the Vision Transformer from week 5. Audio is first converted into a spectrogram, a picture of frequency content over time; its frames become tokens, as in Whisper, or audio can be compressed into discrete codec tokens for generation. Video is the most expensive: a minute of video contains thousands of frames, each with hundreds of patches, so models sample frames or group patches across time. Discrete tokenizers based on the VQ-VAE idea from week 2 let a language model read and write images and audio as tokens.

- **Images:** Cut an image into small patches. Convert each patch to numbers and add its position. This is a Vision Transformer (ViT).

- **Audio:** A spectrogram shows sound frequencies over time. Whisper uses log-mel features; some generators use audio-codec token IDs.

- **Video:** Use image patches from several frames, with their time positions. A tubelet groups patches across nearby frames.

- **Discrete tokens:** A codebook stores learned pattern vectors. A VQ tokenizer replaces an image or audio segment with their IDs (week 2).

## Slide 14: Audio as input: the Whisper pipeline

Whisper, released by OpenAI in 2022 and available as open weights, is our audio representation example. The waveform, sampled at sixteen kilohertz, becomes a log-mel spectrogram. The encoder processes these time-frequency features. The decoder generates transcript tokens while attending to the audio, using the encoder-decoder design from week 5. It was trained on 680,000 hours of weakly supervised audio with web transcripts. Large datasets still require data-quality checks and evaluation across accents and recording conditions. Students transcribe real speech with Whisper in the lab.

- Whisper's **encoder** reads audio features. Its **decoder** generates text using those features.

- **Cross-attention** connects the decoder to the audio representation.

- The model supports speech recognition, translation into English and timestamps; quality varies by input.

## Slide 15: Shared embedding spaces

A shared embedding space is the direct answer to the warm-up. If a photo of a dog on a beach and the text a dog on a beach at sunset map to nearby vectors, search becomes a nearest-neighbour lookup. The space enables many applications without retraining: retrieval in both directions, zero-shot classification by comparing an image with class descriptions, deduplication and clustering of image collections. CLAP does this for audio and text. Meta's ImageBind in 2023 aligned six modalities, including audio, depth and thermal images, by pairing each with images. These encoders also supply the features used by generators and multimodal LLMs.

- Train matching content, such as a dog photo and its caption, to receive a high similarity score.

- The **shared embedding space** gives image and text vectors the same number of dimensions so we can compare them.

- Use the scores for search or classification. Similarity does not prove every detail matches.

- Examples: CLIP/SigLIP for images and words; CLAP for audio and words; ImageBind links six data types.

## Slide 16: Contrastive learning: CLIP and SigLIP

Section three is the technical core. We describe CLIP, its training data and objective, write the contrastive loss, and then look at what CLIP can do without further training, zero-shot classification and retrieval, using real results from the lab run on CIFAR-10. We then introduce SigLIP's sigmoid loss and the modality gap. About thirty minutes; take the break afterwards.

- How do you teach a model that a picture and a sentence mean the same thing?

## Slide 17: CLIP: Contrastive Language–Image Pre-training (OpenAI, 2021)

CLIP, by Radford and colleagues in 2021, is one of the most influential models in generative AI. Its data was four hundred million image-text pairs collected from the internet. Its architecture is a dual encoder: an image encoder, a Vision Transformer in the model we use, and a text transformer, each projecting into a shared five-hundred-and-twelve-dimensional space. The training objective is contrastive: within each batch, every image must pick out its own caption from all the captions, and every caption its own image. CLIP's zero-shot ImageNet accuracy matched a supervised ResNet-50 without using ImageNet labels, and its text encoder became part of Stable Diffusion.

- CLIP was trained on **400 million** web image-text pairs (OpenAI, 2021).

- An image encoder and a text encoder each produce a vector. Projection layers map them to a shared space.

- In each training batch, find the right caption for each image and the right image for each caption.

- A batch of 32,768 pairs supplied many **negative pairs**: non-matching image-caption combinations.

- After training, CLIP can classify using class descriptions without learning that task's labels.

## Slide 18: The contrastive objective as a matrix

Picture a batch of N image-caption pairs as an N by N matrix of similarities: row i is image i, column j is caption j. The correct pairs are on the diagonal. Training increases the diagonal similarities and decreases the off-diagonal ones, treating each row as a classification problem, which caption belongs to this image, and each column as a classification problem, which image belongs to this caption. All other captions in the batch serve as negative examples for free, larger batches provide more comparisons, though ambiguous or false-negative pairs still need consideration.

## Slide 19: The contrastive (InfoNCE) loss

Write the loss on the board. The similarity s i j is the cosine similarity between image i and text j, divided by a temperature; CLIP learns the temperature and it ends up around one hundredth, which sharpens the softmax, the same temperature idea as in week 1. For each image, the softmax over all captions should put its probability on the matching caption; the loss is the negative log of that probability, a cross-entropy with the diagonal as the target. The same is done column-wise for texts, and the two directions are averaged. Uniform choice assigns probability 1/N to each caption, giving loss ln N.

Read the CLIP image-to-text loss in words: for each image, take the negative log probability of its matching caption, then average over the batch. The complete equation appears in Section 3 above.

Each score is the dot product of normalised image and text vectors, divided by the temperature. Average the image-to-text and text-to-image losses to get the symmetric CLIP loss.

- $u_i$ and $v_j$ are normalised vectors. Their dot product is cosine similarity; divide by temperature $\tau$ to scale the scores.

- Turn each row into caption-choice probabilities with **softmax**. The correct caption is at position i on the diagonal.

- Use **cross-entropy** to penalise low probability for the correct choice. Average both directions; equal probabilities give loss ln N.

| Symbol | Meaning |
|---|---|
| $u_i$ / $v_j$ | image / caption vectors with length 1 |
| $\tau$ | temperature: controls how sharply scores become probabilities |
| N | number of image-caption pairs in the batch |

## Slide 20: The loss in numbers (from the lab)

These numbers come from the lab run: six CIFAR-10 images from different classes and six matching captions, embedded with the pretrained CLIP. With the correct pairing the loss is far below the uniform-choice baseline of the log of six, about 1.79, because each image is most similar to its own caption. When the captions are shuffled the loss rises above that baseline, because the model is now confidently wrong. Students compute exactly this in part four of the lab, which makes the abstract equation concrete.

- {{LOSS_NOTES}}

- Low loss means the labelled pair receives a high probability among the candidates.

- After shuffling captions, the diagonal targets no longer match. Inspect the measured loss against the uniform-choice value ln N.

## Slide 21: Zero-shot classification: describe the classes, then compare

Zero-shot classification with CLIP takes no training. Write each class name into a caption template such as a photo of a cat, embed all the captions with the text encoder, embed the image with the image encoder, and predict the class whose caption is most similar. The classifier is defined entirely by text, so changing the classes means changing a list of words. This is also why prompt engineering matters for CLIP, as the next slide shows.

## Slide 22: Real results: prompt templates matter for CLIP too

Real results from the lab: CLIP ViT-B/32 classifying CIFAR-10 test images zero-shot, with only the text prompt changing. Here the bare class name did slightly worse than the template a photo of a, because CLIP's text encoder was trained on natural captions. Averaging several templates, including ones that mention blur and low resolution, which suits CIFAR's tiny images, helped a little more. On 1,000 images a one-point difference is ten images, so check whether such gaps hold on other data. The lesson from week 7 transfers: prompts are part of the model and must be evaluated. Chance level is ten percent, so CLIP is doing a great deal without any CIFAR training.

- {{ZS_NOTES}}

- "A photo of a car" resembles CLIP's training captions more than the single word "car".

- **Prompt ensembling** averages vectors from several class-description templates. Measure whether it helps your data.

## Slide 23: Where zero-shot CLIP goes wrong (real confusion matrix)

The confusion matrix from the lab run shows where the zero-shot classifier fails. The diagonal is strong, and the largest off-diagonal counts are between classes that look alike at low resolution: deer predicted as horses, dogs and frogs predicted as cats, and airplanes predicted as ships. CIFAR's thirty-two by thirty-two images are upscaled to CLIP's input size, which is a distribution shift from the web photos it was trained on. This is a good example of error analysis from week 7: the errors are understandable and suggest remedies, such as better templates, higher-resolution data or a small amount of fine-tuning.

- In this run, the largest confusions were deer → horse, dog → cat, frog → cat and airplane → ship.

- CIFAR-10 images start at 32 × 32 pixels. Resizing to 224 × 224 cannot restore detail that was absent.

## Slide 24: Text-to-image retrieval (real)

Retrieval uses the same embeddings in the other direction. The query text is embedded once and compared with the stored embeddings of all images; the top results are returned. The strip shows the top six CIFAR images for a free-text query about a red sports car, and the bullet gives the measured precision at k for class-name queries. Because image embeddings are computed once and stored in a vector index, this scales to millions of images, which is how semantic search in photo apps works and how multimodal retrieval-augmented generation retrieves images in week 12.

- Encode the query once. Compare it with stored image vectors and return the highest scores.

- {{PREC_NOTES}}

- The same idea supports photo search and image-aware document search (week 12).

## Slide 25: SigLIP: a sigmoid loss instead of softmax (Zhai et al., 2023)

SigLIP, from Google in 2023, replaces the softmax over the batch with an independent sigmoid, a logistic regression, for every image-text pair: matching pairs should score high, all others low. Because each pair is scored separately, there is no need to normalise over the whole batch, which reduces communication between devices and works well even with smaller batches. SigLIP image encoders are widely used as the vision backbone of recent open vision-language models. The lab's optional extension compares SigLIP's zero-shot accuracy with CLIP's.

Read the SigLIP loss in words: label matching pairs +1 and non-matching pairs −1. Scale and shift each similarity, apply the sigmoid, and average the negative log probabilities. The complete equation appears in Section 3 above.

- For every image-caption combination, ask one **binary** question: matching or non-matching?

- The **sigmoid** converts each pair's score to a probability. It does not normalise a whole row as CLIP does.

- This changes training and batch requirements; SigLIP encoders are also used in vision-language models.

| Symbol | Meaning |
|---|---|
| $\sigma$ | sigmoid: maps a score to a value between 0 and 1 |
| t / b | learned score multiplier / offset |
| $z_{ij}$ | +1 for a labelled match; −1 for a non-match |

## Slide 26: The modality gap (real embeddings from the lab)

A surprising property of CLIP-style spaces is visible when we project real embeddings to two dimensions with PCA: images and texts form two separate clouds. This modality gap, studied by Liang and colleagues in 2022, arises from initialisation and the contrastive objective. It does not break retrieval, because what matters is which caption is closest to an image relative to the others. But it has practical consequences: image-to-image similarities are typically much higher than image-to-text similarities, so thresholds must be set separately for each comparison type.

- A **modality gap** means image and text vectors form different regions in the shared space (Liang et al., 2022).

- We can still rank images for a text query: relative scores can identify good matches.

- An image-text score and an image-image score need not mean the same thing.

## Slide 27: Quiz

**Question.** Why did CLIP use very large batches (32,768 pairs)?

- **A.** Fewer model parameters are then needed
- **B.** More negative pair comparisons per step
- **C.** Large image files must be loaded together
- **D.** Batch size has no effect on what is learned

Answer: B. In the contrastive loss, every other caption in the batch is a negative for a given image. With a larger batch, each image must be distinguished from more, and more similar, captions, which forces the embeddings to capture finer distinctions. A and C are irrelevant. D is wrong for softmax-based contrastive learning, although SigLIP's sigmoid loss reduces the dependence on batch size, which is one of its selling points.

## Slide 28: Multimodal architectures and vision-language transformers

Welcome back. Section four surveys the architectures named in the descriptor: dual encoders like CLIP, fusion encoders, encoder-decoders, and a common design connecting a vision encoder to a language model. We trace the history of vision-language transformers and finish with a map of important models.

- How can a language model use visual features to answer an image question?

## Slide 29: Four families of multimodal architecture

Four families. Dual encoders, like CLIP and SigLIP, encode inputs separately and compare their vectors. They support fast retrieval and zero-shot classification; a similarity score alone cannot answer a detailed visual question. Fusion encoders read image and text jointly with attention. Encoder-decoders read one modality and generate another, as in captioning and speech recognition. The fourth family connects a vision encoder to an LLM through a bridge. It is a common multimodal assistant design and the focus of next week.

## Slide 30: Vision-language transformers: a short history

The history shows a convergence. Early vision-language transformers in 2019 extended BERT to image regions detected by an object detector, with either one shared stream or two streams linked by cross-attention. CLIP and Google's ALIGN in 2021 showed the power of contrastive learning at web scale. DeepMind's Flamingo in 2022 connected a frozen language model to image features with gated cross-attention, enabling few-shot visual tasks. BLIP-2 and LLaVA in 2023 showed that a pretrained vision encoder and a pretrained LLM can be connected cheaply by a small bridge. Frontier models now train natively on interleaved modalities.

- **2019:** ViLBERT, LXMERT and VisualBERT combine image-region features with words.

- **2021:** CLIP and ALIGN learn separate encoders by matching images with text.

- **2022:** Flamingo connects a fixed LLM to images with cross-attention; BLIP supports captions and search.

- **2023:** BLIP-2 uses a Q-Former bridge; LLaVA uses a projector and visual instruction examples.

- Later models train on mixed text, image, audio and video inputs (week 10).

## Slide 31: The LLaVA recipe: connect a vision encoder to an LLM

LLaVA, by Liu and colleagues in 2023, made this design popular because it is remarkably simple. A pretrained CLIP vision encoder turns the image into a grid of patch features; a small projection network maps each patch feature into the LLM's token-embedding space, producing visual tokens; the LLM reads them followed by the text prompt and generates an answer. Training has two stages: first the projector alone is trained on captions to align the spaces, then the projector and LLM are fine-tuned on image-based conversations, called visual instruction tuning. Next week we use models built this way and discuss their evaluation.

- **Image:** e.g. 336 × 336 photo, chart or document

- **Vision encoder:** Pretrained CLIP or SigLIP ViT (often frozen)

- **Projector:** Small MLP maps patch features to LLM token embeddings

- **LLM:** Reads visual tokens + text tokens; generates the answer

- A **projector** is a small neural network that maps image features to vectors the LLM can read.

- First train the bridge with image-caption pairs. Then use image-question-answer conversations for instruction tuning.

- Image resolution limits the available detail, including small printed text.

## Slide 32: A map of important multimodal models

This table is a map for revision and for choosing components in the project. Dual encoders for search and zero-shot tasks; bridge-to-LLM models for question answering and chat about images; encoder-decoders for converting speech to text; audio-text contrastive models such as CLAP for sound search; and ImageBind for experiments across many modalities. Most are available as open weights on Hugging Face. Next week adds the natively multimodal assistants and generators.

| Model (year) | Modalities | Type | Typical use |
| --- | --- | --- | --- |
| CLIP (2021), SigLIP (2023) | Image–text | Dual encoder, contrastive | Retrieval, zero-shot, features for generators |
| ALIGN (2021) | Image–text | Dual encoder, noisy web data | Retrieval |
| Flamingo (2022) | Image/video + text | Frozen LLM + cross-attention | Few-shot visual QA |
| BLIP-2 (2023) | Image + text | Q-Former bridge to LLM | Captioning, VQA |
| LLaVA (2023) | Image + text | Projector + LLM | Visual chat; open research baseline |
| Whisper (2022) | Audio → text | Encoder–decoder | Speech recognition, translation |
| CLAP (2022–23) | Audio–text | Dual encoder, contrastive | Audio search and tagging |
| ImageBind (2023) | Six modalities | Encoders bound via images | Cross-modal retrieval |

## Slide 33: Audio, evaluation and responsible AI

The final section returns to audio with real Whisper results, defines word error rate, summarises how multimodal models are evaluated, and discusses the responsible-AI issues of multimodal data, which are serious: consent, harmful content in scraped datasets, bias in zero-shot classification, and surveillance. About twenty minutes.

- How well does it work, and at what cost to whom?

## Slide 34: Speech recognition with Whisper (real)

An illustrative spectrogram of the first LibriSpeech clip from the lab, with time on the horizontal axis and linear frequency bins on the vertical. This plot is not Whisper's exact preprocessing, which uses log-mel features. Bright bands show harmonics of the speaker's voice. Whisper transcribed ten clips in the lab run with the word error rate reported in the bullet. LibriSpeech is clean read audiobook speech, one of the easier benchmarks. Accuracy can be lower for noisy audio, strong accents, overlapping speakers and low-resource languages, which is a fairness issue we return to on the responsible-AI slide.

- {{WER_NOTES}}

- The plot shows sound energy by frequency and time. Whisper itself uses log-mel features.

## Slide 35: Word error rate (WER)

Word error rate is the standard metric for speech recognition. The hypothesis is aligned to the reference with the minimum number of word edits, the edit or Levenshtein distance used in the lab's TODO, and the errors are divided by the number of reference words. Work through the example. Note that WER can exceed one hundred percent if the system inserts many words, and that it treats all errors equally, so a wrong name costs the same as a wrong article. Text normalisation, such as case and punctuation, changes WER substantially, so it must be reported.

Read WER as: changed words plus missing words plus extra words, divided by the number of words in the reference. The complete equation appears in Section 5 above.

- Compare the generated transcript with the known text, word by word.

- Count changed words S, missing words D and extra words I. Divide their sum by the reference length N.

- Reference: "the cat sat on the mat"; transcript: "the cat sat on mat". One missing word out of 6 gives WER = 1/6, about 17%.

| Symbol | Meaning |
|---|---|
| S | substituted words |
| D | deleted words |
| I | inserted words |
| N | reference word count |

## Slide 36: How multimodal models are evaluated

Each capability has standard metrics and benchmarks. Retrieval uses recall at k in both directions, typically on MS-COCO and Flickr30k. Captioning uses reference-based metrics such as CIDEr plus CLIPScore and human rating. Visual question answering has many benchmarks, including ones for charts and documents that matter for business applications. Speech recognition uses word error rate on benchmarks such as LibriSpeech and the multilingual FLEURS. The last row is the most important for responsible deployment and is usually missing from leaderboards: performance must be broken down by domain, accent and demographic group.

| Capability | Metric | Common benchmark |
| --- | --- | --- |
| Zero-shot classification | Top-1 / top-5 accuracy | ImageNet and variants, CIFAR |
| Retrieval | Recall@k (image→text and text→image) | MS-COCO, Flickr30k |
| Captioning | CIDEr, BLEU, CLIPScore, human rating | MS-COCO Captions, NoCaps |
| Visual question answering | Accuracy | VQAv2, TextVQA, ChartQA, DocVQA |
| Speech recognition | Word error rate | LibriSpeech, Common Voice, FLEURS |
| Robustness and fairness | Performance across groups, domains, accents | Custom audits |

## Slide 37: Responsible AI lens: multimodal data and models

Multimodal data raises the sharpest responsible-AI issues of the module. Scraped images, voices and videos rarely come with consent. In December 2023 the Stanford Internet Observatory reported child sexual abuse material in LAION-5B, a dataset used to train open image models; LAION withdrew it and released a cleaned version in 2024. CLIP's own paper and follow-up audits reported biased misclassification of people into offensive categories. OpenAI showed that writing a word on an object can fool CLIP. Face and voice matching enable surveillance, and speech systems perform worse for some accents. The EU AI Act restricts some biometric uses. Students should evaluate across groups in any project using these models.

- **Data rights:** check permission and licences for photos, voices and videos used in training or evaluation.

- **Dataset audits:** a 2023 audit found child sexual abuse material in LAION-5B; the dataset was withdrawn and later cleaned.

- **Bias:** CLIP's evaluation found unequal harmful classification errors across demographic groups.

- **Attacks:** text printed in a picture can mislead CLIP; altered audio can mislead speech systems.

- **Privacy and access:** face or voice matching can enable tracking; speech accuracy differs across accents and languages.

## Slide 38: Lab 9: CLIP embeddings, zero-shot classification and Whisper

The lab implements the descriptor's tutorial: understand modalities and embeddings; explore text-image embeddings, alignment and retrieval with pretrained models like CLIP. It adds the contrastive loss, so students connect the equation to real embeddings, and a speech part with Whisper and word error rate, so that audio is covered as the descriptor's modality list requires. The slide results came from this notebook. The chained speech-to-image retrieval previews the cross-modal systems of week 10.

- Make image and text vectors with CLIP; inspect the similarity table.

- Compare class-description prompts on 1,000 CIFAR-10 images; inspect wrong labels.

- Search images with text; calculate precision@k.

- Implement contrastive loss; compare matching and shuffled pairs with ln N.

- Use **PCA** to compress vectors to a 2D plot; discuss what detail it loses.

- Transcribe with Whisper, calculate WER, then search images with the transcript. Optional: SigLIP.

- **Runtime:** Colab T4 GPU or CPU (slower). About 1 min on a 4 GB laptop GPU after downloads.

- **Hand in:** notebook with similarity plots, accuracy table, retrieval metrics, loss values, WER table and three answers.

## Slide 39: Summary

The six cards on the slide condense these eight summary points. Students should be able to define the key terms, write and explain the contrastive loss, and describe how zero-shot classification and retrieval work. Next week we build on these foundations: multimodal LLMs that chat about images and documents, generation across modalities including speech and video, how multimodal generation is evaluated, and responsible use including deepfakes and content provenance.

- A modality is a data type. Encoders turn inputs into vectors.

- Alignment matches corresponding content; fusion combines information.

- Shared vector spaces support cross-modal search and classification.

- CLIP learns which image and caption belong together, in both directions.

- Zero-shot image classification compares an image with class descriptions.

- SigLIP checks pairs separately; image and text vectors can still show a modality gap.

- A visual projector can connect an image encoder to an LLM; Whisper connects audio to text.

- Measure task accuracy, retrieval precision, speech word errors and performance across groups.

## Slide 40: Resources

The CLIP paper is long, but its first sections are very readable and its bias analysis is worth reading. SigLIP and Whisper are the other two primary sources for today. Hugging Face's blog post on vision language models is an excellent, current overview of architectures. The computer vision and audio courses from Hugging Face have hands-on units on exactly today's models. The Welch Labs video explains CLIP visually. Foster's chapter 13 covers multimodal models including DALL-E and Flamingo. See the source-verification report for the latest checks and access limitations.

- **Paper:** [Radford et al. (2021) CLIP](https://arxiv.org/abs/2103.00020). Learning Transferable Visual Models from Natural Language Supervision.
- **Paper:** [Zhai et al. (2023) SigLIP](https://arxiv.org/abs/2303.15343). Sigmoid loss for language-image pre-training.
- **Paper:** [Radford et al. (2022) Whisper](https://arxiv.org/abs/2212.04356). Robust speech recognition via large-scale weak supervision.
- **Reference:** [Hugging Face blog: Vision language models explained](https://huggingface.co/blog/vlms). Architectures and training.
- **Course:** [Hugging Face Community Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome). Multimodal units (CLIP, VLMs).
- **Course:** [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction). Spectrograms, Whisper, speech models.
- **Video:** [3Blue1Brown / Welch Labs: How AI images and videos work](https://www.youtube.com/watch?v=iv-5mZ_9CPY). Includes a clear CLIP explanation.
- **Book:** [Foster (2023) Generative Deep Learning, ch. 13 'Multimodal Models'](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/). Core module text.

