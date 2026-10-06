"""Authored foundations content. Regenerates only revised weeks 1-4."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'content' / 'revised'
OUT.mkdir(parents=True, exist_ok=True)

def concept(title, explanation, example, misconception, correction, question, answer):
    return dict(title=title, explanation=explanation, example=example,
                misconception=misconception, correction=correction,
                question=question, answer=answer)

weeks = []
weeks.append(dict(
    n=1, title='Generative AI foundations and responsible tool use', outcomes=['LO1','LO4'],
    overview='Learn what a generative model does, use a small text generator, and check the usefulness and accuracy of its output. The first practical task is deliberately small: explain every stage before relying on a large AI tool.',
    prerequisites='Bring basic Python list, dictionary and function knowledge. A supplied notebook introduces execution, seeds and simple percentages; no prior generative AI or GPU is assumed.',
    objectives=['Distinguish generation, prediction, retrieval and the surrounding AI application.', 'Calculate a simple conditional probability and explain training versus generation.', 'Compare outputs against a stated task and record one limitation and one improvement.'],
    concepts=[
        concept('What is generated?',
            'Generative AI learns patterns from examples and produces a sample such as a sentence, image, sound or video. A classifier may label a review positive or negative; a generator may write a reply to it. Both rely on prediction, so the distinction concerns the modelling task and the output. Retrieval is different again: a search system selects existing material. An AI assistant can combine a generative model with search, a calculator and an interface. Ask what each component contributes. The model proposes content; the application decides which information and tools it receives and what checks surround it. A fluent result can be useful without being a reliable statement of fact. This week treats every output as something to inspect.',
            'A student asks when the library closes. A search engine returns the opening-hours page. A classifier labels the question as a library query. A generative model writes a friendly reply. A useful assistant might retrieve the current page and then write a reply based on it. If the page is missing, the model can still write a convincing time; that is exactly the failure the class should notice.',
            'A generated answer is a retrieved fact.',
            'Generation proposes content. A source must be retrieved and checked separately when the task needs factual support.',
            'Which component supplies current library opening hours: the model weights or a current official page?',
            'The current official page provides evidence. The model may have learned old hours, but its fluent wording does not establish which hours apply today.'),
        concept('Probability and sampling',
            'A language model assigns probabilities to possible continuations given the preceding text. Write this as P(next token | earlier tokens). A token may be a word, part of a word or punctuation, depending on the tokenizer. A simple classroom model can count which word follows another and convert the counts into probabilities. A large model learns much richer relationships, but probability and sampling remain useful foundations. Choosing the highest-probability option is greedy decoding. Sampling selects an option according to a distribution and can produce different continuations. A random seed makes an experiment easier to repeat in a fixed environment. Probability here concerns model predictions, not the probability that a whole statement is true. A likely continuation can still be unsupported.',
            'After the word "please", a tiny training corpus contains "read" six times, "check" three times and "wait" once. The estimated probabilities are 0.6, 0.3 and 0.1. A random draw of 0.75 falls in the "check" interval from 0.6 to 0.9. Another draw can choose "read". These are invented teaching counts, not a performance claim about any commercial model.',
            'A token with probability 0.9 makes the answer 90% factually correct.',
            'Token probability describes the model distribution for a continuation. Factual correctness requires separate evidence and evaluation.',
            'If the counts are 4, 3 and 1, what is the probability of the first continuation?',
            'There are eight observations, so the probability is 4/8 = 0.5. It is an estimate from this tiny corpus, not a universal language rule.'),
        concept('Training, inference and model families',
            'Training changes a model using examples and a learning objective. Inference uses the learned model to produce an output. Changing a prompt normally changes the input, not the learned weights. Fine-tuning is additional training on a narrower dataset; it is different from inserting a document into a prompt. The module introduces four important families. A variational autoencoder learns a probabilistic latent representation and decoder. A generative adversarial network trains a generator with feedback from a discriminator. A diffusion model learns a denoising process. A transformer uses attention to combine information across token positions and can support language generation. These families are not four successive versions of the same tool. Different systems can combine components, and model choice depends on the task and available resources.',
            'Imagine generating simple icons. A VAE samples a latent code and decodes it. A GAN generator maps random input to an icon after adversarial training. A diffusion system generates through learned denoising steps. A language model can write a description or SVG code for an icon. All can contribute to a design workflow, but the representation and training objective differ.',
            'Every chat interaction retrains the underlying model.',
            'Ordinary inference uses existing weights. Product memory, conversation context and any provider training policy are separate mechanisms.',
            'A user pastes three examples into a prompt. Is that necessarily fine-tuning?',
            'No. It is in-context use of examples unless a separate training process changes learned parameters.'),
        concept('Useful outputs need a check',
            'Begin evaluation by stating what success means for the task. A summary may need correct facts, appropriate length and readable language. An extracted record may need the right fields, types and source values. Choose a few examples, write expected answers or a rubric, and compare every attempted output. Record the denominator: eight correct answers out of ten is more informative than "80% accurate" alone. Keep some cases separate for later checking so improvement does not become memorising the examples. Human review complements numeric measures, especially for misleading wording and harmful assumptions. Use public or fictional data in initial tool practice. Record the tool, date, prompt, settings and corrections so another learner can understand how the evidence was produced.',
            'A fictional note says: "The robotics club meets on Thursday at 16:00 in room B2." A generated summary says Tuesday at 16:00 in B2. Its grammar is excellent, but one of the three factual fields is wrong. A field checklist catches the day error. A readability-only score would miss it. The class should correct the summary using the note and save both versions.',
            'If a reply sounds confident and polite, it is ready to use.',
            'Style and correctness are separate criteria. Compare factual claims with the relevant evidence before use.',
            'Why should the test set contain a question the source cannot answer?',
            'It checks whether the system recognises missing evidence instead of inventing a plausible answer. A useful system may need to say it does not know.')
    ],
    demo=dict(title='Train and inspect a tiny next-word generator',steps=[
        'Read the short fictional training sentences and identify the words used as tokens.',
        'Run the supplied counting code and inspect a next-word probability table.',
        'Generate three short samples using recorded seeds; compare repetition and missing context.',
        'Change one training sentence, rebuild the counts and predict which continuation changes.',
        'Compare a generated reply with its source using a three-field correctness checklist.'
    ],expected='A working small statistical generator and a visible explanation of why its output changes. It is a learned count model, not an LLM or evidence of large-model capability.',offline='The count-based generator uses included text and runs locally; the optional browser-tool comparison can use an instructor-saved response with its provenance clearly labelled.'),
    practice=[
        dict(question='A system returns the exact paragraph containing a requested date. Is generation necessary?',answer='No. Retrieval may be sufficient. Generation can improve presentation, but adds a separate opportunity to change or invent a detail.'),
        dict(question='Two of ten summaries contain invented dates. Give a suitable metric and its limit.',answer='Date correctness is 8/10 = 80% on these ten examples. It does not measure style, all factual claims or future performance; describe the sample and failures.'),
        dict(question='Why is a next-word counting model a useful first lab despite its weak text?',answer='It exposes training data, conditional probabilities and sampling in a form learners can inspect. It illustrates a principle without claiming to reproduce a transformer LLM.')
    ],
    lab=dict(title='A small generator and an output evidence sheet',goal='Build a baseline understanding of generation, then judge an output against its task.',steps=[
        'Run the notebook from its first cell and record the environment and seed.',
        'Inspect the corpus, calculate one probability by hand and check the code agrees.',
        'Generate three outputs and label one useful pattern and one limitation.',
        'Change one corpus item or sampling setting and explain the observed effect.',
        'Complete a source-check table for a fictional summary and write a 100-word reflection.'
    ],deliverable='Executed notebook, one probability calculation, before/after outputs and a short checked-output table.'),
    references=['https://developers.google.com/machine-learning/crash-course/llm','https://developers.google.com/machine-learning/crash-course'],
    extension='Compare the count model with an institution-approved chat tool on the same harmless task. Identify which capabilities the small baseline lacks; do not rank tools from one example.',
    timing={'lecture':120,'lab':120}
))

weeks.append(dict(
    n=2, title='VAEs and GANs: learning to generate', outcomes=['LO1','LO3','LO4'],
    overview='Study two different ways to learn a generative model. Use simple numerical examples and a small implementation to connect latent variables, training objectives and generated samples.',
    prerequisites='Week 1 probability and training/inference vocabulary. Review mean, squared error and a basic neural-network forward pass using the supplied worked examples.',
    objectives=['Explain the encoder, sampled latent variable and decoder in a VAE.', 'Explain the generator/discriminator training loop in a GAN.', 'Modify a small implementation and compare training evidence, sample quality and diversity.'],
    concepts=[
        concept('A VAE learns a distribution of latent codes',
            'An ordinary autoencoder maps an input into a code and reconstructs it. A variational autoencoder instead learns parameters of a distribution over possible latent codes for each input. In a common example, the encoder produces a mean and variance for a Gaussian distribution. A sampled code z is passed through a decoder to reconstruct the input. Sampling allows nearby codes to produce varied outputs. To create a new sample after training, draw z from the chosen prior and decode it. The latent space is a learned mathematical representation; its axes do not automatically mean human concepts such as happiness or colour. Reconstruction helps us inspect learned information, while generation asks whether samples from the prior are useful and diverse.',
            'For one toy input, suppose the encoder gives mean 0.5 and standard deviation 0.2. With a noise draw epsilon = -1, the sampled latent value is z = 0.5 + 0.2 × (-1) = 0.3. The decoder maps 0.3 to an output. A different noise draw changes z while using the same encoder and decoder. These illustrative values make the sampling step visible.',
            'A VAE is just a file compressor that always returns the exact input.',
            'A VAE is a probabilistic generative model with a reconstruction objective. Information can be lost, and different latent samples can produce different reconstructions.',
            'For mean 1, standard deviation 0.5 and epsilon 2, what is z?',
            'z = 1 + 0.5 × 2 = 2. The sampled value is then passed to the decoder.'),
        concept('Reconstruction and regularisation',
            'A useful VAE must reconstruct examples while keeping its latent distributions organised enough to sample from. A common training loss has a reconstruction term plus a Kullback–Leibler divergence term, often written KL. Reconstruction penalises differences between input and output. KL measures how far the inferred latent distribution is from the chosen prior, typically a standard Gaussian. The standard objective can be understood as maximising an evidence lower bound, or equivalently minimising its negative. For classroom comparison we may weight KL by beta and inspect the trade-off. Too little pressure toward the prior can make sampling unreliable; too much can reduce useful information in the latent code. We inspect components separately rather than declaring that one smaller total loss proves better images.',
            'An illustrative run has reconstruction loss 0.12 and KL 0.08. With beta = 1, the weighted total is 0.20; with beta = 0.5 it is 0.16 for those same component values. This arithmetic does not predict the losses after retraining. Changing beta also changes the learned solution, so the experiment must generate and inspect fresh results.',
            'A lower reported VAE loss always means sharper or more useful samples.',
            'The loss depends on the objective, scaling and data. Inspect reconstruction, prior samples, coverage and the downstream task using a consistent comparison.',
            'Why record the two loss components instead of only their sum?',
            'Their sum can hide a trade-off. Good reconstruction with poor prior matching differs from strong regularisation with weak reconstruction, even when total values are similar.'),
        concept('A GAN learns through adversarial feedback',
            'A generative adversarial network contains a generator G and discriminator D. The generator maps a random latent input to a sample. The discriminator learns to distinguish training examples from generated samples. Training alternates: update the discriminator using real and generated data, then update the generator so its samples are more likely to be judged real. Gradients carry the feedback; the discriminator does not write editing instructions. In the original minimax formulation, the discriminator maximises the expected log score for real data plus the expected log score for rejecting generated data. A commonly used generator loss is -log D(G(z)). The exact loss must match the implementation. At inference, generating a sample normally needs the generator, not the discriminator.',
            'Suppose D gives one generated sample a realness score of 0.2. The non-saturating generator loss for this sample is -ln(0.2), approximately 1.61. If its score becomes 0.8, the loss is about 0.22. This illustrates the direction of the objective. It does not prove that a human will prefer the sample or that all modes of the training distribution are represented.',
            'The discriminator generates the final picture after checking it.',
            'The generator produces the sample. The discriminator provides a learned training signal, and ordinary generation uses G alone.',
            'Why do we alternate updates rather than treating the discriminator as a fixed answer key?',
            'The discriminator is also learned and must adapt as the generator changes. The two models influence each other, so training can be unstable.'),
        concept('Compare sample quality and coverage',
            'Generative evaluation needs more than a loss curve. A model can produce attractive samples while repeating the same kind of output. In a GAN this can appear as mode collapse: the generator covers only a narrow part of the data distribution. A VAE can reconstruct common shapes yet blur details or poorly represent rare examples. Compare samples with a held-out reference distribution and inspect several random seeds. For a tiny one-dimensional exercise, histograms, means and the fraction of samples in each region are understandable checks. For images, use a transparent human rubric and recognise that sophisticated metrics have assumptions. Record the data, implementation and training budget. A short CPU exercise teaches the mechanism, not a general ranking of the two model families.',
            'A fictional training set contains equal numbers of circles and triangles. Generator A returns ten excellent circles. Generator B returns five recognisable circles and five recognisable triangles with some rough edges. A sharpness-only rubric may favour A, but a coverage check exposes its missing triangles. The class should report both observations rather than collapsing them into an unexplained single score.',
            'One convincing image demonstrates that the generator learned the whole dataset.',
            'One sample shows only one outcome. Check diversity, missing modes and held-out evidence across multiple generated samples.',
            'What simple test detects a generator that always returns nearly the same value?',
            'Generate many samples from different latent inputs, inspect their spread or histogram and compare it with the reference distribution. Very low variation is a warning, not a full diagnosis.')
    ],
    demo=dict(title='Trace two real learning loops',steps=[
        'Draw the VAE encoder → distribution → sampled z → decoder path and the GAN latent z → generator path.',
        'Inspect the small notebook models, data split and losses; point to the actual trainable parameters.',
        'Run the short training examples and compare generated samples before and after learning.',
        'Change one permitted parameter, keeping the data and comparison method fixed.',
        'Explain one gain, one failure and why a favourable plot is not a proof of general quality.'
    ],expected='Actual parameter updates and samples from small models, accompanied by a loss explanation and a limited comparison. Exact numeric scores depend on the run.',offline='Use the local small-model notebook. Instructor checkpoints support explanation during an outage, but learners still complete the implementation and modification exercise when the environment is available.'),
    practice=[
        dict(question='Name the VAE component used when generating from a prior sample.',answer='The decoder. An encoder is needed to infer a latent distribution from an input, but new generation can start by sampling z from the prior.'),
        dict(question='A GAN has lower generator loss but less sample diversity. What should the report say?',answer='Report the lower loss and reduced diversity separately. The discriminator-based objective does not alone establish better overall generation.'),
        dict(question='Why should a learner change a real trainable model instead of only moving a point in a hand-drawn latent plot?',answer='The real model connects the claimed architecture to parameter learning and measurable outputs. A diagram is useful for explanation but does not demonstrate implementation.')
    ],
    lab=dict(title='Modify a small generative model',goal='Connect model components, parameter updates and evaluation using manageable data.',steps=[
        'Inspect the small dataset and mark which examples are reserved for checking.',
        'Execute the supplied VAE and GAN learning examples and save their outputs.',
        'Label where each implementation samples z and where its parameters change.',
        'Change one training or latent setting, rerun the chosen model and compare using the same samples or seeds where appropriate.',
        'Submit a two-column explanation of mechanism and evidence, including one failure case.'
    ],deliverable='Executed model notebook, an annotated architecture sketch, a controlled modification and a quality/coverage comparison.'),
    references=['https://arxiv.org/abs/1312.6114','https://arxiv.org/abs/1406.2661','https://www.tensorflow.org/tutorials/generative/cvae','https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'],
    extension='Follow the official image-based VAE or DCGAN tutorial with a permitted dataset and appropriate compute. Keep the original small experiment as the basis for a clear explanation.',
    timing={'lecture':120,'lab':120}
))

weeks.append(dict(
    n=3, title='Diffusion and generated media', outcomes=['LO1','LO2','LO4'],
    overview='Explain generation through learned denoising, distinguish a training example from a generation step, and use a clear rubric to judge generated media. Image generation is the central example; audio and video establish the wider landscape before the multimodal week.',
    prerequisites='Week 1 probability and Week 2 training objectives. Review mean squared error and the role of random noise.',
    objectives=['Explain the forward noising process and the learned reverse process.', 'Identify what a denoiser receives during training and during generation.', 'Compare media outputs using prompt adherence, quality, diversity and responsible-use criteria.'],
    concepts=[
        concept('Start by adding known noise',
            'Diffusion training begins with data and a specified process that adds noise. At a selected time step, the original sample is combined with Gaussian noise at a chosen strength. One common expression is x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon. Here x_0 is the clean training sample, epsilon is sampled noise, and alpha_bar_t controls how much original signal remains. The subscript t is a noise level, not the time a camera captured an image. During training we know the clean sample and the added noise, so we can construct learning targets. The forward process is not the learned act of image creation. It prepares noisy examples from which the network learns information needed for the reverse process.',
            'Take a single illustrative pixel x_0 = 0.8, alpha_bar_t = 0.64 and noise epsilon = -0.5. The noisy value is sqrt(0.64) × 0.8 + sqrt(0.36) × (-0.5) = 0.64 - 0.30 = 0.34. A real image has many pixels or latent values, each with sampled noise. The calculation explains one step without claiming to be an image generator.',
            'Diffusion training only shows the model fully random images with no targets.',
            'Training constructs noisy examples from known data and noise. The objective uses the chosen target, such as the added noise, to learn denoising behaviour.',
            'As alpha_bar_t approaches zero, which part increasingly dominates x_t?',
            'The noise term dominates because its coefficient approaches one while the clean-signal coefficient approaches zero.'),
        concept('Learn a reverse process',
            'A denoising network is trained to predict a quantity that helps recover structure from a noisy sample, often the added noise. Its inputs include the noisy sample and the noise level; conditional models also receive information such as a text representation. A noise-prediction objective can use mean squared error between the actual added noise and the prediction. At generation time, begin from random noise and repeatedly apply the learned prediction with a scheduler. The scheduler determines the update rules across steps. Crucially, the original clean target is unavailable for a new generated sample. A classroom animation that blends a noisy image back toward its known original illustrates restoration, but it is not a trained generative reverse process. The practical work should make that boundary visible.',
            'Suppose the known training noise vector is [0.4, -0.2] and the network predicts [0.1, -0.1]. The squared errors are 0.09 and 0.01, so mean squared error is 0.05. Training adjusts parameters to reduce this error over many examples. Later generation must use the learned prediction alone; supplying the original noise vector at inference would give the algorithm information it should not have.',
            'During generation the model compares every step with the exact image it is trying to reveal.',
            'A new sample has no hidden clean target supplied to the model. Generation uses learned parameters, current noisy state, time information and any conditioning.',
            'What is wrong with a claimed diffusion generator that repeatedly moves toward an already stored clean image?',
            'It uses the target as an oracle and demonstrates interpolation or restoration, not learned generation from an unknown target.'),
        concept('Conditioning and controllable generation',
            'Conditioning gives a generative system information about the desired output. A text-to-image pipeline may encode a prompt, use that representation during denoising and decode an image from a learned latent space. A latent diffusion system performs much of its denoising in a compressed representation rather than directly in full-resolution pixels. Controls can include the initial random seed, number of inference steps and guidance strength, where supported. These controls interact with the particular model and scheduler. More steps or stronger guidance does not universally improve a result. Compare one change at a time using the same task and a fixed rubric. Retain the model and settings with each output so classmates can understand the experiment without assuming another platform has equivalent controls.',
            'The brief is "a blue paper boat on a plain white background". Generate two outputs while changing only the number of inference steps, keeping the model and seed fixed where the tool allows it. Check colour, object, background and visible artifacts separately. If a tool does not expose seeds, record that limitation and use several samples instead of claiming a tightly controlled comparison.',
            'A stronger guidance setting guarantees every instruction will be followed.',
            'Guidance changes the sampling behaviour and can introduce trade-offs. Measure adherence and visual quality on actual outputs.',
            'Why record the scheduler as well as the number of steps?',
            'The scheduler determines the numerical update procedure. Equal step counts with different schedulers need not mean equivalent computation or results.'),
        concept('Images, audio and video need different checks',
            'Generated media covers more than attractive pictures. Image tasks include generation, editing and inpainting; audio tasks include speech and music; video tasks add change over time. Many architectures are used, so not every media generator is a diffusion model. Evaluation must match the task. For an image, inspect object count, lettering and composition. For speech, check the transcript, intelligibility and timing. For video, look for inconsistent objects and actions across frames. Also ask whether the input data and requested likeness are appropriate for the exercise, and disclose synthetic content when relevant. A generated portrait is not evidence that a person attended an event. Keep responsible-use discussion connected to a concrete output rather than treating visual realism as proof of truth.',
            'A fictional event campaign includes a poster, a spoken invitation and a five-second clip. The poster shows the wrong date, the speech mispronounces the venue and the video changes the number of chairs between frames. All three may look or sound polished. Separate checklists expose different failures, and a verified text brief acts as the shared reference for the whole campaign.',
            'Realistic media is reliable evidence of a real event.',
            'Generation can create plausible scenes that never happened. Verify provenance and factual claims independently of perceptual quality.',
            'What extra criterion does video need beyond judging isolated still images?',
            'Temporal consistency: objects, identities, counts, motion and actions should remain coherent across frames for the intended task.')
    ],
    demo=dict(title='From noisy training examples to learned denoising',steps=[
        'Show the forward-noise equation with the one-pixel example, then a small array.',
        'Inspect the small denoiser training inputs and targets; identify the actual loss.',
        'Train the supplied small model and compare a learned prediction with its held-out target.',
        'Inspect a generation or sampling demonstration and confirm the clean target is not supplied at inference.',
        'Review one generated-media output against the brief and separate technical quality from factual support.'
    ],expected='A numerical noise calculation, evidence from an actual learned denoiser and a clear statement of what the small demonstration can and cannot generate.',offline='The small numerical experiment runs locally. A documented instructor output can support the media review when a hosted image tool is unavailable; label the output as a saved example.'),
    practice=[
        dict(question='Why does the model receive the noise level t?',answer='The amount of corruption changes across steps. Time or noise-level information helps the network choose a prediction appropriate to the current state.'),
        dict(question='A classroom code cell subtracts the exact noise it added. Has it learned to denoise?',answer='No. It reverses a known calculation using privileged information. A learned denoiser must predict from allowed inputs and be checked on separate examples.'),
        dict(question='A poster is visually excellent but invents the event date. How should it be graded?',answer='Record visual quality and date correctness separately. It fails a factual requirement and must be corrected before use.')
    ],
    lab=dict(title='Denoising evidence and a media review',goal='Understand the training mechanism and judge media with explicit criteria.',steps=[
        'Calculate one noisy value by hand and verify the notebook result.',
        'Run the learned small-denoiser experiment and identify its parameters, training inputs and target.',
        'Change one noise or training setting and compare held-out error using the same cases.',
        'Inspect the sampling demonstration and state whether it is unconditional, conditioned or only a denoising component.',
        'Complete image/audio/video review examples and write one limitation for each modality.'
    ],deliverable='Executed notebook, the noise calculation, a controlled denoiser comparison and a media-specific review table.'),
    references=['https://arxiv.org/abs/2006.11239','https://huggingface.co/docs/diffusers/en/using-diffusers/unconditional_image_generation','https://huggingface.co/docs/diffusers/en/quicktour'],
    extension='Use an approved pretrained diffusion pipeline to compare two settings on the same harmless brief. Record hardware, model licence, runtime and any access requirements; do not assume the optional pipeline runs on every laptop.',
    timing={'lecture':120,'lab':120}
))

weeks.append(dict(
    n=4, title='Transformers and large language models', outcomes=['LO1','LO3'],
    overview='Connect tokens, embeddings, attention and next-token generation without hiding the essential mathematics. Use small arrays to inspect the mechanism and a small generator to see why a full language model needs more than an attention grid.',
    prerequisites='Week 1 conditional probability and Week 2 neural-network training. The worksheet refreshes dot products, weighted averages and normalisation.',
    objectives=['Trace text through tokenisation, embeddings and a transformer block.', 'Calculate simple attention weights and apply a causal mask.', 'Distinguish encoder, decoder and encoder-decoder roles, and explain the limits of fluent generation.'],
    concepts=[
        concept('Tokens, embeddings and position',
            'A model first represents text in a form that a neural network can process. A tokenizer splits text into tokens and maps them to vocabulary IDs. Tokens can be whole words, subword pieces or other units; token count is not generally word count. A learned embedding maps each ID to a vector. The vector is not a dictionary definition, but a set of learned numbers used by the model. Position information is also needed because a sequence is more than a bag of words. Different transformer designs encode position differently. A context window limits what can be processed in a request, and the product may reserve part of that capacity for output or other context. Do not infer exact product limits from a toy notebook.',
            'The sentences "the dog chased the cat" and "the cat chased the dog" contain the same word types but describe different events. A bag of word counts loses the roles. Ordered tokens and position information let a model distinguish their arrangements. In a tiny teaching vocabulary, a word can have an ID; a real subword tokenizer may split the same word into several tokens.',
            'Every English word is one token, so words and tokens are interchangeable units.',
            'Tokenisation depends on the tokenizer, spelling and language. Measure with the actual tokenizer when a real system has a token budget.',
            'Why is an embedding ID different from the embedding vector?',
            'The ID selects an entry in a learned table; the vector is the numerical representation used in computation. Adjacent IDs do not necessarily represent similar meanings.'),
        concept('Attention is a weighted information mix',
            'Self-attention lets a token combine information from permitted positions in the same sequence. Learned projections produce queries Q, keys K and values V. Query-key similarities provide scores; dividing by the square root of the key dimension controls their scale. Softmax turns the scores into positive weights that sum to one. Multiplying those weights by V produces a weighted mixture. The standard expression is Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Start with one query and a few scalar scores before looking at the matrix form. Multiple heads learn different projections and combine their outputs. Attention is one part of a transformer block; feed-forward layers, residual connections and normalisation also matter. A heatmap shows a computation, not a complete causal explanation of a model answer.',
            'For illustrative scaled scores [1, 2, 0], softmax gives approximately [0.245, 0.665, 0.090]. With scalar values [10, 20, 30], the output is 0.245 × 10 + 0.665 × 20 + 0.090 × 30 ≈ 18.45. The largest weight belongs to the middle value, but the result combines all three. These supplied scores demonstrate the calculation rather than a learned linguistic relationship.',
            'The largest attention weight reveals the model\'s true reason for its final answer.',
            'It identifies a strong contribution within one attention calculation. The full model has many layers and interactions, so the weight alone is not a faithful explanation.',
            'If attention weights are [0.25, 0.75] and values are [4, 8], what is the mixed output?',
            '0.25 × 4 + 0.75 × 8 = 7. It is a weighted mixture, not simply a copy of the highest-weight value.'),
        concept('Causal generation and model roles',
            'An autoregressive decoder predicts the next token from earlier tokens. During training it must not look at the future token it is supposed to predict. A causal attention mask blocks those future positions before softmax, producing zero attention weight there. At inference, the model generates a token, adds it to the context and repeats until it stops or reaches a limit. Encoder models usually build contextual representations using both sides of an input; BERT is a familiar masked-language encoder example, not a drop-in chat generator. Encoder-decoder models first represent an input and then generate an output conditioned on it, as in translation. These roles organise the concepts, while individual modern systems can combine architectures and training objectives in different ways.',
            'A decoder is learning to continue "the library opens". When predicting the next token, it may use "the", "library" and "opens", but it must not see the correct answer already written later in the sequence. For scores [1, 2, 0], masking the last position gives weights approximately [0.269, 0.731, 0]. The remaining weights are renormalised rather than left unchanged.',
            'A decoder mask hides future tokens only during the final demonstration, so training can use them.',
            'The causal constraint matters during training too; otherwise the model could exploit the answer it is meant to predict.',
            'Why is comparing BERT and a chat decoder on free conversation without task adaptation misleading?',
            'Their standard architectures and objectives support different interfaces and tasks. Compare appropriately configured systems on a stated task rather than assuming all language models are interchangeable chatbots.'),
        concept('From language modelling to an assistant',
            'A transformer architecture becomes useful through data, training and an application around it. Pretraining learns broad regularities using a prediction objective. Further instruction training and preference-based methods can shape response behaviour. At use time, prompts, supplied documents, retrieval and tools influence the answer without necessarily changing weights. Sampling settings control how tokens are selected; lower temperature usually concentrates probability on more likely options, but does not guarantee factual correctness. A model may generate a persuasive solution or explanation that contains an error. Evaluate the answer and its evidence, not the length of its reasoning text. A small classroom generator is valuable when learners can identify what was actually trained and changed. It should not be presented as matching a large production assistant.',
            'A model writes "7 × 8 = 54" in an otherwise clear explanation. The text can be checked by a simple independent multiplication, giving 56. Asking for a longer explanation may produce more fluent text without fixing the arithmetic. A well-designed application can route the multiplication to a calculator and then explain the checked result, connecting model output with a more appropriate tool.',
            'Low temperature and a long step-by-step answer make a response trustworthy.',
            'They affect output behaviour and presentation. Verify the result with task-appropriate evidence, tests or tools; explanatory text is not guaranteed to expose internal reasoning.',
            'What evidence would show that a notebook changed model parameters rather than only changed the prompt?',
            'A training step with gradients or another update method, changed parameter values and a controlled before/after evaluation. A new prompt alone does not establish training.')
    ],
    demo=dict(title='Compute attention and inspect generation',steps=[
        'Tokenise two short sentences and inspect IDs, vector shapes and order.',
        'Calculate one attention row from supplied Q, K and V values.',
        'Apply a causal mask and verify forbidden weights are zero and each allowed row sums to one.',
        'Run the supplied small text-generation example and distinguish its learned components from the illustrative arithmetic.',
        'Change a permitted input or setting, compare outputs and identify one error that needs an independent check.'
    ],expected='Correct attention arithmetic, a visible causal-mask effect and a reasoned explanation of the small generation model\'s capabilities and limits.',offline='The supplied arrays and small model are local. An optional large-model example can be replaced by a dated saved response for discussion, without relabelling the replay as a new inference run.'),
    practice=[
        dict(question='Why is softmax applied after the causal mask?',answer='Disallowed scores are made effectively negative infinity before normalisation so their probabilities become zero and the allowed positions are renormalised.'),
        dict(question='Which simplified architecture role suits reading an input sentence and writing its translation?',answer='An encoder-decoder: the encoder represents the source and the decoder generates a translation conditioned on that representation.'),
        dict(question='An attention worksheet works correctly. Does it implement a complete transformer LLM?',answer='No. A complete model also needs learned embeddings and projections, multiple layers and supporting components, an output distribution, training and decoding. The worksheet isolates one mechanism.')
    ],
    lab=dict(title='Attention, masks and a small language generator',goal='Use mathematics and code to explain the path from context to a generated token.',steps=[
        'Run the token and embedding examples; label each array dimension.',
        'Compute a dot product and softmax row by hand, then compare with the notebook.',
        'Apply a causal mask and test that no future-position weight remains.',
        'Run and modify the supplied small generation example, recording what is learned and what is fixed.',
        'Submit a model-role comparison and a checked explanation of one generated error.'
    ],deliverable='Executed notebook, one worked attention calculation, mask checks and a short model/application comparison.'),
    references=['https://arxiv.org/abs/1706.03762','https://developers.google.com/machine-learning/crash-course/llm/transformers','https://developers.google.com/machine-learning/crash-course/llm'],
    extension='Trace a full small transformer block including feed-forward and residual paths. Compare tokenisation for two languages using one documented tokenizer without treating token counts as a measure of language complexity.',
    timing={'lecture':120,'lab':120}
))

for week in weeks:
    (OUT / f"week_{week['n']:02}.json").write_text(json.dumps(week, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"W{week['n']}: {len(week['concepts'])} concepts; explanations " + ', '.join(str(len(c['explanation'].split())) for c in week['concepts']))
