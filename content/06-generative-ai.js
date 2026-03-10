// 06 - Generative AI
(function() {
  const content = {

    // ============================================================
    // GENERATIVE MODELS
    // ============================================================

    gans: `# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs), introduced by Ian Goodfellow in 2014, are a class of generative models that learn to produce realistic data through a competitive two-player game. A **generator** network creates synthetic samples while a **discriminator** network attempts to distinguish real data from generated fakes. This adversarial dynamic pushes both networks to improve, ultimately producing a generator capable of creating highly realistic outputs. GANs revolutionized image synthesis and remain foundational to generative AI despite newer alternatives.

## Key Concepts

The GAN framework rests on a **minimax game** between two networks:

\`\`\`
  Random Noise (z)
       |
       v
  [ Generator G ] ----> Generated Image
       |                      |
       |                      v
       |              [ Discriminator D ] ---> Real or Fake?
       |                      ^
       |                      |
  Real Images ----------------/
\`\`\`

- **Generator (G):** Maps random noise z ~ N(0,1) to data space, learning the data distribution
- **Discriminator (D):** Binary classifier that outputs P(real | input)
- **Adversarial Loss:** min_G max_D [ E[log D(x)] + E[log(1 - D(G(z)))] ]
- **Nash Equilibrium:** Ideal convergence where G produces perfect samples and D outputs 0.5 for all inputs

## How It Works

Training alternates between updating D and G. The discriminator maximizes its classification accuracy, while the generator minimizes the discriminator's ability to detect fakes. **Mode collapse** occurs when G produces limited variety, and **training instability** arises from the delicate balance between the two networks.

| GAN Variant | Innovation | Best For |
|-------------|-----------|----------|
| **DCGAN** | Convolutional architecture with batch norm | Stable image generation |
| **StyleGAN/StyleGAN3** | Style-based mapping network, alias-free ops | High-res face synthesis |
| **CycleGAN** | Unpaired image-to-image via cycle consistency | Domain transfer (horse to zebra) |
| **Pix2Pix** | Paired image-to-image with conditional GAN | Supervised translation (sketch to photo) |
| **ProGAN** | Progressive growing from low to high resolution | Training stability at high res |
| **WGAN-GP** | Wasserstein distance with gradient penalty | Stable training, meaningful loss |

## Applications

- **Face generation and editing:** StyleGAN produces photorealistic faces with fine-grained style control
- **Image-to-image translation:** Converting satellite images to maps, sketches to photos, day to night scenes
- **Data augmentation:** Generating synthetic training data for medical imaging and rare-event detection
- **Super-resolution:** SRGAN and ESRGAN upscale low-resolution images with perceptual quality

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Goodfellow introduces GANs |
| 2015 | DCGAN establishes convolutional GAN architecture |
| 2017 | Pix2Pix and CycleGAN enable image translation |
| 2017 | WGAN introduces Wasserstein distance for stable training |
| 2018 | ProGAN and BigGAN scale to high-resolution generation |
| 2019 | StyleGAN achieves photorealistic face synthesis |
| 2021 | StyleGAN3 eliminates aliasing artifacts |
| 2022+ | Diffusion models begin surpassing GANs in image quality benchmarks |`,

    vaes: `# Variational Autoencoders (VAEs)

Variational Autoencoders (VAEs), introduced by Kingma and Welling in 2013, are probabilistic generative models that learn a structured latent representation of data. Unlike standard autoencoders that produce deterministic encodings, VAEs map inputs to a **probability distribution** in latent space, enabling both meaningful representation learning and generation of new samples. VAEs provide a principled probabilistic framework grounded in Bayesian inference, making them foundational to modern generative modeling and latent space methods used in diffusion models.

## Key Concepts

VAEs combine deep learning with variational Bayesian inference:

\`\`\`
  Input x ---> [ Encoder q(z|x) ] ---> mu, sigma ---> z (sampled) ---> [ Decoder p(x|z) ] ---> Reconstructed x'
                                            |
                                  Reparameterization:
                                  z = mu + sigma * epsilon
                                  epsilon ~ N(0, 1)
\`\`\`

- **Encoder q(z|x):** Approximates the posterior distribution, outputting mean (mu) and variance (sigma)
- **Decoder p(x|z):** Reconstructs data from sampled latent codes
- **Latent Space:** Continuous, structured space where similar data points cluster together
- **Prior p(z):** Typically a standard Gaussian N(0, I) that regularizes the latent space

## How It Works

The VAE optimizes the **Evidence Lower Bound (ELBO):**

**ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))**

The first term is the **reconstruction loss** (how well the decoder recovers the input), and the second is the **KL divergence** (how close the learned posterior is to the prior). The **reparameterization trick** enables backpropagation through the sampling step by expressing z = mu + sigma * epsilon where epsilon ~ N(0,1).

| VAE Variant | Innovation | Benefit |
|-------------|-----------|---------|
| **Beta-VAE** | Weighted KL term (beta > 1) | Disentangled latent factors |
| **VQ-VAE** | Discrete latent codes via vector quantization | Sharper outputs, used in DALL-E |
| **VQ-VAE-2** | Hierarchical discrete codes | High-fidelity image generation |
| **NVAE** | Deep hierarchical architecture | State-of-the-art VAE image quality |
| **CVAE** | Conditional generation with labels | Controlled generation by class |

## Applications

- **Latent space exploration:** Smooth interpolation between data points for creative exploration
- **Anomaly detection:** Measuring reconstruction error to identify out-of-distribution samples
- **Drug discovery:** Generating novel molecular structures by sampling the latent chemical space
- **Representation learning:** Disentangled features for downstream tasks and transfer learning

## Evolution

| Year | Milestone |
|------|-----------|
| 2013 | Kingma & Welling introduce VAEs |
| 2014 | Conditional VAEs (CVAEs) enable label-conditioned generation |
| 2017 | Beta-VAE achieves disentangled representations |
| 2017 | VQ-VAE introduces discrete latent codes |
| 2019 | VQ-VAE-2 achieves high-fidelity hierarchical image generation |
| 2020 | NVAE sets new benchmarks for VAE image quality |
| 2021 | VQ-VAE concepts underpin DALL-E's discrete image tokens |
| 2022 | Latent diffusion models build on VAE encoder-decoder architecture |`,

    diffusion_models: `# Diffusion Models

Diffusion models are a class of generative models that learn to produce data by reversing a gradual noising process. Introduced formally as **Denoising Diffusion Probabilistic Models (DDPM)** by Ho et al. in 2020, they have become the dominant paradigm for image generation, surpassing GANs in quality and diversity. The core idea is elegantly simple: systematically destroy data by adding noise step by step, then train a neural network to reverse each step. This framework underpins DALL-E 2, Stable Diffusion, Midjourney, and Sora.

## Key Concepts

\`\`\`
Forward Process (Fixed):
  x_0 -----> x_1 -----> x_2 -----> ... -----> x_T
  (clean)   (+noise)   (+noise)            (pure noise)

Reverse Process (Learned):
  x_T -----> x_{T-1} --> x_{T-2} --> ... -----> x_0
  (noise)   (denoise)   (denoise)            (generated image)
\`\`\`

- **Forward Process:** Gradually adds Gaussian noise over T steps following a noise schedule: q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
- **Reverse Process:** A neural network (typically a U-Net) predicts the noise to remove at each step
- **Score Matching:** Alternative formulation where the model learns the gradient of the log-density (score function)
- **Noise Schedule:** Controls the rate of noise addition; linear, cosine, and learned schedules affect generation quality

## How It Works

The model is trained to predict the noise epsilon added at each timestep. The **simplified training objective** is:

**L = E[ || epsilon - epsilon_theta(x_t, t) ||^2 ]**

At inference, starting from pure Gaussian noise x_T, the model iteratively denoises to produce a clean sample. **DDIM** (Denoising Diffusion Implicit Models) accelerates this by allowing deterministic, fewer-step sampling.

| Technique | Steps | Quality | Speed |
|-----------|-------|---------|-------|
| **DDPM** | 1000 | Excellent | Slow |
| **DDIM** | 50-100 | Excellent | Moderate |
| **DPM-Solver** | 10-20 | Very Good | Fast |
| **Consistency Models** | 1-2 | Good | Very Fast |
| **Rectified Flow** | 1-5 | Very Good | Fast |

## Applications

- **Text-to-image synthesis:** DALL-E 2, Stable Diffusion, Midjourney, Imagen
- **Video generation:** Sora, Runway Gen-2, Stable Video Diffusion
- **Audio synthesis:** AudioLDM, Riffusion, Stable Audio
- **3D generation:** Point-E, DreamFusion, Zero-1-to-3 for 3D asset creation

## Evolution

| Year | Milestone |
|------|-----------|
| 2015 | Sohl-Dickstein introduces diffusion-based generative models |
| 2020 | Ho et al. publish DDPM, making diffusion practical |
| 2021 | Dhariwal & Nichol show diffusion beats GANs on ImageNet |
| 2021 | DDIM enables faster deterministic sampling |
| 2022 | Latent diffusion and Stable Diffusion democratize generation |
| 2023 | Consistency models enable single-step generation |
| 2024 | Rectified flow models (Flux, SD3) improve efficiency and quality |
| 2025 | Diffusion transformers (DiT) unify diffusion with Transformer backbones |`,

    flow_models: `# Normalizing Flow Models

Normalizing flows are generative models that transform a simple base distribution (like a Gaussian) into a complex data distribution through a sequence of **invertible, differentiable transformations**. Unlike GANs or VAEs, flows provide **exact likelihood computation** because the density of the transformed distribution can be calculated analytically via the change of variables formula. This mathematical elegance makes flows uniquely valuable for density estimation, and their invertibility enables both efficient generation and exact inference.

## Key Concepts

\`\`\`
Base Distribution z ~ N(0, I)
       |
       v
  [ f1: Invertible Transform ]
       |
       v
  [ f2: Invertible Transform ]
       |
       v
  [ fK: Invertible Transform ]
       |
       v
  Generated Data x = fK(...f2(f1(z)))
\`\`\`

- **Change of Variables:** log p(x) = log p(z) - sum of log |det(df_i/dz_i)| for each transform
- **Invertibility:** Each transform must be bijective, allowing exact mapping between data and latent space
- **Jacobian Determinant:** The key computational challenge; efficient architectures use triangular Jacobians
- **Coupling Layers:** Split dimensions and transform one half conditioned on the other, yielding tractable Jacobians

## How It Works

The model chains K invertible layers. Each layer must allow efficient computation of both the forward pass (generation) and the log-determinant of its Jacobian (for training via maximum likelihood).

| Flow Model | Key Innovation | Jacobian Strategy |
|------------|---------------|-------------------|
| **NICE** | Additive coupling layers | Volume-preserving (det = 1) |
| **RealNVP** | Affine coupling with scale and shift | Triangular Jacobian (efficient) |
| **Glow** | 1x1 invertible convolutions, actnorm | LU decomposition for det |
| **Neural Spline Flows** | Monotonic rational-quadratic splines | Flexible, expressive transforms |
| **Continuous Flows (FFJORD)** | ODE-based continuous transforms | Hutchinson trace estimator |

**Affine coupling** (RealNVP): Given input split into (x_a, x_b), output is (x_a, x_b * exp(s(x_a)) + t(x_a)) where s and t are arbitrary neural networks. The Jacobian is triangular, making the determinant a simple product of exp(s) values.

## Applications

- **Density estimation:** Exact log-likelihood for anomaly detection and model comparison
- **Variational inference:** Flows as flexible posterior approximations in VAEs (improving ELBO)
- **Audio synthesis:** WaveGlow uses flows for real-time speech synthesis
- **Physics simulations:** Boltzmann generators for molecular dynamics and lattice field theory

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | NICE introduces additive coupling layers |
| 2016 | RealNVP adds affine coupling for expressiveness |
| 2018 | Glow achieves high-resolution face synthesis with invertible 1x1 convs |
| 2019 | Neural Spline Flows improve flexibility with spline-based transforms |
| 2019 | FFJORD introduces continuous normalizing flows via neural ODEs |
| 2023 | Flow matching and rectified flows revive interest for fast diffusion sampling |
| 2024 | Flux (Black Forest Labs) uses rectified flow for state-of-the-art image generation |`,

    autoregressive: `# Autoregressive Models

Autoregressive models generate data one element at a time, conditioning each new element on all previously generated elements. They decompose the joint probability of a sequence into a product of conditional probabilities: **P(x) = P(x_1) * P(x_2|x_1) * P(x_3|x_1,x_2) * ...**. This simple factorization, combined with powerful neural architectures, underpins the most successful generative models in history, including GPT, WaveNet, and PixelCNN. The autoregressive principle is the foundation of modern large language models.

## Key Concepts

\`\`\`
Autoregressive Generation (Left to Right):

  Step 1: P(x_1)               ---> Generate x_1
  Step 2: P(x_2 | x_1)         ---> Generate x_2
  Step 3: P(x_3 | x_1, x_2)    ---> Generate x_3
  ...
  Step N: P(x_N | x_1, ..., x_{N-1}) ---> Generate x_N
\`\`\`

- **Causal Masking:** Prevents the model from seeing future tokens during training (used in GPT's masked self-attention)
- **Teacher Forcing:** During training, uses ground-truth previous tokens rather than model predictions
- **Exact Likelihood:** Provides tractable log-likelihood via the chain rule of probability
- **Sequential Generation:** Inherently sequential at inference time, one token at a time

## How It Works

Training is highly parallelizable via teacher forcing: all positions can be computed simultaneously since ground-truth context is available. However, **generation is inherently sequential**, creating a speed bottleneck. Each step samples from the predicted conditional distribution.

| Model | Domain | Architecture | Key Contribution |
|-------|--------|-------------|------------------|
| **PixelCNN/PixelCNN++** | Images | Masked convolutions | Pixel-by-pixel image generation |
| **WaveNet** | Audio | Dilated causal convolutions | Raw audio waveform synthesis |
| **GPT-1/2/3/4** | Text | Transformer decoder with causal mask | Scaled language modeling |
| **ImageGPT** | Images | Transformer on pixel sequences | Unified image generation as sequence |
| **NADE** | Tabular | Orderless autoregressive | Density estimation on arbitrary orderings |

**Sampling strategies** control generation diversity and quality:

| Strategy | Description | Effect |
|----------|------------|--------|
| **Greedy** | Always pick highest probability token | Deterministic, repetitive |
| **Temperature** | Scale logits by T before softmax | T<1: focused; T>1: diverse |
| **Top-k** | Sample from k most likely tokens | Limits unlikely choices |
| **Top-p (Nucleus)** | Sample from smallest set with cumulative P >= p | Adaptive vocabulary size |
| **Min-p** | Filter tokens below min_p * max_probability | Dynamic threshold filtering |

## Applications

- **Large language models:** GPT-4, Claude, Llama, and all modern LLMs use autoregressive generation
- **Speech synthesis:** WaveNet powers Google's text-to-speech with natural prosody
- **Image generation:** PixelCNN family and ImageGPT generate images pixel by pixel or patch by patch
- **Code completion:** Copilot, CodeLlama, and StarCoder autoregressively predict code tokens

## Evolution

| Year | Milestone |
|------|-----------|
| 2016 | WaveNet generates raw audio autoregressively |
| 2016 | PixelCNN generates images pixel by pixel |
| 2018 | GPT-1 demonstrates autoregressive pretraining for NLP |
| 2019 | GPT-2 scales to 1.5B parameters with strong zero-shot abilities |
| 2020 | GPT-3 (175B) shows in-context learning and few-shot capabilities |
| 2023 | GPT-4 achieves human-level performance on many benchmarks |
| 2024 | Autoregressive image models (VAR, MAR) challenge diffusion approaches |
| 2025 | Hybrid autoregressive-diffusion architectures emerge for multimodal generation |`,

    // ============================================================
    // GENAI APPLICATIONS
    // ============================================================

    text_generation: `# Text Generation with LLMs

Text generation is the process of producing coherent natural language sequences using large language models (LLMs). Modern LLMs are autoregressive Transformers trained on vast text corpora, capable of generating everything from conversational responses to essays, stories, code, and structured data. The quality of generated text depends critically on the model architecture, training data, alignment techniques, and **sampling strategies** that control the tradeoff between creativity and coherence.

## Key Concepts

\`\`\`
Prompt: "The future of AI is"
         |
         v
  [ LLM (Transformer Decoder) ]
         |
         v
  Logits over vocabulary (50k+ tokens)
         |
         v
  [ Sampling Strategy ] ---> Next token: "bright"
         |
         v
  Append and repeat: "The future of AI is bright"
\`\`\`

- **Tokenization:** Text is split into subword tokens (BPE, SentencePiece, tiktoken) before processing
- **Context Window:** Maximum sequence length the model can attend to (4K to 1M+ tokens)
- **KV Cache:** Cached key-value pairs from previous tokens to avoid redundant computation during generation
- **Logits:** Raw unnormalized scores over the vocabulary before softmax

## How It Works

At each step, the model outputs a probability distribution over the vocabulary. **Sampling strategies** shape this distribution to control output quality:

| Parameter | Range | Low Value Effect | High Value Effect |
|-----------|-------|-----------------|-------------------|
| **Temperature** | 0.0-2.0 | Deterministic, focused | Creative, random |
| **Top-k** | 1-100 | Conservative | Diverse |
| **Top-p (Nucleus)** | 0.0-1.0 | Narrow selection | Broad selection |
| **Frequency Penalty** | -2.0-2.0 | Allows repetition | Discourages repetition |
| **Presence Penalty** | -2.0-2.0 | Allows topic revisits | Encourages new topics |
| **Min-p** | 0.0-1.0 | No filtering | Aggressive low-prob filtering |

**Structured generation** constrains outputs to valid formats:
- **JSON mode:** Forces valid JSON output structure
- **Grammar-constrained decoding:** Uses context-free grammars to restrict token choices
- **Function calling:** Model outputs structured tool invocations

## Applications

- **Conversational AI:** ChatGPT, Claude, Gemini for dialogue and assistance
- **Content creation:** Blog posts, marketing copy, email drafting, summarization
- **Reasoning:** Chain-of-thought prompting, tree-of-thought for complex problem solving
- **Agentic workflows:** LLM-powered agents that plan, use tools, and execute multi-step tasks

## Evolution

| Year | Milestone |
|------|-----------|
| 2018 | GPT-1 demonstrates generative pretraining for language |
| 2019 | GPT-2 shows emergent zero-shot capabilities at scale |
| 2020 | GPT-3 enables few-shot learning via in-context examples |
| 2022 | ChatGPT (RLHF-aligned GPT-3.5) launches consumer AI era |
| 2023 | GPT-4, Claude 2, Llama 2 advance reasoning and safety |
| 2024 | Claude 3.5, GPT-4o, Llama 3 push multimodal and efficiency frontiers |
| 2025 | Extended thinking, tool use, and agentic capabilities become standard |`,

    image_synthesis: `# Image Synthesis

AI image synthesis encompasses techniques for generating, modifying, and enhancing images using neural networks. From creating entirely new images from text descriptions to filling in missing regions, extending image boundaries, and upscaling resolution, modern image synthesis systems combine diffusion models, GANs, and Transformer architectures to achieve photorealistic results. This field has transformed creative workflows in art, design, advertising, film, and gaming.

## Key Concepts

\`\`\`
Text-to-Image Pipeline:
  "A cat astronaut on Mars" ---> [ Text Encoder (CLIP/T5) ]
                                         |
                                         v
                               [ Diffusion U-Net / DiT ]
                                         |
                                         v
                               [ VAE Decoder ] ---> Generated Image
\`\`\`

- **Text-to-Image:** Generating images from natural language descriptions using CLIP-guided diffusion
- **Image Inpainting:** Filling in masked or missing regions of an image while maintaining context coherence
- **Outpainting:** Extending an image beyond its original boundaries with plausible new content
- **Super-Resolution:** Upscaling low-resolution images to higher resolution with added detail
- **Image Editing:** Modifying specific aspects of existing images via text instructions

## How It Works

| Task | Technique | Key Models |
|------|-----------|-----------|
| **Text-to-Image** | Conditional diffusion with text embedding | DALL-E 3, Midjourney, Flux |
| **Inpainting** | Masked diffusion with context conditioning | Stable Diffusion Inpaint, Adobe Firefly |
| **Outpainting** | Extended canvas with boundary-aware generation | DALL-E 2, Photoshop Generative Expand |
| **Super-Resolution** | Diffusion-based upscaling or GAN-based (ESRGAN) | Stable Diffusion Upscaler, Real-ESRGAN |
| **Image Editing** | Instruction-following diffusion models | InstructPix2Pix, Imagen Editor |
| **Style Transfer** | Feature matching across content and style images | Neural style transfer, ControlNet |

**Guidance** controls how strongly the output follows the text prompt:

- **Classifier-Free Guidance (CFG):** Interpolates between conditional and unconditional predictions; CFG scale of 7-12 is typical
- **Negative Prompts:** Text describing what to avoid, subtracted from the generation process
- **ControlNet:** Additional conditioning on poses, edges, depth maps for spatial control

## Applications

- **Creative design:** Concept art, illustrations, marketing visuals, brand assets
- **Photo editing:** Professional-grade inpainting, background replacement, object removal
- **E-commerce:** Product visualization, virtual try-on, catalog generation at scale
- **Film and gaming:** Pre-visualization, texture generation, environment concept art

## Evolution

| Year | Milestone |
|------|-----------|
| 2015 | Neural style transfer popularizes AI art |
| 2021 | DALL-E and CLIP connect text and images |
| 2022 | DALL-E 2, Stable Diffusion, Midjourney launch the text-to-image revolution |
| 2023 | ControlNet adds spatial conditioning; SDXL improves quality |
| 2023 | DALL-E 3 achieves strong text rendering and prompt following |
| 2024 | Flux, SD3, and DiT-based architectures improve coherence and aesthetics |
| 2025 | Real-time generation, native text rendering, and precise editing become standard |`,

    video_generation: `# Video Generation

AI video generation creates moving image sequences from text prompts, still images, or other video inputs. Building on the success of image diffusion models, video generation extends the challenge into the temporal dimension, requiring **temporal consistency** (coherent motion across frames), **physical plausibility** (realistic dynamics), and **narrative coherence** (meaningful progression). This field has rapidly advanced from short, jittery clips to cinematic-quality video, with implications for filmmaking, advertising, education, and entertainment.

## Key Concepts

\`\`\`
Video Diffusion Pipeline:

  Text Prompt -----> [ Text Encoder ]
                          |
  (Optional)              v
  Image/Video ----> [ Temporal Diffusion Model ] ----> Denoised Latent Frames
                          |                                    |
                          v                                    v
                   [ Temporal Attention ]              [ VAE Frame Decoder ]
                   (consistency across frames)                 |
                                                               v
                                                        Generated Video
\`\`\`

- **Temporal Consistency:** Adjacent frames must be coherent; objects should not flicker, appear, or disappear randomly
- **Motion Modeling:** Capturing realistic physics, camera movements, and object dynamics
- **Temporal Attention:** Attention layers that operate across the time dimension to maintain frame coherence
- **Keyframe Interpolation:** Generating intermediate frames between specified keyframes

## How It Works

Video diffusion models extend image diffusion by adding temporal dimensions to the latent space. A 3D U-Net or spatiotemporal DiT processes frames jointly, with **temporal attention blocks** ensuring cross-frame consistency.

| Model | Developer | Key Features |
|-------|-----------|-------------|
| **Sora** | OpenAI | Minute-long videos, strong physics understanding, variable aspect ratios |
| **Gen-3 Alpha** | Runway | High fidelity, fine-grained control, lip sync |
| **Kling** | Kuaishou | 2-minute videos, realistic motion, strong physics |
| **Veo 2** | Google DeepMind | 4K resolution, cinematic quality, extended duration |
| **Wan** | Alibaba | Open-source, image-to-video, video-to-video editing |
| **Stable Video Diffusion** | Stability AI | Open-weight, image-to-video, research-friendly |
| **HunyuanVideo** | Tencent | Open-source, 5s+ clips, strong text-following |

**Key challenges** in video generation:

| Challenge | Description | Current Solutions |
|-----------|------------|-------------------|
| Temporal flickering | Inconsistent details between frames | Temporal attention, 3D convolutions |
| Physics violations | Objects defying gravity or clipping | Physics-informed training data, longer context |
| Identity preservation | Characters changing appearance mid-video | Reference frame conditioning, identity tokens |
| Long-form coherence | Maintaining narrative over minutes | Hierarchical planning, keyframe strategies |

## Applications

- **Filmmaking:** Pre-visualization, VFX shots, storyboard animation, B-roll generation
- **Advertising:** Rapid video ad prototyping, product demos, personalized video content
- **Education:** Animated explanations, historical recreations, scientific visualizations
- **Social media:** Short-form content creation, video effects, avatar animation

## Evolution

| Year | Milestone |
|------|-----------|
| 2022 | Make-A-Video and Imagen Video demonstrate text-to-video |
| 2023 | Runway Gen-2, Pika, and Stable Video Diffusion launch commercially |
| 2024 | Sora preview shows cinematic-quality minute-long videos |
| 2024 | Kling, Veo, and Gen-3 Alpha push quality and duration limits |
| 2025 | Open-source video models (Wan, HunyuanVideo) democratize video generation |
| 2025 | Real-time video generation and interactive video editing emerge |`,

    code_generation: `# AI Code Generation

AI code generation uses large language models trained on vast codebases to assist developers by writing, completing, explaining, debugging, and refactoring code. These models understand programming languages, frameworks, APIs, and software engineering patterns, transforming natural language intent into functional code. AI-assisted development has become one of the most impactful applications of generative AI, fundamentally changing how software is built by augmenting developer productivity and lowering barriers to programming.

## Key Concepts

\`\`\`
Developer Workflow with AI Code Generation:

  Natural Language / Code Context
           |
           v
  [ Code LLM (Transformer) ]
           |
           v
  Code Completion / Generation / Explanation
           |
           v
  [ Developer Review & Integration ]
\`\`\`

- **Code Completion:** Suggesting the next lines of code based on current context and cursor position
- **Code Generation:** Creating entire functions, classes, or files from natural language descriptions
- **Code Explanation:** Analyzing existing code and producing human-readable explanations
- **Bug Detection and Fixing:** Identifying errors and suggesting corrections
- **Refactoring:** Restructuring code for readability, performance, or maintainability

## How It Works

Code LLMs are trained on massive datasets of source code and technical documentation. They use the same autoregressive Transformer architecture as language models but are specialized for code understanding through **fill-in-the-middle (FIM)** training and multi-language support.

| Model / Tool | Developer | Parameters | Key Strengths |
|-------------|-----------|-----------|---------------|
| **GitHub Copilot** | GitHub / OpenAI | GPT-4+ based | IDE integration, inline suggestions |
| **Claude Code** | Anthropic | Claude-based | Agentic coding, terminal integration, large context |
| **Cursor** | Cursor Inc. | Multi-model | AI-native IDE, codebase-aware editing |
| **CodeLlama** | Meta | 7B-70B | Open-weight, Python specialist, FIM support |
| **StarCoder 2** | BigCode | 3B-15B | Open-source, 600+ languages, long context |
| **DeepSeek Coder V2** | DeepSeek | MoE-based | Strong reasoning, competitive benchmarks |
| **Codestral** | Mistral | 22B | Fast inference, 80+ languages |

**Evaluation benchmarks** for code models:

| Benchmark | Description | Measures |
|-----------|------------|----------|
| **HumanEval** | 164 Python problems | Function-level correctness |
| **MBPP** | 974 crowd-sourced Python tasks | Basic programming ability |
| **SWE-bench** | Real GitHub issues | End-to-end bug fixing and feature implementation |
| **LiveCodeBench** | Rolling competitive programming | Reasoning under novel problems |

## Applications

- **IDE-integrated assistance:** Real-time completions, inline suggestions, chat-driven development
- **Agentic development:** AI agents that plan, write, test, and debug entire features autonomously
- **Code review:** Automated identification of bugs, security vulnerabilities, and style issues
- **Documentation:** Auto-generating docstrings, READMEs, and API documentation from code

## Evolution

| Year | Milestone |
|------|-----------|
| 2020 | GPT-3 demonstrates code generation capabilities |
| 2021 | Codex and GitHub Copilot launch AI pair programming |
| 2023 | Code Llama, StarCoder, and WizardCoder advance open-source code models |
| 2023 | GPT-4 achieves strong performance on coding benchmarks |
| 2024 | Claude Code, Cursor, and Windsurf introduce agentic coding workflows |
| 2024 | SWE-bench becomes the standard for evaluating real-world coding ability |
| 2025 | AI coding agents handle multi-file refactoring and autonomous feature development |`,

    music_gen: `# AI Music Generation

AI music generation uses neural networks to compose, produce, and transform musical content from text descriptions, melodies, or other audio inputs. Combining techniques from audio synthesis, language modeling, and diffusion models, AI music systems can generate full songs with vocals, instrumentals, and complex arrangements. This field bridges generative AI with creative expression, enabling both professional musicians and casual users to create music in ways previously requiring years of training and expensive equipment.

## Key Concepts

\`\`\`
Text-to-Music Pipeline:

  "Upbeat jazz piano with a walking bassline"
           |
           v
  [ Text Encoder (CLAP / T5) ]
           |
           v
  [ Audio Generation Model ]
  (Diffusion / Autoregressive / Hybrid)
           |
           v
  [ Audio Decoder / Vocoder ]
           |
           v
  Generated Audio Waveform
\`\`\`

- **Audio Tokenization:** Converting continuous audio waveforms into discrete tokens using codecs (EnCodec, SoundStream)
- **CLAP:** Contrastive Language-Audio Pretraining; aligns text and audio in a shared embedding space (like CLIP for audio)
- **Mel Spectrograms:** Frequency-time representations of audio used as intermediate representations
- **Symbolic vs. Audio Generation:** Generating MIDI/notation (symbolic) versus raw audio waveforms directly

## How It Works

| Model | Developer | Approach | Key Features |
|-------|-----------|----------|-------------|
| **MusicLM** | Google | Autoregressive on audio tokens | Text-to-music, hierarchical generation |
| **Udio** | Udio | Diffusion-based | Full songs with vocals, commercial quality |
| **Suno** | Suno | Hybrid architecture | Lyrics + music, verse/chorus structure |
| **Stable Audio** | Stability AI | Latent diffusion on audio | Open-weight, variable length, timing control |
| **MusicGen** | Meta | Autoregressive Transformer on EnCodec tokens | Open-source, melody conditioning, text-to-music |
| **Jukebox** | OpenAI | VQ-VAE + autoregressive | Raw audio with singing, multiple genres |
| **Riffusion** | Open-source | Diffusion on spectrograms | Real-time, spectrogram-based generation |

**Key technical challenges:**

| Challenge | Description | Approach |
|-----------|------------|----------|
| Long-range structure | Songs need verse-chorus-bridge coherence | Hierarchical generation, structural prompting |
| Vocal quality | Realistic singing with lyrics is extremely hard | Dedicated vocal models, audio codec improvements |
| Musical coherence | Maintaining key, tempo, and harmony throughout | Music-theory-aware conditioning |
| High sample rates | Music needs 44.1kHz+ (much higher than speech) | Efficient audio codecs, multi-scale generation |

## Applications

- **Music production:** Generating backing tracks, stems, and demo recordings for professional producers
- **Content creation:** Royalty-free background music for videos, podcasts, and games
- **Creative exploration:** Rapid prototyping of musical ideas, genre blending, style experimentation
- **Interactive experiences:** Dynamic game soundtracks and personalized music for applications

## Evolution

| Year | Milestone |
|------|-----------|
| 2020 | OpenAI Jukebox generates raw audio music with singing |
| 2023 | MusicLM demonstrates high-quality text-to-music generation |
| 2023 | MusicGen (Meta) releases open-source text-to-music model |
| 2023 | Suno and Udio launch commercial AI music platforms |
| 2024 | Stable Audio 2.0 enables open-weight full-track generation |
| 2024 | AI-generated music triggers copyright and licensing debates |
| 2025 | Multi-track stem generation and real-time music co-creation mature |`,

    // ============================================================
    // GENAI ECOSYSTEM
    // ============================================================

    dalle_midjourney: `# DALL-E and Midjourney

DALL-E (OpenAI) and Midjourney represent two of the most influential text-to-image generation systems, each taking distinct approaches to AI image creation. **DALL-E** emerged from OpenAI's research lineage, progressing from discrete VAE tokens to CLIP-guided diffusion to native integration with ChatGPT. **Midjourney** was built by an independent lab focused on aesthetic quality and artistic style, quickly becoming the preferred tool for artists and designers. Together, they catalyzed the text-to-image revolution that began in 2022.

## Key Concepts

- **CLIP Conditioning:** Both systems use CLIP (Contrastive Language-Image Pretraining) to align text descriptions with visual features
- **Prompt Engineering:** The art of crafting text descriptions to guide generation; Midjourney developed an especially rich prompt vocabulary
- **Upscaling:** Multi-stage pipelines that generate at low resolution and upscale with detail-adding models
- **Style Control:** Mechanisms for controlling artistic style, photorealism, and aesthetic qualities

## How It Works

| Feature | DALL-E 1 (2021) | DALL-E 2 (2022) | DALL-E 3 (2023) |
|---------|----------------|-----------------|-----------------|
| Architecture | dVAE + Autoregressive Transformer | CLIP + Diffusion (unCLIP) | Diffusion with improved text encoder |
| Resolution | 256x256 | 1024x1024 | 1024x1024+ |
| Text Understanding | Basic composition | Improved but limited | Strong prompt following, text rendering |
| Access | Research preview | API and ChatGPT Plus | Native ChatGPT integration |
| Key Innovation | First text-to-image at scale | CLIP guidance + inpainting | Prompt rewriting via GPT-4 |

| Feature | Midjourney v1-3 (2022) | Midjourney v4-5 (2023) | Midjourney v6+ (2024) |
|---------|----------------------|----------------------|---------------------|
| Quality | Artistic, painterly | Photorealistic capable | Near-photographic |
| Text in Images | Poor | Limited | Improved text rendering |
| Prompt Adherence | Moderate | Good | Strong compositional understanding |
| Style | Strong default aesthetic | Versatile with style control | Precise style and consistency modes |
| Access | Discord bot | Discord + web alpha | Full web interface |

**Architectural differences:**

- **DALL-E 3** uses a text-to-prompt pipeline: GPT-4 rewrites user prompts into detailed descriptions, dramatically improving generation quality without requiring prompt engineering expertise
- **Midjourney** uses a proprietary architecture emphasizing aesthetic quality, with extensive community-driven discovery of effective style parameters and prompt techniques

## Applications

- **Concept art:** Rapid visual ideation for games, film, and product design
- **Marketing and advertising:** Campaign visuals, social media content, brand imagery
- **Publishing:** Book covers, editorial illustrations, educational materials
- **Personal creativity:** Artistic expression, gift creation, hobby projects

## Evolution

| Year | Milestone |
|------|-----------|
| 2021 | DALL-E 1 demonstrates text-to-image generation at scale |
| 2022 | DALL-E 2 launches with CLIP-guided diffusion and inpainting |
| 2022 | Midjourney v1-v3 launch via Discord, attract massive creative community |
| 2023 | Midjourney v5 achieves photorealistic quality |
| 2023 | DALL-E 3 integrates with ChatGPT with prompt rewriting |
| 2024 | Midjourney v6 adds text rendering, web interface, and style consistency |
| 2025 | Both platforms compete with Flux, Imagen 3, and open-source models |`,

    stable_diffusion: `# Stable Diffusion

Stable Diffusion is an open-weight text-to-image generation model developed by Stability AI in collaboration with researchers from CompVis (LMU Munich) and Runway. Released in August 2022, it was the first high-quality image generation model made freely available to the public, sparking an unprecedented open-source ecosystem. Its key innovation is **latent diffusion**, performing the denoising process in a compressed latent space rather than pixel space, dramatically reducing computational requirements. Stable Diffusion democratized AI image generation and enabled an explosion of community-built tools, fine-tunes, and applications.

## Key Concepts

\`\`\`
Stable Diffusion Architecture (Latent Diffusion):

  Text Prompt --> [ CLIP Text Encoder ] --> Text Embeddings
                                                  |
                                                  v
  Random Noise --> [ U-Net (Denoising in Latent Space) ] <-- Timestep
                            |
                            v
                    Denoised Latents
                            |
                            v
                    [ VAE Decoder ] --> Generated Image (512x512 / 1024x1024)
\`\`\`

- **Latent Diffusion:** Compressing images to a smaller latent space (e.g., 64x64x4) via a VAE before applying diffusion, reducing compute by ~50x
- **U-Net Backbone:** Convolutional network with cross-attention layers for text conditioning at multiple resolutions
- **CLIP Text Encoder:** Converts text prompts into embeddings that guide the denoising process
- **Classifier-Free Guidance (CFG):** Controls adherence to the text prompt; higher values follow the prompt more strictly

## How It Works

| Version | Release | Resolution | Text Encoder | Key Improvement |
|---------|---------|-----------|-------------|----------------|
| **SD 1.5** | 2022 | 512x512 | CLIP ViT-L/14 | Community standard, massive ecosystem |
| **SD 2.0/2.1** | 2022 | 768x768 | OpenCLIP ViT-H | Higher resolution, depth model |
| **SDXL** | 2023 | 1024x1024 | Dual CLIP (ViT-L + OpenCLIP ViT-G) | Larger U-Net, refiner model, better quality |
| **SDXL Turbo** | 2023 | 1024x1024 | Dual CLIP | Adversarial diffusion distillation, 1-4 steps |
| **SD 3 / 3.5** | 2024 | 1024x1024 | Triple (CLIP x2 + T5-XXL) | DiT backbone (MMDiT), flow matching |

**Community ecosystem** is Stable Diffusion's greatest strength:

| Tool / Technique | Purpose |
|-----------------|---------|
| **ControlNet** | Spatial conditioning via poses, edges, depth, and more |
| **LoRA** | Lightweight fine-tuning for custom styles and characters (4-100MB files) |
| **Textual Inversion** | Learning new concepts via small embedding vectors |
| **DreamBooth** | Fine-tuning to learn specific subjects from few images |
| **ComfyUI** | Node-based visual workflow editor for complex generation pipelines |
| **Automatic1111** | Feature-rich web UI for Stable Diffusion with extensions |
| **IP-Adapter** | Image prompt conditioning for style and subject transfer |

## Applications

- **Rapid prototyping:** Designers generate concepts locally without API costs or rate limits
- **Custom model training:** Fine-tuning on proprietary data for brand-specific generation
- **Integration:** Embedding generation into applications, games, and creative tools via local inference
- **Research:** Open weights enable academic study and architectural experimentation

## Evolution

| Year | Milestone |
|------|-----------|
| 2022 | Stable Diffusion 1.4/1.5 released open-weight, igniting the community |
| 2022 | ControlNet, LoRA, and community tools transform the ecosystem |
| 2023 | SDXL raises quality to 1024x1024 with dual text encoders |
| 2023 | SDXL Turbo achieves generation in 1-4 steps via distillation |
| 2024 | SD3 adopts Diffusion Transformer (MMDiT) architecture and flow matching |
| 2024 | Flux (by Black Forest Labs, from ex-Stability researchers) emerges as a competitor |
| 2025 | Open-source ecosystem includes video, 3D, and audio diffusion variants |`,

    ai_art: `# AI Art

AI art encompasses the creation of visual artwork using artificial intelligence systems, spanning generative models, neural style transfer, interactive creative tools, and autonomous artistic agents. What began as experimental academic projects has grown into a global movement that challenges traditional notions of authorship, creativity, and artistic value. AI art sits at the intersection of technology, aesthetics, philosophy, and law, prompting fundamental questions about what it means to create and who deserves credit for machine-assisted works.

## Key Concepts

- **Prompt-Based Art:** Using text descriptions to guide AI image generation; the prompt becomes a creative medium
- **Human-AI Co-Creation:** Collaborative process where artists use AI as a tool, iterating and curating outputs
- **Procedural Aesthetics:** AI systems develop emergent visual styles not explicitly programmed by humans
- **Digital Provenance:** Tracking the origin and creation process of AI-generated images
- **Style Transfer:** Applying the visual style of one image to the content of another via neural networks

## How It Works

AI art creation involves multiple approaches and tools:

| Approach | Description | Tools |
|----------|------------|-------|
| **Text-to-Image** | Generating art from written descriptions | Midjourney, DALL-E 3, Stable Diffusion |
| **Image-to-Image** | Transforming existing images with AI guidance | ControlNet, img2img, InstructPix2Pix |
| **Style Fine-Tuning** | Training models on specific artistic styles | LoRA, DreamBooth, Textual Inversion |
| **Neural Style Transfer** | Blending content and style from separate images | Neural style transfer networks |
| **Generative Agents** | Autonomous systems that create art based on goals | AARON, creative coding with LLMs |
| **Interactive Tools** | Real-time AI-assisted drawing and painting | Adobe Firefly, Krea AI, Playground |

**The copyright debate** is central to AI art:

| Position | Argument |
|----------|---------|
| **Pro-AI-art** | AI is a tool like a camera; human creativity lies in curation and prompting |
| **Anti-AI-art** | Models trained on copyrighted art without consent; outputs may replicate styles |
| **Legal status** | US Copyright Office: purely AI-generated images cannot be copyrighted; human authorship required |
| **Training data** | Debate over fair use of training data; lawsuits from artists and publishers ongoing |
| **Opt-out movements** | Artists demand consent and compensation; tools like Glaze protect style from mimicry |

## Applications

- **Professional illustration:** Concept art, book covers, game assets, architectural visualization
- **Fine art:** AI-generated works exhibited in galleries and sold at auction
- **Therapeutic art:** AI-assisted creative expression for art therapy and accessibility
- **Fashion and design:** Pattern generation, textile design, virtual fashion

## Evolution

| Year | Milestone |
|------|-----------|
| 2015 | DeepDream and neural style transfer captivate public imagination |
| 2018 | GAN-generated "Portrait of Edmond de Belamy" sells for $432,500 at Christie's |
| 2022 | "Theatre D'opera Spatial" (Midjourney) wins Colorado State Fair art prize |
| 2022 | Stability AI, Midjourney, and DALL-E 2 launch the text-to-image era |
| 2023 | US Copyright Office rules on AI art authorship requirements |
| 2023 | Artists file class-action lawsuits against AI companies over training data |
| 2024 | Content authenticity standards (C2PA) emerge for AI-generated media labeling |
| 2025 | Legal frameworks and licensing models for AI art training data continue evolving |`,

    multimodal_gen: `# Multimodal Generation

Multimodal generation refers to AI systems that can create, understand, and transform content across multiple modalities simultaneously, including text, images, audio, video, and 3D. Rather than separate models for each medium, the field is converging toward **unified architectures** that natively handle any combination of inputs and outputs. This represents the frontier of generative AI: systems that reason about and produce rich, multi-sensory content in the way humans naturally communicate and create.

## Key Concepts

\`\`\`
Unified Multimodal Model:

  Inputs (Any Combination):        Outputs (Any Combination):
  [Text] ----\\                      /----> [Text]
  [Image] ----+---> [ Unified ] ---+----> [Image]
  [Audio] ----+     [ Model  ]    +----> [Audio]
  [Video] ---/                      \\----> [Video]
\`\`\`

- **Multimodal Tokens:** Converting all modalities into a shared token space for unified processing
- **Cross-Modal Attention:** Attention mechanisms that relate information across different modalities
- **Any-to-Any Generation:** Models that accept and produce arbitrary combinations of modalities
- **Modality Alignment:** Training to create shared representations where similar concepts across modalities are close in embedding space
- **Interleaved Generation:** Producing outputs that naturally mix modalities (e.g., text with inline images)

## How It Works

| Model | Developer | Modalities | Key Innovation |
|-------|-----------|-----------|----------------|
| **GPT-4o** | OpenAI | Text, image, audio (in/out) | Native multimodal with real-time voice |
| **Gemini 2.0** | Google | Text, image, audio, video (in/out) | Natively multimodal from training |
| **Claude 3.5** | Anthropic | Text, image (in), text (out) | Strong vision-language reasoning |
| **Chameleon** | Meta | Text, image (in/out) | Early-fusion mixed-modal tokens |
| **CoDi** | Microsoft | Text, image, audio, video (in/out) | Any-to-any composable generation |
| **Unified-IO 2** | Allen AI | Text, image, audio, video, actions | Single model for all modalities |
| **Janus** | DeepSeek | Text, image (in/out) | Decoupled visual encoding for understanding vs generation |

**Architecture approaches** for multimodal generation:

| Approach | Description | Example |
|----------|------------|---------|
| **Early Fusion** | Tokenize all modalities into single sequence | Chameleon, Gemini |
| **Late Fusion** | Separate encoders with cross-modal attention | Flamingo, LLaVA |
| **Modular** | Specialized models connected via shared latent space | CoDi, NExT-GPT |
| **Diffusion + LLM** | LLM for reasoning, diffusion for generation | DALL-E 3 + GPT-4, Emu |

**Multimodal alignment** ensures coherent cross-modal generation:
- **CLIP/CLAP:** Contrastive pretraining for text-image and text-audio alignment
- **Shared tokenizers:** Discrete tokens representing all modalities in one vocabulary
- **Cross-attention bridges:** Conditioning generation in one modality on context from another

## Applications

- **Creative workflows:** Generate storyboards with text, images, and narration in one pass
- **Accessibility:** Convert content between modalities (image to audio description, text to sign language video)
- **Education:** Interactive lessons combining explanations, diagrams, animations, and quizzes
- **Virtual assistants:** Agents that see, hear, speak, and create visual content natively

## Evolution

| Year | Milestone |
|------|-----------|
| 2021 | CLIP demonstrates powerful text-image alignment |
| 2022 | Flamingo shows few-shot multimodal learning |
| 2023 | GPT-4V adds vision to language models; LLaVA opens multimodal research |
| 2023 | CoDi demonstrates any-to-any generation across four modalities |
| 2024 | GPT-4o achieves real-time multimodal conversation with native audio |
| 2024 | Gemini 2.0 trained natively on interleaved multimodal data |
| 2025 | Unified models generating text, image, audio, and video from single architectures become reality |`,

  };

  // Register all content on the global AI_DOCS object
  Object.assign(window.AI_DOCS, content);
})();
