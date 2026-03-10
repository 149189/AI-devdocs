// 15 - Emerging Frontiers
(function () {
  const content = {
    quantum_ml: `# Quantum Machine Learning

Quantum Machine Learning (QML) explores using quantum computers to speed up or improve machine learning algorithms. It combines quantum computing principles with ML to solve problems intractable for classical computers.

## Key Concepts

- **Qubit**: Quantum bit that can be 0, 1, or superposition of both
- **Superposition**: Qubit exists in multiple states simultaneously
- **Entanglement**: Qubits correlated so measuring one affects the other
- **Quantum Gate**: Operations on qubits (analogous to logic gates)
- **Quantum Circuit**: Sequence of quantum gates forming an algorithm
- **Quantum Advantage**: When quantum computer outperforms classical

## Quantum ML Approaches

| Approach | Description | Status |
|----------|-------------|--------|
| Quantum Kernel Methods | Quantum-computed kernel functions | Research |
| Variational Quantum Circuits | Parameterized quantum circuits trained classically | Active research |
| Quantum Neural Networks | Neural nets on quantum hardware | Early stage |
| Quantum Sampling | Boltzmann machines on quantum hardware | Demonstrated |
| Quantum Optimization | QAOA for combinatorial problems | Near-term |
| Quantum Data Analysis | Quantum PCA, HHL algorithm | Theoretical advantage |

## Frameworks

| Framework | Provider | Purpose |
|-----------|----------|---------|
| Qiskit | IBM | Quantum programming and ML |
| PennyLane | Xanadu | Quantum ML library |
| Cirq | Google | Quantum circuit programming |
| Amazon Braket | AWS | Cloud quantum computing |
| TensorFlow Quantum | Google | Quantum ML with TensorFlow |

## Current Limitations

- **Noise**: Current quantum computers are noisy (NISQ era)
- **Qubit Count**: Limited qubits (100-1000 range currently)
- **Coherence Time**: Qubits lose quantum state quickly
- **Error Correction**: Requires many physical qubits per logical qubit
- **Practical Advantage**: Not yet demonstrated for real ML tasks

## Evolution

- **2019**: Google claims quantum supremacy (Sycamore processor)
- **2020**: Quantum ML frameworks (PennyLane, TF Quantum) released
- **2023**: IBM releases 1,121-qubit processor (Condor)
- **2024**: Error correction milestones achieved
- **2030+**: Fault-tolerant quantum computers expected to enable true QML advantage`,

    neuromorphic: `# Neuromorphic Computing

Neuromorphic computing designs hardware that mimics the structure and function of biological neural networks. It promises extreme energy efficiency and real-time processing for edge AI applications.

## Key Concepts

- **Spiking Neural Networks (SNNs)**: Neurons communicate via discrete spikes (like biological neurons)
- **Event-Driven**: Process only when input changes (vs clock-driven in traditional chips)
- **In-Memory Computing**: Computation happens where data is stored (no von Neumann bottleneck)
- **Synaptic Plasticity**: Connections strengthen/weaken based on activity (learning on chip)
- **Temporal Coding**: Information encoded in timing of spikes

## Hardware

| Chip | Org | Neurons | Synapses | Power |
|------|-----|---------|----------|-------|
| TrueNorth | IBM | 1M | 256M | 70 mW |
| Loihi 2 | Intel | 1M | 120M | <1 W |
| SpiNNaker 2 | Manchester | 10M+ | Programmable | ~1 W |
| BrainChip Akida | BrainChip | Commercial | On-chip learning | <500 mW |

## Advantages over Traditional AI

| Aspect | Traditional GPU | Neuromorphic |
|--------|----------------|-------------|
| Power | 300-700W | < 1W |
| Paradigm | Batch processing | Event-driven |
| Latency | Milliseconds | Microseconds |
| Learning | Offline training | On-chip learning |
| Best for | Large models, training | Edge, real-time, always-on |

## Applications

- Always-on sensor processing (gesture, voice)
- Robotics (real-time sensory processing)
- Edge AI with extreme power constraints
- Brain-computer interfaces
- Autonomous systems requiring microsecond latency

## Evolution

- **2011**: IBM unveils TrueNorth concept
- **2014**: TrueNorth chip with 1M neurons
- **2017**: Intel releases Loihi neuromorphic chip
- **2021**: Loihi 2 with improved architecture
- **2025+**: Neuromorphic computing finds niche applications in edge and robotics`,

    foundation_models: `# Foundation Models

Foundation Models are large-scale AI models trained on broad data that can be adapted to many downstream tasks. They represent a paradigm shift from task-specific to general-purpose AI.

## Key Characteristics

- **Scale**: Billions to trillions of parameters
- **Broad Training**: Trained on diverse, massive datasets
- **Emergent Abilities**: Capabilities that appear at scale but not in smaller models
- **Transfer**: Adapt to new tasks via fine-tuning or prompting
- **Multimodal**: Increasingly handle text, images, audio, video

## Evolution of Foundation Models

| Year | Model | Params | Modality | Key Innovation |
|------|-------|--------|----------|---------------|
| 2018 | BERT | 340M | Text | Bidirectional pretraining |
| 2020 | GPT-3 | 175B | Text | Few-shot learning at scale |
| 2021 | CLIP | 400M | Text+Image | Contrastive vision-language |
| 2022 | ChatGPT | 175B+ | Text | RLHF for conversation |
| 2023 | GPT-4 | ~1.8T (est.) | Multimodal | Vision + text reasoning |
| 2023 | LLaMA | 7-70B | Text | Open-source foundation |
| 2024 | Claude 3 | Unknown | Multimodal | Long context, safety |
| 2024 | Gemini | Unknown | Multimodal | Native multimodal |

## Types

| Type | Examples | Domain |
|------|---------|--------|
| Language Models | GPT-4, Claude, LLaMA | Text understanding/generation |
| Vision Models | CLIP, DINOv2, SAM | Image understanding |
| Multimodal | GPT-4V, Gemini | Text + image + audio |
| Code Models | Codex, StarCoder, CodeLlama | Code generation |
| Scientific | AlphaFold, ESM | Protein/molecular science |
| Audio | Whisper, MusicLM | Speech and audio |

## Impact

- **Democratization**: Open-source models enable broad access
- **Emergent behavior**: Abilities not explicitly trained for appear at scale
- **Few-shot learning**: Solve new tasks with just a few examples
- **Research paradigm**: Fine-tune foundation models rather than train from scratch

## Evolution

- **2018**: BERT establishes foundation model concept
- **2020**: GPT-3 demonstrates few-shot learning
- **2022**: ChatGPT brings foundation models to mainstream
- **2023**: Open-source models (LLaMA, Mistral) challenge closed models
- **2025+**: Specialized foundation models for every domain`,

    multimodal_ai: `# Multimodal AI

Multimodal AI systems process and relate information across multiple data types - text, images, audio, video, and more. They move AI beyond single-modality specialists toward human-like understanding.

## Key Concepts

- **Modality**: A type of data (text, image, audio, video, sensor data)
- **Cross-Modal Learning**: Learning relationships between different modalities
- **Fusion**: Combining information from multiple modalities
- **Contrastive Learning**: Aligning representations across modalities (CLIP)
- **Vision-Language Models**: Models that understand both images and text

## Fusion Strategies

| Strategy | When | How |
|----------|------|-----|
| Early Fusion | Before processing | Concatenate raw inputs |
| Late Fusion | After processing | Combine separate model outputs |
| Cross-Attention | During processing | Attend across modalities |
| Unified | End-to-end | Single model handles all modalities |

## Key Models

| Model | Modalities | Capability |
|-------|-----------|------------|
| CLIP | Image + Text | Zero-shot image classification |
| GPT-4V/o | Text + Image | Visual reasoning and description |
| Gemini | Text + Image + Audio + Video | Native multimodal understanding |
| LLaVA | Image + Text | Open-source vision-language |
| Whisper | Audio -> Text | Multilingual speech recognition |
| ImageBind | 6 modalities | Unified embedding space |

## Applications

- Visual question answering ("What is in this image?")
- Image captioning and description
- Video understanding and summarization
- Multimodal search (search images with text)
- Document understanding (text + layout + images)
- Accessibility (describe images for visually impaired)

## Evolution

- **2021**: CLIP connects vision and language with contrastive learning
- **2022**: Flamingo and BLIP advance vision-language models
- **2023**: GPT-4V, Gemini enable multimodal reasoning in production
- **2024**: Video understanding and generation mature (Sora, Gemini 1.5)
- **2025+**: Unified models handle any combination of modalities natively`,

    world_models: `# World Models

World Models are AI systems that build internal representations of the environment to predict future states and plan actions. They learn a compressed model of how the world works.

## Key Concepts

- **World Model**: Internal representation of environment dynamics
- **Imagination**: Simulating future scenarios without real interaction
- **Latent Space**: Compressed representation of world state
- **Model-Based RL**: Use learned world model for planning
- **Predictive Coding**: Brain-inspired prediction of sensory input
- **Dreaming**: Training agent inside the world model

## Architecture

\`\`\`
World Model Components:
1. Vision Model (V): Encode observations into latent space
   Raw pixels -> z (latent representation)

2. Memory Model (M): Predict future latent states
   z_t, a_t -> z_{t+1} (predict next state)

3. Controller (C): Select actions based on latent state
   z_t -> a_t (choose action)

Ha & Schmidhuber (2018) World Model:
- VAE encodes observations
- RNN predicts future latent states
- Small controller learns to act in "dream" environment
\`\`\`

## Key Work

| Model | Year | Innovation |
|-------|------|-----------|
| World Models (Ha) | 2018 | VAE + RNN + Controller |
| Dreamer | 2019 | World model with actor-critic |
| DreamerV2 | 2021 | Discrete latent space |
| DreamerV3 | 2023 | Generalist world model |
| JEPA (LeCun) | 2022 | Joint Embedding Predictive Architecture |
| Sora | 2024 | Video generation as world simulation |
| Genie 2 | 2024 | Interactive 3D world generation |

## Applications

- Robotics (simulate actions before executing)
- Autonomous driving (predict traffic scenarios)
- Video game AI (plan strategies mentally)
- Video generation (Sora as implicit world model)
- Scientific simulation (predicting physical systems)

## Evolution

- **2018**: Ha & Schmidhuber introduce "World Models" paper
- **2019**: Dreamer learns behaviors from world model predictions
- **2022**: LeCun proposes JEPA as path toward autonomous intelligence
- **2024**: Sora demonstrates video generation as world simulation
- **2025+**: World models as a path toward general intelligence`,

    ai_science: `# AI for Science

AI for Science applies machine learning to accelerate scientific discovery, from protein structure prediction to materials design, drug discovery, and climate modeling.

## Key Applications

| Domain | AI Application | Impact |
|--------|---------------|--------|
| Biology | AlphaFold (protein structure) | Predicted 200M+ protein structures |
| Drug Discovery | Molecular generation, virtual screening | Faster drug candidates |
| Climate | Weather prediction, climate modeling | More accurate forecasts |
| Materials | Material property prediction | New material discovery |
| Physics | Particle physics, gravitational waves | Pattern detection |
| Mathematics | Theorem proving, conjecture generation | New mathematical insights |

## Landmark Results

\`\`\`
AlphaFold 2 (2020):
- Predicts protein 3D structure from amino acid sequence
- Solved 50-year grand challenge in biology
- 200M+ structures predicted (AlphaFold DB)
- Nobel Prize in Chemistry 2024

GraphCast (2023):
- Weather prediction using graph neural networks
- 10-day forecasts more accurate than traditional methods
- 1000x faster than numerical weather prediction
- Runs in under a minute vs hours for traditional models

GNoME (2023):
- Graph Networks for Materials Exploration
- Discovered 2.2 million new crystal structures
- 380,000+ stable new materials identified
- Could transform energy, electronics, computing
\`\`\`

## AI-Driven Scientific Methods

| Method | Description |
|--------|-------------|
| Surrogate Models | ML approximation of expensive simulations |
| Active Learning | AI guides which experiments to run next |
| Generative Design | Generate novel molecules, materials, proteins |
| Automated Labs | Self-driving labs with robotic experimentation |
| Literature Mining | Extract knowledge from scientific papers |

## Evolution

- **2017**: AI applied to protein structure prediction
- **2020**: AlphaFold 2 solves protein folding
- **2023**: GraphCast for weather; GNoME for materials
- **2024**: AlphaFold wins Nobel Prize; AI labs go mainstream
- **2025+**: AI becomes integral to every scientific discipline`,

    robotics_ai: `# Robotics & Embodied AI

Embodied AI creates intelligent agents that interact with the physical world through robotic bodies. It combines perception, planning, and control to build robots that can manipulate objects, navigate environments, and collaborate with humans.

## Key Concepts

- **Embodiment**: Physical body that can sense and act in the real world
- **Manipulation**: Grasping, moving, and using objects
- **Navigation**: Moving through environments
- **Sim-to-Real Transfer**: Training in simulation, deploying in reality
- **Foundation Models for Robotics**: Using LLMs/VLMs for robot reasoning

## Robot Types

| Type | Example | Application |
|------|---------|------------|
| Industrial | KUKA, ABB arms | Manufacturing, assembly |
| Collaborative (Cobot) | Universal Robots | Human-robot collaboration |
| Mobile | Boston Dynamics Spot | Inspection, delivery |
| Humanoid | Tesla Optimus, Figure 01 | General-purpose |
| Surgical | da Vinci | Minimally invasive surgery |
| Autonomous Vehicle | Waymo, Tesla | Transportation |

## Modern AI Approaches

\`\`\`
Traditional Robotics:
Sense -> Model -> Plan -> Act (explicit programming)

Modern AI Robotics:
1. Foundation Models for Reasoning:
   - LLMs plan high-level actions ("pick up the red cup")
   - VLMs understand visual scene
   - Example: SayCan, RT-2, PaLM-E

2. Imitation Learning:
   - Learn from human demonstrations
   - Behavior cloning, DAGGER

3. RL for Control:
   - Learn manipulation/locomotion policies
   - Sim-to-real with domain randomization

4. Robot Foundation Models:
   - RT-X: Cross-embodiment robot dataset
   - Octo: Open robot foundation model
   - Train once, deploy on many robots
\`\`\`

## Applications

- Warehouse automation (Amazon, Ocado)
- Last-mile delivery (Starship, Nuro)
- Home assistance (cooking, cleaning)
- Agriculture (harvesting, weeding)
- Construction (autonomous machines)

## Evolution

- **1961**: Unimate (first industrial robot)
- **2016**: Boston Dynamics Atlas demonstrates dynamic locomotion
- **2022**: SayCan connects LLMs to robot actions
- **2023**: RT-2 and PaLM-E (vision-language-action models)
- **2025+**: Humanoid robots with foundation model reasoning`,

    bci: `# Brain-Computer Interfaces

Brain-Computer Interfaces (BCIs) create direct communication pathways between the brain and external devices. AI plays a crucial role in decoding neural signals and enabling new forms of interaction.

## Types of BCIs

| Type | Method | Invasiveness | Resolution |
|------|--------|-------------|------------|
| Non-invasive | EEG, fNIRS | None | Low (scalp recording) |
| Semi-invasive | ECoG | Moderate | Medium (brain surface) |
| Invasive | Microelectrode arrays | High | High (inside brain tissue) |

## Key Concepts

- **Neural Decoding**: Using ML to interpret brain signals
- **Motor Imagery**: Imagining movements to control devices
- **P300**: Brain response to expected stimuli (used for spelling)
- **Spike Sorting**: Identifying individual neuron activity
- **Neural Prosthetics**: Devices controlled by brain signals

## AI in BCIs

\`\`\`
AI Pipeline for BCIs:
1. Signal Acquisition: Record brain activity (EEG/implant)
2. Preprocessing: Filter noise, artifact removal
3. Feature Extraction: Extract relevant signal patterns
4. Decoding: ML model translates neural patterns to commands

AI Models Used:
- CNNs: Spatial pattern recognition in EEG
- RNNs/Transformers: Temporal pattern decoding
- Reinforcement Learning: Adaptive decoder calibration
- Diffusion Models: Neural signal denoising
\`\`\`

## Key Players

| Company | Approach | Achievement |
|---------|----------|-------------|
| Neuralink | Invasive implant | First human implant (2024) |
| BrainGate | Microelectrode array | Cursor control, typing for paralyzed |
| Synchron | Endovascular (blood vessel) | Less invasive brain interface |
| OpenBCI | Open-source EEG | Accessible research hardware |

## Applications

- Restoring communication for locked-in patients
- Prosthetic limb control
- Treating neurological disorders (epilepsy, depression)
- Cognitive enhancement (attention, memory)
- Direct brain-to-text communication

## Evolution

- **1998**: First human BCI implant (BrainGate)
- **2016**: BrainGate enables typing at 8 words/minute
- **2021**: BCI achieves 90+ words/minute handwriting decoding
- **2024**: Neuralink first human implant; Synchron clinical trials
- **2030+**: High-bandwidth, long-lasting implants for wider use`,

    agi: `# Artificial General Intelligence (AGI)

AGI refers to AI systems with human-level intelligence across all cognitive domains - reasoning, learning, creativity, common sense, and adaptability. It remains one of the most debated and sought-after goals in AI.

## Key Concepts

- **Narrow AI (ANI)**: AI excelling at specific tasks (current state)
- **General AI (AGI)**: Human-level intelligence across all domains
- **Superintelligence (ASI)**: Intelligence surpassing all humans
- **Common Sense**: Understanding everyday world knowledge
- **Transfer**: Applying skills learned in one domain to completely new ones
- **Metacognition**: Understanding and monitoring own thought processes

## AGI vs Current AI

| Capability | Current AI | AGI |
|-----------|-----------|-----|
| Narrow tasks | Superhuman | Human-level |
| Common sense | Limited | Full |
| Novel situations | Struggles | Adapts |
| Cross-domain transfer | Minimal | Natural |
| Physical world understanding | Learned from text | Embodied experience |
| Self-awareness | None | Debated |

## Proposed Paths to AGI

| Approach | Description | Proponents |
|----------|-------------|-----------|
| Scaling | Scale up current LLMs | OpenAI (Sam Altman) |
| Hybrid Neuro-Symbolic | Combine neural nets with symbolic reasoning | Gary Marcus |
| World Models | Learn internal models of reality | Yann LeCun (JEPA) |
| Embodied AI | Ground intelligence in physical interaction | Robotics researchers |
| Whole Brain Emulation | Simulate the brain in detail | Neuroscience approach |
| Evolutionary Approaches | Evolve intelligent systems | OpenELM, evolutionary computation |

## Safety Considerations

- **Alignment**: Ensuring AGI pursues human-beneficial goals
- **Control Problem**: Maintaining human oversight of superhuman systems
- **Value Loading**: How to encode human values into AGI
- **Containment**: Preventing unintended AGI capabilities
- **Coordination**: International cooperation on AGI development

## Evolution

- **1956**: AGI goal articulated at Dartmouth Conference
- **1990s-2000s**: AI winter; AGI seen as distant
- **2020**: GPT-3 reignites AGI discussions
- **2023**: "Sparks of AGI" paper about GPT-4; OpenAI states AGI as mission
- **2025+**: Debate intensifies on timelines and approaches; no consensus on when or if`,

    ai_blockchain: `# AI & Blockchain

The intersection of AI and blockchain combines artificial intelligence with decentralized ledger technology to create transparent, verifiable, and decentralized AI systems.

## Key Intersections

| Direction | Description | Examples |
|-----------|-------------|---------|
| AI for Blockchain | AI improves blockchain systems | Smart contract auditing, fraud detection |
| Blockchain for AI | Blockchain verifies AI outputs | Model provenance, data authenticity |
| Decentralized AI | AI training/inference on decentralized networks | Federated learning, compute markets |
| Tokenized AI | AI services traded as tokens | GPU compute markets |

## Key Concepts

- **Decentralized Compute**: Distribute AI training across many providers
- **Model Provenance**: Track model training data and version history on-chain
- **Data Marketplaces**: Buy/sell training data with smart contracts
- **Proof of Computation**: Verify AI computations were performed correctly
- **AI DAOs**: Decentralized organizations governed by AI agents

## Applications

\`\`\`
1. Decentralized AI Training:
   - Pool GPU resources from many providers
   - Compensate contributors with tokens
   - Examples: Gensyn, Together AI

2. AI Model Verification:
   - Prove a model was trained on specific data
   - Verify model hasn't been tampered with
   - Cryptographic proofs of ML inference

3. Data Provenance:
   - Track training data origins
   - Ensure data consent and licensing
   - Immutable audit trail

4. AI-Powered Smart Contracts:
   - ML oracles for real-world data
   - AI agents executing trades
   - Automated compliance checking
\`\`\`

## Projects

| Project | Focus |
|---------|-------|
| Bittensor | Decentralized AI network |
| Gensyn | Distributed ML training |
| Ocean Protocol | Data marketplace |
| Fetch.ai | Autonomous AI agents |
| SingularityNET | AI service marketplace |

## Challenges

- Scalability of blockchain for AI workloads
- Energy consumption of both AI and blockchain
- Verifying complex AI computations on-chain
- Regulatory uncertainty

## Evolution

- **2017**: SingularityNET proposes decentralized AI marketplace
- **2020**: Federated learning + blockchain for privacy
- **2023**: GPU compute markets gain traction
- **2024**: AI agent + blockchain integration grows
- **2025+**: Decentralized AI infrastructure matures`,
  };

  Object.assign(window.AI_DOCS, content);
})();
