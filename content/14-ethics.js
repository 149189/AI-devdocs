// 14 - AI Ethics & Governance
(function () {
  const content = {
    bias_fairness: `# Bias & Fairness in AI

AI systems can perpetuate and amplify societal biases present in training data, algorithm design, and deployment contexts. Ensuring fairness is a critical challenge in responsible AI development.

## Types of Bias

| Type | Source | Example |
|------|--------|---------|
| Historical | Past societal inequities in data | Hiring models trained on biased hiring history |
| Representation | Underrepresentation of groups | Face recognition worse for darker skin |
| Measurement | Flawed data collection | Proxy variables correlating with protected attributes |
| Aggregation | One model for diverse populations | Medical model trained on one demographic |
| Evaluation | Biased testing benchmarks | Benchmarks not representing all groups |
| Deployment | Context of use | Same model applied in inappropriate contexts |

## Fairness Metrics

\`\`\`
Demographic Parity:
P(Y=1 | Group=A) = P(Y=1 | Group=B)
Each group gets positive outcomes at equal rates

Equalized Odds:
P(Y=1 | Y_true=1, Group=A) = P(Y=1 | Y_true=1, Group=B)
Equal true positive and false positive rates across groups

Individual Fairness:
Similar individuals receive similar predictions
Requires defining "similarity" metric

Counterfactual Fairness:
Prediction wouldn't change if individual belonged to different group
\`\`\`

## Mitigation Strategies

- **Pre-processing**: Rebalance training data, remove biased features
- **In-processing**: Adversarial debiasing, fairness constraints during training
- **Post-processing**: Calibrate predictions to achieve fairness metrics
- **Auditing**: Regular bias audits with disaggregated evaluation

## Tools

| Tool | Provider | Purpose |
|------|----------|---------|
| Fairlearn | Microsoft | Fairness assessment and mitigation |
| AI Fairness 360 (AIF360) | IBM | Bias detection toolkit |
| What-If Tool | Google | Interactive model analysis |
| Aequitas | U. Chicago | Bias audit toolkit |

## Evolution

- **2016**: ProPublica COMPAS investigation highlights algorithmic bias
- **2018**: Gender Shades study shows face recognition disparities
- **2020**: IBM, Microsoft restrict facial recognition sales
- **2023**: EU AI Act mandates bias testing for high-risk systems
- **2024+**: Bias testing becomes standard in model development pipelines`,

    xai: `# Explainability (XAI)

Explainable AI (XAI) makes ML model decisions interpretable and understandable to humans. It answers "why did the model make this prediction?" - critical for trust, debugging, and regulatory compliance.

## Types of Explanations

| Type | Description | Example |
|------|-------------|---------|
| Global | Explain overall model behavior | Feature importance rankings |
| Local | Explain single prediction | Why was this loan denied? |
| Model-specific | Built into model type | Decision tree rules |
| Model-agnostic | Works on any model | SHAP, LIME |
| Ante-hoc | Inherently interpretable models | Linear models, decision trees |
| Post-hoc | Applied after model training | Saliency maps, SHAP values |

## Key Methods

\`\`\`python
# SHAP (SHapley Additive exPlanations)
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
# Shows each feature's contribution to predictions

# LIME (Local Interpretable Model-agnostic Explanations)
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train, feature_names=feature_names)
explanation = explainer.explain_instance(X_test[0], model.predict_proba)
explanation.show_in_notebook()
# Approximates model locally with interpretable model

# Attention Visualization (for transformers)
# Visualize which tokens the model attends to
# Built into most transformer implementations
\`\`\`

## Methods Comparison

| Method | Scope | Speed | Fidelity |
|--------|-------|-------|----------|
| SHAP | Local + Global | Slow | High (game-theoretic) |
| LIME | Local | Fast | Approximate |
| Integrated Gradients | Local | Medium | Exact (for neural nets) |
| Attention Maps | Local | Fast | Correlative (not causal) |
| Feature Importance | Global | Fast | Model-dependent |
| Counterfactual | Local | Medium | Actionable |

## Applications

- Healthcare: Explain diagnosis predictions to doctors
- Finance: Justify loan/credit decisions (regulatory requirement)
- Legal: Explain recidivism risk scores
- Autonomous driving: Explain driving decisions for safety analysis

## Evolution

- **2016**: LIME introduces local interpretable explanations
- **2017**: SHAP provides unified approach based on Shapley values
- **2019**: Integrated Gradients for neural network attribution
- **2023**: LLMs can self-explain reasoning (chain-of-thought)
- **2024+**: Explainability required by EU AI Act for high-risk systems`,

    ai_safety: `# AI Safety & Alignment

AI Safety ensures that AI systems behave in ways that are beneficial and aligned with human values. Alignment research focuses on making AI pursue intended goals rather than unintended harmful behaviors.

## Key Concepts

- **Alignment**: Ensuring AI objectives match human intentions
- **Outer Alignment**: Specifying the right objective function
- **Inner Alignment**: Ensuring the model actually optimizes for that objective
- **Reward Hacking**: AI finding unintended ways to maximize reward
- **Corrigibility**: Ability to safely shut down or modify an AI system
- **Scalable Oversight**: Maintaining control as AI systems become more capable

## Alignment Approaches

| Approach | Description | Used By |
|----------|-------------|---------|
| RLHF | Human feedback to align behavior | OpenAI, most LLM labs |
| Constitutional AI | Principles-based self-improvement | Anthropic (Claude) |
| Debate | AIs argue, humans judge | OpenAI research |
| Iterated Distillation | Train on curated outputs | Various |
| Process Supervision | Reward reasoning steps, not just outcomes | OpenAI |

## Safety Challenges

\`\`\`
Current Risks:
- Hallucination: Models generating false but plausible information
- Jailbreaking: Users bypassing safety guardrails
- Dual-use: Beneficial technology used for harm
- Bias amplification: Encoding societal biases at scale
- Privacy leaks: Models memorizing and revealing training data

Long-term Risks:
- Goal misgeneralization: AI pursues wrong goals in new contexts
- Deceptive alignment: AI appears aligned during training but isn't
- Power-seeking behavior: AI acquiring resources beyond its mandate
- Value lock-in: Permanently encoding current values into AI
\`\`\`

## Organizations

| Organization | Focus |
|-------------|-------|
| Anthropic | Constitutional AI, alignment research |
| OpenAI | Superalignment team (safety research) |
| DeepMind | Safety team, technical alignment |
| MIRI | Mathematical foundations of alignment |
| ARC (Alignment Research Center) | Evaluating dangerous capabilities |

## Evolution

- **2014**: Bostrom publishes "Superintelligence" raising alignment concerns
- **2016**: Amodei et al. publish "Concrete Problems in AI Safety"
- **2022**: Constitutional AI introduced by Anthropic
- **2023**: AI safety becomes mainstream concern; government attention increases
- **2024+**: Safety evaluations become standard; international AI safety summits`,

    data_privacy: `# Data Privacy & GDPR

Data privacy in AI addresses how personal data is collected, used, and protected in ML systems. Regulations like GDPR and the EU AI Act set legal requirements for AI systems handling personal data.

## Key Regulations

| Regulation | Region | Key Requirements |
|-----------|--------|-----------------|
| GDPR | EU | Consent, right to explanation, data minimization |
| CCPA/CPRA | California | Consumer data rights, opt-out |
| EU AI Act | EU | Risk-based AI regulation, transparency |
| PIPEDA | Canada | Consent and accountability principles |
| LGPD | Brazil | Similar to GDPR |

## Privacy-Preserving ML Techniques

\`\`\`
Differential Privacy:
- Add calibrated noise to data or model outputs
- Guarantees: Individual data points cannot be identified
- Used by: Apple, Google, US Census Bureau
- epsilon-differential privacy: lower epsilon = more privacy

Federated Learning:
- Train models on distributed data without centralizing it
- Each device trains locally, shares only model updates
- Used by: Google Keyboard, Apple Siri

Homomorphic Encryption:
- Compute on encrypted data without decrypting
- Result when decrypted matches result on plaintext
- Slow but mathematically guaranteed privacy

Secure Multi-Party Computation:
- Multiple parties jointly compute without revealing their inputs
- Used for privacy-preserving analytics
\`\`\`

## GDPR Rights Relevant to AI

| Right | AI Implication |
|-------|---------------|
| Right to Explanation | Must explain automated decisions |
| Right to Erasure | Remove individual's data from training |
| Data Minimization | Only collect necessary data |
| Purpose Limitation | Use data only for stated purpose |
| Right to Object | Opt out of automated decision-making |

## Applications

- Healthcare ML with patient privacy
- Financial models with customer data protection
- HR AI systems with employee data
- Government AI with citizen data

## Evolution

- **2016**: GDPR adopted (effective 2018)
- **2020**: CCPA takes effect in California
- **2023**: EU AI Act proposed and negotiated
- **2024**: EU AI Act enters into force
- **2025+**: Global AI governance frameworks emerging`,

    responsible_ai: `# Responsible AI

Responsible AI is a governance approach ensuring AI systems are developed and deployed ethically, transparently, and accountably. It encompasses fairness, safety, privacy, and societal impact.

## Principles

| Principle | Description |
|-----------|-------------|
| Fairness | Avoid discrimination and bias |
| Transparency | Explain how AI makes decisions |
| Accountability | Clear ownership of AI outcomes |
| Privacy | Protect personal data |
| Safety | Prevent harm from AI systems |
| Inclusivity | Design for all users and communities |
| Reliability | Consistent, predictable performance |

## Implementation Framework

\`\`\`
Responsible AI in Practice:

1. Design Phase
   - Impact assessment: Who is affected and how?
   - Stakeholder engagement: Include diverse perspectives
   - Define fairness criteria for the specific context

2. Development Phase
   - Data governance: Source, quality, representation
   - Bias testing across demographic groups
   - Transparency documentation (Model Cards, Datasheets)

3. Deployment Phase
   - Human oversight and intervention mechanisms
   - Monitoring for drift and emergent behaviors
   - Feedback channels for affected communities

4. Ongoing Governance
   - Regular audits and impact assessments
   - Update models as society and context evolve
   - Incident response procedures
\`\`\`

## Documentation Standards

| Standard | Purpose |
|----------|---------|
| Model Cards | Document model capabilities and limitations |
| Datasheets for Datasets | Document dataset composition and biases |
| System Cards | Document end-to-end AI system behavior |
| AI Impact Assessments | Evaluate societal impact before deployment |

## Applications

- Enterprise AI governance programs
- Government AI procurement standards
- Healthcare AI certification
- Financial AI regulatory compliance

## Evolution

- **2018**: Google publishes AI Principles
- **2019**: Microsoft, IBM, and others adopt responsible AI frameworks
- **2021**: UNESCO adopts global AI ethics recommendation
- **2023**: US Executive Order on Safe AI
- **2024+**: Responsible AI becomes regulatory requirement, not just best practice`,

    ai_regulation: `# AI Regulation Landscape

The global AI regulatory landscape is rapidly evolving as governments work to balance innovation with safety, rights, and accountability. Different regions take different approaches.

## Major Regulatory Frameworks

| Framework | Region | Approach | Status |
|-----------|--------|----------|--------|
| EU AI Act | European Union | Risk-based classification | In force (2024) |
| Executive Order 14110 | United States | Voluntary + targeted rules | Active (2023) |
| AI Safety Institute | UK | Testing and evaluation | Operational |
| AI Governance Framework | Singapore | Principles-based | Active |
| AI Regulations | China | Content + algorithmic rules | Active |

## EU AI Act Risk Categories

\`\`\`
Unacceptable Risk (BANNED):
- Social scoring by governments
- Real-time biometric surveillance (with exceptions)
- Manipulative AI targeting vulnerabilities

High Risk (STRICT REQUIREMENTS):
- Critical infrastructure
- Education and employment decisions
- Law enforcement and justice
- Credit scoring and insurance
Requirements: Conformity assessment, transparency, human oversight

Limited Risk (TRANSPARENCY):
- Chatbots (must disclose AI)
- Deepfake generation (must label)
- Emotion recognition systems

Minimal Risk (NO RESTRICTIONS):
- Spam filters
- AI in video games
- Most business applications
\`\`\`

## Key Regulatory Themes

| Theme | Description |
|-------|-------------|
| Transparency | Disclose when AI is used; explain decisions |
| Accountability | Define who is responsible for AI outcomes |
| Testing | Require safety testing before deployment |
| Rights | Protect against discriminatory AI decisions |
| Data governance | Rules for training data collection and use |

## Evolution

- **2021**: EU proposes AI Act (first comprehensive AI law)
- **2023**: US Executive Order on AI; UK AI Safety Summit
- **2024**: EU AI Act enters into force; global coordination increases
- **2025+**: Enforcement begins; regulatory fragmentation vs harmonization`,

    deepfakes: `# Deepfakes & Misinformation

Deepfakes use AI to create realistic synthetic media - fake videos, audio, and images that are increasingly difficult to distinguish from real content. They pose significant challenges for trust and information integrity.

## Types of Synthetic Media

| Type | Technology | Example |
|------|-----------|---------|
| Face Swap | GANs, autoencoders | Putting one person's face on another's body |
| Face Reenactment | Facial landmark manipulation | Making someone appear to say something |
| Voice Cloning | TTS models (VALL-E, ElevenLabs) | Synthetic speech matching someone's voice |
| Text-to-Image | Diffusion models | Generating fake photorealistic images |
| Text-to-Video | Video diffusion (Sora) | Creating realistic fake video footage |
| Full Body | Motion transfer | Puppeteering someone's body |

## Detection Methods

\`\`\`
Detection Approaches:

1. Visual Artifacts
   - Inconsistent lighting and shadows
   - Blurring around face edges
   - Unnatural eye blinking patterns
   - Asymmetric facial features

2. Deep Learning Detection
   - Binary classifiers (real vs fake)
   - Trained on datasets of known deepfakes
   - Challenge: Generalization to unseen generation methods

3. Provenance and Watermarking
   - C2PA (Coalition for Content Provenance and Authenticity)
   - Digital watermarks embedded in authentic content
   - Blockchain-based content authentication

4. Metadata Analysis
   - Examine file metadata for manipulation signs
   - Compression artifact analysis
   - Temporal consistency checks for video
\`\`\`

## Impact Areas

- **Elections**: Fake videos of political candidates
- **Fraud**: Voice cloning for financial scams
- **Harassment**: Non-consensual intimate imagery
- **Journalism**: Undermining trust in media
- **Entertainment**: Authorized use in film and games

## Countermeasures

| Approach | Organization |
|----------|-------------|
| C2PA Standard | Adobe, Microsoft, BBC |
| SynthID | Google DeepMind (watermarking) |
| Content Credentials | Adobe (provenance metadata) |
| Deepfake Detection Challenge | Facebook/Meta |

## Evolution

- **2017**: "Deepfakes" term coined on Reddit
- **2019**: Deepfake detection challenges launched
- **2022**: ElevenLabs makes voice cloning accessible
- **2023**: AI-generated misinformation becomes major concern
- **2024+**: C2PA and content provenance tools gain adoption`,

    environmental_impact: `# Environmental Impact of AI

Training and running large AI models consumes significant energy and water resources. Understanding and mitigating AI's environmental footprint is increasingly important as models scale up.

## Energy Consumption

| Model | Training Energy | CO2 Equivalent |
|-------|----------------|----------------|
| BERT (110M params) | ~1,500 kWh | ~650 kg CO2 |
| GPT-3 (175B params) | ~1,300 MWh | ~550 tons CO2 |
| GPT-4 (est.) | ~50,000+ MWh | ~20,000+ tons CO2 |
| LLaMA 2 70B | ~1,700 MWh | ~700 tons CO2 |

*Estimates vary based on hardware, data center efficiency, and energy source*

## Where Energy Is Used

\`\`\`
AI Energy Breakdown:
- Training: 20-40% (one-time but massive for large models)
- Inference: 60-80% (ongoing, scales with users)
- Data Centers: Cooling, networking, storage overhead

Compute Growth:
- Training compute doubles every ~6 months
- 300,000x increase from AlexNet (2012) to GPT-4 (2023)
- LLM inference requests growing exponentially
\`\`\`

## Water Consumption

- Data centers use water for cooling
- GPT-3 training estimated to consume ~700,000 liters of water
- Microsoft reported 34% increase in water use partly due to AI (2023)
- Water scarcity makes this a significant concern

## Mitigation Strategies

| Strategy | Impact |
|----------|--------|
| Efficient architectures | Mixture of Experts uses fewer resources per token |
| Model compression | Quantization and pruning reduce inference cost |
| Renewable energy | Google, Microsoft committing to carbon-free energy |
| Efficient hardware | Each GPU generation more energy-efficient |
| Carbon-aware computing | Schedule training in regions with clean energy |
| Smaller models | Use appropriately-sized models for tasks |

## Applications of AI for Environment

- Climate modeling and weather prediction
- Energy grid optimization
- Deforestation monitoring from satellite imagery
- Carbon capture optimization
- Biodiversity tracking and conservation

## Evolution

- **2019**: Strubell et al. estimate NLP model carbon footprint
- **2021**: Google commits to 24/7 carbon-free energy by 2030
- **2022**: AI energy consumption becomes mainstream discussion
- **2024**: EU AI Act includes energy efficiency reporting requirements
- **2025+**: Green AI principles influence model development decisions`,
  };

  Object.assign(window.AI_DOCS, content);
})();
