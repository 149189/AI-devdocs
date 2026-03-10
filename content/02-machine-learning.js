// 02 - Machine Learning
(function() {
  const content = {

    // ========================================================================
    // LEARNING PARADIGMS
    // ========================================================================

    supervised_learning: `# Supervised Learning

**Supervised learning** is the most widely adopted machine learning paradigm, where models learn from labeled training data consisting of input-output pairs. The algorithm discovers a mapping function from inputs to desired outputs, enabling it to predict correct labels for unseen data. This paradigm forms the backbone of most production ML systems today.

## Key Concepts

- **Labeled Data**: Each training example includes an input \`x\` and a corresponding target label \`y\`
- **Loss Function**: Measures the discrepancy between predicted outputs and true labels (e.g., MSE, cross-entropy)
- **Generalization**: The model's ability to perform accurately on new, unseen data beyond the training set
- **Overfitting vs. Underfitting**: Balancing model complexity to avoid memorizing noise or being too simplistic
- **Bias-Variance Tradeoff**: Managing the tension between model flexibility and stability across datasets

## How It Works

1. **Data Collection**: Gather a dataset of input features \`X\` paired with labels \`Y\`
2. **Data Splitting**: Divide into training, validation, and test sets (commonly 70/15/15)
3. **Model Selection**: Choose an appropriate algorithm (linear model, tree, neural network, etc.)
4. **Training**: Optimize model parameters by minimizing the loss function on training data
5. **Validation**: Tune hyperparameters using the validation set to prevent overfitting
6. **Evaluation**: Assess final performance on the held-out test set using metrics like accuracy, F1, or AUC

| Task Type | Output | Example Metrics | Use Case |
|---|---|---|---|
| **Classification** | Discrete labels | Accuracy, Precision, Recall, F1 | Spam detection, image labeling |
| **Regression** | Continuous values | MSE, MAE, R-squared | Price prediction, demand forecasting |

## Applications

- **Medical Diagnosis**: Classifying X-ray or MRI scans to detect diseases such as pneumonia or tumors
- **Email Filtering**: Distinguishing spam from legitimate messages using text features
- **Credit Scoring**: Predicting loan default probability based on financial history
- **Speech Recognition**: Mapping audio waveforms to text transcriptions

## Evolution

| Year | Milestone |
|---|---|
| 1957 | Rosenblatt's Perceptron introduced supervised binary classification |
| 1986 | Backpropagation enabled training of multi-layer neural networks |
| 2001 | Random Forests combined ensemble methods with decision trees |
| 2012 | AlexNet won ImageNet, demonstrating deep supervised learning at scale |
| 2020 | GPT-3 showed supervised fine-tuning on massive pretrained models |`,

    unsupervised_learning: `# Unsupervised Learning

**Unsupervised learning** deals with discovering hidden patterns and structures in data without any labeled examples. The algorithm receives only input data and must find meaningful organization on its own, such as groupings, associations, or compressed representations. This paradigm is essential when labeled data is scarce, expensive, or unavailable.

## Key Concepts

- **Unlabeled Data**: Training examples consist solely of input features with no target variable
- **Clustering**: Grouping similar data points together based on distance or density measures
- **Dimensionality Reduction**: Projecting high-dimensional data into fewer dimensions while preserving structure
- **Density Estimation**: Modeling the underlying probability distribution of the data
- **Anomaly Detection**: Identifying data points that deviate significantly from normal patterns

## How It Works

1. **Data Collection**: Gather raw, unlabeled input features
2. **Preprocessing**: Normalize, scale, and handle missing values to ensure fair comparisons
3. **Algorithm Selection**: Choose between clustering (K-Means, DBSCAN), reduction (PCA, t-SNE), or association methods
4. **Model Fitting**: The algorithm iteratively discovers structure without guidance from labels
5. **Interpretation**: Analyze discovered clusters, components, or rules for business insights
6. **Validation**: Use internal metrics (silhouette score, elbow method) since external labels are absent

\`\`\`python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering example
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Dimensionality reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
\`\`\`

## Applications

- **Customer Segmentation**: Grouping customers by purchasing behavior for targeted marketing
- **Anomaly Detection**: Identifying fraudulent transactions or network intrusions
- **Gene Expression Analysis**: Discovering groups of co-expressed genes in bioinformatics
- **Topic Modeling**: Extracting themes from large document collections without labeled topics

## Evolution

| Year | Milestone |
|---|---|
| 1957 | Lloyd's algorithm (K-Means) proposed for signal quantization |
| 1901/1933 | PCA formalized by Pearson and Hotelling for dimensionality reduction |
| 1996 | DBSCAN introduced density-based clustering for arbitrary-shaped groups |
| 2008 | t-SNE enabled powerful nonlinear visualization of high-dimensional data |
| 2018 | UMAP provided faster, more scalable manifold learning for embeddings |`,

    semi_supervised: `# Semi-supervised Learning

**Semi-supervised learning** bridges the gap between supervised and unsupervised learning by leveraging a small amount of labeled data alongside a large pool of unlabeled data. This paradigm is highly practical because labeling data is often expensive and time-consuming, while unlabeled data is abundant. Semi-supervised methods can significantly improve model performance compared to using labeled data alone.

## Key Concepts

- **Label Scarcity**: Only a fraction (often 1-10%) of the training data has labels
- **Smoothness Assumption**: Nearby data points in feature space are likely to share the same label
- **Cluster Assumption**: Data tends to form discrete clusters, and points in the same cluster share a label
- **Manifold Assumption**: High-dimensional data lies on a lower-dimensional manifold; labels vary smoothly along it
- **Pseudo-labeling**: Using model predictions on unlabeled data as provisional labels for retraining

## How It Works

1. **Initial Training**: Train a base model using the small labeled dataset
2. **Pseudo-label Generation**: Use the trained model to predict labels for unlabeled data
3. **Confidence Filtering**: Select only high-confidence predictions as pseudo-labels
4. **Combined Training**: Retrain the model on both labeled data and pseudo-labeled data
5. **Iteration**: Repeat the pseudo-labeling and retraining cycle until convergence
6. **Evaluation**: Assess on a held-out labeled test set

\`\`\`python
# Pseudo-labeling approach
model.fit(X_labeled, y_labeled)

# Generate pseudo-labels for unlabeled data
pseudo_labels = model.predict(X_unlabeled)
confidence = model.predict_proba(X_unlabeled).max(axis=1)

# Filter high-confidence predictions (threshold = 0.95)
mask = confidence > 0.95
X_combined = concat(X_labeled, X_unlabeled[mask])
y_combined = concat(y_labeled, pseudo_labels[mask])

# Retrain with expanded dataset
model.fit(X_combined, y_combined)
\`\`\`

## Applications

- **Medical Imaging**: Using few expert-annotated scans with thousands of unlabeled images
- **Web Content Classification**: Categorizing web pages where manual labeling covers a tiny fraction
- **Speech Recognition**: Training ASR systems when only a small subset of audio has transcriptions
- **Satellite Imagery**: Classifying land use from satellite photos with sparse ground-truth labels

## Evolution

| Year | Milestone |
|---|---|
| 1998 | Transductive SVMs introduced semi-supervised margin-based learning |
| 2004 | Co-training formalized using multiple views of unlabeled data |
| 2006 | Graph-based methods leveraged data topology for label propagation |
| 2013 | Pseudo-labeling gained popularity in deep learning workflows |
| 2020 | FixMatch combined consistency regularization with pseudo-labels to set new benchmarks |`,

    self_supervised: `# Self-supervised Learning

**Self-supervised learning** (SSL) is a powerful paradigm where the model generates its own supervisory signal from the structure of the input data itself, eliminating the need for human-annotated labels. By designing **pretext tasks** that require understanding data structure, models learn rich, transferable representations. SSL has become the foundation of modern large language models and vision transformers.

## Key Concepts

- **Pretext Task**: An automatically generated task that forces the model to learn useful representations (e.g., predicting masked tokens)
- **Contrastive Learning**: Training the model to pull similar pairs closer and push dissimilar pairs apart in embedding space
- **Masked Prediction**: Hiding portions of the input and training the model to reconstruct them
- **Negative Sampling**: Using unrelated data points as negative examples to avoid representation collapse
- **Downstream Transfer**: Using learned representations as features or initialization for supervised tasks

## How It Works

1. **Design a Pretext Task**: Choose a self-supervision strategy (masking, rotation prediction, contrastive pairing)
2. **Generate Training Signal**: Automatically create labels from the data itself (e.g., masked token identity)
3. **Pretrain the Model**: Train on the pretext task using large amounts of unlabeled data
4. **Learn Representations**: The model's hidden layers capture generalizable features about data structure
5. **Fine-tune**: Transfer the pretrained model to downstream tasks with a small labeled dataset
6. **Evaluate**: Measure performance on target benchmarks to validate representation quality

| Method | Strategy | Domain | Example |
|---|---|---|---|
| **BERT** | Masked token prediction | NLP | Predicting hidden words in sentences |
| **GPT** | Next token prediction | NLP | Autoregressive language modeling |
| **SimCLR** | Contrastive augmentations | Vision | Comparing augmented views of the same image |
| **MAE** | Masked image patches | Vision | Reconstructing hidden regions of images |

## Applications

- **Language Models**: GPT and BERT learn grammar, facts, and reasoning through text prediction
- **Visual Representations**: Models like DINO learn object features without any labeled images
- **Audio Understanding**: Wav2Vec learns speech representations from raw audio waveforms
- **Protein Folding**: ESM models learn protein structure from amino acid sequences

## Evolution

| Year | Milestone |
|---|---|
| 2018 | BERT popularized masked language modeling as a pretraining strategy |
| 2019 | GPT-2 demonstrated the power of autoregressive self-supervision at scale |
| 2020 | SimCLR and MoCo brought contrastive self-supervised learning to vision |
| 2021 | DINO and BEiT showed self-supervised ViTs matching supervised performance |
| 2022 | Masked Autoencoders (MAE) proved efficient self-supervised pretraining for images |`,

    rl_intro: `# Reinforcement Learning Introduction

**Reinforcement Learning** (RL) is a learning paradigm where an **agent** learns to make sequential decisions by interacting with an **environment**, receiving **rewards** or **penalties** based on its actions. Unlike supervised learning, there are no explicit correct answers; the agent must discover optimal behavior through trial and error. RL has achieved superhuman performance in games, robotics, and is central to aligning modern LLMs.

## Key Concepts

- **Agent**: The learner and decision-maker that interacts with the environment
- **Environment**: The external system the agent operates within and observes
- **State (s)**: A representation of the current situation in the environment
- **Action (a)**: A choice made by the agent that affects the environment
- **Reward (r)**: A scalar feedback signal indicating how good an action was
- **Policy (pi)**: A strategy mapping states to actions; the agent's behavior function
- **Value Function V(s)**: The expected cumulative future reward from a given state
- **Discount Factor (gamma)**: Controls how much future rewards are valued relative to immediate ones

## How It Works

1. **Observe State**: The agent perceives the current state \`s_t\` of the environment
2. **Select Action**: Using its policy, the agent chooses action \`a_t\`
3. **Environment Response**: The environment transitions to a new state \`s_{t+1}\` and emits reward \`r_t\`
4. **Update Policy**: The agent updates its strategy to maximize cumulative reward over time
5. **Repeat**: This loop continues across many episodes until the policy converges
6. **Exploit**: The trained agent uses its learned policy to act optimally

\`\`\`
Agent-Environment Loop:
    s_t --> [Agent: pi(s_t) = a_t] --> Environment
                                          |
    s_{t+1}, r_t <------------------------+
\`\`\`

| Approach | Description | Example Algorithm |
|---|---|---|
| **Value-based** | Learn the value of states/actions, derive policy | Q-Learning, DQN |
| **Policy-based** | Directly learn the policy function | REINFORCE, PPO |
| **Actor-Critic** | Combine value estimation with policy learning | A2C, SAC |
| **Model-based** | Learn a model of the environment to plan | Dyna-Q, MuZero |

## Applications

- **Game Playing**: AlphaGo, AlphaZero, and OpenAI Five achieved superhuman play in Go, Chess, and Dota 2
- **Robotics**: Training robotic arms for manipulation, locomotion, and dexterous tasks
- **RLHF**: Aligning large language models with human preferences using reward models
- **Resource Management**: Optimizing data center cooling, network routing, and traffic signals

## Evolution

| Year | Milestone |
|---|---|
| 1992 | TD-Gammon learned backgammon through self-play and temporal difference learning |
| 2013 | DQN combined deep learning with Q-learning to play Atari games from pixels |
| 2016 | AlphaGo defeated world champion Lee Sedol in the game of Go |
| 2017 | PPO became the de facto policy optimization algorithm for practical RL |
| 2022 | RLHF used to align ChatGPT, bringing RL into mainstream AI alignment |`,

    online_learning: `# Online Learning

**Online learning** is a paradigm where a model is trained incrementally as new data arrives one example (or mini-batch) at a time, rather than having access to the entire dataset upfront. This is essential for real-world systems where data streams continuously, distributions shift over time, and storing all historical data is infeasible. The model updates its parameters after each observation.

## Key Concepts

- **Sequential Updates**: The model processes one data point at a time and updates immediately
- **Regret Minimization**: The goal is to minimize cumulative loss relative to the best fixed model in hindsight
- **Concept Drift**: The underlying data distribution changes over time, requiring the model to adapt
- **Learning Rate Decay**: Gradually reducing the step size to stabilize convergence as more data is seen
- **Forgetting Factor**: Weighting recent observations more heavily than older ones to handle non-stationarity

## How It Works

1. **Initialize Model**: Start with initial parameters (random or from a pretrained model)
2. **Receive Example**: A single data point \`(x_t, y_t)\` arrives from the stream
3. **Make Prediction**: The model predicts \`y_hat_t\` using current parameters
4. **Compute Loss**: Calculate the error between prediction and true value
5. **Update Parameters**: Adjust model weights using gradient descent on the single example
6. **Repeat**: Continue for each new data point indefinitely

\`\`\`python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss='log_loss')

# Online learning loop
for X_batch, y_batch in data_stream:
    model.partial_fit(X_batch, y_batch, classes=[0, 1])
    # Model is ready for prediction after each update
    prediction = model.predict(X_new)
\`\`\`

| Variant | Description | Key Feature |
|---|---|---|
| **Stochastic Gradient Descent** | Updates on single examples | Foundation of online optimization |
| **Online Convex Optimization** | Theoretical framework for online decisions | Regret bounds |
| **Bandit Algorithms** | Balance exploration and exploitation | Limited feedback |
| **Streaming Algorithms** | Process data in a single pass | Memory efficient |

## Applications

- **Stock Price Prediction**: Continuously updating forecasts as market data streams in
- **Ad Click Prediction**: Adapting click-through-rate models in real time as user behavior evolves
- **Spam Filtering**: Updating email classifiers as new spam patterns emerge
- **Recommendation Systems**: Adjusting recommendations based on the latest user interactions

## Evolution

| Year | Milestone |
|---|---|
| 1958 | Perceptron algorithm introduced as the first online learning method |
| 1997 | Online convex optimization framework formalized by Zinkevich |
| 2002 | Bandit algorithms (UCB) provided principled exploration-exploitation tradeoffs |
| 2011 | Follow The Regularized Leader (FTRL) became standard for large-scale online ads |
| 2019 | Online learning integrated into production recommendation engines at massive scale |`,

    transfer_learning: `# Transfer Learning

**Transfer learning** is a paradigm where knowledge gained from training a model on one task (the **source**) is reused to improve performance on a different but related task (the **target**). Instead of training from scratch, models start from pretrained weights and adapt them, dramatically reducing the data, compute, and time required. This paradigm has become the dominant approach in modern deep learning.

## Key Concepts

- **Source Domain/Task**: The original task and dataset the model was initially trained on
- **Target Domain/Task**: The new task where we want to apply the transferred knowledge
- **Feature Extraction**: Using pretrained layers as fixed feature extractors without updating their weights
- **Fine-tuning**: Unfreezing some or all pretrained layers and retraining them on target data with a small learning rate
- **Domain Adaptation**: Techniques to handle distribution shifts between source and target domains
- **Negative Transfer**: When transferring hurts performance because source and target are too dissimilar

## How It Works

1. **Pretrain**: Train a large model on a data-rich source task (e.g., ImageNet, large text corpus)
2. **Select Layers**: Decide which layers of the pretrained model to reuse
3. **Adapt Architecture**: Replace or add task-specific layers (e.g., new classification head)
4. **Feature Extraction or Fine-tune**: Either freeze pretrained layers or unfreeze them for further training
5. **Train on Target**: Train the adapted model on the (typically smaller) target dataset
6. **Evaluate**: Validate performance on the target task

\`\`\`python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classification head for new task
model.fc = nn.Linear(2048, num_target_classes)

# Fine-tune only the new head (or unfreeze last few layers)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
\`\`\`

## Applications

- **Medical Imaging**: Fine-tuning ImageNet-pretrained CNNs for X-ray diagnosis with limited labeled scans
- **NLP Tasks**: Using BERT or GPT pretrained on general text for sentiment analysis, NER, or QA
- **Autonomous Driving**: Transferring models trained in simulation to real-world driving scenarios
- **Low-resource Languages**: Transferring multilingual models to languages with very little training data

## Evolution

| Year | Milestone |
|---|---|
| 1976 | Bozinovski and Fulgosi first described transfer in neural network training |
| 2014 | ImageNet-pretrained CNNs became standard starting features for vision tasks |
| 2018 | ULMFiT and BERT established transfer learning as the default approach in NLP |
| 2020 | GPT-3 demonstrated few-shot transfer via in-context learning without fine-tuning |
| 2023 | Foundation models made transfer learning the universal paradigm across modalities |`,

    meta_learning: `# Meta Learning

**Meta learning**, often called "learning to learn," is a paradigm where models are trained across many tasks so they can rapidly adapt to new, unseen tasks with very few examples. Rather than optimizing for a single task, meta learning optimizes the learning process itself, discovering initialization strategies, update rules, or model architectures that facilitate fast adaptation.

## Key Concepts

- **Task Distribution**: A collection of related tasks from which episodes are sampled during training
- **Few-shot Learning**: The ability to learn a new task from only a handful of labeled examples (1-shot, 5-shot)
- **Support Set**: The small set of labeled examples provided for a new task
- **Query Set**: The examples used to evaluate the model's performance on the new task
- **Inner Loop**: Fast adaptation to a specific task using the support set
- **Outer Loop**: Slow optimization of meta-parameters across many tasks

## How It Works

1. **Sample Task**: Draw a task \`T_i\` from the task distribution
2. **Provide Support Set**: Give the model a few labeled examples from \`T_i\`
3. **Inner Adaptation**: The model adapts its parameters to the support set (fast learning)
4. **Evaluate on Query Set**: Measure performance on held-out examples from the same task
5. **Meta-Update**: Update the meta-parameters to improve adaptation across all tasks (outer loop)
6. **Repeat**: Continue sampling tasks and updating until meta-parameters converge

\`\`\`python
# MAML-style meta-learning (simplified)
for episode in range(num_episodes):
    task = sample_task(task_distribution)
    support_X, support_y = task.support_set()
    query_X, query_y = task.query_set()

    # Inner loop: adapt to task
    adapted_params = model.params.clone()
    for step in range(inner_steps):
        loss = compute_loss(model(support_X, adapted_params), support_y)
        adapted_params -= inner_lr * grad(loss, adapted_params)

    # Outer loop: meta-update
    meta_loss = compute_loss(model(query_X, adapted_params), query_y)
    model.params -= outer_lr * grad(meta_loss, model.params)
\`\`\`

| Approach | Method | Key Idea |
|---|---|---|
| **Optimization-based** | MAML, Reptile | Learn good initial parameters for fast gradient adaptation |
| **Metric-based** | Prototypical Networks, Siamese | Learn an embedding space where classes are separable by distance |
| **Model-based** | SNAIL, Memory-Augmented NN | Use external memory or recurrence to store task information |

## Applications

- **Few-shot Image Classification**: Recognizing new object categories from just 1-5 example images
- **Drug Discovery**: Rapidly predicting molecular properties for novel compounds with limited experimental data
- **Robotics**: Enabling robots to learn new manipulation tasks in minutes rather than hours
- **Personalization**: Quickly adapting recommendation models to individual user preferences

## Evolution

| Year | Milestone |
|---|---|
| 1998 | Thrun and Pratt formalized "Learning to Learn" as a research area |
| 2016 | Matching Networks introduced metric-based few-shot learning |
| 2017 | MAML (Model-Agnostic Meta-Learning) established optimization-based meta learning |
| 2019 | Prototypical Networks and task-augmentation strategies improved few-shot benchmarks |
| 2023 | In-context learning in LLMs recognized as an emergent form of meta learning |`,

    active_learning: `# Active Learning

**Active learning** is a paradigm where the model intelligently selects which data points should be labeled next by a human annotator, rather than passively receiving randomly labeled data. By strategically choosing the most informative examples for labeling, active learning achieves better performance with far fewer labeled samples, reducing annotation costs significantly.

## Key Concepts

- **Query Strategy**: The method used to select which unlabeled instances to request labels for
- **Uncertainty Sampling**: Selecting examples where the model is least confident in its prediction
- **Query by Committee**: Using an ensemble of models and selecting points where they disagree most
- **Expected Model Change**: Choosing examples that would cause the greatest update to model parameters
- **Annotation Budget**: The limited number of labels the annotator can provide
- **Oracle**: The entity (human expert) that provides ground-truth labels when queried

## How It Works

1. **Initialize**: Train an initial model on a small seed set of labeled data
2. **Predict on Pool**: Use the model to make predictions on the unlabeled data pool
3. **Apply Query Strategy**: Rank unlabeled examples by their informativeness score
4. **Select Top-k**: Choose the most informative examples for human labeling
5. **Annotate**: The oracle provides labels for the selected examples
6. **Retrain**: Add newly labeled data to the training set and retrain the model
7. **Repeat**: Continue the cycle until the budget is exhausted or performance is satisfactory

\`\`\`python
import numpy as np

def uncertainty_sampling(model, X_unlabeled, n_queries=10):
    """Select examples where the model is most uncertain."""
    probs = model.predict_proba(X_unlabeled)
    # Entropy-based uncertainty
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    # Select top-n most uncertain examples
    query_indices = np.argsort(entropy)[-n_queries:]
    return query_indices

# Active learning loop
for cycle in range(num_cycles):
    indices = uncertainty_sampling(model, X_pool)
    new_labels = oracle.label(X_pool[indices])
    X_train = concat(X_train, X_pool[indices])
    y_train = concat(y_train, new_labels)
    X_pool = remove(X_pool, indices)
    model.fit(X_train, y_train)
\`\`\`

## Applications

- **Medical Annotation**: Prioritizing which medical images need expert radiologist review
- **NLP Labeling**: Selecting the most ambiguous text samples for human annotators
- **Autonomous Driving**: Identifying edge-case driving scenarios that need manual labeling
- **Document Classification**: Efficiently labeling legal or regulatory documents

## Evolution

| Year | Milestone |
|---|---|
| 1994 | Uncertainty sampling introduced by Lewis and Gale for text classification |
| 2007 | Settles' survey systematized active learning strategies and benchmarks |
| 2017 | Deep active learning combined neural networks with query strategies |
| 2020 | Batch active learning methods scaled to large datasets and deep models |
| 2023 | Active learning integrated with LLM-based annotation pipelines |`,

    federated_learning: `# Federated Learning

**Federated learning** (FL) is a decentralized machine learning paradigm where multiple devices or institutions collaboratively train a shared model without exchanging their raw data. Each participant trains locally on their own data and only shares model updates (gradients or weights), which are aggregated by a central server. This preserves data privacy while enabling learning from distributed data sources.

## Key Concepts

- **Data Privacy**: Raw data never leaves the local device or institution
- **Local Training**: Each client trains on its own data independently
- **Model Aggregation**: A central server combines updates from all clients into a global model
- **Communication Rounds**: The iterative process of distributing the model, local training, and aggregation
- **Non-IID Data**: Participants often have non-identically distributed data, creating optimization challenges
- **Differential Privacy**: Adding noise to updates to provide mathematical privacy guarantees

## How It Works

1. **Initialize**: The server creates a global model and distributes it to all participating clients
2. **Local Training**: Each client trains the model on its local data for several epochs
3. **Upload Updates**: Clients send their model updates (weight deltas) to the central server
4. **Aggregate**: The server combines updates using an aggregation strategy (e.g., FedAvg)
5. **Distribute**: The updated global model is sent back to all clients
6. **Repeat**: The process continues for multiple communication rounds until convergence

\`\`\`python
# Federated Averaging (FedAvg) - simplified
def federated_round(global_model, clients):
    local_updates = []
    for client in clients:
        # Distribute global model
        local_model = copy(global_model)
        # Local training
        local_model.train(client.data, epochs=local_epochs)
        # Collect update
        local_updates.append(local_model.get_weights())

    # Aggregate: weighted average by dataset size
    total_samples = sum(c.num_samples for c in clients)
    new_weights = weighted_average(
        local_updates,
        weights=[c.num_samples / total_samples for c in clients]
    )
    global_model.set_weights(new_weights)
    return global_model
\`\`\`

## Applications

- **Mobile Keyboard Prediction**: Google's Gboard trains next-word prediction across millions of phones
- **Healthcare**: Hospitals collaboratively train diagnostic models without sharing patient records
- **Financial Fraud Detection**: Banks jointly improve fraud models while keeping transaction data private
- **Edge IoT**: Smart devices collectively learn patterns without uploading sensor data to the cloud

## Evolution

| Year | Milestone |
|---|---|
| 2016 | Google introduced Federated Learning and FedAvg algorithm |
| 2017 | Federated learning deployed in production for Gboard next-word prediction |
| 2019 | FedProx addressed data heterogeneity challenges in non-IID settings |
| 2020 | FATE and PySyft frameworks enabled open-source federated learning research |
| 2023 | Federated learning applied to cross-institutional LLM fine-tuning |`,

    continual_learning: `# Continual Learning

**Continual learning** (also called lifelong or incremental learning) is a paradigm where models learn from a continuous stream of tasks or data distributions over time without forgetting previously acquired knowledge. The central challenge is **catastrophic forgetting**, where training on new tasks overwrites the neural network weights important for earlier tasks.

## Key Concepts

- **Catastrophic Forgetting**: The tendency of neural networks to abruptly lose old knowledge when learning new information
- **Plasticity-Stability Dilemma**: Balancing the ability to learn new tasks (plasticity) with retaining old knowledge (stability)
- **Task Incremental**: New classes or tasks arrive sequentially, and the model must handle all of them
- **Experience Replay**: Storing and revisiting a subset of old data to maintain previous performance
- **Regularization-based**: Adding penalty terms that prevent important weights from changing too much
- **Progressive Expansion**: Growing the model architecture to accommodate new tasks

## How It Works

1. **Learn Task 1**: Train the model normally on the first task
2. **Identify Important Weights**: Determine which parameters are critical for the current task
3. **New Task Arrives**: Receive data for a new task or shifted distribution
4. **Constrained Learning**: Train on the new task while protecting important weights or replaying old data
5. **Evaluate All Tasks**: Measure performance across all tasks seen so far (not just the latest)
6. **Repeat**: Continue as new tasks arrive over the model's lifetime

\`\`\`python
# Elastic Weight Consolidation (EWC) - simplified
class EWC:
    def __init__(self, model, old_data, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Compute Fisher Information for each parameter
        self.fisher = compute_fisher(model, old_data)
        # Store optimal parameters from previous task
        self.old_params = {n: p.clone() for n, p in model.named_parameters()}

    def penalty(self):
        loss = 0
        for name, param in self.model.named_parameters():
            # Penalize changes to important weights
            loss += (self.fisher[name] * (param - self.old_params[name])**2).sum()
        return self.lambda_ewc * loss

# Training loop for new task
for x, y in new_task_data:
    task_loss = criterion(model(x), y)
    ewc_loss = ewc.penalty()
    total_loss = task_loss + ewc_loss
    total_loss.backward()
    optimizer.step()
\`\`\`

| Strategy | Method | Mechanism |
|---|---|---|
| **Regularization** | EWC, SI, MAS | Penalize changes to important parameters |
| **Replay** | Experience Replay, GEM | Store and revisit old task examples |
| **Architecture** | Progressive Nets, PackNet | Allocate new capacity for new tasks |
| **Generative Replay** | DGR, CGAN-based | Use a generative model to synthesize old data |

## Applications

- **Robotics**: Robots learning new manipulation skills without forgetting earlier ones
- **Chatbots**: Dialogue systems expanding to new domains while maintaining quality on existing ones
- **Autonomous Vehicles**: Adapting to new road types and conditions over the vehicle's lifetime
- **Personal Assistants**: Continuously learning user preferences without losing general capabilities

## Evolution

| Year | Milestone |
|---|---|
| 1989 | McCloskey and Cohen first documented catastrophic forgetting in neural networks |
| 2017 | Elastic Weight Consolidation (EWC) provided a principled regularization approach |
| 2017 | Progressive Neural Networks proposed architecture expansion for continual learning |
| 2019 | Experience replay methods showed strong results on complex continual benchmarks |
| 2023 | Continual learning techniques applied to adapting large foundation models over time |`,

    // ========================================================================
    // CLASSICAL ALGORITHMS
    // ========================================================================

    regression: `# Linear & Logistic Regression

**Regression** methods are among the most foundational algorithms in machine learning. **Linear regression** models the relationship between input features and a continuous output as a linear function. **Logistic regression**, despite its name, is a classification algorithm that uses the logistic (sigmoid) function to predict class probabilities. Both are interpretable, efficient, and serve as important baselines.

## Key Concepts

- **Linear Regression**: Predicts a continuous value as a weighted sum of features: \`y = w^T * x + b\`
- **Logistic Regression**: Passes the linear output through a sigmoid function for binary classification
- **Cost Function**: MSE (mean squared error) for linear; cross-entropy (log loss) for logistic
- **Gradient Descent**: Iterative optimization of weights to minimize the cost function
- **Regularization**: L1 (Lasso) for sparsity, L2 (Ridge) for weight decay, Elastic Net for both
- **Multicollinearity**: When input features are highly correlated, destabilizing weight estimates

## How It Works

1. **Initialize Weights**: Set model parameters \`w\` and bias \`b\` (typically to zeros or small random values)
2. **Compute Prediction**: Calculate \`y_hat = w^T * x + b\` (linear) or \`sigma(w^T * x + b)\` (logistic)
3. **Calculate Loss**: Compute MSE or cross-entropy between predictions and true values
4. **Compute Gradients**: Determine the direction and magnitude of weight updates
5. **Update Weights**: Adjust parameters using gradient descent: \`w = w - lr * gradient\`
6. **Converge**: Repeat until the loss stops decreasing significantly

\`\`\`python
import numpy as np

# Linear Regression (closed-form solution)
# w = (X^T X)^{-1} X^T y
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Logistic Regression with gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(num_epochs):
    z = X @ w + b
    y_hat = sigmoid(z)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    dw = (1/m) * X.T @ (y_hat - y)
    db = (1/m) * np.sum(y_hat - y)
    w -= learning_rate * dw
    b -= learning_rate * db
\`\`\`

| Variant | Type | Regularization | Best For |
|---|---|---|---|
| **Linear Regression** | Regression | None | Simple linear relationships |
| **Ridge Regression** | Regression | L2 | Multicollinear features |
| **Lasso Regression** | Regression | L1 | Feature selection |
| **Logistic Regression** | Classification | L1/L2 | Binary/multi-class classification |

## Applications

- **House Price Prediction**: Estimating property values from features like area, location, and bedrooms
- **Medical Risk Scoring**: Predicting disease probability from patient vitals and lab results
- **Marketing Attribution**: Measuring the impact of advertising channels on sales revenue
- **Click-through Rate**: Predicting the probability a user clicks an ad (logistic regression)

## Evolution

| Year | Milestone |
|---|---|
| 1805 | Legendre published the method of least squares for linear regression |
| 1958 | Cox formalized logistic regression for binary outcome analysis |
| 1970 | Ridge regression (Tikhonov regularization) addressed multicollinearity |
| 1996 | Lasso regression introduced L1 regularization for sparse models |
| 2005 | Elastic Net combined L1 and L2 penalties for improved regularization |`,

    decision_trees: `# Decision Trees & Random Forests

**Decision trees** are intuitive models that learn hierarchical if-then rules by recursively splitting data on feature values. They partition the feature space into regions and assign predictions to each region. **Random forests** are an ensemble of many decision trees that reduces overfitting and improves generalization by combining diverse trees trained on random subsets of data and features.

## Key Concepts

- **Root Node**: The topmost decision point that initiates the first split
- **Splitting Criterion**: A metric to determine the best feature and threshold for splitting (Gini, Entropy, MSE)
- **Gini Impurity**: Measures the probability of misclassifying a randomly chosen element
- **Information Gain**: Reduction in entropy after a split (used in ID3/C4.5)
- **Pruning**: Removing branches that provide little predictive power to reduce overfitting
- **Bagging**: Bootstrap Aggregating -- training each tree on a random sample with replacement
- **Feature Randomness**: Each split considers only a random subset of features (Random Forest)

## How It Works

**Decision Tree:**
1. **Start at Root**: Consider all training examples at the root node
2. **Find Best Split**: For each feature, evaluate all possible thresholds using the splitting criterion
3. **Split Data**: Partition data into child nodes based on the best feature/threshold
4. **Recurse**: Repeat splitting for each child node
5. **Stop**: When a stopping criterion is met (max depth, min samples, pure node)
6. **Predict**: Traverse the tree for new inputs and return the leaf node's majority class or mean value

**Random Forest:**
1. Create \`n_estimators\` bootstrap samples from training data
2. Train a decision tree on each sample, using random feature subsets at each split
3. Aggregate predictions by majority vote (classification) or averaging (regression)

\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Single decision tree
dt = DecisionTreeClassifier(max_depth=5, criterion='gini')
dt.fit(X_train, y_train)

# Random forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',  # sqrt(n_features) at each split
    random_state=42
)
rf.fit(X_train, y_train)
importance = rf.feature_importances_
\`\`\`

| Property | Decision Tree | Random Forest |
|---|---|---|
| **Interpretability** | High (visualizable rules) | Lower (ensemble of trees) |
| **Overfitting Risk** | High | Low (bagging reduces variance) |
| **Training Speed** | Fast | Moderate (many trees) |
| **Feature Importance** | Available | More robust aggregated importance |

## Applications

- **Credit Risk Assessment**: Banks use tree-based models for transparent loan approval decisions
- **Medical Diagnosis**: Decision trees provide explainable diagnostic pathways for clinicians
- **Fraud Detection**: Random forests identify fraudulent patterns across many transaction features
- **Customer Churn**: Predicting which customers will cancel subscriptions based on behavior features

## Evolution

| Year | Milestone |
|---|---|
| 1984 | CART (Classification and Regression Trees) published by Breiman et al. |
| 1986 | ID3 algorithm by Quinlan introduced information gain splitting |
| 1993 | C4.5 improved ID3 with pruning, continuous features, and missing values |
| 2001 | Random Forests introduced by Breiman, combining bagging with random feature selection |
| 2016 | XGBoost and LightGBM pushed gradient-boosted trees to competition-winning dominance |`,

    svm: `# Support Vector Machines

**Support Vector Machines** (SVMs) are powerful supervised learning models that find the optimal **hyperplane** separating data into classes with the maximum possible **margin**. By maximizing the distance between the decision boundary and the nearest data points (**support vectors**), SVMs achieve strong generalization. The **kernel trick** enables SVMs to handle non-linearly separable data by implicitly mapping features into higher-dimensional spaces.

## Key Concepts

- **Hyperplane**: The decision boundary that separates classes in feature space
- **Margin**: The distance between the hyperplane and the closest data points from either class
- **Support Vectors**: The critical training points that lie closest to the decision boundary and define it
- **Hard Margin**: Requires perfect separation; only works for linearly separable data
- **Soft Margin**: Allows some misclassifications using a penalty parameter \`C\` for flexibility
- **Kernel Trick**: Maps data into higher dimensions without explicitly computing the transformation

## How It Works

1. **Define the Objective**: Maximize the margin between classes while minimizing classification errors
2. **Formulate as Optimization**: Solve a constrained quadratic programming problem
3. **Identify Support Vectors**: The data points that lie on or within the margin boundaries
4. **Apply Kernel (if needed)**: Use a kernel function to handle non-linear separation
5. **Compute Decision Boundary**: The hyperplane defined by support vectors and their weights
6. **Classify New Data**: Determine which side of the hyperplane a new point falls on

\`\`\`python
from sklearn.svm import SVC

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

# RBF Kernel SVM for non-linear data
svm_rbf = SVC(kernel='rbf', C=10.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

# Support vectors
print(f"Number of support vectors: {len(svm_rbf.support_vectors_)}")
\`\`\`

| Kernel | Formula | Best For |
|---|---|---|
| **Linear** | \`K(x,y) = x^T y\` | Linearly separable data, text classification |
| **Polynomial** | \`K(x,y) = (x^T y + c)^d\` | Image processing, interaction features |
| **RBF (Gaussian)** | \`K(x,y) = exp(-gamma * ||x-y||^2)\` | Most non-linear problems, default choice |
| **Sigmoid** | \`K(x,y) = tanh(alpha * x^T y + c)\` | Neural network-like behavior |

## Applications

- **Text Classification**: SVMs excel at high-dimensional sparse data like TF-IDF document vectors
- **Handwriting Recognition**: Classifying handwritten digits (MNIST) with kernel SVMs
- **Bioinformatics**: Classifying proteins, gene expression data, and drug activity
- **Image Classification**: Object recognition using histogram of gradients (HOG) features with SVMs

## Evolution

| Year | Milestone |
|---|---|
| 1963 | Vapnik and Chervonenkis introduced the original linear classifier concept |
| 1992 | Boser, Guyon, and Vapnik introduced the kernel trick for non-linear SVMs |
| 1995 | Cortes and Vapnik published the soft-margin SVM formulation |
| 1998 | SVMs became the dominant method for text classification and handwriting recognition |
| 2001 | Sequential Minimal Optimization (SMO) made SVM training practical for large datasets |`,

    knn: `# k-Nearest Neighbors

**k-Nearest Neighbors** (k-NN) is one of the simplest and most intuitive machine learning algorithms. It is a **non-parametric**, **instance-based** (lazy) learning method that makes predictions by finding the \`k\` closest training examples to a query point and aggregating their labels. k-NN stores all training data and defers computation until prediction time, making it a "lazy learner" with no explicit training phase.

## Key Concepts

- **Instance-based Learning**: No explicit model is trained; the algorithm memorizes the entire training set
- **Distance Metric**: The measure used to determine closeness (Euclidean, Manhattan, Cosine, Minkowski)
- **k Parameter**: The number of nearest neighbors to consider when making a prediction
- **Majority Vote**: For classification, the predicted class is the most common among k neighbors
- **Weighted Voting**: Closer neighbors can be given higher influence in the prediction
- **Curse of Dimensionality**: Performance degrades in high-dimensional spaces as distances become uniform

## How It Works

1. **Store Training Data**: Save all training examples in memory (no model fitting)
2. **Receive Query Point**: A new data point arrives for prediction
3. **Compute Distances**: Calculate the distance from the query to every training example
4. **Find k Nearest**: Select the \`k\` training points with smallest distances
5. **Aggregate**: For classification, take majority vote; for regression, compute mean/weighted mean
6. **Return Prediction**: Output the aggregated result

\`\`\`python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# k-NN classifier
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean',
    weights='distance'  # weight by inverse distance
)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Manual implementation for intuition
def knn_predict(X_train, y_train, x_query, k=5):
    distances = np.sqrt(np.sum((X_train - x_query)**2, axis=1))
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return np.bincount(nearest_labels).argmax()
\`\`\`

| Distance Metric | Formula | Best For |
|---|---|---|
| **Euclidean** | \`sqrt(sum((x_i - y_i)^2))\` | Continuous features, default choice |
| **Manhattan** | \`sum(|x_i - y_i|)\` | Grid-like data, sparse features |
| **Cosine** | \`1 - (x . y)/(||x|| * ||y||)\` | Text/document similarity |
| **Minkowski** | \`(sum(|x_i - y_i|^p))^(1/p)\` | Generalized distance (p=1: Manhattan, p=2: Euclidean) |

## Applications

- **Recommendation Systems**: Finding similar users or items for collaborative filtering
- **Anomaly Detection**: Points with very distant neighbors are flagged as outliers
- **Image Recognition**: Classifying handwritten digits by comparing pixel patterns
- **Imputation**: Filling missing values using the average of nearest neighbors' values

## Evolution

| Year | Milestone |
|---|---|
| 1951 | Fix and Hodges introduced the nearest neighbor classification rule |
| 1967 | Cover and Hart proved that k-NN error rate converges to at most 2x the Bayes optimal rate |
| 1975 | KD-trees introduced to accelerate nearest neighbor search in low dimensions |
| 2002 | Locality-Sensitive Hashing (LSH) enabled approximate nearest neighbor search at scale |
| 2019 | FAISS and Annoy libraries made k-NN practical for billion-scale vector retrieval |`,

    naive_bayes: `# Naive Bayes

**Naive Bayes** is a family of probabilistic classifiers based on **Bayes' theorem** with the "naive" assumption that all features are **conditionally independent** given the class label. Despite this simplifying assumption rarely holding in practice, Naive Bayes classifiers are remarkably effective, computationally efficient, and particularly well-suited for high-dimensional data like text classification.

## Key Concepts

- **Bayes' Theorem**: \`P(class|features) = P(features|class) * P(class) / P(features)\`
- **Prior Probability P(class)**: The base rate of each class in the training data
- **Likelihood P(features|class)**: The probability of observing the features given a class
- **Posterior P(class|features)**: The updated probability of the class after observing features
- **Conditional Independence**: The naive assumption that features are independent given the class
- **Laplace Smoothing**: Adding a small count to all feature probabilities to avoid zero-probability issues

## How It Works

1. **Compute Priors**: Calculate \`P(class)\` for each class from training label frequencies
2. **Estimate Likelihoods**: For each feature, compute \`P(feature|class)\` from training data
3. **Apply Independence Assumption**: Decompose joint likelihood as product of individual feature likelihoods
4. **Classify New Instance**: Compute posterior for each class: \`P(class) * product(P(feature_i|class))\`
5. **Select Maximum**: Assign the class with the highest posterior probability
6. **Apply Smoothing**: Use Laplace smoothing to handle unseen feature values

\`\`\`python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Gaussian Naive Bayes (continuous features)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Multinomial Naive Bayes (text classification with word counts)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(documents)
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
mnb.fit(X_counts, labels)

# Manual Bayes calculation
# P(spam|words) proportional to P(spam) * P(word1|spam) * P(word2|spam) * ...
\`\`\`

| Variant | Feature Type | Distribution Assumed | Common Use |
|---|---|---|---|
| **Gaussian NB** | Continuous | Gaussian (normal) | General classification |
| **Multinomial NB** | Discrete counts | Multinomial | Text classification (word counts) |
| **Bernoulli NB** | Binary | Bernoulli | Text classification (word presence) |
| **Complement NB** | Discrete counts | Complement of Multinomial | Imbalanced text datasets |

## Applications

- **Spam Filtering**: One of the earliest and most successful applications of Naive Bayes
- **Sentiment Analysis**: Classifying reviews as positive or negative from word frequencies
- **Document Categorization**: Assigning news articles to topics (sports, politics, tech)
- **Medical Diagnosis**: Quick probabilistic assessment of disease given symptoms

## Evolution

| Year | Milestone |
|---|---|
| 1763 | Thomas Bayes' theorem published posthumously, laying the theoretical foundation |
| 1961 | Naive Bayes first applied to pattern recognition and text classification |
| 1998 | McCallum and Nigam compared Multinomial vs. Bernoulli NB for text categorization |
| 2002 | Naive Bayes demonstrated competitive performance with SVMs on many text tasks |
| 2010 | Complement Naive Bayes improved handling of imbalanced text datasets |`,

    clustering: `# Clustering (K-Means, DBSCAN, Hierarchical)

**Clustering** is the task of grouping unlabeled data points into meaningful clusters such that points within a cluster are more similar to each other than to points in other clusters. It is a core unsupervised learning technique used for discovering hidden structure in data. The three most prominent approaches are **K-Means** (centroid-based), **DBSCAN** (density-based), and **Hierarchical** clustering (connectivity-based).

## Key Concepts

- **Centroid**: The center point of a cluster (used in K-Means)
- **Inertia**: Sum of squared distances from each point to its assigned centroid
- **Density Reachability**: Points are in the same cluster if connected through dense regions (DBSCAN)
- **Dendrogram**: A tree diagram showing the hierarchical merging or splitting of clusters
- **Silhouette Score**: Measures how similar a point is to its own cluster vs. other clusters (-1 to 1)
- **Epsilon (eps)**: The neighborhood radius parameter in DBSCAN

## How It Works

**K-Means:**
1. Choose \`k\` (number of clusters) and randomly initialize \`k\` centroids
2. Assign each point to the nearest centroid
3. Recompute centroids as the mean of assigned points
4. Repeat steps 2-3 until centroids stabilize

**DBSCAN:**
1. For each point, count neighbors within radius \`eps\`
2. Points with >= \`min_samples\` neighbors are **core points**
3. Connect core points that are within \`eps\` of each other into clusters
4. Assign border points to nearby clusters; label the rest as noise

**Hierarchical (Agglomerative):**
1. Start with each point as its own cluster
2. Merge the two closest clusters (by linkage criterion)
3. Repeat until all points are in one cluster (or desired \`k\` is reached)

\`\`\`python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels_km = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)

# Hierarchical Agglomerative
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X)
\`\`\`

| Algorithm | Cluster Shape | Handles Noise | Requires k | Complexity |
|---|---|---|---|---|
| **K-Means** | Spherical/convex | No | Yes | O(n * k * iterations) |
| **DBSCAN** | Arbitrary | Yes (labels outliers) | No | O(n log n) with index |
| **Hierarchical** | Arbitrary | No | Optional | O(n^2 log n) |

## Applications

- **Customer Segmentation**: Grouping customers by behavior for targeted marketing strategies
- **Image Segmentation**: Partitioning pixels into regions for object boundary detection
- **Genomics**: Identifying groups of genes with similar expression patterns
- **Document Clustering**: Organizing news articles or research papers into topical groups

## Evolution

| Year | Milestone |
|---|---|
| 1957 | Lloyd proposed the K-Means algorithm for pulse-code modulation |
| 1973 | Hierarchical clustering formalized with various linkage criteria |
| 1996 | DBSCAN introduced density-based clustering for arbitrary-shaped clusters |
| 2007 | HDBSCAN combined hierarchical and density-based approaches |
| 2017 | Deep clustering methods (DEC, DeepCluster) merged neural nets with clustering |`,

    dim_reduction: `# Dimensionality Reduction (PCA, t-SNE, UMAP)

**Dimensionality reduction** transforms high-dimensional data into a lower-dimensional representation while preserving as much meaningful structure as possible. This is essential for visualization, noise reduction, computational efficiency, and combating the **curse of dimensionality**. The three most widely used methods are **PCA** (linear), **t-SNE** (nonlinear, local), and **UMAP** (nonlinear, global+local).

## Key Concepts

- **Curse of Dimensionality**: As dimensions increase, data becomes sparse and distances lose meaning
- **Variance Preservation**: PCA maximizes the variance captured in fewer dimensions
- **Eigendecomposition**: Computing eigenvectors and eigenvalues of the covariance matrix
- **Perplexity**: t-SNE parameter controlling the effective number of local neighbors considered
- **Manifold Learning**: Assuming high-dimensional data lies on a lower-dimensional manifold
- **Explained Variance Ratio**: The fraction of total variance captured by each principal component

## How It Works

**PCA (Principal Component Analysis):**
1. Center the data by subtracting the mean of each feature
2. Compute the covariance matrix of the centered data
3. Calculate eigenvectors and eigenvalues of the covariance matrix
4. Sort eigenvectors by descending eigenvalue (variance explained)
5. Select top \`k\` eigenvectors as principal components
6. Project data onto these components

**t-SNE:**
1. Compute pairwise similarities in high-dimensional space using Gaussian kernels
2. Define a similar probability distribution in low-dimensional space using Student-t distribution
3. Minimize the KL divergence between the two distributions using gradient descent

**UMAP:**
1. Construct a weighted k-nearest neighbor graph in high dimensions
2. Optimize a low-dimensional layout that preserves the graph topology
3. Uses cross-entropy loss between fuzzy simplicial sets

\`\`\`python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# PCA: fast, linear, preserves global variance
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# t-SNE: non-linear, good for visualization, slow
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP: non-linear, fast, preserves global + local structure
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X)
\`\`\`

| Method | Type | Speed | Global Structure | Local Structure | Invertible |
|---|---|---|---|---|---|
| **PCA** | Linear | Very fast | Excellent | Limited | Yes |
| **t-SNE** | Nonlinear | Slow | Poor | Excellent | No |
| **UMAP** | Nonlinear | Fast | Good | Excellent | Approximate |

## Applications

- **Data Visualization**: Reducing 100+ dimensions to 2D/3D for visual exploration of clusters
- **Noise Reduction**: PCA removes low-variance components that often represent noise
- **Feature Engineering**: Creating compact, decorrelated features for downstream models
- **Genomics**: Visualizing single-cell RNA sequencing data to identify cell types

## Evolution

| Year | Milestone |
|---|---|
| 1901 | Karl Pearson introduced PCA for fitting lines and planes to data |
| 1933 | Hotelling formalized PCA with eigendecomposition of covariance matrices |
| 2008 | van der Maaten and Hinton published t-SNE for high-dimensional visualization |
| 2018 | McInnes et al. introduced UMAP as a faster alternative preserving global structure |
| 2020 | PaCMAP and TriMap offered new nonlinear reduction methods balancing speed and quality |`,

    ensemble: `# Ensemble Methods (Bagging, Boosting, Stacking)

**Ensemble methods** combine multiple individual models ("base learners") to produce a single, stronger predictive model. The core insight is that aggregating diverse models reduces errors, since individual model mistakes tend to cancel out. The three main strategies are **bagging** (parallel, variance reduction), **boosting** (sequential, bias reduction), and **stacking** (meta-learning over model outputs).

## Key Concepts

- **Base Learner**: An individual model (often a decision tree) used as a building block
- **Diversity**: Ensembles work best when base learners make different errors
- **Bagging**: Trains models independently on bootstrap samples and averages their predictions
- **Boosting**: Trains models sequentially, each correcting the errors of the previous one
- **Stacking**: Trains a meta-model that learns how to best combine base model predictions
- **Weak Learner**: A model only slightly better than random chance (boosting can turn these into strong learners)

## How It Works

**Bagging (Bootstrap Aggregating):**
1. Create \`n\` bootstrap samples (sampling with replacement) from the training data
2. Train an independent base model on each bootstrap sample
3. For prediction, aggregate all models: majority vote (classification) or average (regression)

**Boosting (e.g., AdaBoost, Gradient Boosting):**
1. Train a weak learner on the full dataset
2. Increase weights on misclassified examples (or fit residuals for gradient boosting)
3. Train the next weak learner focusing on the hard examples
4. Combine all learners with weighted voting based on their accuracy

**Stacking:**
1. Train multiple diverse base models on the training data
2. Use each base model's predictions as features for a new "meta-learner"
3. Train the meta-learner (e.g., logistic regression) to combine predictions optimally

\`\`\`python
from sklearn.ensemble import (
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Bagging
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=100
)

# AdaBoost
adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)

# Gradient Boosting
gboost = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3
)

# Stacking
stacking = StackingClassifier(
    estimators=[('rf', RandomForestClassifier()), ('svm', SVC())],
    final_estimator=LogisticRegression()
)
\`\`\`

| Method | Training | Error Reduction | Key Algorithm |
|---|---|---|---|
| **Bagging** | Parallel (independent) | Reduces **variance** | Random Forest |
| **Boosting** | Sequential (dependent) | Reduces **bias** | XGBoost, LightGBM, AdaBoost |
| **Stacking** | Two-level (meta-learning) | Reduces both | Blending diverse model families |

## Applications

- **Kaggle Competitions**: Ensemble methods (especially XGBoost and stacking) dominate ML competitions
- **Credit Scoring**: Gradient boosting models are industry standard for credit risk assessment
- **Search Ranking**: LambdaMART (boosted trees) powers major web search ranking engines
- **Medical Prediction**: Ensembles provide robust predictions for clinical decision support

## Evolution

| Year | Milestone |
|---|---|
| 1996 | Breiman introduced Bagging as a general variance-reduction technique |
| 1997 | Freund and Schapire published AdaBoost, the first practical boosting algorithm |
| 2001 | Random Forests combined bagging with random feature selection |
| 2016 | XGBoost became the dominant algorithm on Kaggle with regularized gradient boosting |
| 2017 | LightGBM and CatBoost introduced histogram-based and categorical feature optimizations |`,

  };

  Object.assign(window.AI_DOCS, content);
})();
