// 03 - Neural Networks
(function() {
  const content = {

    // ============================================================
    // CORE ARCHITECTURES
    // ============================================================

    perceptrons: `# Perceptrons

The perceptron is the simplest form of a neural network, introduced by Frank Rosenblatt in 1958. It is a single-layer binary classifier that maps input features to an output by computing a weighted sum and applying a threshold activation function. Despite its simplicity, the perceptron laid the groundwork for all modern neural network architectures and remains fundamental to understanding how artificial neurons process information.

## Architecture

A perceptron consists of input nodes, weights, a bias term, and an activation function. Given inputs x1, x2, ..., xn, the perceptron computes:

\`\`\`
   x1 ---(w1)---\\
   x2 ---(w2)----+---> [ Sum + Bias ] ---> [ Step Function ] ---> Output
   x3 ---(w3)---/
\`\`\`

**Output:** y = f(w . x + b), where f is typically a step function:
- f(z) = 1 if z >= 0
- f(z) = 0 if z < 0

## How It Works

The perceptron learning algorithm adjusts weights based on prediction errors. For each misclassified sample, the weight update rule is:

**w_new = w_old + alpha * (y_true - y_pred) * x**

Where alpha is the learning rate. This process repeats over the dataset until convergence or a maximum number of epochs is reached. The Perceptron Convergence Theorem guarantees convergence for linearly separable data.

\`\`\`python
import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sign(self.linear(x))
\`\`\`

## Applications

- Binary classification tasks (spam vs. not spam)
- Logic gate simulation (AND, OR, NAND)
- Foundation for understanding neural network learning rules
- Feature-based decision boundaries in low-dimensional spaces

## Evolution

| Year | Milestone |
|------|-----------|
| 1943 | McCulloch-Pitts neuron model proposed |
| 1958 | Rosenblatt introduces the Perceptron |
| 1969 | Minsky & Papert show XOR limitation |
| 1986 | Multi-layer networks overcome the limitation via backpropagation |
| 2000s | Perceptron concepts underpin kernel methods and SVMs |`,

    mlps: `# Multi-Layer Perceptrons (MLPs)

The Multi-Layer Perceptron (MLP) is a feedforward neural network consisting of multiple layers of neurons with nonlinear activation functions. MLPs overcome the linear separability limitation of single perceptrons by introducing hidden layers that can learn complex, nonlinear decision boundaries. They form the backbone of deep learning and are used as building blocks within more specialized architectures.

## Architecture

An MLP has three types of layers: an input layer, one or more hidden layers, and an output layer. Each layer is fully connected (dense) to the next.

\`\`\`
Input Layer       Hidden Layer 1     Hidden Layer 2     Output Layer
  [x1] --------\\     [h1] --------\\     [h1] --------\\     [y1]
  [x2] ---------+--> [h2] ---------+--> [h2] ---------+--> [y2]
  [x3] --------/     [h3] --------/     [h3] --------/
\`\`\`

Each connection carries a learnable weight, and each neuron applies: **z = activation(W * x + b)**

## How It Works

MLPs use the Universal Approximation Theorem: a network with at least one hidden layer and nonlinear activations can approximate any continuous function to arbitrary precision, given sufficient neurons. Training uses backpropagation with gradient descent to minimize a loss function.

\`\`\`python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10)  # MNIST classifier
\`\`\`

## Applications

- Tabular data classification and regression
- Function approximation and interpolation
- Feature extraction in hybrid architectures
- Reinforcement learning value and policy networks

## Evolution

| Year | Milestone |
|------|-----------|
| 1986 | Rumelhart, Hinton & Williams popularize backpropagation for MLPs |
| 1989 | Cybenko proves the Universal Approximation Theorem |
| 1990s | MLPs become standard for pattern recognition tasks |
| 2010s | Deep MLPs revived with ReLU, dropout, and batch normalization |
| 2020s | MLP-Mixer shows competitive vision results with pure MLP architectures |`,

    cnns: `# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are specialized neural architectures designed to process grid-structured data such as images, video, and spectrograms. Inspired by the visual cortex, CNNs use local receptive fields, shared weights, and spatial hierarchies to automatically learn translation-invariant features. They revolutionized computer vision and remain the dominant architecture for image-related tasks.

## Architecture

A typical CNN consists of convolutional layers, pooling layers, and fully connected layers:

\`\`\`
Input Image --> [Conv + ReLU] --> [Pool] --> [Conv + ReLU] --> [Pool] --> [Flatten] --> [FC] --> Output
  (H x W x C)    (filters)     (downsample)   (filters)    (downsample)           (classes)
\`\`\`

**Key components:**
- **Convolutional Layer:** Slides learnable filters (kernels) across the input, producing feature maps
- **Pooling Layer:** Reduces spatial dimensions (Max Pooling, Average Pooling)
- **Stride & Padding:** Control output spatial dimensions

## How It Works

Each convolutional filter computes a dot product with a local patch of the input. For a filter of size k x k applied to input with C channels:

**Output(i,j) = SUM over c,m,n of [ Filter(c,m,n) * Input(c, i+m, j+n) ] + bias**

Parameter sharing means the same filter is applied across all spatial positions, drastically reducing parameters compared to fully connected layers.

\`\`\`python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
\`\`\`

## Applications

- Image classification (ResNet, EfficientNet)
- Object detection (YOLO, Faster R-CNN)
- Medical imaging (tumor detection, X-ray analysis)
- Video analysis and action recognition

## Evolution

| Year | Milestone |
|------|-----------|
| 1989 | LeCun applies backprop to CNNs (LeNet) |
| 2012 | AlexNet wins ImageNet, sparking the deep learning revolution |
| 2015 | ResNet introduces skip connections (152 layers) |
| 2019 | EfficientNet achieves state-of-the-art with compound scaling |
| 2021 | Vision Transformers (ViT) challenge CNN dominance |`,

    rnns: `# Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks designed for sequential data processing. Unlike feedforward networks, RNNs maintain a hidden state that acts as memory, allowing them to capture temporal dependencies in sequences. This recurrent connection enables the network to process inputs of variable length, making them suitable for time series, natural language, and any data with sequential structure.

## Architecture

An RNN processes one element at a time while maintaining a hidden state vector that carries information from previous time steps:

\`\`\`
     x_t-1          x_t          x_t+1
       |              |              |
       v              v              v
  [RNN Cell] --> [RNN Cell] --> [RNN Cell] -->
       |     h_t-1    |     h_t     |     h_t+1
       v              v              v
     y_t-1          y_t          y_t+1
\`\`\`

**Unrolled computation at each step:**
- h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
- y_t = W_hy * h_t + b_y

## How It Works

At each time step, the RNN takes the current input and the previous hidden state to produce a new hidden state and optional output. The same weight matrices (W_hh, W_xh, W_hy) are shared across all time steps, enabling the network to generalize across sequence positions. Training uses Backpropagation Through Time (BPTT), which unrolls the RNN and applies standard backpropagation.

\`\`\`python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.rnn(x)       # out: all hidden states
        return self.fc(out[:, -1, :]) # last time step
\`\`\`

**Vanishing/Exploding Gradient Problem:** During BPTT, gradients are multiplied by the recurrent weight matrix at each step. Over long sequences, gradients can shrink to near zero (vanishing) or grow unboundedly (exploding), making it difficult to learn long-range dependencies.

## Applications

- Language modeling and text generation
- Speech recognition pipelines
- Time series forecasting (stock prices, weather)
- Music composition and sequence prediction

## Evolution

| Year | Milestone |
|------|-----------|
| 1986 | Rumelhart introduces recurrent connections |
| 1990 | Elman networks formalize simple RNN architecture |
| 1997 | LSTM addresses vanishing gradient problem |
| 2014 | GRU offers a simplified gating mechanism |
| 2017 | Transformers largely replace RNNs for NLP tasks |`,

    lstm_gru: `# LSTM & GRU

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are advanced recurrent architectures that solve the vanishing gradient problem through gating mechanisms. These gates control what information to retain, forget, or output, enabling the network to learn dependencies across hundreds or thousands of time steps. They dominated sequence modeling before the Transformer era and remain highly relevant for real-time and resource-constrained applications.

## Architecture

**LSTM Cell:**
\`\`\`
        c_{t-1} ----[x]--------[+]--------> c_t
                     |          |
                  [Forget]   [Input * Candidate]
                   Gate        Gate
  x_t ---|                                   |--> h_t
  h_{t-1}|---[Concat]----> Gates compute --> [Output Gate] * tanh(c_t) --> h_t
\`\`\`

**LSTM equations:**
- Forget gate: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
- Input gate: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
- Candidate: c_hat = tanh(W_c * [h_{t-1}, x_t] + b_c)
- Cell state: c_t = f_t * c_{t-1} + i_t * c_hat
- Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
- Hidden state: h_t = o_t * tanh(c_t)

**GRU** simplifies LSTM by merging cell and hidden state using two gates (reset and update) instead of three.

## How It Works

The cell state acts as a conveyor belt carrying information across time steps with minimal interference. The forget gate decides what to discard, the input gate selects new information to store, and the output gate determines what to expose. GRU achieves similar performance with fewer parameters by using a single update gate that balances forgetting and remembering.

\`\`\`python
import torch.nn as nn

# LSTM-based sequence classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))
\`\`\`

## Applications

- Machine translation (seq2seq with attention)
- Speech recognition and synthesis
- Sentiment analysis and text classification
- Time-series anomaly detection and forecasting

## LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | Cell state + hidden state | Hidden state only |
| Parameters | More (larger model) | Fewer (faster training) |
| Long sequences | Slightly better | Comparable |
| Training speed | Slower | Faster |`,

    autoencoders: `# Autoencoders

Autoencoders are unsupervised neural networks that learn compressed representations of data by training to reconstruct their input through a bottleneck architecture. The network consists of an encoder that maps input to a lower-dimensional latent space and a decoder that reconstructs the original input from this representation. By forcing data through a bottleneck, autoencoders learn meaningful features without labeled data.

## Architecture

\`\`\`
Input -----> [ Encoder ] -----> Latent Code (z) -----> [ Decoder ] -----> Reconstruction
(784)        (512->256)            (32)                (256->512)           (784)
                              (Bottleneck)
\`\`\`

**Variants:**

| Variant | Key Idea |
|---------|----------|
| Vanilla AE | Deterministic bottleneck, MSE loss |
| Variational AE (VAE) | Probabilistic latent space, KL divergence + reconstruction loss |
| Denoising AE | Trained to reconstruct clean input from corrupted input |
| Sparse AE | Enforces sparsity constraint on latent activations |
| Contractive AE | Penalizes sensitivity of latent code to input perturbations |

## How It Works

The encoder function g(x) maps input x to latent code z, and the decoder f(z) reconstructs x_hat. Training minimizes the reconstruction loss:

**L = ||x - f(g(x))||^2** (MSE for continuous data)

For VAEs, the encoder outputs parameters (mean, variance) of a probability distribution, and the loss includes a KL divergence term to regularize the latent space:

**L_VAE = E[||x - x_hat||^2] + KL(q(z|x) || p(z))**

\`\`\`python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
\`\`\`

## Applications

- Dimensionality reduction and data visualization
- Anomaly detection (high reconstruction error signals outliers)
- Image denoising and inpainting
- Generative modeling (VAEs for sampling new data)
- Feature learning for downstream tasks

## Evolution

| Year | Milestone |
|------|-----------|
| 1986 | Rumelhart & Hinton introduce autoencoders |
| 2008 | Vincent et al. propose Denoising Autoencoders |
| 2013 | Kingma & Welling introduce Variational Autoencoders |
| 2017 | VQ-VAE enables discrete latent codes for generation |
| 2020s | Autoencoder components integrated into diffusion and LLM architectures |`,

    gnns: `# Graph Neural Networks (GNNs)

Graph Neural Networks are a class of neural networks designed to operate directly on graph-structured data. Unlike images (grids) or text (sequences), graphs have irregular topology with nodes connected by edges in arbitrary patterns. GNNs learn representations for nodes, edges, or entire graphs by aggregating information from local neighborhoods, enabling deep learning on relational and structural data.

## Architecture

GNNs follow a message-passing paradigm where each node updates its representation by aggregating features from its neighbors:

\`\`\`
     [A]---[B]           Layer 1: Each node aggregates 1-hop neighbors
      |   / |             Layer 2: Each node aggregates 2-hop neighborhood
     [C]   [D]            Layer k: k-hop receptive field
      |
     [E]

Node Update: h_v^(k) = UPDATE(h_v^(k-1), AGGREGATE({h_u^(k-1) : u in N(v)}))
\`\`\`

**Common GNN variants:**

| Variant | Aggregation Method |
|---------|--------------------|
| GCN (Graph Convolutional Network) | Normalized mean of neighbor features |
| GraphSAGE | Sampling + aggregation (mean, LSTM, pool) |
| GAT (Graph Attention Network) | Attention-weighted neighbor aggregation |
| GIN (Graph Isomorphism Network) | Sum aggregation with MLP (maximally expressive) |
| MPNN (Message Passing NN) | General framework with message and update functions |

## How It Works

For a GCN layer, the update rule is:

**H^(l+1) = sigma(D^(-1/2) * A_hat * D^(-1/2) * H^(l) * W^(l))**

Where A_hat = A + I (adjacency with self-loops), D is the degree matrix, and W is a learnable weight matrix. Stacking multiple layers expands each node's receptive field to capture multi-hop relationships.

\`\`\`python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
\`\`\`

## Applications

- Social network analysis (community detection, link prediction)
- Molecular property prediction (drug discovery)
- Recommendation systems (user-item interaction graphs)
- Traffic forecasting and route optimization
- Knowledge graph reasoning

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Spectral graph convolutions introduced |
| 2017 | Kipf & Welling propose GCN (semi-supervised classification) |
| 2017 | GraphSAGE enables inductive learning on large graphs |
| 2018 | GAT introduces attention to graph learning |
| 2020s | GNNs scale to billion-node graphs; used in AlphaFold for protein structure |`,

    attention: `# Attention Mechanism

The attention mechanism allows neural networks to dynamically focus on relevant parts of the input when producing each element of the output. Instead of compressing an entire sequence into a fixed-size vector, attention computes a weighted combination of all input positions, where the weights reflect relevance to the current decoding step. This breakthrough solved the information bottleneck in sequence-to-sequence models and became the foundation for the Transformer architecture.

## Key Concepts

**Core idea:** For each output position, compute a compatibility score between a "query" and all "keys", then use those scores to weight the corresponding "values".

\`\`\`
  Query (Q) --|
              +--> Attention Scores --> Softmax --> Weights --> Weighted Sum of Values
  Keys  (K) --|                                                       |
  Values (V) ---------------------------------------------------------+---> Output
\`\`\`

**Types of Attention:**

| Type | Description |
|------|-------------|
| Additive (Bahdanau) | score(q, k) = V * tanh(W1*q + W2*k) |
| Dot-Product | score(q, k) = q . k |
| Scaled Dot-Product | score(q, k) = (q . k) / sqrt(d_k) |
| Multi-Head | Multiple parallel attention heads, concatenated |
| Self-Attention | Q, K, V all derived from the same sequence |
| Cross-Attention | Q from one sequence, K/V from another |

## How It Works

Scaled dot-product attention, the core of the Transformer:

**Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V**

Multi-head attention runs h parallel attention heads with different learned projections, then concatenates:

**MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O**
**where head_i = Attention(Q * W_Q^i, K * W_K^i, V * W_V^i)**

\`\`\`python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights
\`\`\`

## Applications

- Machine translation (Bahdanau attention with seq2seq)
- Transformer-based language models (GPT, BERT)
- Image captioning and visual question answering
- Speech recognition (Listen, Attend and Spell)
- Protein structure prediction (AlphaFold)

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Bahdanau et al. introduce additive attention for NMT |
| 2015 | Luong et al. propose dot-product attention variants |
| 2017 | Vaswani et al. introduce self-attention in "Attention Is All You Need" |
| 2018 | BERT uses bidirectional self-attention for pretraining |
| 2020s | Efficient attention variants (Flash Attention, linear attention) enable long contexts |`,

    // ============================================================
    // TRAINING & OPTIMIZATION
    // ============================================================

    backprop: `# Backpropagation

Backpropagation (backward propagation of errors) is the fundamental algorithm for training neural networks. It efficiently computes the gradient of the loss function with respect to every weight in the network by applying the chain rule of calculus in reverse through the computation graph. Combined with gradient descent, backpropagation enables neural networks to learn from data by iteratively adjusting weights to minimize prediction errors.

## Key Concepts

Backpropagation operates in two phases:

\`\`\`
Forward Pass:                          Backward Pass:
Input --> [Layer 1] --> [Layer 2] --> Loss    Loss --> [dL/dW2] --> [dL/dW1]
   x    W1*x + b1    W2*h1 + b2     L(y,y_hat)   Gradients flow backward
\`\`\`

**Chain rule application:** For a network y = f3(f2(f1(x))):
- dL/dW1 = dL/dy * dy/df2 * df2/df1 * df1/dW1

## How It Works

1. **Forward pass:** Compute activations layer by layer, caching intermediate values
2. **Compute loss:** Evaluate the loss function at the output
3. **Backward pass:** Compute gradients starting from the loss, propagating through each layer
4. **Weight update:** w = w - lr * dL/dw

For a single neuron with z = wx + b, a = sigma(z):
- dL/dw = dL/da * da/dz * dz/dw = dL/da * sigma'(z) * x

\`\`\`python
import torch
import torch.nn as nn

# PyTorch autograd handles backprop automatically
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training step
x, y = torch.randn(16, 10), torch.randn(16, 1)
y_hat = model(x)              # Forward pass
loss = criterion(y_hat, y)    # Compute loss
loss.backward()               # Backward pass (computes gradients)
optimizer.step()              # Update weights
optimizer.zero_grad()         # Reset gradients
\`\`\`

**Computational graph:** Modern frameworks (PyTorch, TensorFlow) build dynamic or static computation graphs that automatically compute gradients via autograd, abstracting away manual differentiation.

## Applications

- Training all feedforward and recurrent neural networks
- Backpropagation Through Time (BPTT) for RNNs
- Gradient-based hyperparameter optimization
- Neural architecture search (differentiable)

## Evolution

| Year | Milestone |
|------|-----------|
| 1970 | Linnainmaa describes automatic differentiation |
| 1986 | Rumelhart, Hinton & Williams popularize backprop for neural networks |
| 2012 | GPU-accelerated backprop enables training of deep CNNs |
| 2015 | Automatic differentiation frameworks (TensorFlow, PyTorch) emerge |
| 2022 | Flash Attention and memory-efficient backprop for large Transformers |`,

    activations: `# Activation Functions

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns beyond linear mappings. Without activations, stacking multiple layers would be equivalent to a single linear transformation. The choice of activation function significantly impacts training dynamics, gradient flow, and model performance. Modern architectures carefully select activations based on the task and architecture type.

## Key Concepts

An activation function f is applied element-wise after the linear transformation z = Wx + b:

**output = f(z)**

**Desirable properties:**
- Nonlinear (enables learning complex functions)
- Differentiable (enables gradient-based training)
- Zero-centered output (faster convergence)
- Avoids saturating gradients (prevents vanishing gradients)

## Comparison Table

| Function | Formula | Range | Pros | Cons |
|----------|---------|-------|------|------|
| **Sigmoid** | 1 / (1 + e^(-x)) | (0, 1) | Smooth, probabilistic output | Vanishing gradients, not zero-centered |
| **Tanh** | (e^x - e^(-x)) / (e^x + e^(-x)) | (-1, 1) | Zero-centered | Still saturates at extremes |
| **ReLU** | max(0, x) | [0, inf) | Fast, no saturation for x > 0 | Dead neurons (zero gradient for x < 0) |
| **Leaky ReLU** | max(0.01x, x) | (-inf, inf) | No dead neurons | Small negative slope is a hyperparameter |
| **GELU** | x * Phi(x) | ~ (-0.17, inf) | Smooth ReLU; used in Transformers | Computationally heavier |
| **Swish** | x * sigmoid(x) | ~ (-0.28, inf) | Self-gated, smooth | Slightly more expensive |
| **SiLU** | x * sigmoid(x) | ~ (-0.28, inf) | Same as Swish | Renamed in many frameworks |

## How It Works

\`\`\`python
import torch
import torch.nn.functional as F

x = torch.linspace(-5, 5, 100)

# Common activations
relu_out    = F.relu(x)
sigmoid_out = torch.sigmoid(x)
tanh_out    = torch.tanh(x)
gelu_out    = F.gelu(x)
swish_out   = x * torch.sigmoid(x)  # SiLU/Swish

# In a model
class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 4)
        self.fc2 = torch.nn.Linear(dim * 4, dim)
        self.act = torch.nn.GELU()  # Standard in Transformers

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
\`\`\`

**Modern conventions:**
- **CNNs:** ReLU or Leaky ReLU
- **Transformers:** GELU (BERT, GPT) or SiLU/Swish (LLaMA)
- **Output layers:** Sigmoid (binary), Softmax (multiclass), Linear (regression)

## Applications

- ReLU: Default for hidden layers in CNNs and MLPs
- GELU: Standard in Transformer architectures (BERT, GPT, ViT)
- Sigmoid: Binary classification output, gating in LSTMs
- Softmax: Multi-class classification output layers

## Evolution

| Year | Milestone |
|------|-----------|
| 1943 | Step function in McCulloch-Pitts neuron |
| 1989 | Sigmoid dominates early neural networks |
| 2010 | Nair & Hinton introduce ReLU, accelerating deep learning |
| 2016 | Hendrycks & Gimpel propose GELU |
| 2017 | Swish discovered via automated search (Google Brain) |`,

    loss_functions: `# Loss Functions

Loss functions (also called cost functions or objective functions) measure the discrepancy between a model's predictions and the ground truth labels. They define the optimization landscape that gradient descent navigates during training. The choice of loss function directly shapes what the model learns, as the network's weights are adjusted to minimize this quantity. Different tasks require different loss functions tailored to the output type and training objective.

## Key Concepts

A loss function L(y_pred, y_true) maps predictions and targets to a scalar value. The total loss over a dataset is typically averaged:

**J(theta) = (1/N) * SUM_{i=1}^{N} L(y_pred_i, y_true_i)**

## Comparison Table

| Loss Function | Formula | Use Case |
|---------------|---------|----------|
| **MSE** | (1/N) SUM(y - y_hat)^2 | Regression |
| **MAE (L1)** | (1/N) SUM(abs(y - y_hat)) | Robust regression |
| **Huber** | MSE if abs(e)<d, else MAE | Regression with outliers |
| **Cross-Entropy** | -SUM(y * log(y_hat)) | Classification |
| **Binary CE** | -[y*log(p) + (1-y)*log(1-p)] | Binary classification |
| **Hinge** | max(0, 1 - y * y_hat) | SVM, margin classifiers |
| **KL Divergence** | SUM(p * log(p/q)) | Distribution matching (VAEs) |
| **CTC** | Alignment-free sequence loss | Speech recognition, OCR |
| **Focal Loss** | -alpha * (1-p)^gamma * log(p) | Imbalanced classification |

## How It Works

**Cross-Entropy** for multiclass classification (C classes):

**L = -SUM_{c=1}^{C} y_c * log(y_hat_c)**

Where y is a one-hot vector and y_hat is the softmax output. This loss heavily penalizes confident wrong predictions, providing strong gradient signal.

\`\`\`python
import torch
import torch.nn as nn

# Classification losses
ce_loss = nn.CrossEntropyLoss()     # combines LogSoftmax + NLLLoss
bce_loss = nn.BCEWithLogitsLoss()   # binary with sigmoid built-in
focal = lambda p, t, g=2: -((1-p)**g * t * torch.log(p)).mean()

# Regression losses
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
huber_loss = nn.HuberLoss(delta=1.0)

# Example usage
logits = torch.randn(32, 10)        # batch of 32, 10 classes
targets = torch.randint(0, 10, (32,))
loss = ce_loss(logits, targets)
loss.backward()
\`\`\`

## Applications

- **MSE:** Regression tasks (house prices, temperature prediction)
- **Cross-Entropy:** Image classification, NLP classification
- **Hinge Loss:** Support Vector Machines, ranking tasks
- **KL Divergence:** VAE latent space regularization, knowledge distillation
- **Focal Loss:** Object detection with class imbalance (RetinaNet)

## Evolution

| Year | Milestone |
|------|-----------|
| 1805 | Legendre introduces least squares (MSE precursor) |
| 1948 | Shannon formalizes cross-entropy in information theory |
| 1963 | Hinge loss formalized for margin classifiers |
| 2006 | CTC loss enables end-to-end speech recognition |
| 2017 | Focal Loss addresses class imbalance in object detection |`,

    optimizers: `# Optimizers (SGD, Adam, AdamW, RMSProp)

Optimizers are algorithms that update neural network parameters to minimize the loss function. They determine how gradient information is used to adjust weights, controlling the speed, stability, and convergence quality of training. From basic Stochastic Gradient Descent to adaptive methods like Adam, the choice of optimizer profoundly impacts training efficiency and final model performance.

## Key Concepts

All gradient-based optimizers follow the general update rule:

**theta_{t+1} = theta_t - lr * g(gradients, history)**

Where the function g varies by algorithm. Key considerations include learning rate, momentum, and per-parameter adaptivity.

## Comparison Table

| Optimizer | Update Rule (simplified) | Key Feature |
|-----------|--------------------------|-------------|
| **SGD** | w -= lr * grad | Simple, requires tuning |
| **SGD+Momentum** | v = beta*v + grad; w -= lr*v | Accelerates convergence |
| **RMSProp** | s = beta*s + (1-beta)*grad^2; w -= lr*grad/sqrt(s+eps) | Per-param adaptive LR |
| **Adam** | Combines momentum + RMSProp with bias correction | Default choice for most tasks |
| **AdamW** | Adam + decoupled weight decay | Standard for Transformers |
| **LAMB** | Layer-wise Adam with trust ratios | Large batch training |
| **Lion** | Sign-based momentum optimizer | Memory efficient |

## How It Works

**Adam** maintains two moving averages per parameter:
- First moment (mean): m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
- Second moment (variance): v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
- Bias correction: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
- Update: theta -= lr * m_hat / (sqrt(v_hat) + epsilon)

**AdamW** decouples weight decay from the gradient update, applying it directly to weights: theta -= lr * (m_hat / (sqrt(v_hat) + eps) + lambda * theta)

\`\`\`python
import torch.optim as optim

model = ...  # your neural network

# Standard SGD with momentum
opt_sgd = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

# Adam (default for many tasks)
opt_adam = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

# AdamW (standard for Transformers/LLMs)
opt_adamw = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# RMSProp (good for RNNs)
opt_rms = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99)
\`\`\`

## Applications & Conventions

- **SGD + Momentum:** CNNs (ResNet, EfficientNet), often achieves best final accuracy
- **Adam/AdamW:** Transformers, LLMs, GANs, default starting point
- **RMSProp:** Recurrent networks, reinforcement learning (DQN, A3C)
- **LAMB:** Distributed training with very large batch sizes (BERT pretraining)

## Evolution

| Year | Milestone |
|------|-----------|
| 1951 | Robbins & Monro introduce stochastic approximation (SGD) |
| 1986 | Momentum added to SGD for acceleration |
| 2012 | RMSProp proposed by Hinton (unpublished lecture notes) |
| 2014 | Kingma & Ba introduce Adam |
| 2019 | Loshchilov & Hutter formalize AdamW (decoupled weight decay) |`,

    regularization: `# Regularization (Dropout, BatchNorm, LayerNorm, Weight Decay)

Regularization techniques prevent neural networks from overfitting to training data by constraining the model's capacity or adding noise during training. Overfitting occurs when a model memorizes training examples rather than learning generalizable patterns. Modern deep learning relies on a combination of regularization methods to achieve robust generalization, from explicit constraints like weight decay to implicit regularizers like dropout and normalization layers.

## Key Concepts

| Technique | Mechanism | Where Applied |
|-----------|-----------|---------------|
| **Dropout** | Randomly zeros activations with probability p | Hidden layers during training |
| **Batch Normalization** | Normalizes activations across the batch dimension | After linear/conv layers |
| **Layer Normalization** | Normalizes activations across the feature dimension | Standard in Transformers |
| **Weight Decay (L2)** | Adds lambda * ||w||^2 penalty to loss | Applied to all weights |
| **L1 Regularization** | Adds lambda * ||w||_1 penalty (promotes sparsity) | Feature selection |
| **Early Stopping** | Stop training when validation loss increases | Training loop |
| **Data Augmentation** | Transforms training data (flip, crop, noise) | Input pipeline |

## How It Works

**Dropout** during training randomly sets neurons to zero with probability p. At inference, all neurons are active but scaled by (1 - p). This forces the network to learn redundant representations and prevents co-adaptation of neurons.

**Batch Normalization** normalizes each feature across the batch:
- BN(x) = gamma * (x - mean_batch) / sqrt(var_batch + eps) + beta
- gamma and beta are learnable scale and shift parameters

**Layer Normalization** normalizes across features for each sample (batch-independent):
- LN(x) = gamma * (x - mean_features) / sqrt(var_features + eps) + beta

\`\`\`python
import torch.nn as nn

class RegularizedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # BatchNorm for CNNs
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        # LayerNorm for Transformers
        self.transformer_block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

# AdamW includes decoupled weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
\`\`\`

## BatchNorm vs LayerNorm

| Property | BatchNorm | LayerNorm |
|----------|-----------|-----------|
| Normalization axis | Batch dimension | Feature dimension |
| Batch dependence | Yes (needs batch stats) | No (per-sample) |
| Best for | CNNs | Transformers, RNNs |
| Inference behavior | Uses running mean/var | Same as training |

## Applications

- **Dropout:** Standard in fully connected and attention layers
- **BatchNorm:** CNNs (ResNet, EfficientNet)
- **LayerNorm:** Transformers (GPT, BERT, ViT)
- **Weight Decay:** Universal; critical for AdamW-based training

## Evolution

| Year | Milestone |
|------|-----------|
| 1991 | Weight decay (L2 regularization) applied to neural networks |
| 2012 | Srivastava & Hinton introduce Dropout |
| 2015 | Ioffe & Szegedy introduce Batch Normalization |
| 2016 | Ba et al. introduce Layer Normalization |
| 2019 | RMSNorm proposed as a simpler alternative to LayerNorm |`,

    lr_scheduling: `# Learning Rate Scheduling

Learning rate scheduling dynamically adjusts the learning rate during training to improve convergence speed and final model quality. Starting with a high learning rate enables rapid initial progress, while gradually reducing it allows the optimizer to settle into sharper minima. Modern training recipes combine warmup phases, decay strategies, and restarts to navigate the loss landscape effectively, and the schedule is often as important as the optimizer choice itself.

## Key Concepts

The learning rate (lr) controls the step size of parameter updates. A fixed lr faces a tradeoff: too high causes divergence, too low causes slow convergence. Scheduling resolves this by varying lr over the training process.

\`\`\`
Learning Rate Over Time:

  lr |  Warmup   Cosine Decay
     | /------\\
     |/        \\
     |          \\___________
     |________________________> Steps
\`\`\`

## Common Schedules

| Schedule | Formula / Behavior | Use Case |
|----------|-------------------|----------|
| **Step Decay** | lr *= gamma every N epochs | Classic CNN training |
| **Exponential** | lr = lr_0 * gamma^epoch | Smooth monotonic decay |
| **Cosine Annealing** | lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T)) | Transformers, modern CNNs |
| **Linear Warmup** | lr = lr_max * (step / warmup_steps) | First phase of Transformer training |
| **Warmup + Cosine** | Linear warmup then cosine decay | Standard for LLMs (GPT, BERT) |
| **Cosine with Restarts** | Cosine with periodic lr resets | Avoiding local minima |
| **OneCycleLR** | Warmup to peak, then anneal to near zero | Fast convergence (super-convergence) |
| **ReduceOnPlateau** | Reduce lr when metric stops improving | Adaptive, metric-driven |

## How It Works

**Warmup** prevents large initial gradient updates from destabilizing randomly initialized parameters. This is especially critical for Transformers with attention layers.

**Cosine annealing** smoothly decreases the learning rate following a half-cosine curve, providing gentler decay than step schedules:

**lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))**

\`\`\`python
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, OneCycleLR,
    LinearLR, SequentialLR
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Step decay: multiply lr by 0.1 every 30 epochs
sched_step = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing over 100 epochs
sched_cosine = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Warmup + Cosine (common for Transformers)
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=50000)
sched_warmup_cosine = SequentialLR(optimizer, [warmup, cosine], milestones=[1000])

# OneCycleLR for super-convergence
sched_1cycle = OneCycleLR(optimizer, max_lr=1e-3, total_steps=10000)
\`\`\`

## Applications

- **Warmup + Cosine:** GPT, BERT, ViT, and most LLM pretraining
- **Step Decay:** ResNet and classic CNN training (reduce every 30 epochs)
- **OneCycleLR:** Rapid training with limited compute budget
- **ReduceOnPlateau:** Fine-tuning when optimal schedule is unknown

## Evolution

| Year | Milestone |
|------|-----------|
| 1998 | Step decay used in LeNet training |
| 2016 | Cosine annealing proposed by Loshchilov & Hutter (SGDR) |
| 2017 | Warmup shown critical for Transformer training |
| 2018 | Smith & Topin introduce super-convergence with OneCycleLR |
| 2020s | Warmup + cosine decay becomes the universal LLM training recipe |`,

  };

  // Register all content on the global AI_DOCS object
  Object.assign(window.AI_DOCS, content);
})();
