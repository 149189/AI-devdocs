// 04 - Deep Learning Specializations
(function() {
  const content = {

    // ============================================================
    // VISION TASKS
    // ============================================================

    object_detection: `# Object Detection

Object detection is the computer vision task of identifying and localizing multiple objects within an image by predicting both their class labels and bounding box coordinates. Unlike image classification, which assigns a single label to an entire image, object detection must handle varying numbers of objects, different scales, overlapping instances, and cluttered backgrounds. It is one of the most commercially important tasks in deep learning, powering autonomous vehicles, surveillance systems, and retail analytics.

## Key Concepts

- **Bounding Boxes:** Rectangular regions defined by (x, y, width, height) or (x_min, y_min, x_max, y_max) that localize objects
- **Anchor Boxes:** Pre-defined boxes of various aspect ratios and scales used as reference templates for prediction
- **Intersection over Union (IoU):** Metric measuring overlap between predicted and ground-truth boxes: IoU = Area(Intersection) / Area(Union)
- **Non-Maximum Suppression (NMS):** Post-processing step that removes duplicate detections by keeping only the highest-confidence box among overlapping predictions
- **Feature Pyramid Networks (FPN):** Multi-scale feature extraction for detecting objects at different sizes

\`\`\`
Input Image --> Backbone CNN --> Feature Maps --> Detection Head --> Bounding Boxes + Classes
                                     |
                              [Anchor Boxes]
                                     |
                          [NMS Post-processing]
\`\`\`

## How It Works

Detection models fall into two paradigms:

**Two-Stage Detectors (R-CNN Family):**
1. A Region Proposal Network (RPN) generates candidate object regions
2. Each region is classified and its bounding box is refined
3. Slower but generally more accurate for small objects

**One-Stage Detectors (YOLO, SSD):**
1. Divide the image into a grid of cells
2. Each cell predicts bounding boxes and class probabilities simultaneously
3. Faster inference, suitable for real-time applications

\`\`\`python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict("image.jpg", conf=0.5)
for box in results[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf:.2f}")
    print(f"Bounding Box: {box.xyxy}")
\`\`\`

| Detector | Type | Speed | Accuracy |
|----------|------|-------|----------|
| **Faster R-CNN** | Two-stage | ~5 FPS | High |
| **SSD** | One-stage | ~30 FPS | Medium |
| **YOLOv8** | One-stage | ~80 FPS | High |
| **DETR** | Transformer | ~15 FPS | High |

## Applications

- Autonomous driving (pedestrian, vehicle, sign detection)
- Video surveillance and security monitoring
- Medical imaging (tumor, lesion detection)
- Retail (shelf inventory, checkout-free stores)
- Drone and satellite imagery analysis

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | R-CNN introduces region-based detection with CNNs |
| 2015 | Faster R-CNN adds Region Proposal Networks |
| 2016 | YOLO (v1) and SSD enable real-time detection |
| 2020 | DETR applies transformers to detection, removing anchors |
| 2023 | YOLOv8 achieves state-of-the-art speed-accuracy tradeoff |`,

    image_classification: `# Image Classification

Image classification is the foundational computer vision task of assigning a single categorical label to an entire image. It serves as the backbone of visual understanding and has driven much of the progress in deep learning, from AlexNet's breakthrough in 2012 to modern EfficientNet architectures. Convolutional Neural Networks (CNNs) dominate this task by learning hierarchical feature representations that capture edges, textures, shapes, and high-level semantic patterns directly from pixel data.

## Key Concepts

- **Convolutional Layers:** Apply learned filters to extract spatial features with parameter sharing and translation invariance
- **Pooling Layers:** Downsample feature maps to reduce dimensionality and increase receptive field (max pooling, average pooling)
- **Residual Connections:** Skip connections in ResNet that allow gradients to flow through very deep networks: **y = F(x) + x**
- **Transfer Learning:** Reusing pre-trained weights from ImageNet as initialization for downstream classification tasks
- **Data Augmentation:** Applying random transformations (flips, rotations, crops, color jitter) to increase training set diversity

\`\`\`
Input Image (224x224x3)
    |
[Conv 3x3] --> [ReLU] --> [MaxPool 2x2]    // Low-level features (edges)
    |
[Conv 3x3] --> [ReLU] --> [MaxPool 2x2]    // Mid-level features (textures)
    |
[Conv 3x3] --> [ReLU] --> [Global AvgPool]  // High-level features (objects)
    |
[Fully Connected] --> [Softmax] --> Class Probabilities
\`\`\`

## How It Works

A CNN processes an input image through stacked convolutional layers that learn increasingly abstract features. The final feature map is flattened or globally pooled and passed through fully connected layers to produce class probabilities via softmax. Training minimizes cross-entropy loss using backpropagation.

\`\`\`python
import torchvision.models as models
import torch.nn as nn

# Transfer learning with ResNet-50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, num_classes)  # Replace final layer

# Freeze backbone for fine-tuning
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
\`\`\`

| Architecture | Year | Top-1 Acc (ImageNet) | Parameters |
|-------------|------|---------------------|------------|
| **AlexNet** | 2012 | 63.3% | 61M |
| **VGG-16** | 2014 | 74.4% | 138M |
| **ResNet-50** | 2015 | 76.1% | 25M |
| **EfficientNet-B7** | 2019 | 84.3% | 66M |
| **ConvNeXt-L** | 2022 | 87.8% | 198M |

## Applications

- Medical diagnosis from X-rays, MRIs, and pathology slides
- Content moderation and image filtering on social platforms
- Agricultural crop disease identification from field photos
- Quality control in manufacturing via defect classification
- Wildlife species identification from camera trap images

## Evolution

| Year | Milestone |
|------|-----------|
| 2012 | AlexNet wins ImageNet, sparking the deep learning revolution |
| 2014 | VGGNet demonstrates depth matters; GoogLeNet introduces Inception |
| 2015 | ResNet enables training of 150+ layer networks |
| 2019 | EfficientNet achieves optimal scaling of depth, width, resolution |
| 2021 | Vision Transformers (ViT) challenge CNN dominance |`,

    segmentation: `# Image Segmentation

Image segmentation is the pixel-level classification task that assigns a label to every pixel in an image. Unlike detection which outputs bounding boxes, segmentation produces precise masks that delineate exact object boundaries. This makes it essential for applications requiring fine-grained spatial understanding such as medical imaging, autonomous driving, and satellite analysis. Three main variants exist: semantic segmentation (per-pixel class labels), instance segmentation (separating individual object instances), and panoptic segmentation (combining both).

## Key Concepts

- **Semantic Segmentation:** Every pixel receives a class label, but individual instances of the same class are not distinguished
- **Instance Segmentation:** Each object instance gets a unique mask, distinguishing between separate objects of the same class
- **Panoptic Segmentation:** Unifies semantic and instance segmentation into a single coherent output
- **Encoder-Decoder Architecture:** Encoder compresses spatial information; decoder recovers spatial resolution through upsampling
- **Skip Connections:** Connect encoder layers to decoder layers to preserve fine spatial details lost during downsampling

\`\`\`
Encoder (Downsampling)              Decoder (Upsampling)
[Input 256x256] --conv--> [128x128] ----skip----> [128x128] --conv--> [256x256 Mask]
                --conv--> [64x64]   ----skip----> [64x64]
                --conv--> [32x32]   --bottleneck--> [32x32]
\`\`\`

## How It Works

**U-Net** (the dominant architecture) uses a symmetric encoder-decoder with skip connections. The encoder extracts multi-scale features through successive convolution and pooling. The decoder upsamples features using transposed convolutions, concatenating skip connection features to recover spatial detail. The output is a per-pixel class probability map.

\`\`\`python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)
# Output shape: (batch, num_classes, H, W)
# Apply argmax along class dim for final mask
\`\`\`

| Model | Type | Key Innovation |
|-------|------|---------------|
| **FCN** | Semantic | First end-to-end pixel classification |
| **U-Net** | Semantic | Skip connections for fine detail recovery |
| **Mask R-CNN** | Instance | Adds mask branch to Faster R-CNN |
| **DeepLab v3+** | Semantic | Atrous convolutions for multi-scale context |
| **Mask2Former** | Panoptic | Transformer-based unified segmentation |

## Applications

- Medical imaging (organ boundaries, tumor segmentation, cell counting)
- Autonomous driving (road, lane, pedestrian pixel-level parsing)
- Satellite imagery (land use mapping, building footprint extraction)
- Video editing (background removal, green-screen replacement)
- Robotics (scene understanding for manipulation and navigation)

## Evolution

| Year | Milestone |
|------|-----------|
| 2015 | Fully Convolutional Networks (FCN) enable end-to-end segmentation |
| 2015 | U-Net dominates medical image segmentation |
| 2017 | Mask R-CNN unifies detection and instance segmentation |
| 2018 | DeepLab v3+ achieves strong results with atrous spatial pyramid pooling |
| 2022 | Segment Anything Model (SAM) introduces promptable zero-shot segmentation |`,

    image_generation: `# Image Generation with GANs

Generative Adversarial Networks (GANs) are a framework for training generative models through an adversarial game between two neural networks: a Generator that creates synthetic images and a Discriminator that distinguishes real images from generated ones. Introduced by Ian Goodfellow in 2014, GANs revolutionized image synthesis by producing photorealistic outputs without explicit density estimation. The generator learns to map random noise vectors to realistic images, while the discriminator provides training signal by trying to detect fakes.

## Key Concepts

- **Generator (G):** Neural network that maps latent noise z ~ N(0,1) to synthetic images G(z)
- **Discriminator (D):** Neural network that classifies images as real (from dataset) or fake (from generator)
- **Adversarial Loss:** Minimax objective: min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]
- **Mode Collapse:** Failure mode where the generator produces limited variety of outputs
- **Latent Space:** Continuous vector space where arithmetic operations produce meaningful visual changes (e.g., smiling face - neutral face + frowning face)

\`\`\`
Random Noise z --> [Generator G] --> Fake Image G(z)
                                          |
                                    [Discriminator D] --> Real or Fake?
                                          |
Real Image x ----------------------------|
\`\`\`

## How It Works

Training alternates between updating D and G. The discriminator is trained to maximize its ability to distinguish real from fake. The generator is trained to minimize the discriminator's accuracy, effectively learning to produce increasingly convincing images. Training is notoriously unstable, requiring careful balancing of the two networks.

\`\`\`python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z.view(-1, 100, 1, 1))
\`\`\`

| GAN Variant | Key Innovation |
|-------------|---------------|
| **DCGAN** | Stable CNN architecture for GANs |
| **WGAN** | Wasserstein distance for stable training |
| **ProGAN** | Progressive growing for high-resolution synthesis |
| **StyleGAN3** | Style-based generator with alias-free operations |
| **GigaGAN** | Scaling GANs to text-to-image at billion parameters |

## Applications

- Photorealistic face generation and editing
- Image super-resolution and enhancement
- Data augmentation for training other models
- Art creation and creative tools
- Image inpainting (filling missing regions)

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Goodfellow introduces Generative Adversarial Networks |
| 2016 | DCGAN establishes stable convolutional GAN training |
| 2018 | ProGAN generates 1024x1024 photorealistic faces |
| 2019 | StyleGAN enables unprecedented control over generated imagery |
| 2022 | Diffusion models begin to surpass GANs for image quality |`,

    style_transfer: `# Neural Style Transfer

Neural style transfer is a technique that applies the artistic style of one image (the style reference) to the content of another image (the content target), producing a new image that preserves the original scene's structure while adopting the visual aesthetics of the style reference. The method exploits the hierarchical feature representations learned by deep CNNs, where lower layers capture texture and style patterns while higher layers encode semantic content and object structure.

## Key Concepts

- **Content Representation:** High-level feature maps from deep CNN layers that encode spatial structure and object arrangement
- **Style Representation:** Gram matrices computed from feature maps that capture texture patterns, color distributions, and brushstroke characteristics independent of spatial layout
- **Gram Matrix:** G_ij = sum_k(F_ik * F_jk), measuring correlations between feature channels to capture texture statistics
- **Perceptual Loss:** Loss computed in feature space (not pixel space) using a pre-trained VGG network as a fixed feature extractor
- **Total Variation Loss:** Regularization term that encourages spatial smoothness in the output image

\`\`\`
Content Image ----> VGG Features (Layer conv4_2) ----> Content Loss
                                                            |
Output Image -----> VGG Features (Multiple Layers) ----> Total Loss --> Optimize
                                                            |
Style Image ------> VGG Features (Gram Matrices) -----> Style Loss
\`\`\`

## How It Works

**Optimization-Based (Gatys et al.):**
Start with a random or content-initialized image. Iteratively optimize the pixel values to minimize a combined loss: content loss (feature distance to content image at deep layers) + style loss (Gram matrix distance to style image across multiple layers).

**Feed-Forward (Johnson et al.):**
Train a neural network to perform style transfer in a single forward pass, trading flexibility for speed.

\`\`\`python
import torchvision.models as models

vgg = models.vgg19(pretrained=True).features.eval()

# Content loss at deeper layer
content_loss = F.mse_loss(
    features_output["conv4_2"],
    features_content["conv4_2"]
)

# Style loss via Gram matrices across layers
def gram_matrix(x):
    b, c, h, w = x.size()
    F = x.view(b, c, h * w)
    return torch.bmm(F, F.transpose(1, 2)) / (c * h * w)

style_loss = sum(
    F.mse_loss(gram_matrix(features_output[l]),
               gram_matrix(features_style[l]))
    for l in style_layers
)
\`\`\`

| Approach | Speed | Flexibility | Quality |
|----------|-------|-------------|---------|
| **Optimization-based** | ~minutes | Any style | Highest |
| **Feed-forward network** | Real-time | Fixed style | High |
| **AdaIN** | Real-time | Arbitrary style | Good |
| **Diffusion-based** | ~seconds | Text-guided | Very High |

## Applications

- Artistic photo filters (mobile apps like Prisma)
- Creative artwork generation and digital art tools
- Video stylization for film and media production
- Architectural visualization with artistic rendering
- Game and virtual world texture generation

## Evolution

| Year | Milestone |
|------|-----------|
| 2015 | Gatys et al. introduce neural style transfer via optimization |
| 2016 | Johnson et al. achieve real-time transfer with feed-forward networks |
| 2017 | AdaIN enables arbitrary style transfer without retraining |
| 2019 | Video-consistent style transfer with temporal coherence |
| 2023 | Diffusion models enable text-guided artistic stylization at scale |`,

    // ============================================================
    // NLP FOUNDATIONS
    // ============================================================

    tokenization: `# Tokenization

Tokenization is the process of converting raw text into a sequence of discrete tokens that serve as the input vocabulary for language models. It is a critical preprocessing step that directly impacts model performance, vocabulary size, and the ability to handle rare or out-of-vocabulary words. Modern subword tokenization methods strike a balance between character-level flexibility (handling any word) and word-level semantics (preserving meaningful units), enabling models to process any text in any language with a finite, manageable vocabulary.

## Key Concepts

- **Word-Level Tokenization:** Splits on whitespace/punctuation; simple but creates huge vocabularies and cannot handle unseen words
- **Character-Level Tokenization:** Uses individual characters as tokens; tiny vocabulary but loses word semantics and creates very long sequences
- **Subword Tokenization:** Splits words into frequently occurring sub-units, combining the benefits of word and character approaches
- **Byte Pair Encoding (BPE):** Iteratively merges the most frequent pair of adjacent tokens until vocabulary size is reached
- **Special Tokens:** Reserved tokens like [CLS], [SEP], [PAD], [MASK], <|endoftext|> that serve structural roles

\`\`\`
"unbelievably" -->  Word-level:    ["unbelievably"]          (OOV risk)
                    Character:     ["u","n","b","e","l",...]  (too long)
                    BPE subword:   ["un", "believ", "ably"]   (balanced)
\`\`\`

## How It Works

**BPE Algorithm:**
1. Start with all individual characters as the initial vocabulary
2. Count all adjacent token pairs in the training corpus
3. Merge the most frequent pair into a new single token
4. Repeat until the desired vocabulary size is reached

\`\`\`python
from transformers import AutoTokenizer

# GPT-2 uses BPE
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("Tokenization is fundamental")
# ['Token', 'ization', ' is', ' fundamental']
ids = tokenizer.encode("Tokenization is fundamental")
# [30642, 1634, 318, 7531]
\`\`\`

| Method | Used By | Vocab Size | Approach |
|--------|---------|------------|----------|
| **BPE** | GPT-2, GPT-4, LLaMA | 50k-100k | Frequency-based merging |
| **WordPiece** | BERT, DistilBERT | 30k | Likelihood-based merging |
| **SentencePiece** | T5, ALBERT, mBART | 32k-250k | Language-agnostic, treats text as raw bytes |
| **Unigram** | XLNet, ALBERT | Variable | Probabilistic subword selection |
| **Byte-level BPE** | GPT-4, LLaMA 2 | 100k+ | Operates on raw bytes, no UNK tokens |

## Applications

- Preprocessing for all modern language models (GPT, BERT, T5)
- Multilingual NLP where scripts and morphology vary widely
- Code generation models that tokenize programming syntax
- Domain-specific models (biomedical, legal) requiring specialized vocabularies
- Efficient compression of text for storage and transmission

## Evolution

| Year | Milestone |
|------|-----------|
| 1994 | BPE originally developed for data compression |
| 2016 | Sennrich applies BPE to neural machine translation |
| 2018 | WordPiece used in BERT; SentencePiece enables language-agnostic tokenization |
| 2019 | Byte-level BPE in GPT-2 eliminates out-of-vocabulary tokens |
| 2023 | Tiktoken and optimized tokenizers handle 100k+ vocabularies efficiently |`,

    word_embeddings: `# Word Embeddings

Word embeddings are dense, low-dimensional vector representations of words that capture semantic relationships in continuous vector space. Unlike one-hot encoding where every word is an orthogonal sparse vector, embeddings map semantically similar words to nearby points in vector space. This enables models to generalize across related words and perform arithmetic on meaning. Word embeddings were a transformative breakthrough that made neural NLP practical by providing a way to represent discrete symbols as continuous, learnable features.

## Key Concepts

- **Distributional Hypothesis:** Words appearing in similar contexts have similar meanings ("You shall know a word by the company it keeps")
- **Embedding Dimension:** Typically 100-1024 dimensions; higher dimensions capture more nuance but require more data
- **Cosine Similarity:** Measures semantic similarity between embeddings: sim(a,b) = (a . b) / (||a|| * ||b||)
- **Analogy Completion:** Vector arithmetic captures semantic relationships: king - man + woman = queen
- **Contextual Embeddings:** Modern embeddings (BERT, GPT) produce different vectors for the same word based on surrounding context

\`\`\`
One-Hot (dim = vocab_size):   "cat" = [0, 0, 1, 0, ..., 0]  (sparse, 50k dims)
Embedding (dim = 300):        "cat" = [0.21, -0.05, 0.83, ..., 0.12]  (dense)

Semantic Space:
  "king" -----> "queen"     (gender direction)
    |              |
  "man" ------> "woman"     (same offset vector)
\`\`\`

## How It Works

**Word2Vec (Skip-gram):** Predicts surrounding context words given a target word. Trains by maximizing the probability of context words appearing near the target. Uses negative sampling to make training efficient.

**GloVe:** Constructs a word co-occurrence matrix from the corpus, then factorizes it to produce embeddings that capture global statistical information.

\`\`\`python
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

# Load pre-trained GloVe vectors
glove = api.load("glove-wiki-gigaword-300")
print(glove.most_similar("python"))
# [('perl', 0.78), ('java', 0.75), ...]

# Analogy: king - man + woman = ?
result = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"]
)  # [('queen', 0.87), ...]
\`\`\`

| Method | Year | Type | Key Strength |
|--------|------|------|-------------|
| **Word2Vec** | 2013 | Static | Fast training, captures syntax/semantics |
| **GloVe** | 2014 | Static | Global co-occurrence statistics |
| **FastText** | 2016 | Static | Subword n-grams handle rare/OOV words |
| **ELMo** | 2018 | Contextual | Different vectors per context |
| **BERT embeddings** | 2018 | Contextual | Deep bidirectional context |

## Applications

- Semantic similarity and text clustering
- Named entity recognition and part-of-speech tagging
- Sentiment analysis feature representation
- Information retrieval and document ranking
- Cross-lingual transfer (aligned multilingual embeddings)

## Evolution

| Year | Milestone |
|------|-----------|
| 2003 | Bengio's neural language model learns word representations |
| 2013 | Word2Vec makes large-scale embedding training practical |
| 2014 | GloVe combines global matrix factorization with local context windows |
| 2016 | FastText introduces subword embeddings for morphologically rich languages |
| 2018 | ELMo and BERT shift the field to contextual embeddings |`,

    sequence_models: `# Sequence-to-Sequence Models

Sequence-to-Sequence (Seq2Seq) models map variable-length input sequences to variable-length output sequences using an encoder-decoder architecture. The encoder processes the input into a fixed or variable-length representation, and the decoder generates the output one token at a time. Originally built with RNNs and LSTMs, Seq2Seq models were the dominant paradigm for machine translation, summarization, and dialogue before the transformer architecture unified and surpassed them.

## Key Concepts

- **Encoder:** Processes the input sequence and compresses it into a context representation (hidden states)
- **Decoder:** Generates the output sequence autoregressively, conditioned on the encoder output and previously generated tokens
- **Attention Mechanism:** Allows the decoder to focus on different parts of the input at each generation step, solving the information bottleneck
- **Teacher Forcing:** Training technique where the decoder receives the ground-truth previous token instead of its own prediction
- **Beam Search:** Decoding strategy that maintains top-k candidate sequences at each step instead of greedy selection

\`\`\`
Encoder:  [x1] --> [x2] --> [x3] --> [Context Vector]
                                            |
Decoder:                              [<start>] --> [y1] --> [y2] --> [<end>]

With Attention:
Encoder:  [h1]    [h2]    [h3]
            \\      |      /
         [Attention Weights: 0.1, 0.7, 0.2]
                   |
Decoder:    [context] + [prev_output] --> [y_t]
\`\`\`

## How It Works

1. The encoder (RNN/LSTM/GRU) reads the input token by token, producing hidden states
2. The final hidden state (or all states with attention) forms the context
3. The decoder generates output tokens one at a time, using the context and attending to encoder states
4. The attention score at each decoder step is: **alpha_i = softmax(score(s_t, h_i))**
5. Context vector: **c_t = sum(alpha_i * h_i)**

\`\`\`python
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(src_vocab, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(tgt_vocab, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab)

    def forward(self, src, tgt):
        enc_out, (h, c) = self.encoder(src)
        dec_out, _ = self.decoder(tgt, (h, c))
        return self.fc(dec_out)
\`\`\`

| Architecture | Strength | Limitation |
|-------------|----------|------------|
| **Vanilla Seq2Seq** | Simple | Fixed-size bottleneck |
| **+ Bahdanau Attention** | Dynamic context | Sequential computation |
| **+ Copy Mechanism** | Handles rare words | Added complexity |
| **Transformer** | Parallel, scalable | Quadratic memory in sequence length |

## Applications

- Machine translation (English to French, Chinese to English)
- Text summarization (document to summary)
- Conversational AI and chatbots
- Code generation from natural language descriptions
- Speech recognition (audio sequence to text sequence)

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Sutskever et al. introduce Seq2Seq with LSTMs for translation |
| 2015 | Bahdanau attention eliminates the fixed-context bottleneck |
| 2016 | Google Neural Machine Translation (GNMT) deployed at scale |
| 2017 | Transformer replaces RNNs with self-attention for Seq2Seq |
| 2020 | Pre-trained encoder-decoders (T5, BART) dominate Seq2Seq tasks |`,

    text_classification: `# Text Classification

Text classification is the task of assigning predefined categories to text documents. It is one of the most widely deployed NLP applications, handling tasks from spam detection and sentiment analysis to topic categorization and intent recognition. Deep learning approaches have largely replaced traditional bag-of-words and TF-IDF methods by learning rich text representations that capture word order, context, and semantic nuance directly from raw text.

## Key Concepts

- **Sentiment Analysis:** Classifying text as positive, negative, or neutral based on expressed opinion
- **Multi-Label Classification:** Assigning multiple non-exclusive labels to a single document (e.g., topic tags)
- **Fine-Tuning:** Adapting a pre-trained language model to a specific classification task with a task-specific output head
- **Text Preprocessing:** Lowercasing, removing stopwords, lemmatization (traditional); tokenization into subwords (modern)
- **Class Imbalance:** Handling skewed label distributions via oversampling, weighted loss, or focal loss

\`\`\`
Input Text: "This movie was absolutely fantastic!"
    |
[Tokenization] --> [101, 2023, 3185, 2001, 7078, 10392, 999, 102]
    |
[Pre-trained Encoder (BERT)] --> [CLS] hidden state (768-dim)
    |
[Classification Head (Linear)] --> [Positive: 0.96, Negative: 0.04]
\`\`\`

## How It Works

**CNN for Text:** Apply 1D convolutions over word embeddings with varying kernel sizes (e.g., 3, 4, 5 for n-gram features), max-pool across sequence length, then classify.

**RNN/LSTM for Text:** Process the token sequence sequentially, use the final hidden state or attention-pooled states for classification.

**Transformer Fine-Tuning (Dominant):** Add a classification layer on top of the [CLS] token output from BERT or similar pre-trained models.

\`\`\`python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("zero-shot-classification",
                       model="facebook/bart-large-mnli")
result = classifier(
    "The quarterly earnings exceeded expectations",
    candidate_labels=["finance", "sports", "technology"]
)
# {'labels': ['finance', ...], 'scores': [0.94, ...]}

# Fine-tuned sentiment analysis
sentiment = pipeline("sentiment-analysis")
sentiment("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
\`\`\`

| Approach | Speed | Accuracy | Training Data Needed |
|----------|-------|----------|---------------------|
| **Naive Bayes + TF-IDF** | Very Fast | Moderate | Small |
| **TextCNN** | Fast | Good | Medium |
| **BiLSTM + Attention** | Medium | Good | Medium |
| **BERT Fine-tuned** | Slow | Excellent | Small-Medium |
| **Zero-shot (BART)** | Medium | Good | None (zero-shot) |

## Applications

- Email spam and phishing detection
- Customer review sentiment analysis
- News article topic categorization
- Support ticket routing and priority classification
- Content moderation (hate speech, toxicity detection)

## Evolution

| Year | Milestone |
|------|-----------|
| 2012 | Word2Vec + simple classifiers improve over bag-of-words |
| 2014 | Kim's TextCNN applies convolutional networks to text |
| 2015 | Attention-based BiLSTMs achieve strong sequence classification |
| 2018 | BERT fine-tuning sets new baselines across all text classification benchmarks |
| 2020 | Zero-shot classification with large language models eliminates task-specific training |`,

    ner: `# Named Entity Recognition (NER)

Named Entity Recognition is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, monetary values, and more. NER is a fundamental component of information extraction pipelines, enabling systems to understand who, what, where, and when within unstructured text. Modern NER systems frame the problem as a sequence labeling task, assigning a tag to each token in the input sequence.

## Key Concepts

- **BIO Tagging Scheme:** B-TYPE marks the beginning of an entity, I-TYPE marks inside tokens, O marks non-entity tokens
- **Entity Types:** Common categories include PER (person), ORG (organization), LOC (location), DATE, MONEY, MISC
- **Span-Based NER:** Alternative to BIO that directly predicts entity spans (start, end, type) tuples
- **Nested Entities:** Handling overlapping entities (e.g., "New York University" contains LOC "New York" and ORG "New York University")
- **Conditional Random Fields (CRF):** Structured prediction layer often added on top of neural models to enforce valid tag sequences

\`\`\`
Input:   "Barack  Obama   visited  the  United  Nations  in  New   York"
BIO:      B-PER   I-PER   O        O    B-ORG   I-ORG    O   B-LOC  I-LOC
\`\`\`

## How It Works

Modern NER uses a transformer encoder (e.g., BERT) to produce contextual token representations, followed by a classification head that predicts a BIO tag for each token. An optional CRF layer models tag transition probabilities to ensure valid sequences (e.g., I-PER cannot follow B-ORG).

\`\`\`python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER",
               aggregation_strategy="simple")
results = ner("Elon Musk founded SpaceX in Hawthorne, California")
# [
#   {'entity_group': 'PER', 'word': 'Elon Musk', 'score': 0.99},
#   {'entity_group': 'ORG', 'word': 'SpaceX', 'score': 0.98},
#   {'entity_group': 'LOC', 'word': 'Hawthorne', 'score': 0.97},
#   {'entity_group': 'LOC', 'word': 'California', 'score': 0.99}
# ]

# spaCy NER
import spacy
nlp = spacy.load("en_core_web_trf")
doc = nlp("Apple is opening a new store in London")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Apple: ORG, London: GPE
\`\`\`

| Model | F1 (CoNLL-2003) | Approach |
|-------|----------------|----------|
| **BiLSTM-CRF** | 91.2% | Sequential encoding + structured prediction |
| **BERT + Linear** | 92.8% | Transformer + token classification |
| **BERT + CRF** | 93.1% | Transformer + structured prediction |
| **RoBERTa-Large** | 93.6% | Larger pre-trained model fine-tuned |
| **UniversalNER** | 93.9% | LLM-based instruction-tuned NER |

## Applications

- Information extraction from documents, contracts, and reports
- Knowledge graph construction from unstructured text
- Search engine query understanding and entity linking
- Medical NER for drug names, diseases, and symptoms
- Financial NER for company names, tickers, and monetary values

## Evolution

| Year | Milestone |
|------|-----------|
| 2003 | CoNLL shared task establishes standard NER benchmarks |
| 2015 | BiLSTM-CRF becomes the dominant neural NER architecture |
| 2018 | BERT fine-tuning surpasses all prior NER systems |
| 2020 | Few-shot NER with pre-trained models reduces annotation needs |
| 2023 | LLM-based NER (UniversalNER) achieves zero-shot entity extraction |`,

    // ============================================================
    // SPEECH & AUDIO
    // ============================================================

    asr: `# Automatic Speech Recognition (ASR)

Automatic Speech Recognition is the task of converting spoken language audio into written text. ASR systems must handle diverse accents, background noise, varying speaking speeds, and domain-specific vocabulary. Modern ASR has evolved from complex multi-component pipelines (acoustic model, pronunciation model, language model) to end-to-end neural architectures that directly map audio waveforms or spectrograms to text transcriptions, dramatically simplifying the system while achieving superhuman accuracy on many benchmarks.

## Key Concepts

- **Mel Spectrogram:** Time-frequency representation of audio that mimics human auditory perception, used as input features for neural ASR
- **Connectionist Temporal Classification (CTC):** Loss function that handles alignment between variable-length audio and text without explicit frame-level labels
- **Encoder-Decoder with Attention:** Architecture where an audio encoder produces frame representations and a text decoder generates transcription with cross-attention
- **Beam Search Decoding:** Searches multiple candidate transcriptions in parallel for optimal output
- **Word Error Rate (WER):** Primary metric: WER = (Substitutions + Insertions + Deletions) / Total Reference Words

\`\`\`
Audio Waveform --> [Mel Spectrogram] --> [Encoder (Conformer/Transformer)]
                                                    |
                                              [CTC / Attention]
                                                    |
                                              [Text Decoder] --> "The quick brown fox"
\`\`\`

## How It Works

**End-to-End ASR (Whisper):** The audio is converted to an 80-channel log-Mel spectrogram. A transformer encoder processes the spectrogram features. A transformer decoder autoregressively generates text tokens, using cross-attention over encoder outputs. Whisper is trained on 680,000 hours of multilingual labeled audio.

\`\`\`python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
# "The quick brown fox jumps over the lazy dog"

# With language detection and timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)
for segment in result["segments"]:
    print(f"[{segment['start']:.1f}s] {segment['text']}")
\`\`\`

| Model | Year | WER (LibriSpeech) | Approach |
|-------|------|-------------------|----------|
| **DeepSpeech2** | 2016 | 5.3% | CTC + RNN |
| **wav2vec 2.0** | 2020 | 1.8% | Self-supervised pre-training + CTC |
| **Conformer** | 2020 | 1.9% | Conv + Transformer hybrid |
| **Whisper Large** | 2022 | 2.7% | Encoder-decoder, massively multilingual |
| **Universal-1** | 2023 | 1.5% | Commercial state-of-the-art |

## Applications

- Voice assistants (Siri, Alexa, Google Assistant)
- Meeting transcription and real-time captioning
- Medical dictation and clinical documentation
- Call center analytics and compliance monitoring
- Accessibility tools for deaf and hard-of-hearing users

## Evolution

| Year | Milestone |
|------|-----------|
| 1952 | Bell Labs Audrey recognizes spoken digits |
| 2012 | Deep learning replaces GMM-HMM acoustic models |
| 2016 | End-to-end CTC models simplify ASR pipelines |
| 2020 | wav2vec 2.0 enables self-supervised speech representation learning |
| 2022 | Whisper achieves robust multilingual ASR trained on web-scale data |`,

    tts: `# Text-to-Speech (TTS)

Text-to-Speech synthesis converts written text into natural-sounding human speech. Modern neural TTS systems produce audio that is nearly indistinguishable from human recordings, handling prosody, intonation, emphasis, and emotional expression. The typical pipeline consists of a text analysis frontend (grapheme-to-phoneme conversion, prosody prediction), an acoustic model that generates mel spectrograms, and a vocoder that converts spectrograms to audio waveforms.

## Key Concepts

- **Mel Spectrogram:** Intermediate acoustic representation that captures frequency content aligned with human perception
- **Vocoder:** Neural network that converts mel spectrograms to raw audio waveforms (WaveNet, HiFi-GAN, WaveGlow)
- **Prosody:** Rhythm, stress, and intonation patterns that make speech sound natural rather than robotic
- **Grapheme-to-Phoneme (G2P):** Converts written text to phonetic pronunciation (e.g., "read" -> /rid/ or /rEd/ depending on context)
- **Speaker Embedding:** Vector that captures speaker identity, enabling multi-speaker or voice cloning capabilities

\`\`\`
Text: "Hello world"
  |
[Text Frontend] --> Phonemes: /h@loU w3:ld/
  |
[Acoustic Model (Tacotron/VITS)] --> Mel Spectrogram
  |
[Vocoder (HiFi-GAN)] --> Audio Waveform --> Speaker Output
\`\`\`

## How It Works

**Tacotron 2:** An encoder converts text/phonemes to hidden representations. An autoregressive decoder with attention generates mel spectrogram frames one at a time. A WaveNet or HiFi-GAN vocoder synthesizes the final waveform.

**VITS (End-to-End):** Combines text processing, acoustic modeling, and waveform generation into a single model using variational inference and adversarial training, producing speech directly from text in one forward pass.

\`\`\`python
# Using Coqui TTS
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/vits")
tts.tts_to_file(text="Hello, how are you today?",
                file_path="output.wav")

# Multi-speaker voice cloning
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="This is a cloned voice.",
    speaker_wav="reference_audio.wav",
    language="en",
    file_path="cloned_output.wav"
)
\`\`\`

| Model | Year | Key Innovation |
|-------|------|---------------|
| **WaveNet** | 2016 | Autoregressive neural vocoder, natural quality |
| **Tacotron 2** | 2018 | Attention-based spectrogram prediction |
| **FastSpeech 2** | 2020 | Non-autoregressive, parallel generation |
| **VITS** | 2021 | End-to-end with variational inference |
| **XTTS** | 2023 | Cross-lingual zero-shot voice cloning |

## Applications

- Virtual assistants and conversational AI interfaces
- Audiobook narration and content accessibility
- Voice cloning for personalized digital assistants
- Language learning with native pronunciation examples
- Assistive technology for individuals with speech impairments

## Evolution

| Year | Milestone |
|------|-----------|
| 1968 | First computer-generated speech at Bell Labs |
| 2016 | WaveNet produces near-human-quality speech |
| 2018 | Tacotron 2 makes end-to-end neural TTS practical |
| 2021 | VITS achieves single-model end-to-end synthesis |
| 2023 | Zero-shot voice cloning with minimal reference audio becomes possible |`,

    speaker_recognition: `# Speaker Recognition

Speaker recognition is the task of identifying or verifying a person's identity based on their voice characteristics. It encompasses two subtasks: speaker identification (determining who is speaking from a set of known speakers) and speaker verification (confirming whether a speech sample matches a claimed identity). The technology leverages the fact that each person's voice has unique acoustic properties determined by vocal tract shape, speaking habits, and articulatory patterns that form a distinctive voiceprint.

## Key Concepts

- **Speaker Verification:** One-to-one comparison that answers "Is this person who they claim to be?" (accept/reject decision)
- **Speaker Identification:** One-to-many comparison that answers "Which known speaker produced this utterance?"
- **Speaker Embedding:** Fixed-dimensional vector extracted from variable-length audio that represents speaker identity (analogous to face embeddings)
- **Speaker Diarization:** Segmenting audio into speaker turns, answering "Who spoke when?" in multi-speaker recordings
- **Enrollment:** Process of registering a speaker's voiceprint by extracting embeddings from reference audio samples

\`\`\`
Audio Segment --> [Feature Extraction (Mel/MFCC)]
                           |
                  [Speaker Encoder (ECAPA-TDNN)]
                           |
                  [Speaker Embedding (192-dim)]
                           |
              [Cosine Similarity / PLDA Scoring]
                           |
           Verification: score > threshold? --> Accept/Reject
           Identification: argmax(scores) --> Speaker ID
\`\`\`

## How It Works

A speaker encoder network (e.g., ECAPA-TDNN or ResNet) processes mel-spectrogram features and produces a fixed-size embedding vector. For verification, the cosine similarity between the test embedding and the enrolled speaker embedding is compared against a threshold. Training uses metric learning losses like AAM-Softmax or contrastive loss to ensure embeddings from the same speaker are close and different speakers are far apart.

\`\`\`python
from speechbrain.inference import SpeakerRecognition

verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Verify if two audio samples are from the same speaker
score, prediction = verifier.verify_files(
    "speaker1_enroll.wav",
    "speaker1_test.wav"
)
print(f"Score: {score:.3f}, Same speaker: {prediction}")
# Score: 0.891, Same speaker: True

# Extract speaker embedding
embedding = verifier.encode_batch(waveform)
# Shape: (1, 192)
\`\`\`

| Model | Year | EER (VoxCeleb1) | Architecture |
|-------|------|-----------------|-------------|
| **x-vector** | 2018 | 3.1% | TDNN + statistics pooling |
| **ECAPA-TDNN** | 2020 | 0.87% | SE-blocks + attentive stats pooling |
| **ResNetSE34** | 2020 | 1.12% | ResNet with squeeze-excitation |
| **WavLM + ECAPA** | 2022 | 0.56% | Self-supervised features + ECAPA |
| **Whisper-AT** | 2023 | 0.71% | Whisper encoder adapted for speaker tasks |

## Applications

- Biometric authentication (banking, device unlock)
- Forensic speaker analysis for law enforcement
- Call center caller verification and fraud detection
- Meeting transcription with speaker diarization
- Personalized voice assistant responses

## Evolution

| Year | Milestone |
|------|-----------|
| 1995 | GMM-UBM becomes the standard speaker modeling approach |
| 2011 | i-vectors replace GMMs with compact speaker representations |
| 2018 | x-vectors introduce deep neural network speaker embeddings |
| 2020 | ECAPA-TDNN achieves sub-1% error on VoxCeleb benchmarks |
| 2022 | Self-supervised models (WavLM) further improve speaker embeddings |`,

    emotion_recognition: `# Emotion Recognition

Emotion recognition in speech and multimodal data is the task of automatically detecting and classifying the emotional state of a speaker from audio, text, facial expressions, or combinations thereof. Speech emotion recognition (SER) analyzes acoustic features like pitch, energy, speaking rate, and spectral characteristics to identify emotions such as happiness, sadness, anger, fear, surprise, and neutral states. Multimodal approaches fuse information from audio, text transcription, and visual cues for more robust and accurate emotion detection.

## Key Concepts

- **Categorical Emotions:** Discrete classes like happiness, sadness, anger, fear, surprise, disgust (Ekman's basic emotions)
- **Dimensional Model:** Continuous axes representing valence (positive-negative), arousal (calm-excited), and dominance (weak-strong)
- **Acoustic Features:** Fundamental frequency (F0/pitch), energy, MFCCs, jitter, shimmer, speaking rate, spectral centroid
- **Multimodal Fusion:** Combining audio, text, and visual modalities; can be early fusion (feature-level), late fusion (decision-level), or attention-based
- **Domain Shift:** Major challenge where models trained on acted speech datasets perform poorly on spontaneous real-world emotions

\`\`\`
Audio -----> [Acoustic Encoder] -----> Audio Embedding ----\\
                                                            |
Text ------> [Text Encoder] ---------> Text Embedding -----+--> [Fusion] --> Emotion
                                                            |
Video -----> [Face Encoder] ---------> Visual Embedding ---/

Dimensional Output:
  Valence: [-1.0 (sad) ... +1.0 (happy)]
  Arousal: [-1.0 (calm) ... +1.0 (excited)]
\`\`\`

## How It Works

**Speech Emotion Recognition:** Audio is converted to log-mel spectrograms or handcrafted feature sets (eGeMAPS). A CNN or transformer encoder extracts temporal and spectral patterns. An attention mechanism highlights emotionally salient frames. The pooled representation is classified into emotion categories or mapped to dimensional values.

\`\`\`python
from transformers import pipeline

# Speech emotion recognition with Hugging Face
emotion_pipe = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
result = emotion_pipe("speech_sample.wav")
# [{'label': 'angry', 'score': 0.85},
#  {'label': 'neutral', 'score': 0.08}, ...]

# Using openSMILE for acoustic features
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
features = smile.process_file("audio.wav")
# 88 acoustic features (F0, energy, MFCCs, jitter, shimmer, etc.)
\`\`\`

| Model | Modality | Dataset | Accuracy |
|-------|----------|---------|----------|
| **wav2vec2 + Linear** | Audio | IEMOCAP | 72.3% |
| **HuBERT + Attention** | Audio | IEMOCAP | 75.1% |
| **Text BERT + Audio CNN** | Multimodal | IEMOCAP | 78.6% |
| **Emotion2Vec** | Audio | Multiple | 77.2% |
| **GPT-4o Multimodal** | Audio + Text | Multiple | ~80% |

## Applications

- Call center quality monitoring and customer satisfaction detection
- Mental health screening through speech pattern analysis
- In-car driver state monitoring for safety alerts
- Interactive gaming and virtual reality with emotional NPCs
- Educational technology that adapts to student frustration or engagement

## Evolution

| Year | Milestone |
|------|-----------|
| 2003 | IEMOCAP dataset establishes benchmark for emotion recognition |
| 2013 | Deep learning surpasses handcrafted features for SER |
| 2018 | Attention mechanisms improve focus on emotionally relevant frames |
| 2021 | Self-supervised speech models (wav2vec2, HuBERT) advance SER accuracy |
| 2024 | Multimodal large language models integrate emotion understanding |`,

    music_generation: `# AI Music Generation

AI music generation uses deep learning models to compose, arrange, and produce musical content ranging from simple melodies to full multi-track productions. These systems learn musical structure, harmony, rhythm, and style from large corpora of existing music and can generate novel compositions in specified genres, moods, or styles. The field spans symbolic music generation (MIDI notes and scores), raw audio synthesis, and hybrid approaches that combine symbolic understanding with high-fidelity audio production.

## Key Concepts

- **Symbolic Music:** Representation as MIDI events (note on/off, pitch, velocity, duration) or piano roll matrices, enabling precise musical structure control
- **Audio Generation:** Direct waveform or spectrogram synthesis producing ready-to-listen audio output
- **Music Language Model:** Treating music tokens (pitch, duration, instrument) as a sequence language modeling problem
- **Conditioning:** Guiding generation with text prompts, melody contours, chord progressions, genre labels, or reference audio
- **Audio Codecs:** Neural audio compression (Encodec, SoundStream) that converts audio to discrete token sequences for language model processing

\`\`\`
Text Prompt: "upbeat jazz piano with walking bass"
    |
[Text Encoder] --> Conditioning Signal
    |
[Music Language Model / Diffusion Model]
    |
[Audio Codec Decoder / Vocoder]
    |
Output: Generated Audio (stereo, 44.1kHz)
\`\`\`

## How It Works

**Language Model Approach (MusicLM, MusicGen):** Audio is tokenized into discrete codes using a neural audio codec (Encodec). A transformer language model generates sequences of audio tokens conditioned on text embeddings or other inputs. The audio codec decoder converts tokens back to waveform. Multiple codebook streams can be modeled in parallel for quality.

**Diffusion Approach (MusicLDM, Stable Audio):** A latent diffusion model iteratively denoises random noise into a mel spectrogram or latent audio representation, guided by text or musical conditioning signals.

\`\`\`python
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=15)  # 15 seconds

# Text-conditional generation
wav = model.generate([
    "a cheerful acoustic guitar melody with light percussion",
    "dark ambient electronic with deep bass and reverb"
])
# wav shape: (2, 1, 32000 * 15) -- 2 samples, mono, 15 sec

# Melody-conditional generation
wav = model.generate_with_chroma(
    descriptions=["orchestral version of the melody"],
    melody_wavs=reference_audio,
    melody_sample_rate=32000
)
\`\`\`

| Model | Year | Type | Capability |
|-------|------|------|-----------|
| **MuseNet** | 2019 | Symbolic | Multi-instrument MIDI generation |
| **Jukebox** | 2020 | Audio | Raw audio with lyrics in multiple genres |
| **MusicLM** | 2023 | Audio | Text-to-music with high fidelity |
| **MusicGen** | 2023 | Audio | Open-source text/melody-conditioned generation |
| **Stable Audio 2** | 2024 | Audio | Variable-length, structured music generation |

## Applications

- Background music for video content, podcasts, and games
- Creative composition assistance for musicians and producers
- Personalized music generation for meditation, focus, and mood
- Adaptive game soundtracks that respond to player actions
- Rapid prototyping of musical ideas and arrangements

## Evolution

| Year | Milestone |
|------|-----------|
| 2016 | WaveNet demonstrates neural audio synthesis quality |
| 2019 | MuseNet generates 4-minute multi-instrument MIDI compositions |
| 2020 | Jukebox produces raw audio music with singing in multiple styles |
| 2023 | MusicLM and MusicGen enable text-to-music at production quality |
| 2024 | Suno and Udio bring AI music generation to consumer applications |`,

  };
  Object.assign(window.AI_DOCS, content);
})();