// 05 - Transformers & LLMs
(function() {
  const content = {

    // ============================================================
    // ARCHITECTURE
    // ============================================================

    self_attention: `# Self-Attention Mechanism

Self-attention (also called intra-attention) is the core computational primitive of the Transformer architecture. Introduced in the landmark "Attention Is All You Need" paper (Vaswani et al., 2017), it allows every position in a sequence to attend to every other position, computing a weighted representation based on relevance. Unlike recurrent models that process sequences step by step, self-attention captures dependencies across arbitrary distances in a single operation, enabling massive parallelism and superior long-range modeling.

## Key Concepts

Self-attention operates on three learned projections of the input:
- **Query (Q):** What am I looking for?
- **Key (K):** What do I contain?
- **Value (V):** What information do I provide?

Each token produces a Q, K, and V vector by multiplying the input embedding with learned weight matrices W_Q, W_K, and W_V.

## How It Works

**Scaled Dot-Product Attention:**

\`\`\`
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
\`\`\`

Step-by-step:
1. Compute dot products between each query and all keys: scores = Q * K^T
2. Scale by sqrt(d_k) to prevent softmax saturation for large dimensions
3. Apply softmax to obtain attention weights (each row sums to 1)
4. Multiply weights by values to produce the output

\`\`\`python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Example: sequence of 10 tokens, dimension 64
x = torch.randn(1, 10, 64)
Q = K = V = x  # self-attention: Q, K, V all come from same input
out, weights = scaled_dot_product_attention(Q, K, V)
# out.shape: (1, 10, 64), weights.shape: (1, 10, 10)
\`\`\`

**Causal masking** is used in decoder models (GPT) to prevent tokens from attending to future positions. The mask sets upper-triangle scores to -infinity before softmax.

## Applications

- Language modeling and text generation (GPT family)
- Machine translation (original Transformer)
- Document understanding and summarization
- Vision Transformers (ViT) treating image patches as tokens

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Bahdanau attention for seq-to-seq models |
| 2015 | Luong attention introduces dot-product scoring |
| 2017 | Vaswani et al. introduce self-attention in the Transformer |
| 2020 | Flash Attention optimizes memory and speed for self-attention |
| 2023 | Grouped Query Attention (GQA) reduces KV cache overhead in LLMs |`,

    multi_head_attention: `# Multi-Head Attention

Multi-Head Attention (MHA) extends self-attention by running multiple attention operations in parallel, each with its own learned projection. Instead of computing a single attention function, MHA splits the model dimension into multiple heads, allowing the model to jointly attend to information from different representation subspaces at different positions. This dramatically increases the expressive power of the attention mechanism and is a defining feature of all Transformer architectures.

## Key Concepts

| Component | Description |
|-----------|-------------|
| **Number of heads (h)** | Typically 8, 12, 16, or more; splits d_model into h subspaces |
| **Head dimension (d_k)** | d_model / h (e.g., 768 / 12 = 64 per head) |
| **Per-head projections** | Each head has its own W_Q, W_K, W_V matrices |
| **Output projection** | Concatenated head outputs are projected back via W_O |

**Formula:**
\`\`\`
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
where head_i = Attention(Q * W_Qi, K * W_Ki, V * W_Vi)
\`\`\`

## How It Works

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)
\`\`\`

**Variants for efficiency:**
- **Multi-Query Attention (MQA):** All heads share a single K and V projection; faster inference
- **Grouped Query Attention (GQA):** K/V are shared among groups of heads; used in LLaMA 2/3 and Mistral
- **Multi-Head Latent Attention (MLA):** Compresses KV into a latent space; used in DeepSeek-V2

| Variant | KV Heads | Speed | Quality | Used In |
|---------|----------|-------|---------|---------|
| MHA | h | Baseline | Best | BERT, GPT-2 |
| MQA | 1 | Fastest | Slightly lower | PaLM, Falcon |
| GQA | h / g | Fast | Near MHA | LLaMA 2, Mistral |

## Applications

- Every layer in every Transformer-based model (BERT, GPT, T5, ViT)
- Cross-attention in encoder-decoder models (translation, summarization)
- Multi-modal fusion (attending across text and image tokens)

## Evolution

| Year | Milestone |
|------|-----------|
| 2017 | Multi-Head Attention introduced in the original Transformer |
| 2019 | Multi-Query Attention proposed by Shazeer for faster decoding |
| 2022 | Flash Attention enables memory-efficient MHA computation |
| 2023 | GQA adopted in LLaMA 2 as a balance between MHA and MQA |
| 2024 | MLA introduced in DeepSeek-V2 for compressed KV caches |`,

    positional_encoding: `# Positional Encoding

Positional encoding injects sequence order information into Transformer models, which otherwise treat input tokens as an unordered set. Since self-attention computes pairwise relationships without any inherent notion of position, positional encodings are essential for the model to distinguish "the cat sat on the mat" from "the mat sat on the cat." The design of positional encodings has evolved significantly, becoming a key factor in extending context length for modern LLMs.

## Key Concepts

| Method | Type | Key Property |
|--------|------|-------------|
| **Sinusoidal** | Fixed | Deterministic sin/cos functions; no learned parameters |
| **Learned Absolute** | Learned | Trainable embedding per position; used in BERT, GPT-2 |
| **RoPE** | Relative | Rotates Q/K vectors; encodes relative distance naturally |
| **ALiBi** | Relative | Adds linear bias to attention scores based on distance |
| **Relative Position Bias** | Relative | Learned bias table indexed by relative distance (T5) |

## How It Works

**Sinusoidal Encoding** (original Transformer):
\`\`\`
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

Each dimension uses a different frequency, creating a unique pattern for each position that the model can learn to decode.

**RoPE (Rotary Position Embedding):**
Applies a rotation matrix to Q and K vectors based on position, so the dot product Q*K naturally depends on relative position:

\`\`\`python
import torch

def apply_rope(x, pos, dim):
    """Apply Rotary Position Embedding to input tensor."""
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (seq_len, dim/2)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals
    ], dim=-1).flatten(-2)
    return rotated

# ALiBi adds a linear penalty to attention scores
# score_ij = q_i * k_j - m * |i - j|
# where m is a head-specific slope (no learned parameters)
\`\`\`

**RoPE** is dominant in modern LLMs because it naturally handles relative positions and can be extended to longer sequences via frequency scaling (NTK-aware interpolation, YaRN).

## Applications

- **Sinusoidal:** Original Transformer, educational implementations
- **Learned Absolute:** BERT, GPT-2 (limited to fixed max sequence length)
- **RoPE:** LLaMA, Mistral, Qwen, DeepSeek, most modern open-source LLMs
- **ALiBi:** BLOOM, MPT (no positional parameters, zero-shot length extrapolation)

## Evolution

| Year | Milestone |
|------|-----------|
| 2017 | Sinusoidal positional encoding in original Transformer |
| 2018 | Learned absolute positions in BERT and GPT-2 |
| 2020 | Relative position bias in T5 |
| 2021 | RoPE proposed by Su et al.; ALiBi proposed by Press et al. |
| 2023 | RoPE scaling methods (NTK, YaRN) enable context extension to 100K+ tokens |`,

    enc_dec: `# Encoder-Decoder Architecture

The encoder-decoder architecture is the original Transformer design from "Attention Is All You Need." It consists of two stacks: an encoder that processes the full input sequence bidirectionally, and a decoder that generates the output sequence autoregressively. Cross-attention connects the two, allowing the decoder to focus on relevant parts of the encoded input at each generation step. While modern LLMs often use decoder-only designs, encoder-decoder models remain powerful for tasks that require understanding a complete input before producing structured output.

## Key Concepts

\`\`\`
Input Tokens --> [Encoder Stack] --> Encoded Representations
                                           |
                                     Cross-Attention
                                           |
Output Tokens --> [Decoder Stack] --> Next Token Prediction
\`\`\`

| Component | Role | Attention Type |
|-----------|------|---------------|
| **Encoder** | Processes full input bidirectionally | Self-attention (no causal mask) |
| **Decoder** | Generates output autoregressively | Causal self-attention + cross-attention |
| **Cross-attention** | Decoder attends to encoder output | Q from decoder, K/V from encoder |

## How It Works

Each encoder layer applies:
1. Multi-head self-attention over the input
2. Feed-forward network (FFN)
3. LayerNorm and residual connections

Each decoder layer applies:
1. Masked (causal) multi-head self-attention
2. Multi-head cross-attention (Q from decoder, K/V from encoder)
3. Feed-forward network
4. LayerNorm and residual connections

\`\`\`python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, is_decoder=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.is_decoder = is_decoder
        if is_decoder:
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out=None, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, attn_mask=mask)[0])
        if self.is_decoder and enc_out is not None:
            x = self.norm2(x + self.cross_attn(x, enc_out, enc_out)[0])
        x = self.norm3(x + self.ffn(x))
        return x
\`\`\`

**Architecture comparison:**

| Architecture | Examples | Strengths |
|-------------|----------|-----------|
| Encoder-Decoder | T5, BART, mBART, Flan-T5 | Translation, summarization, structured output |
| Encoder-only | BERT, RoBERTa | Classification, NER, extractive tasks |
| Decoder-only | GPT, LLaMA, Mistral | Text generation, general-purpose LLMs |

## Applications

- Machine translation (English to French, multilingual)
- Text summarization (abstractive)
- Question answering with structured output
- Speech recognition (Whisper uses encoder-decoder)

## Evolution

| Year | Milestone |
|------|-----------|
| 2014 | Sutskever et al. introduce seq-to-seq with RNN encoder-decoder |
| 2017 | Vaswani et al. replace RNNs with Transformer encoder-decoder |
| 2019 | T5 frames all NLP as text-to-text with encoder-decoder |
| 2020 | BART, mBART extend encoder-decoder for denoising and multilingual tasks |
| 2022 | Whisper uses encoder-decoder for robust speech recognition |`,

    // ============================================================
    // EVOLUTION OF LLMs
    // ============================================================

    bert: `# BERT (Bidirectional Encoder Representations from Transformers)

BERT, introduced by Google in 2018, revolutionized NLP by demonstrating that pretraining a deep bidirectional Transformer on unlabeled text, then fine-tuning on downstream tasks, could achieve state-of-the-art results across a wide range of benchmarks. Unlike previous models that read text left-to-right or right-to-left, BERT processes the entire sequence simultaneously, allowing each token to attend to both its left and right context. This bidirectional approach fundamentally changed how machines understand language.

## Key Concepts

| Feature | Detail |
|---------|--------|
| **Architecture** | Encoder-only Transformer |
| **Pretraining Task 1** | Masked Language Modeling (MLM): predict 15% randomly masked tokens |
| **Pretraining Task 2** | Next Sentence Prediction (NSP): is sentence B the actual next sentence? |
| **Input Format** | [CLS] Sentence A [SEP] Sentence B [SEP] |
| **BERT-Base** | 12 layers, 768 hidden, 12 heads, 110M parameters |
| **BERT-Large** | 24 layers, 1024 hidden, 16 heads, 340M parameters |

## How It Works

**Masked Language Modeling (MLM):**
- Randomly mask 15% of input tokens
- Of those: 80% replaced with [MASK], 10% replaced with random token, 10% unchanged
- Model predicts original tokens at masked positions
- Forces bidirectional context understanding

\`\`\`python
from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
predicted = tokenizer.decode(outputs.logits[0, mask_idx].argmax())
# predicted: "paris"
\`\`\`

## BERT Variant Comparison

| Model | Key Change | Parameters | Performance |
|-------|-----------|------------|-------------|
| **BERT** | Original MLM + NSP | 110M / 340M | Baseline |
| **RoBERTa** | Removes NSP, more data, longer training | 125M / 355M | +2-3% over BERT |
| **ALBERT** | Factorized embeddings, shared layers | 12M / 235M | Similar quality, much smaller |
| **DistilBERT** | Knowledge distillation from BERT | 66M | 97% of BERT, 60% faster |
| **DeBERTa** | Disentangled attention + enhanced mask decoder | 134M / 390M | Surpasses human on SuperGLUE |
| **ELECTRA** | Replaced token detection instead of MLM | 14M / 335M | More sample-efficient |

## Applications

- Text classification and sentiment analysis
- Named Entity Recognition (NER)
- Question answering (SQuAD)
- Semantic similarity and sentence embeddings
- Information retrieval and search ranking

## Evolution

| Year | Milestone |
|------|-----------|
| 2018 | BERT released; dominates 11 NLP benchmarks |
| 2019 | RoBERTa, ALBERT, DistilBERT improve on BERT's recipe |
| 2020 | DeBERTa surpasses human baseline on SuperGLUE |
| 2020 | ELECTRA offers more efficient pretraining alternative |
| 2021+ | BERT-style encoders remain standard for classification, retrieval, and embeddings |`,

    gpt_series: `# GPT Series (GPT-1 through GPT-4)

The GPT (Generative Pre-trained Transformer) series from OpenAI represents the most influential line of large language models, demonstrating that scaling decoder-only Transformers yields increasingly powerful and general-purpose AI systems. From GPT-1's modest 117M parameters to GPT-4's rumored trillion-plus parameters, this series has driven the field's understanding of scaling laws, emergent abilities, and the surprising generality of next-token prediction as a training objective.

## Key Concepts

| Model | Year | Parameters | Context | Key Innovation |
|-------|------|-----------|---------|---------------|
| **GPT-1** | 2018 | 117M | 512 | Unsupervised pretraining + supervised fine-tuning |
| **GPT-2** | 2019 | 1.5B | 1024 | Zero-shot task performance; "too dangerous to release" |
| **GPT-3** | 2020 | 175B | 2048 | In-context learning; few-shot prompting |
| **GPT-3.5** | 2022 | ~175B | 4096 | RLHF alignment; powers ChatGPT |
| **GPT-4** | 2023 | Undisclosed | 8K/32K/128K | Multimodal (text + vision); massive quality leap |

## How It Works

All GPT models are **decoder-only Transformers** trained on **causal language modeling** (predict the next token):

\`\`\`
P(x_1, x_2, ..., x_n) = PRODUCT P(x_i | x_1, ..., x_{i-1})
\`\`\`

Each layer applies:
1. Causal (masked) multi-head self-attention
2. Feed-forward network
3. LayerNorm and residual connections

\`\`\`python
# GPT-style generation using Hugging Face
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(output[0]))
\`\`\`

**Scaling Laws** (Kaplan et al., 2020): Loss decreases as a smooth power law with model size, dataset size, and compute. This motivated the massive scale-up from GPT-2 to GPT-3.

**Emergent Abilities:** Capabilities that appear suddenly at certain scales, including chain-of-thought reasoning, code generation, arithmetic, and multilingual transfer. These were largely unexpected and emerged without explicit training.

## Applications

- Conversational AI (ChatGPT, powered by GPT-3.5/4)
- Code generation and understanding (GitHub Copilot)
- Content creation, summarization, and translation
- Complex reasoning and multi-step problem solving
- Multimodal understanding (GPT-4 with vision)

## Evolution

| Year | Milestone |
|------|-----------|
| 2018 | GPT-1 shows unsupervised pretraining transfers to downstream tasks |
| 2019 | GPT-2 demonstrates strong zero-shot performance across tasks |
| 2020 | GPT-3 enables in-context learning via few-shot prompting |
| 2022 | InstructGPT/ChatGPT add RLHF for human-aligned responses |
| 2023 | GPT-4 introduces multimodal capabilities and further reasoning gains |`,

    t5: `# T5 (Text-to-Text Transfer Transformer)

T5, introduced by Google Research in 2019, reframes every NLP task as a text-to-text problem: both the input and output are always text strings. This elegant unification means translation, summarization, classification, question answering, and even regression all use the same model, loss function, and decoding procedure. By prepending a task-specific prefix to the input (e.g., "translate English to German:"), T5 demonstrated that a single encoder-decoder architecture could match or exceed specialized models across dozens of benchmarks.

## Key Concepts

| Feature | Detail |
|---------|--------|
| **Architecture** | Encoder-decoder Transformer |
| **Pretraining** | Span corruption: mask consecutive spans, predict them |
| **Task format** | "task prefix: input text" -> "output text" |
| **Dataset** | C4 (Colossal Clean Crawled Corpus), ~750GB of filtered web text |

**T5 sizes:**

| Variant | Parameters | Encoder/Decoder Layers |
|---------|-----------|----------------------|
| T5-Small | 60M | 6/6 |
| T5-Base | 220M | 12/12 |
| T5-Large | 770M | 24/24 |
| T5-3B | 3B | 24/24 |
| T5-11B | 11B | 24/24 |

## How It Works

**Text-to-text framework examples:**

\`\`\`
# Classification
Input:  "sst2 sentence: This movie is fantastic!"
Output: "positive"

# Translation
Input:  "translate English to French: How are you?"
Output: "Comment allez-vous?"

# Summarization
Input:  "summarize: [long article text]"
Output: "[concise summary]"

# Question Answering
Input:  "question: What is gravity? context: Gravity is a force..."
Output: "a force that attracts objects"
\`\`\`

\`\`\`python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

input_text = "summarize: Machine learning is a subset of AI that enables systems to learn from data..."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\`\`\`

## T5 Family Comparison

| Model | Key Improvement | Parameters |
|-------|----------------|-----------|
| **T5** | Original text-to-text framework | 60M - 11B |
| **mT5** | Multilingual T5 trained on mC4 (101 languages) | 300M - 13B |
| **FLAN-T5** | Instruction-tuned on 1800+ tasks | 80M - 11B |
| **UL2** | Unified pretraining with multiple denoising objectives | 20B |

## Applications

- Multi-task NLP (single model for translation, QA, summarization)
- Instruction following (FLAN-T5)
- Multilingual processing (mT5)
- Structured output generation (tables, code, SQL)

## Evolution

| Year | Milestone |
|------|-----------|
| 2019 | T5 paper explores transfer learning at scale |
| 2020 | mT5 extends to 101 languages |
| 2022 | FLAN-T5 shows instruction tuning dramatically improves zero-shot performance |
| 2022 | UL2 unifies different pretraining objectives |
| 2023 | FLAN-T5 remains a strong baseline for instruction-following research |`,

    llama: `# LLaMA (Large Language Model Meta AI)

LLaMA is Meta's family of open-weight large language models that democratized access to high-quality LLMs. LLaMA 1 proved that smaller, well-trained models could match much larger ones, while LLaMA 2 introduced open commercial licensing and RLHF alignment. LLaMA 3 pushed the frontier further with larger scales, longer context, and multilingual capabilities. The LLaMA family spawned an enormous open-source ecosystem of fine-tuned derivatives, making it the most influential open model family in AI.

## Key Concepts

| Model | Year | Sizes | Context | Key Features |
|-------|------|-------|---------|-------------|
| **LLaMA 1** | Feb 2023 | 7B, 13B, 33B, 65B | 2048 | Research-only; trained on 1.4T tokens |
| **LLaMA 2** | Jul 2023 | 7B, 13B, 70B | 4096 | Commercial license; RLHF chat models |
| **LLaMA 3** | Apr 2024 | 8B, 70B | 8192 | GQA, 15T tokens, expanded vocabulary |
| **LLaMA 3.1** | Jul 2024 | 8B, 70B, 405B | 128K | Longest context; largest open model |

## How It Works

LLaMA models are **decoder-only Transformers** with key architectural choices:

| Feature | LLaMA 1 | LLaMA 2 | LLaMA 3 |
|---------|---------|---------|---------|
| Normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) | RMSNorm (pre-norm) |
| Activation | SiLU (Swish) | SiLU | SiLU |
| Position Encoding | RoPE | RoPE | RoPE |
| Attention | MHA | GQA (70B) | GQA (all sizes) |
| Vocabulary Size | 32K (SentencePiece) | 32K | 128K (tiktoken) |
| Training Data | 1.4T tokens | 2T tokens | 15T+ tokens |

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

messages = [{"role": "user", "content": "Explain transformers in one paragraph."}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
\`\`\`

## Notable Derivatives

| Model | Base | Innovation |
|-------|------|-----------|
| **Alpaca** | LLaMA 7B | Stanford instruction-tuning with 52K examples |
| **Vicuna** | LLaMA 13B | Fine-tuned on ShareGPT conversations |
| **CodeLlama** | LLaMA 2 | Specialized for code generation |
| **Llama Guard** | LLaMA | Safety classification model |

## Applications

- Open-source chatbots and assistants
- Code generation (CodeLlama)
- Research on alignment, fine-tuning, and quantization
- Enterprise deployment with commercial licensing
- Foundation for thousands of community fine-tuned models

## Evolution

| Year | Milestone |
|------|-----------|
| Feb 2023 | LLaMA 1 released for research; leaked and widely adopted |
| Mar 2023 | Alpaca, Vicuna, and other derivatives emerge within weeks |
| Jul 2023 | LLaMA 2 released with commercial license and chat models |
| Apr 2024 | LLaMA 3 raises the bar with 15T training tokens |
| Jul 2024 | LLaMA 3.1 405B becomes the largest open-weight model |`,

    claude: `# Claude Models

Claude is Anthropic's family of large language models distinguished by their focus on safety, helpfulness, and honesty. Developed using Constitutional AI (CAI) and RLHF, Claude models are designed to be helpful assistants that avoid harmful outputs through principled alignment techniques. The Claude series has evolved from Claude 1 to Claude 4, with each generation improving capabilities while maintaining strong safety properties.

## Key Concepts

| Feature | Description |
|---------|-------------|
| **Constitutional AI (CAI)** | Self-critique and revision guided by a set of principles (a "constitution") |
| **RLHF** | Reinforcement Learning from Human Feedback for alignment |
| **Harmlessness Training** | Explicit optimization to refuse harmful requests while remaining helpful |
| **Long Context** | Claude supports up to 200K token context windows |

**Model Generations:**

| Model | Year | Key Capabilities |
|-------|------|-----------------|
| **Claude 1** | Mar 2023 | Initial release; 100K context window |
| **Claude 2** | Jul 2023 | Improved reasoning, coding, and math |
| **Claude 3 (Haiku/Sonnet/Opus)** | Mar 2024 | Tiered model family; multimodal (vision) |
| **Claude 3.5 Sonnet** | Jun 2024 | Major quality leap; near-Opus quality at Sonnet speed |
| **Claude 4 / Opus 4** | 2025 | Frontier capabilities; advanced reasoning |

## How It Works

**Constitutional AI Pipeline:**

\`\`\`
Step 1: Supervised Learning
  Train initial helpful model on human demonstrations

Step 2: Red-Teaming & Self-Critique
  Generate responses to harmful prompts
  Model critiques its own responses using constitutional principles
  Model revises responses to be harmless

Step 3: RLHF
  Train reward model on preference data (revised vs. original)
  Optimize policy using PPO against the reward model
\`\`\`

**Key principles in the constitution include:**
- Choose the response that is most helpful and least harmful
- Avoid assisting with illegal or dangerous activities
- Be honest about uncertainty and limitations
- Respect user autonomy while preventing misuse

**Model tiers (Claude 3 family):**

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **Haiku** | Fastest | Good | Quick tasks, classification, extraction |
| **Sonnet** | Balanced | Very Good | General use, coding, analysis |
| **Opus** | Slower | Best | Complex reasoning, research, nuanced tasks |

## Applications

- Conversational AI assistants
- Code generation, review, and debugging
- Long-document analysis (up to 200K tokens)
- Complex reasoning and multi-step problem solving
- Content creation with safety guardrails

## Evolution

| Year | Milestone |
|------|-----------|
| 2021 | Anthropic founded; Constitutional AI research begins |
| 2023 | Claude 1 and Claude 2 released |
| 2024 | Claude 3 introduces tiered model family with vision capabilities |
| 2024 | Claude 3.5 Sonnet achieves strong benchmark results at efficient cost |
| 2025 | Claude 4 / Opus 4 pushes frontier reasoning and agentic capabilities |`,

    palm_gemini: `# PaLM & Gemini (Google's LLM Family)

PaLM (Pathways Language Model) and its successor Gemini represent Google's flagship large language model efforts. PaLM demonstrated breakthrough performance through massive scale and the Pathways system, while Gemini was designed from the ground up as a natively multimodal model capable of understanding text, images, video, audio, and code. Together they represent Google's strategy of building foundational models that power products from Search to Workspace.

## Key Concepts

| Model | Year | Parameters | Key Innovation |
|-------|------|-----------|---------------|
| **PaLM** | Apr 2022 | 540B | Pathways training across 6144 TPUs; breakthrough reasoning |
| **PaLM 2** | May 2023 | Undisclosed | Improved multilingual, reasoning, and coding; powers Bard |
| **Gemini 1.0** | Dec 2023 | Undisclosed | Natively multimodal (text, image, video, audio, code) |
| **Gemini 1.5** | Feb 2024 | Undisclosed | Mixture-of-Experts; 1M token context window |
| **Gemini 2.0** | Dec 2024 | Undisclosed | Enhanced agentic capabilities; native tool use |

## How It Works

**PaLM Architecture:**
- Standard decoder-only Transformer at massive scale
- Trained using Google's Pathways system for efficient multi-TPU training
- Demonstrated discontinuous improvements ("emergent abilities") at 540B scale
- Showed strong performance on reasoning tasks (BIG-Bench Hard) without chain-of-thought

**Gemini Architecture:**
- Natively multimodal: processes interleaved text, images, audio, and video
- Built on a Mixture-of-Experts (MoE) architecture for Gemini 1.5
- Supports extremely long context (up to 1M tokens in Gemini 1.5 Pro)

**Model tiers (Gemini):**

| Model | Target | Best For |
|-------|--------|----------|
| **Ultra** | Highest capability | Complex reasoning, research |
| **Pro** | Balanced | General use, coding, multimodal tasks |
| **Flash** | Fast and efficient | High-throughput, cost-sensitive apps |
| **Nano** | On-device | Mobile, edge deployment |

\`\`\`python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-1.5-pro")

# Text generation
response = model.generate_content("Explain quantum computing simply")

# Multimodal: image understanding
import PIL.Image
img = PIL.Image.open("diagram.png")
response = model.generate_content(["Describe this diagram:", img])
\`\`\`

## Applications

- Google Search (AI Overviews powered by Gemini)
- Google Workspace (Gemini in Gmail, Docs, Sheets)
- Multimodal understanding (images, video, audio analysis)
- Code generation (AlphaCode 2 builds on Gemini)
- On-device AI (Gemini Nano on Pixel phones)

## Evolution

| Year | Milestone |
|------|-----------|
| 2022 | PaLM 540B demonstrates emergent reasoning abilities |
| 2023 | PaLM 2 powers Bard chatbot and Google products |
| 2023 | Gemini 1.0 launches as natively multimodal model |
| 2024 | Gemini 1.5 achieves 1M token context with MoE architecture |
| 2024 | Gemini 2.0 introduces agentic capabilities and native tool use |`,

    mistral: `# Mistral (Mistral AI)

Mistral AI, a French startup founded by former Google and Meta researchers, has rapidly established itself as a leader in efficient open-weight language models. Their flagship Mistral 7B outperformed LLaMA 2 13B on most benchmarks despite being half the size, demonstrating that architectural innovation and data quality can compensate for raw scale. Mixtral introduced an efficient Mixture-of-Experts approach, and subsequent models have pushed both open and commercial frontiers.

## Key Concepts

| Model | Year | Parameters | Context | Key Innovation |
|-------|------|-----------|---------|---------------|
| **Mistral 7B** | Sep 2023 | 7.3B | 8K (32K with sliding window) | GQA + sliding window attention |
| **Mixtral 8x7B** | Dec 2023 | 46.7B (12.9B active) | 32K | Sparse MoE: 8 experts, 2 active per token |
| **Mixtral 8x22B** | Apr 2024 | 176B (39B active) | 64K | Larger MoE with stronger performance |
| **Mistral Large** | Feb 2024 | Undisclosed | 32K | Commercial flagship; rivals GPT-4 class |
| **Mistral Small / Nemo** | 2024 | 7B - 22B | 128K | Efficient models for diverse deployment |

## How It Works

**Sliding Window Attention (SWA):**
Instead of attending to all previous tokens, each layer attends only to the last W tokens (e.g., W = 4096). Through stacked layers, information propagates across the full sequence:

\`\`\`
Layer 3:  [----window----]                    Effective range: 4 * W
Layer 2:     [----window----]                 Effective range: 3 * W
Layer 1:        [----window----]              Effective range: 2 * W
Layer 0:           [----window----]           Effective range: W
         Token 1  Token 2  ...  Token N
\`\`\`

After L layers, the effective receptive field is L * W tokens, achieving long-range dependency without the quadratic cost of full attention.

**Mixture of Experts (MoE) in Mixtral:**

\`\`\`
Input Token --> Router Network --> selects top-2 experts
                                      |
              Expert 1  Expert 2 ... Expert 8  (each is a FFN)
                                      |
              Weighted sum of top-2 expert outputs
\`\`\`

\`\`\`python
# Conceptual MoE forward pass
def moe_forward(x, experts, router):
    gate_logits = router(x)                          # (batch, n_experts)
    top2_weights, top2_indices = gate_logits.topk(2)  # select 2 experts
    top2_weights = F.softmax(top2_weights, dim=-1)
    output = sum(w * experts[i](x) for w, i in zip(top2_weights.T, top2_indices.T))
    return output
\`\`\`

**Key advantage:** Mixtral 8x7B has 46.7B total parameters but only activates 12.9B per token, providing near-70B quality at 13B inference cost.

## Architecture Comparison

| Feature | Mistral 7B | Mixtral 8x7B | LLaMA 2 13B |
|---------|-----------|-------------|-------------|
| Total Params | 7.3B | 46.7B | 13B |
| Active Params | 7.3B | 12.9B | 13B |
| Attention | GQA | GQA | GQA |
| Position | RoPE | RoPE | RoPE |
| Special | Sliding Window | MoE (8 experts) | Standard |

## Applications

- Cost-effective deployment (high quality at smaller active compute)
- Multilingual applications (strong European language support)
- Enterprise and on-premise deployment (open weights)
- Research into MoE architectures and efficiency

## Evolution

| Year | Milestone |
|------|-----------|
| May 2023 | Mistral AI founded by ex-Google/Meta researchers |
| Sep 2023 | Mistral 7B released; outperforms LLaMA 2 13B |
| Dec 2023 | Mixtral 8x7B demonstrates efficient MoE for open models |
| Apr 2024 | Mixtral 8x22B scales up the MoE approach |
| 2024 | Mistral Large competes with top commercial models |`,

    // ============================================================
    // LLM TECHNIQUES
    // ============================================================

    finetuning: `# Fine-Tuning (Full, LoRA, QLoRA, Adapters, PEFT)

Fine-tuning adapts a pretrained language model to a specific task or domain by continuing training on a targeted dataset. While full fine-tuning updates all model parameters, parameter-efficient methods (PEFT) modify only a tiny fraction of weights, making it feasible to customize billion-parameter models on a single GPU. These techniques have democratized LLM customization, enabling practitioners to create specialized models without massive compute budgets.

## Key Concepts

| Method | Parameters Updated | Memory | Quality |
|--------|--------------------|--------|---------|
| **Full Fine-Tuning** | All (100%) | Very High | Best (but risk of forgetting) |
| **LoRA** | ~0.1-1% via low-rank matrices | Low | Near full fine-tuning |
| **QLoRA** | LoRA + 4-bit quantized base | Very Low | Surprisingly close to full |
| **Adapters** | Small bottleneck layers inserted between layers | Low | Good |
| **Prefix Tuning** | Learnable prefix tokens prepended to each layer | Very Low | Moderate |
| **Prompt Tuning** | Learnable soft prompt tokens at the input | Minimal | Good for specific tasks |

## How It Works

**LoRA (Low-Rank Adaptation):**
Instead of updating a full weight matrix W (d x d), LoRA freezes W and adds a low-rank decomposition:

\`\`\`
W_new = W_frozen + (alpha / r) * B * A
Where: A is (d x r), B is (r x d), and r << d (e.g., r = 8 or 16)
\`\`\`

Only A and B are trained, reducing trainable parameters by 100-1000x.

\`\`\`python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-8B")

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                       # rank of decomposition
    lora_alpha=32,              # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 13.6M || all params: 8B || trainable%: 0.17%
\`\`\`

**QLoRA** combines LoRA with 4-bit NormalFloat quantization of the base model, enabling fine-tuning of a 65B model on a single 48GB GPU:

\`\`\`python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-8B",
    quantization_config=bnb_config,
)
# Then apply LoRA on top of the quantized model
\`\`\`

## Method Comparison

| Method | GPU Needed (7B model) | Training Time | Use Case |
|--------|----------------------|---------------|----------|
| Full Fine-Tuning | 4x A100 80GB | Hours | Maximum quality, ample compute |
| LoRA | 1x A100 40GB | Hours | Standard PEFT, great balance |
| QLoRA | 1x RTX 3090 24GB | Hours | Consumer GPU fine-tuning |
| Prompt Tuning | 1x T4 16GB | Minutes | Quick task adaptation |

## Applications

- Domain adaptation (legal, medical, financial language)
- Instruction following (chat model creation)
- Task specialization (code, math, translation)
- Style and personality customization

## Evolution

| Year | Milestone |
|------|-----------|
| 2019 | Adapter layers introduced by Houlsby et al. |
| 2021 | Prefix Tuning and Prompt Tuning proposed |
| 2021 | LoRA by Hu et al. becomes the dominant PEFT method |
| 2023 | QLoRA enables fine-tuning 65B models on consumer GPUs |
| 2024 | LoRA+ and DoRA further improve parameter-efficient adaptation |`,

    rlhf_technique: `# RLHF (Reinforcement Learning from Human Feedback)

RLHF is the technique that transformed raw language models into helpful, harmless, and honest AI assistants. By training a reward model on human preference data and then optimizing the language model against that reward using reinforcement learning, RLHF aligns model outputs with human values and expectations. This approach powers ChatGPT, Claude, and essentially all modern conversational AI systems. DPO (Direct Preference Optimization) has emerged as a simpler alternative that achieves similar results without explicit RL.

## Key Concepts

| Component | Role |
|-----------|------|
| **SFT Model** | Initial supervised fine-tuning on high-quality demonstrations |
| **Reward Model** | Trained on human preference comparisons to score outputs |
| **Policy Optimization** | RL algorithm (PPO) maximizes reward while staying close to SFT model |
| **KL Penalty** | Prevents the policy from drifting too far from the reference model |

## How It Works

**The RLHF Pipeline:**

\`\`\`
Step 1: Supervised Fine-Tuning (SFT)
  Pretrained LLM --> Fine-tune on human demonstrations --> SFT Model

Step 2: Reward Model Training
  For each prompt, generate multiple responses
  Humans rank: Response A > Response B > Response C
  Train reward model: R(prompt, response) -> scalar score
  Loss: -log(sigmoid(R(chosen) - R(rejected)))   [Bradley-Terry model]

Step 3: RL Optimization (PPO)
  Objective: maximize E[R(prompt, response)] - beta * KL(policy || SFT_model)
  The KL term prevents reward hacking and mode collapse
\`\`\`

\`\`\`python
# Simplified RLHF training loop (conceptual)
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    kl_penalty="kl",          # KL divergence penalty
    init_kl_coef=0.2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=sft_model,       # reference for KL penalty
    tokenizer=tokenizer,
)

for batch in dataloader:
    query_tensors = batch["input_ids"]
    response_tensors = ppo_trainer.generate(query_tensors)
    rewards = reward_model(query_tensors, response_tensors)
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
\`\`\`

**DPO (Direct Preference Optimization):**
Eliminates the reward model entirely by deriving a closed-form loss directly from preference data:

\`\`\`
L_DPO = -log sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))))
\`\`\`

Where y_w is the preferred response and y_l is the dispreferred response.

| Method | Reward Model | RL Algorithm | Stability | Simplicity |
|--------|-------------|-------------|-----------|------------|
| **RLHF (PPO)** | Required | PPO | Can be unstable | Complex pipeline |
| **DPO** | Not needed | None | More stable | Much simpler |
| **KTO** | Not needed | None | Works with binary signal | Simplest |
| **ORPO** | Not needed | None | Combined SFT + preference | Single stage |

## Applications

- Aligning LLMs to be helpful, harmless, and honest
- Creating chat assistants (ChatGPT, Claude, Gemini)
- Reducing toxic, biased, or harmful outputs
- Improving instruction following quality

## Evolution

| Year | Milestone |
|------|-----------|
| 2017 | Christiano et al. introduce learning from human preferences |
| 2020 | Stiennon et al. apply RLHF to summarization |
| 2022 | InstructGPT and ChatGPT demonstrate RLHF at scale |
| 2023 | DPO proposed as a simpler alternative to PPO-based RLHF |
| 2024 | KTO, ORPO, and other preference optimization variants emerge |`,

    prompt_engineering: `# Prompt Engineering

Prompt engineering is the practice of designing input text to elicit desired behaviors from large language models without modifying their weights. It ranges from simple zero-shot instructions to sophisticated multi-step reasoning frameworks. As LLMs have grown more capable, prompt engineering has evolved from a curiosity into a critical skill, enabling users to unlock capabilities that models possess but only exhibit when properly instructed.

## Key Concepts

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Zero-shot** | Direct instruction with no examples | Simple, well-defined tasks |
| **Few-shot** | Include 2-5 examples in the prompt | Tasks requiring specific format or style |
| **Chain-of-Thought (CoT)** | "Think step by step" | Math, logic, multi-step reasoning |
| **ReAct** | Interleave reasoning and action/tool use | Tasks requiring external information |
| **Tree-of-Thought (ToT)** | Explore multiple reasoning paths, backtrack | Complex planning and search problems |
| **Self-Consistency** | Sample multiple CoT paths, vote on final answer | Improving reliability of reasoning |
| **System Prompts** | Set role, constraints, and behavior | Defining assistant personality and rules |

## How It Works

**Zero-Shot vs Few-Shot:**

\`\`\`
# Zero-shot
Classify the sentiment: "This product is amazing!" ->

# Few-shot
Classify the sentiment:
"I love it!" -> Positive
"Terrible experience." -> Negative
"This product is amazing!" ->
\`\`\`

**Chain-of-Thought Prompting:**

\`\`\`
Q: A store sells apples for $2 each. If you buy 3 apples and pay with $10,
   how much change do you get?

A: Let me think step by step.
   1. Cost of 3 apples: 3 * $2 = $6
   2. Change from $10: $10 - $6 = $4
   The answer is $4.
\`\`\`

**ReAct (Reasoning + Action):**

\`\`\`
Question: What is the population of the capital of France?

Thought: I need to find the capital of France first.
Action: search("capital of France")
Observation: Paris is the capital of France.
Thought: Now I need to find the population of Paris.
Action: search("population of Paris")
Observation: The population of Paris is about 2.1 million.
Answer: The population of the capital of France (Paris) is approximately 2.1 million.
\`\`\`

**Tree-of-Thought:**

\`\`\`
Problem: [Complex task]
          |
    +-----+-----+
    |     |     |
  Path A Path B Path C    (explore multiple approaches)
    |     |     |
  Eval   Eval  Eval       (evaluate each path)
    |           |
  Prune    Continue        (backtrack from dead ends)
    |
  Final Answer
\`\`\`

## Comparison of Techniques

| Technique | Reasoning Quality | Cost (tokens) | Latency | Best For |
|-----------|------------------|---------------|---------|----------|
| Zero-shot | Baseline | Low | Fast | Simple tasks |
| Few-shot | Better | Medium | Medium | Format-sensitive tasks |
| CoT | Much better | High | Slower | Math, logic, reasoning |
| Self-Consistency | Highest | Very High | Slowest | High-stakes reasoning |
| ReAct | Grounded | High | Variable | Tasks needing real-time info |
| ToT | Exploratory | Very High | Slowest | Planning, creative problems |

## Applications

- Complex reasoning tasks (math, logic, science)
- Chatbot behavior design via system prompts
- Data extraction and formatting
- Code generation with specific requirements
- Agentic workflows combining reasoning and tool use

## Evolution

| Year | Milestone |
|------|-----------|
| 2020 | GPT-3 demonstrates few-shot in-context learning |
| 2022 | Wei et al. introduce Chain-of-Thought prompting |
| 2022 | ReAct combines reasoning with tool use |
| 2023 | Tree-of-Thought and Graph-of-Thought proposed |
| 2024 | Prompt engineering becomes integral to agentic AI frameworks |`,

    context_scaling: `# Context Window Scaling

Context window scaling addresses one of the most important limitations of Transformer-based LLMs: the finite number of tokens they can process at once. The original Transformer supported only 512 tokens; modern models handle up to 1 million or more. This evolution has been driven by algorithmic innovations in positional encoding, attention computation, and memory management. Longer context enables processing entire books, large codebases, and extended conversations without information loss.

## Key Concepts

| Model | Year | Context Length | Method |
|-------|------|---------------|--------|
| Original Transformer | 2017 | 512 | Sinusoidal positions |
| GPT-2 | 2019 | 1,024 | Learned absolute positions |
| GPT-3 | 2020 | 2,048 | Learned absolute positions |
| Claude 1 | 2023 | 100K | Architectural innovations |
| GPT-4 Turbo | 2023 | 128K | Undisclosed |
| Gemini 1.5 Pro | 2024 | 1M | MoE + long-context training |
| Claude 3 | 2024 | 200K | Extended training and architecture |

**The core challenge:** Standard self-attention has O(n^2) time and memory complexity in sequence length n, making long contexts prohibitively expensive without optimizations.

## How It Works

**RoPE Scaling Methods:**
RoPE (Rotary Position Embedding) encodes position via rotation frequencies. To extend context beyond training length, the frequency base can be modified:

\`\`\`
# Position Interpolation (PI)
# Scale positions to fit within original training range
adjusted_position = position * (L_train / L_target)

# NTK-Aware Interpolation
# Scale the frequency base instead of positions
base_new = base * (L_target / L_train) ^ (dim / (dim - 2))

# YaRN (Yet another RoPE extensioN)
# Combines NTK scaling with attention temperature adjustment
# and targeted frequency band scaling
\`\`\`

**Efficient Attention Mechanisms:**

| Method | Complexity | Key Idea |
|--------|-----------|----------|
| **Standard Attention** | O(n^2) | Full pairwise attention |
| **Flash Attention** | O(n^2) time, O(n) memory | Tiled computation avoids materializing attention matrix |
| **Sliding Window** | O(n * w) | Each token attends only to w nearest tokens |
| **Ring Attention** | O(n^2 / devices) | Distributes sequence across devices in a ring |
| **Sparse Attention** | O(n * sqrt(n)) | Attend to local + strided global tokens |
| **Linear Attention** | O(n) | Kernel approximation of softmax attention |

**Flash Attention** is particularly important. It does not change the computation but restructures it for GPU memory hierarchy:

\`\`\`python
# Flash Attention is integrated into PyTorch 2.0+
import torch.nn.functional as F

# Automatically uses Flash Attention when available
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    is_causal=True  # for decoder models
)
# Avoids materializing the full N x N attention matrix
# Reduces memory from O(N^2) to O(N)
\`\`\`

**Ring Attention** distributes the sequence across multiple devices, with each device computing attention for its local block while passing KV blocks around the ring:

\`\`\`
Device 1: [tokens 1-1024]  ---KV--->  Device 2: [tokens 1025-2048]
     ^                                      |
     |                                      v
Device 4: [tokens 3073-4096] <---KV---  Device 3: [tokens 2049-3072]
\`\`\`

## Applications

- Processing entire books, legal documents, or codebases
- Long-form conversation without losing context
- Multi-document analysis and comparison
- Repository-level code understanding
- Retrieval-augmented generation with large retrieved contexts

## Evolution

| Year | Milestone |
|------|-----------|
| 2022 | Flash Attention by Dao et al. makes long-context training practical |
| 2023 | Position interpolation enables extending RoPE to longer contexts |
| 2023 | Ring Attention distributes sequences across devices for near-infinite length |
| 2024 | YaRN combines multiple scaling techniques for robust extension |
| 2024 | Gemini 1.5 demonstrates 1M token context in a production model |`,

  };

  // Register all content on the global AI_DOCS object
  Object.assign(window.AI_DOCS, content);
})();
