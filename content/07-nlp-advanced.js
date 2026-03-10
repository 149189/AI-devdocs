// 07 - Advanced NLP
(function () {
  const content = {
    summarization: `# Text Summarization

Text Summarization automatically condenses long documents into shorter versions while preserving key information. It is one of the most practically useful NLP tasks.

## Types

| Type | Description | Example |
|------|-------------|---------|
| Extractive | Selects key sentences from original text | TextRank, BertSum |
| Abstractive | Generates new sentences capturing meaning | BART, T5, GPT |
| Hybrid | Combines extraction and generation | Extract-then-abstract pipelines |

## Key Concepts

- **ROUGE Score**: Standard metric (ROUGE-1, ROUGE-2, ROUGE-L) comparing n-gram overlap
- **Extractive methods**: Score sentences by importance, select top-k
- **Abstractive methods**: Encoder-decoder models generate summaries
- **Faithfulness**: Generated summary should not hallucinate facts
- **Compression ratio**: Length of summary vs original document

## How It Works

\`\`\`python
# Extractive: TextRank (graph-based)
# 1. Split text into sentences
# 2. Build similarity graph between sentences
# 3. Run PageRank to score sentences
# 4. Select top-k sentences

# Abstractive: Using HuggingFace
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
result = summarizer(article, max_length=130, min_length=30)
\`\`\`

## Applications

- News article summarization
- Legal document condensation
- Meeting notes generation
- Research paper abstracts
- Email thread summarization

## Evolution

- **2004**: TextRank applies PageRank to sentence extraction
- **2015**: Rush et al. introduce neural abstractive summarization
- **2019**: BART and PEGASUS achieve state-of-the-art results
- **2022**: LLMs enable zero-shot summarization with simple prompts
- **2024**: Long-context models summarize entire books`,

    machine_translation: `# Machine Translation

Machine Translation (MT) automatically translates text between languages. It has evolved from rule-based systems to neural approaches that rival human translation quality.

## Key Concepts

- **Source Language**: Language being translated from
- **Target Language**: Language being translated to
- **BLEU Score**: Standard evaluation metric comparing n-gram overlap with reference translations
- **Attention Mechanism**: Allows model to focus on relevant source words during translation
- **Beam Search**: Decoding strategy that maintains top-k candidate translations

## Approaches

| Era | Approach | Method |
|-----|----------|--------|
| 1950s-1990s | Rule-based (RBMT) | Linguistic rules and dictionaries |
| 1990s-2015 | Statistical (SMT) | Phrase tables, language models |
| 2014-2017 | Seq2Seq + Attention | RNN encoder-decoder with attention |
| 2017+ | Transformer-based (NMT) | Self-attention, parallel processing |

## How It Works

\`\`\`python
# Modern Neural Machine Translation
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = "Machine learning is transforming the world."
tokens = tokenizer(text, return_tensors="pt")
translated = model.generate(**tokens)
result = tokenizer.decode(translated[0], skip_special_tokens=True)
\`\`\`

## Applications

- Google Translate (serves 100+ languages)
- Real-time conversation translation
- Document localization
- Subtitle generation
- Cross-lingual information retrieval

## Evolution

- **1954**: Georgetown-IBM experiment (first MT demo)
- **2014**: Seq2Seq models introduced for MT (Sutskever et al.)
- **2015**: Attention mechanism dramatically improves quality (Bahdanau)
- **2017**: Transformer architecture ("Attention Is All You Need")
- **2020s**: Multilingual models (mBART, NLLB) cover 200+ languages`,

    question_answering: `# Question Answering (QA)

Question Answering systems automatically find answers to natural language questions from a given context, knowledge base, or open-domain sources.

## Types

| Type | Description | Example |
|------|-------------|---------|
| Extractive QA | Extracts answer span from context | SQuAD-style models |
| Abstractive QA | Generates answer in own words | GPT-based QA |
| Open-Domain QA | Searches large corpus first | RAG, DPR + Reader |
| Knowledge-Base QA | Queries structured knowledge | SPARQL over Wikidata |
| Conversational QA | Multi-turn dialogue | CoQA, QuAC |

## Key Concepts

- **Context/Passage**: The text containing the answer
- **Question Encoding**: Representing the question as vectors
- **Answer Extraction**: Predicting start and end positions of answer span
- **Retriever-Reader Pattern**: First retrieve relevant docs, then extract answer
- **F1 Score & Exact Match**: Standard evaluation metrics

## How It Works

\`\`\`python
# Extractive QA with HuggingFace
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
result = qa(
    question="What is machine learning?",
    context="Machine learning is a subset of AI that enables systems to learn from data."
)
# result: {"answer": "a subset of AI that enables systems to learn from data", ...}

# Open-Domain QA (RAG pattern)
# 1. Retrieve: Find relevant documents using dense retrieval
# 2. Read: Extract/generate answer from retrieved documents
\`\`\`

## Applications

- Search engines (featured snippets)
- Customer support chatbots
- Medical Q&A systems
- Educational tutoring
- Enterprise knowledge bases

## Evolution

- **1999**: TREC QA track begins evaluation of QA systems
- **2016**: SQuAD dataset released (100K+ questions)
- **2018**: BERT achieves human-level performance on SQuAD 2.0
- **2020**: RAG combines retrieval with generation
- **2023+**: LLMs provide zero-shot QA across virtually any domain`,

    sentiment_analysis: `# Sentiment Analysis

Sentiment Analysis determines the emotional tone or opinion expressed in text. It classifies text as positive, negative, or neutral, with more advanced systems detecting specific emotions.

## Key Concepts

- **Polarity**: Positive, negative, or neutral classification
- **Subjectivity**: Whether text expresses opinion vs fact
- **Aspect-Based**: Sentiment toward specific aspects (e.g., food vs service in restaurant reviews)
- **Fine-Grained**: 5-star scale rather than binary positive/negative
- **Emotion Detection**: Classifying specific emotions (joy, anger, sadness, fear)

## Approaches

| Method | Description | Example |
|--------|-------------|---------|
| Lexicon-Based | Dictionary of sentiment words | VADER, SentiWordNet |
| Traditional ML | Feature engineering + classifiers | TF-IDF + SVM |
| Deep Learning | Neural text classification | CNN, LSTM for text |
| Transformer-Based | Pre-trained language models | BERT, RoBERTa fine-tuned |
| LLM Zero-Shot | Prompt-based classification | GPT-4, Claude |

## How It Works

\`\`\`python
# Using VADER (rule-based)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
scores = analyzer.polarity_scores("This product is amazing!")
# {'neg': 0.0, 'neu': 0.406, 'pos': 0.594, 'compound': 0.6239}

# Using Transformers
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love this new feature!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
\`\`\`

## Applications

- Brand monitoring on social media
- Product review analysis
- Stock market sentiment indicators
- Customer feedback classification
- Political opinion tracking

## Evolution

- **2002**: Pang & Lee pioneer movie review sentiment classification
- **2011**: VADER introduces rule-based sentiment for social media
- **2018**: BERT fine-tuned for sentiment achieves state-of-the-art
- **2020s**: Aspect-based and multilingual sentiment analysis mature
- **2023+**: LLMs enable nuanced sentiment understanding without fine-tuning`,

    text_to_sql: `# Text-to-SQL

Text-to-SQL converts natural language questions into SQL queries, enabling non-technical users to query databases using plain English.

## Key Concepts

- **Schema Linking**: Mapping natural language entities to database columns/tables
- **SQL Generation**: Producing syntactically correct and semantically accurate SQL
- **Execution Accuracy**: Whether generated SQL returns correct results
- **Schema Encoding**: Representing database structure for the model
- **Multi-turn**: Handling follow-up questions that reference previous queries

## How It Works

\`\`\`
User: "Show me total sales by region for 2024"

Generated SQL:
SELECT region, SUM(amount) as total_sales
FROM sales
WHERE YEAR(sale_date) = 2024
GROUP BY region
ORDER BY total_sales DESC;

Pipeline:
1. Parse natural language question
2. Identify relevant tables and columns (schema linking)
3. Determine SQL operations (SELECT, WHERE, GROUP BY, etc.)
4. Generate SQL query
5. Validate and execute
\`\`\`

## Approaches

| Approach | Method | Example |
|----------|--------|---------|
| Seq2Seq | Encode question, decode SQL | BRIDGE, RAT-SQL |
| LLM Prompting | In-context learning with schema | GPT-4 + schema |
| Fine-tuned LLM | Domain-specific training | SQLCoder, DeFog |
| Agentic | LLM with self-correction | LangChain SQL Agent |

## Benchmarks

- **Spider**: 200+ databases, 10K+ questions, cross-database evaluation
- **WikiSQL**: 80K+ question-SQL pairs from Wikipedia tables
- **BIRD**: Real-world databases with dirty data

## Applications

- Business intelligence dashboards
- Data exploration for non-technical users
- Automated reporting
- Database-backed chatbots
- Enterprise search over structured data

## Evolution

- **2017**: WikiSQL dataset released (simple single-table queries)
- **2018**: Spider benchmark pushes complex, cross-database evaluation
- **2020**: RAT-SQL achieves strong cross-database generalization
- **2023**: LLMs with schema prompting approach expert-level accuracy
- **2024+**: Agentic Text-to-SQL with self-correction and validation`,

    dialogue_systems: `# Dialogue Systems

Dialogue Systems (conversational AI) enable natural language conversation between humans and machines. They range from task-oriented assistants to open-domain chatbots.

## Types

| Type | Goal | Example |
|------|------|---------|
| Task-Oriented | Complete specific tasks | Booking flights, ordering food |
| Open-Domain | General conversation | ChatGPT, Claude |
| Retrieval-Based | Select from predefined responses | FAQ bots |
| Generative | Generate novel responses | LLM-based chatbots |
| Hybrid | Combine retrieval and generation | RAG-based assistants |

## Key Concepts

- **Dialogue State Tracking (DST)**: Maintaining conversation context
- **Intent Recognition**: Identifying user's goal
- **Slot Filling**: Extracting specific information (date, location, etc.)
- **Response Generation**: Producing appropriate replies
- **Turn-Taking**: Managing conversation flow

## Architecture

\`\`\`
Traditional Pipeline:
User Input -> NLU (Intent + Slots) -> Dialogue Manager -> NLG -> Response

Modern LLM-Based:
User Input -> Context Window (history + system prompt) -> LLM -> Response

Hybrid (RAG):
User Input -> Retrieve relevant docs -> Augment prompt -> LLM -> Response
\`\`\`

## Applications

- Customer service chatbots
- Virtual assistants (Siri, Alexa, Google Assistant)
- Mental health support (Woebot)
- Educational tutoring systems
- Enterprise help desks

## Evolution

- **1966**: ELIZA (pattern-matching psychotherapy bot)
- **2011**: Siri launches as first mainstream voice assistant
- **2016**: Seq2Seq dialogue models gain traction
- **2022**: ChatGPT demonstrates human-like conversation ability
- **2024+**: Multi-modal dialogue with vision and voice`,

    information_retrieval: `# Information Retrieval

Information Retrieval (IR) is the science of finding relevant information from large collections. It powers search engines, recommendation systems, and RAG pipelines.

## Key Concepts

- **Query**: User's information need expressed in text
- **Document**: A unit of retrievable content
- **Relevance**: How well a document satisfies the query
- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that are retrieved
- **Index**: Pre-built data structure for fast lookup

## Retrieval Methods

| Method | Type | How It Works |
|--------|------|-------------|
| TF-IDF | Sparse | Term frequency weighted by rarity |
| BM25 | Sparse | Probabilistic TF-IDF variant |
| Dense Retrieval | Dense | Encode query and docs as vectors, compute similarity |
| Hybrid | Both | Combine sparse and dense scores |
| Re-ranking | Two-stage | Retrieve broadly, then re-rank with powerful model |

## How It Works

\`\`\`python
# BM25 (Traditional)
from rank_bm25 import BM25Okapi
corpus = ["machine learning is great", "deep learning uses neural networks"]
bm25 = BM25Okapi([doc.split() for doc in corpus])
scores = bm25.get_scores("neural networks".split())

# Dense Retrieval (Modern)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
query_emb = model.encode("What is deep learning?")
doc_embs = model.encode(corpus)
# Compute cosine similarity between query and documents
\`\`\`

## Applications

- Web search engines (Google, Bing)
- Enterprise search
- RAG systems for LLMs
- Legal document search
- Academic paper discovery

## Evolution

- **1960s**: Boolean retrieval and inverted indices
- **1972**: TF-IDF introduced by Sparck Jones
- **1994**: BM25 (Okapi) becomes IR standard
- **2020**: Dense Passage Retrieval (DPR) for neural search
- **2023+**: Hybrid retrieval powers production RAG systems`,

    document_understanding: `# Document Understanding

Document Understanding combines OCR, layout analysis, and NLP to extract and interpret structured information from documents like invoices, forms, contracts, and reports.

## Key Concepts

- **Layout Analysis**: Understanding spatial arrangement of text, tables, figures
- **OCR**: Converting images of text to machine-readable text
- **Key-Value Extraction**: Identifying field labels and their values
- **Table Extraction**: Parsing tabular data from documents
- **Document Classification**: Categorizing document types

## Approaches

| Model | Method | Strength |
|-------|--------|----------|
| LayoutLM | Text + layout features | Position-aware understanding |
| LayoutLMv3 | Unified text, layout, image | Multimodal understanding |
| Donut | OCR-free end-to-end | No separate OCR needed |
| GPT-4V / Claude | Vision LLMs | Zero-shot document parsing |
| DocTR | Deep learning OCR | Open-source OCR engine |

## How It Works

\`\`\`
Document Understanding Pipeline:
1. Document Image Input (PDF/scan/photo)
2. OCR: Extract text with bounding boxes
3. Layout Analysis: Detect headers, paragraphs, tables
4. Entity Extraction: Identify key fields (name, date, amount)
5. Structure Output: JSON/structured data

Modern approach (Vision LLMs):
1. Send document image to GPT-4V / Claude
2. Prompt: "Extract invoice number, date, and line items"
3. Receive structured JSON output
\`\`\`

## Applications

- Invoice processing and accounts payable automation
- Contract analysis and clause extraction
- Medical record digitization
- Insurance claims processing
- Government form processing

## Evolution

- **1990s**: Rule-based template matching for forms
- **2006**: Tesseract open-sourced by Google
- **2020**: LayoutLM introduces position-aware document AI
- **2022**: Donut enables OCR-free document understanding
- **2024+**: Vision LLMs handle arbitrary documents with zero-shot prompting`,

    multilingual_nlp: `# Multilingual NLP

Multilingual NLP develops models and techniques that work across multiple languages, enabling AI to understand and generate text beyond English.

## Key Concepts

- **Cross-lingual Transfer**: Training on one language, applying to another
- **Multilingual Embeddings**: Shared vector space for multiple languages
- **Code-Switching**: Mixed-language text within a single document
- **Low-Resource Languages**: Languages with limited training data
- **Script Diversity**: Handling different writing systems (Latin, CJK, Arabic, etc.)

## Key Models

| Model | Languages | Approach |
|-------|-----------|----------|
| mBERT | 104 | Multilingual BERT trained on Wikipedia |
| XLM-R | 100 | Cross-lingual RoBERTa with CommonCrawl |
| mT5 | 101 | Multilingual T5 text-to-text |
| NLLB | 200+ | Meta's No Language Left Behind for translation |
| BLOOM | 46 | Open multilingual LLM |

## Challenges

- **Data imbalance**: English dominates training data (60%+ of web)
- **Script diversity**: Tokenizers must handle all writing systems
- **Cultural context**: Idioms, humor, and references vary across cultures
- **Evaluation**: Benchmarks are English-centric

## How It Works

\`\`\`python
# Cross-lingual zero-shot classification
from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")
# Trained on English NLI data, works on other languages
result = classifier(
    "Este producto es excelente",  # Spanish input
    candidate_labels=["positive", "negative", "neutral"]
)
\`\`\`

## Applications

- Cross-border customer support
- Multilingual search engines
- Global content moderation
- Translation memory systems
- Language preservation for endangered languages

## Evolution

- **2018**: Multilingual BERT enables cross-lingual transfer
- **2019**: XLM-R improves on mBERT with better cross-lingual performance
- **2022**: NLLB covers 200+ languages for translation
- **2023**: Large multilingual LLMs (GPT-4, Claude) support 50+ languages natively
- **2024+**: Focus shifts to low-resource language inclusion`,

    nlp_benchmarks: `# NLP Benchmarks & Evaluation

NLP Benchmarks provide standardized tasks and metrics for measuring and comparing model performance. They drive research progress and establish baselines.

## Major Benchmarks

| Benchmark | Tasks | Focus |
|-----------|-------|-------|
| GLUE | 9 tasks | General language understanding |
| SuperGLUE | 8 tasks | Harder language understanding |
| SQuAD 1.1/2.0 | QA | Reading comprehension |
| MMLU | 57 subjects | Massive multitask knowledge |
| HumanEval | Code | Code generation correctness |
| MT-Bench | Dialogue | Multi-turn conversation quality |
| HELM | Holistic | Comprehensive LLM evaluation |

## Key Metrics

- **Accuracy**: Correct predictions / total predictions
- **F1 Score**: Harmonic mean of precision and recall
- **BLEU**: N-gram overlap for translation/generation
- **ROUGE**: Recall-oriented metric for summarization
- **Perplexity**: How well model predicts next token (lower = better)
- **Exact Match (EM)**: Percentage of exact correct answers
- **ELO Rating**: Comparative ranking from human preferences (Chatbot Arena)

## How Evaluation Works

\`\`\`python
# Computing ROUGE for summarization
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(
    "The cat sat on the mat",  # reference
    "A cat was sitting on a mat"  # prediction
)
# Returns precision, recall, fmeasure for each ROUGE variant

# MMLU evaluation (multiple choice)
# For each question: provide 4 options (A/B/C/D)
# Measure: accuracy across 57 academic subjects
\`\`\`

## Challenges

- **Benchmark saturation**: Models surpass human baselines
- **Contamination**: Test data leaking into training data
- **Narrow focus**: Benchmarks may not reflect real-world usefulness
- **Gaming**: Optimizing for benchmark rather than genuine capability
- **Human evaluation**: Expensive but often more meaningful

## Evolution

- **2018**: GLUE benchmark released; BERT surpasses human baseline
- **2019**: SuperGLUE introduced for harder evaluation
- **2020**: GPT-3 demonstrates few-shot benchmark performance
- **2023**: MMLU becomes standard LLM evaluation; Chatbot Arena for human preference
- **2024+**: Focus shifts to real-world evaluation, safety benchmarks, and contamination-aware testing`,
  };

  Object.assign(window.AI_DOCS, content);
})();
