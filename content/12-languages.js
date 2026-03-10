// 12 - Programming Languages for AI
(function () {
  const content = {
    python_ai: `# Python for AI/ML

Python is the dominant programming language for AI and machine learning. Its readability, vast ecosystem of ML libraries, and strong community make it the default choice for AI development.

## Why Python Dominates AI

- **Rich ecosystem**: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, HuggingFace
- **Readability**: Clean syntax that maps well to mathematical notation
- **Community**: Largest AI/ML community with extensive documentation
- **Prototyping speed**: Rapid experimentation and iteration
- **Interoperability**: C/C++ extensions for performance-critical code
- **Jupyter Notebooks**: Interactive development and visualization

## Key Libraries

| Library | Domain | Purpose |
|---------|--------|---------|
| NumPy | Numerics | Array operations, linear algebra |
| Pandas | Data | DataFrames, data manipulation |
| Scikit-learn | ML | Classical ML algorithms |
| PyTorch | DL | Deep learning research framework |
| TensorFlow | DL | Production deep learning |
| HuggingFace | NLP/LLM | Pretrained models and pipelines |
| LangChain | LLM Apps | LLM application framework |
| Matplotlib/Plotly | Viz | Data visualization |

## Example: Full ML Pipeline

\`\`\`python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
\`\`\`

## Limitations

- **Speed**: Slower than compiled languages (mitigated by C extensions)
- **GIL**: Global Interpreter Lock limits true multi-threading
- **Mobile**: Not native for mobile/edge deployment
- **Memory**: Higher memory usage than C/C++/Rust

## Evolution

- **1991**: Python released by Guido van Rossum
- **2006**: NumPy 1.0 establishes scientific computing foundation
- **2008**: Pandas released for data analysis
- **2015**: TensorFlow and Keras make Python the DL language
- **2022+**: Python dominates 80%+ of AI/ML development`,

    javascript_ai: `# JavaScript/TypeScript for AI

JavaScript is expanding its role in AI through browser-based inference, Node.js backends, and LLM application frameworks. TypeScript adds type safety for production AI systems.

## Key Capabilities

- **Browser ML**: Run models directly in the browser (TensorFlow.js, ONNX.js)
- **LLM Applications**: Build AI apps with LangChain.js, Vercel AI SDK
- **Full-Stack AI**: Unified language for frontend + AI backend
- **Edge Deployment**: Run models on edge with serverless functions
- **Real-time**: WebSocket-based streaming for LLM responses

## Key Libraries

| Library | Purpose |
|---------|---------|
| TensorFlow.js | Train and run models in browser/Node.js |
| ONNX Runtime Web | Run ONNX models in browser |
| LangChain.js | LLM application framework |
| Vercel AI SDK | React hooks for AI streaming |
| Transformers.js | HuggingFace models in JS |
| ml5.js | Friendly ML for creative coding |
| brain.js | Neural networks in JS |

## Example: LLM App with Vercel AI SDK

\`\`\`typescript
// Next.js API route with streaming
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';

export async function POST(req: Request) {
  const { messages } = await req.json();
  const result = streamText({
    model: openai('gpt-4'),
    messages,
  });
  return result.toDataStreamResponse();
}

// React component
import { useChat } from 'ai/react';

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat();
  return (
    <form onSubmit={handleSubmit}>
      {messages.map(m => <div key={m.id}>{m.content}</div>)}
      <input value={input} onChange={handleInputChange} />
    </form>
  );
}
\`\`\`

## Applications

- AI-powered web applications (chatbots, assistants)
- Browser-based image classification and object detection
- Real-time AI features in web apps
- Serverless AI functions (Vercel, Cloudflare Workers)

## Evolution

- **2018**: TensorFlow.js enables in-browser ML
- **2022**: LangChain.js brings LLM chains to JavaScript
- **2023**: Vercel AI SDK simplifies streaming AI UIs
- **2024**: Transformers.js runs HuggingFace models in the browser
- **2025+**: JS becomes second most important AI language`,

    rust_ml: `# Rust for ML

Rust is emerging in the ML ecosystem for performance-critical components. Its memory safety, zero-cost abstractions, and C-level speed make it ideal for ML infrastructure, tokenizers, and inference engines.

## Why Rust for ML

- **Performance**: C/C++ level speed without garbage collection
- **Memory Safety**: Prevents null pointers, buffer overflows, data races at compile time
- **Concurrency**: Fearless concurrency with ownership system
- **WebAssembly**: Compile to WASM for browser deployment
- **Interop**: Easy FFI with Python via PyO3/Maturin

## Key Projects

| Project | Purpose |
|---------|---------|
| HuggingFace Tokenizers | Fast tokenization (used by Transformers) |
| Candle | Minimalist ML framework by HuggingFace |
| Burn | Deep learning framework in pure Rust |
| tch-rs | Rust bindings for libtorch (PyTorch) |
| ort | ONNX Runtime bindings |
| Polars | Fast DataFrame library (alternative to Pandas) |
| tiktoken | OpenAI's tokenizer (Rust + Python) |

## Example

\`\`\`rust
use candle_core::{Device, Tensor, DType};
use candle_nn::{Linear, Module};

// Simple linear layer
let device = Device::Cpu;
let weight = Tensor::randn(0f32, 1.0, (10, 784), &device)?;
let bias = Tensor::zeros((10,), DType::F32, &device)?;
let linear = Linear::new(weight, Some(bias));

let input = Tensor::randn(0f32, 1.0, (1, 784), &device)?;
let output = linear.forward(&input)?;
\`\`\`

## Rust's Role in AI

- **Tokenizers**: HuggingFace tokenizers are Rust with Python bindings (100x faster)
- **Inference engines**: Candle runs LLMs efficiently
- **Data processing**: Polars outperforms Pandas on large datasets
- **Production serving**: Memory-safe, high-performance model servers
- **Edge/WASM**: Compile ML models to WebAssembly

## Evolution

- **2019**: tch-rs brings PyTorch bindings to Rust
- **2021**: HuggingFace tokenizers rewritten in Rust (massive speedup)
- **2023**: Candle framework for running LLMs in Rust
- **2024**: Burn framework matures; Polars becomes major data tool
- **2025+**: Rust solidifies role as ML infrastructure language`,

    julia: `# Julia for Scientific Computing

Julia is a high-performance programming language designed for scientific computing and numerical analysis. It combines Python-like syntax with C-like speed, making it attractive for computational ML research.

## Key Features

- **Multiple Dispatch**: Core paradigm enabling flexible function specialization
- **JIT Compilation**: LLVM-based compilation to native machine code
- **Speed**: Approaches C/Fortran performance for numerical code
- **Unicode Math**: Write code that looks like mathematical notation
- **Interop**: Call Python (PyCall), C, and Fortran directly
- **Differentiable Programming**: Zygote.jl for automatic differentiation

## Key Packages

| Package | Purpose |
|---------|---------|
| Flux.jl | Deep learning framework |
| MLJ.jl | Unified ML interface (like scikit-learn) |
| Zygote.jl | Source-to-source automatic differentiation |
| DifferentialEquations.jl | Solving ODEs/PDEs (world-class) |
| Turing.jl | Probabilistic programming |
| Plots.jl | Data visualization |
| DataFrames.jl | Tabular data manipulation |

## Example

\`\`\`julia
using Flux

# Define a neural network
model = Chain(
    Dense(784, 128, relu),
    Dropout(0.2),
    Dense(128, 10),
    softmax
)

# Train
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM(0.001)
Flux.train!(loss, params(model), data, opt)

# Scientific computing strength
using DifferentialEquations
f(u, p, t) = 1.01 * u
prob = ODEProblem(f, 0.5, (0.0, 1.0))
sol = solve(prob, Tsit5())
\`\`\`

## Julia vs Python

| Aspect | Julia | Python |
|--------|-------|--------|
| Speed | Near C | Slow (needs C extensions) |
| Ecosystem | Growing | Massive |
| Learning curve | Moderate | Easy |
| Community | Small but active | Huge |
| Best for | Scientific computing, simulations | General ML, LLMs |

## Evolution

- **2012**: Julia 0.1 released at MIT
- **2018**: Julia 1.0 stable release
- **2020**: Flux.jl matures for deep learning
- **2022**: SciML ecosystem for scientific ML
- **2024+**: Niche but strong in scientific computing and differential equations`,

    r_language: `# R for Statistical Learning

R is a programming language built specifically for statistical computing and data analysis. It excels in statistical modeling, visualization, and is widely used in academia, biostatistics, and data science.

## Key Features

- **Statistical Heritage**: Built by statisticians for statistics
- **Visualization**: ggplot2 produces publication-quality graphics
- **CRAN**: 20,000+ packages for every statistical method
- **Tidyverse**: Modern, consistent data manipulation ecosystem
- **RMarkdown/Quarto**: Reproducible research documents
- **Shiny**: Interactive web dashboards

## Key Packages

| Package | Purpose |
|---------|---------|
| caret / tidymodels | Unified ML training interface |
| ggplot2 | Grammar of graphics visualization |
| dplyr / tidyr | Data manipulation (tidyverse) |
| xgboost / ranger | Gradient boosting, random forests |
| keras (R) | Deep learning via TensorFlow |
| torch (R) | PyTorch in R |
| Shiny | Interactive web applications |

## Example

\`\`\`r
library(tidyverse)
library(tidymodels)

# Load and prepare data
data <- read_csv("data.csv") %>%
  mutate(target = as_factor(target))

# Split data
split <- initial_split(data, prop = 0.8)
train <- training(split)
test <- testing(split)

# Define model and workflow
model_spec <- rand_forest(trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

workflow <- workflow() %>%
  add_model(model_spec) %>%
  add_formula(target ~ .)

# Fit and evaluate
fit <- workflow %>% fit(train)
predictions <- predict(fit, test)
accuracy <- accuracy(bind_cols(test, predictions), target, .pred_class)
\`\`\`

## R vs Python

| Aspect | R | Python |
|--------|---|--------|
| Best for | Statistics, visualization | General ML, DL, LLMs |
| Visualization | ggplot2 (superior) | Matplotlib (good) |
| Statistics | Native strength | Needs libraries |
| Deep Learning | Possible but limited | Dominant |
| Industry adoption | Decreasing | Increasing |

## Evolution

- **1995**: R released as open-source S-language implementation
- **2009**: ggplot2 revolutionizes data visualization
- **2014**: Tidyverse modernizes R data science workflow
- **2020**: Tidymodels provides unified ML interface
- **2024+**: R remains strong in biostatistics and academic research`,

    cpp_ml: `# C++ for Performance ML

C++ is the performance backbone of ML frameworks. While not used for day-to-day ML development, it powers the engines beneath PyTorch, TensorFlow, and inference runtimes where microsecond latency matters.

## Why C++ in ML

- **Raw Performance**: Zero-overhead abstractions, manual memory management
- **GPU Programming**: CUDA kernels written in C/C++
- **Framework Internals**: PyTorch (libtorch), TensorFlow core are C++
- **Inference Engines**: TensorRT, ONNX Runtime, TVM written in C++
- **Embedded/Edge**: Required for resource-constrained devices
- **Low Latency**: Trading systems, real-time robotics

## Key Uses

| Domain | C++ Role |
|--------|----------|
| Framework Core | PyTorch libtorch, TF runtime |
| CUDA Kernels | Custom GPU operations |
| Inference | TensorRT, ONNX Runtime |
| Edge Deployment | TF Lite C++ API, NCNN |
| Robotics | ROS, real-time control |
| Game AI | Unreal Engine ML integration |

## Example

\`\`\`cpp
// PyTorch C++ Frontend (libtorch)
#include <torch/torch.h>

struct Net : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        return fc2->forward(x);
    }
};

// Inference
auto model = Net();
torch::load(model, "model.pt");
auto output = model.forward(input_tensor);
\`\`\`

## Applications

- High-frequency trading with ML models
- Robotics inference (real-time constraints)
- Game AI (in-engine inference)
- Custom CUDA kernels for novel operations
- Embedded ML on microcontrollers

## Evolution

- **2011**: C++11 modernizes the language (lambdas, smart pointers)
- **2015**: TensorFlow uses C++ for its computation engine
- **2017**: PyTorch libtorch C++ frontend released
- **2020**: C++20 adds ranges, concepts, coroutines
- **2024+**: C++ remains essential for ML infrastructure and edge deployment`,

    sql_data: `# SQL for Data Pipelines

SQL (Structured Query Language) is essential for AI/ML data pipelines. It handles data extraction, transformation, feature engineering, and serves as the interface for Text-to-SQL AI applications.

## Role in ML

- **Data Extraction**: Pull training data from databases and warehouses
- **Feature Engineering**: Compute aggregates, joins, window functions
- **Data Quality**: Validate, clean, and deduplicate data
- **Feature Stores**: Define and serve features for ML models
- **Analytics**: Evaluate model performance on business metrics

## Key Operations for ML

\`\`\`sql
-- Feature engineering with window functions
SELECT
    user_id,
    COUNT(*) OVER (PARTITION BY user_id ORDER BY date
                   ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as purchases_30d,
    AVG(amount) OVER (PARTITION BY user_id) as avg_spend,
    NTILE(10) OVER (ORDER BY total_spend) as spend_decile
FROM transactions;

-- Training data extraction
SELECT
    f.*, l.churn_label
FROM features f
JOIN labels l ON f.user_id = l.user_id
WHERE f.snapshot_date = '2024-01-01';

-- Model evaluation
SELECT
    segment,
    COUNT(*) as total,
    SUM(CASE WHEN prediction = actual THEN 1 ELSE 0 END) as correct,
    ROUND(100.0 * SUM(CASE WHEN prediction = actual THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_pct
FROM model_predictions
GROUP BY segment;
\`\`\`

## Data Platforms

| Platform | Type | ML Integration |
|----------|------|---------------|
| BigQuery | Cloud warehouse | BigQuery ML (train in SQL) |
| Snowflake | Cloud warehouse | Snowpark for Python ML |
| Databricks | Lakehouse | Unity Catalog, MLflow |
| PostgreSQL | RDBMS | pgvector for embeddings |
| DuckDB | Embedded OLAP | Fast local analytics |

## Applications

- ETL pipelines for ML training data
- Feature store backends
- Text-to-SQL with LLMs
- In-database ML (BigQuery ML, MindsDB)
- Data validation and monitoring

## Evolution

- **1970s**: SQL invented at IBM (Codd's relational model)
- **2000s**: Data warehouses become ML data sources
- **2019**: BigQuery ML enables training models with SQL
- **2023**: Text-to-SQL with LLMs democratizes data access
- **2024+**: SQL remains the lingua franca of data`,

    mojo: `# Mojo Language

Mojo is a new programming language developed by Modular that combines Python's usability with systems-level performance. It aims to be a superset of Python that can match C++ speed for AI workloads.

## Key Features

- **Python Superset**: Valid Python is valid Mojo (progressive adoption)
- **Compiled Performance**: MLIR-based compilation to native code
- **Systems Programming**: Ownership, manual memory management when needed
- **SIMD Support**: First-class vectorization for numerical code
- **Hardware Targeting**: Compile for CPUs, GPUs, and custom accelerators
- **Zero-Cost Abstractions**: High-level code compiles to efficient machine code

## How It Works

\`\`\`python
# Mojo code (Python-compatible mode)
def hello():
    print("Hello from Mojo!")

# Mojo with performance features
fn compute_sum(data: Tensor[DType.float32]) -> Float32:
    var total: Float32 = 0.0
    for i in range(data.num_elements()):
        total += data[i]
    return total

# SIMD vectorization
fn vectorized_add(a: SIMD[DType.float32, 8], b: SIMD[DType.float32, 8]) -> SIMD[DType.float32, 8]:
    return a + b

# Struct with value semantics (like Rust)
struct MyTensor:
    var data: Pointer[Float32]
    var size: Int

    fn __init__(inout self, size: Int):
        self.size = size
        self.data = Pointer[Float32].alloc(size)
\`\`\`

## Mojo vs Python vs C++

| Aspect | Python | Mojo | C++ |
|--------|--------|------|-----|
| Syntax | Easy | Easy (Python-like) | Complex |
| Speed | Slow (interpreted) | Fast (compiled) | Fast (compiled) |
| Memory Safety | GC managed | Optional ownership | Manual |
| AI Libraries | Vast ecosystem | Growing | Framework internals |
| Learning Curve | Low | Low-Medium | High |

## Applications

- High-performance ML kernels
- Custom operators for deep learning
- Edge deployment with optimal performance
- Replacing C++ in ML framework internals
- Unified language for research and production

## Evolution

- **2023 (May)**: Mojo announced by Chris Lattner (creator of LLVM, Swift)
- **2023 (Sep)**: Mojo available for local development
- **2024**: Open-source components, MAX platform for deployment
- **2025+**: Growing ecosystem; potential to unify Python ease with C++ speed`,
  };

  Object.assign(window.AI_DOCS, content);
})();
