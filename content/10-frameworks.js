// 10 - Frameworks & Tools
(function () {
  const content = {
    tensorflow: `# TensorFlow

TensorFlow is Google's open-source ML framework. It provides a comprehensive ecosystem for building, training, and deploying ML models across platforms from servers to mobile devices and browsers.

## Key Features

- **Eager Execution**: Run operations immediately (default since TF 2.0)
- **Keras Integration**: High-level API built directly into TensorFlow
- **TF Serving**: Production model serving infrastructure
- **TF Lite**: Mobile and edge deployment
- **TF.js**: Run models in the browser with JavaScript
- **TPU Support**: Optimized for Google's Tensor Processing Units

## How It Works

\`\`\`python
import tensorflow as tf

# Build a model with Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test)

# Save and deploy
model.save('my_model')
# Convert for mobile: tf.lite.TFLiteConverter
# Serve in production: TensorFlow Serving
\`\`\`

## Ecosystem

| Component | Purpose |
|-----------|---------|
| TensorFlow Core | Training and inference engine |
| Keras | High-level model building API |
| TF Serving | REST/gRPC model serving |
| TF Lite | Mobile and embedded deployment |
| TF.js | Browser-based ML |
| TF Extended (TFX) | Production ML pipelines |
| TensorBoard | Training visualization |

## Evolution

- **2015**: TensorFlow 1.0 released by Google Brain (static computation graphs)
- **2017**: TensorFlow Lite for mobile; TF.js for browsers
- **2019**: TensorFlow 2.0 (eager execution, Keras as default API)
- **2021**: TF 2.x becomes mature production framework
- **2024+**: Focus shifts to JAX at Google; TF remains widely deployed`,

    pytorch: `# PyTorch

PyTorch is Meta's open-source deep learning framework known for its Pythonic design, dynamic computation graphs, and strong research community. It has become the dominant framework in AI research.

## Key Features

- **Dynamic Computation Graphs**: Define-by-run, natural Python debugging
- **Autograd**: Automatic differentiation for gradient computation
- **torch.nn**: Neural network building blocks
- **Distributed Training**: Multi-GPU and multi-node support
- **TorchScript**: JIT compilation for production deployment
- **CUDA Integration**: Seamless GPU acceleration

## How It Works

\`\`\`python
import torch
import torch.nn as nn

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
\`\`\`

## Ecosystem

| Component | Purpose |
|-----------|---------|
| PyTorch Core | Training engine with autograd |
| torchvision | CV datasets, models, transforms |
| torchaudio | Audio processing and models |
| torchtext | NLP utilities |
| PyTorch Lightning | High-level training framework |
| torch.compile | Graph optimization (2.0+) |

## Evolution

- **2016**: PyTorch released by Facebook AI Research (FAIR)
- **2018**: PyTorch 1.0 merges Caffe2 for production
- **2020**: PyTorch dominates ML research papers
- **2022**: PyTorch 2.0 introduces torch.compile for speed
- **2024+**: PyTorch is the most-used framework in AI research and increasingly in production`,

    jax: `# JAX

JAX is Google's high-performance numerical computing library that combines NumPy's API with automatic differentiation, GPU/TPU acceleration, and functional transformations. It is increasingly used for cutting-edge ML research.

## Key Features

- **NumPy API**: Familiar interface (jax.numpy replaces numpy)
- **Automatic Differentiation**: jax.grad for arbitrary Python functions
- **JIT Compilation**: jax.jit compiles functions with XLA for GPU/TPU
- **Vectorization**: jax.vmap for automatic batching
- **Parallelism**: jax.pmap for multi-device parallelism
- **Functional Design**: Pure functions, no hidden state

## How It Works

\`\`\`python
import jax
import jax.numpy as jnp

# Pure function + grad
def loss_fn(params, x, y):
    pred = jnp.dot(x, params['w']) + params['b']
    return jnp.mean((pred - y) ** 2)

grad_fn = jax.grad(loss_fn)  # automatic differentiation
jit_grad_fn = jax.jit(grad_fn)  # JIT compile for speed

# Training step
params = {'w': jnp.zeros(10), 'b': 0.0}
grads = jit_grad_fn(params, x_batch, y_batch)
params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

# Vectorize over batch dimension
batched_predict = jax.vmap(predict_single)

# Parallelize across devices
parallel_train_step = jax.pmap(train_step)
\`\`\`

## JAX Ecosystem

| Library | Purpose |
|---------|---------|
| Flax | Neural network library (Google) |
| Haiku | Neural network library (DeepMind) |
| Optax | Gradient processing and optimizers |
| Orbax | Checkpointing and serialization |
| Equinox | Pythonic neural networks |

## JAX vs PyTorch

| Aspect | JAX | PyTorch |
|--------|-----|---------|
| Paradigm | Functional | Object-oriented |
| Graphs | Compiled (XLA) | Dynamic (eager) |
| TPU support | Native | Limited |
| Debugging | Harder (functional) | Easier (Pythonic) |
| Community | Growing (research) | Dominant |

## Evolution

- **2018**: JAX released by Google Brain
- **2020**: Flax and Haiku emerge as neural network libraries
- **2022**: Google DeepMind adopts JAX for key research (AlphaFold, Gemini)
- **2023**: JAX used to train Gemini models
- **2024+**: Growing adoption in research; production use at Google`,

    keras: `# Keras

Keras is a high-level neural network API that provides a user-friendly interface for building and training deep learning models. Originally standalone, it is now the official high-level API of TensorFlow.

## Key Features

- **Sequential API**: Stack layers linearly for simple models
- **Functional API**: Build complex architectures with shared layers and multiple inputs/outputs
- **Subclassing**: Full customization via Python class inheritance
- **Pre-trained Models**: ImageNet models (ResNet, EfficientNet, etc.) ready to use
- **Callbacks**: Training hooks for logging, early stopping, checkpointing

## How It Works

\`\`\`python
import tensorflow as tf
from tensorflow import keras

# Sequential API (simplest)
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Functional API (flexible)
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[
    keras.callbacks.EarlyStopping(patience=3),
    keras.callbacks.ModelCheckpoint('best_model.keras')
])
\`\`\`

## Applications

- Rapid prototyping of deep learning models
- Transfer learning with pretrained models
- Production deployment via TF Serving / TF Lite
- Educational tool (most-used in ML courses)

## Evolution

- **2015**: Keras released by Francois Chollet (standalone library)
- **2017**: Keras adopted as TensorFlow's official high-level API
- **2019**: tf.keras becomes the recommended way to use TensorFlow
- **2023**: Keras 3.0 released (multi-backend: TF, PyTorch, JAX)
- **2024+**: Keras 3 enables framework-agnostic model development`,

    sklearn: `# Scikit-learn

Scikit-learn is the most popular Python library for classical machine learning. It provides simple and efficient tools for data mining, analysis, and modeling with a consistent API.

## Key Features

- **Consistent API**: fit(), predict(), transform() pattern for all algorithms
- **Comprehensive**: Classification, regression, clustering, dimensionality reduction
- **Preprocessing**: Scaling, encoding, imputation, feature selection
- **Model Selection**: Cross-validation, grid search, train-test split
- **Pipeline**: Chain preprocessing and modeling steps

## How It Works

\`\`\`python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Grid search for best hyperparameters
param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None]}
search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)

# Evaluate
y_pred = search.predict(X_test)
print(classification_report(y_test, y_pred))
\`\`\`

## Algorithm Coverage

| Category | Algorithms |
|----------|-----------|
| Classification | SVM, Random Forest, Gradient Boosting, k-NN, Naive Bayes |
| Regression | Linear, Ridge, Lasso, ElasticNet, SVR |
| Clustering | K-Means, DBSCAN, Hierarchical, Spectral |
| Dim. Reduction | PCA, t-SNE, UMAP (via umap-learn) |
| Preprocessing | StandardScaler, OneHotEncoder, LabelEncoder |
| Model Selection | Cross-validation, GridSearchCV, RandomizedSearchCV |

## Evolution

- **2007**: Scikit-learn started as Google Summer of Code project
- **2010**: First stable release (0.1)
- **2015**: Becomes the standard for classical ML in Python
- **2020**: 1.0 release with improved API consistency
- **2024+**: Remains essential for tabular data; complements deep learning frameworks`,

    fastai: `# FastAI

FastAI is a deep learning library built on top of PyTorch that provides high-level abstractions for training models with best practices built in. It emphasizes practical, state-of-the-art results with minimal code.

## Key Features

- **Layered API**: High-level (one line training) to low-level (full customization)
- **Built-in Best Practices**: Learning rate finder, one-cycle policy, mixup, progressive resizing
- **Data Blocks API**: Flexible data loading and augmentation
- **Transfer Learning**: Fine-tuning pretrained models made simple
- **Applications**: Vision, text, tabular, collaborative filtering

## How It Works

\`\`\`python
from fastai.vision.all import *

# Image classification in 4 lines
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path/'images'),
    pat=r'(.+)_\\d+.jpg$', item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)

# Learning rate finder
learn.lr_find()  # plots loss vs learning rate

# Text classification
from fastai.text.all import *
dls = TextDataLoaders.from_folder(path, valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)
learn.fine_tune(4, 1e-2)
\`\`\`

## Key Innovations

| Feature | Description |
|---------|-------------|
| Learning Rate Finder | Automatically find optimal learning rate |
| One-Cycle Policy | Cyclical learning rate for faster convergence |
| Progressive Resizing | Start training with small images, increase size |
| Mixup | Blend training examples for regularization |
| Discriminative LR | Different learning rates for different layers |

## Evolution

- **2018**: FastAI v1 released alongside fast.ai course (Jeremy Howard)
- **2020**: FastAI v2 rebuilt from scratch with layered architecture
- **2021**: FastAI textbook published (O'Reilly)
- **2023+**: Continues as a practical ML education and research tool`,

    huggingface: `# HuggingFace Transformers

HuggingFace Transformers is the most popular open-source library for working with pretrained language models and transformer architectures. It provides access to thousands of models with a unified API.

## Key Features

- **Model Hub**: 500K+ pretrained models for download
- **Pipeline API**: One-line inference for common tasks
- **Auto Classes**: Automatically select correct model architecture
- **Tokenizers**: Fast tokenization with Rust backend
- **Trainer**: Simplified training with distributed support
- **Datasets**: Efficient dataset loading and processing

## How It Works

\`\`\`python
from transformers import pipeline, AutoModel, AutoTokenizer

# Pipeline API (simplest)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")  # [{'label': 'POSITIVE', 'score': 0.99}]

generator = pipeline("text-generation", model="gpt2")
text = generator("AI is transforming", max_length=50)

# Auto Classes (flexible)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)  # hidden states, attentions, etc.

# Fine-tuning with Trainer
from transformers import Trainer, TrainingArguments
args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
\`\`\`

## Ecosystem

| Component | Purpose |
|-----------|---------|
| Transformers | Model library (NLP, Vision, Audio) |
| Datasets | Dataset loading and processing |
| Tokenizers | Fast tokenization |
| Accelerate | Multi-GPU/TPU training |
| PEFT | Parameter-efficient fine-tuning (LoRA, etc.) |
| TRL | Transformer Reinforcement Learning (RLHF) |
| Gradio/Spaces | Model demos and deployment |

## Evolution

- **2018**: HuggingFace releases pytorch-transformers (BERT wrapper)
- **2019**: Renamed to Transformers; supports multiple architectures
- **2020**: Model Hub launched; becomes community standard
- **2022**: Diffusers library for image generation
- **2024+**: 500K+ models, supports LLMs, vision, audio, and multimodal`,

    onnx: `# ONNX Runtime

ONNX (Open Neural Network Exchange) is an open format for representing ML models, enabling interoperability between frameworks. ONNX Runtime is Microsoft's high-performance inference engine.

## Key Concepts

- **ONNX Format**: Framework-agnostic model representation (.onnx files)
- **Graph Optimization**: Fuse operations, eliminate redundancy
- **Hardware Acceleration**: GPU, CPU, TensorRT, DirectML, OpenVINO backends
- **Quantization**: Reduce model precision for faster inference
- **Cross-Platform**: Run same model on server, mobile, browser, edge

## How It Works

\`\`\`python
# Export PyTorch model to ONNX
import torch

model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'], output_names=['output'])

# Run inference with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
result = session.run(None, {"input": input_data.numpy()})

# Quantize for faster inference
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quantized.onnx",
                 weight_type=QuantType.QInt8)
\`\`\`

## Supported Frameworks

| Framework | Export Support |
|-----------|--------------|
| PyTorch | torch.onnx.export |
| TensorFlow | tf2onnx converter |
| Keras | keras2onnx / tf2onnx |
| Scikit-learn | skl2onnx |
| JAX | jax2onnx |

## Evolution

- **2017**: ONNX launched by Facebook and Microsoft
- **2019**: ONNX Runtime released (high-performance inference)
- **2020**: Support for transformer models and NLP workloads
- **2022**: ONNX Runtime Web enables browser-based inference
- **2024+**: Standard interchange format for model deployment pipelines`,

    tensorrt: `# TensorRT

TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It maximizes GPU throughput for production deployment by optimizing models for NVIDIA hardware.

## Key Features

- **Layer Fusion**: Combine multiple operations into single GPU kernels
- **Precision Calibration**: FP16, INT8, FP8 quantization with minimal accuracy loss
- **Kernel Auto-Tuning**: Select optimal GPU kernel for each operation
- **Dynamic Shapes**: Support variable input sizes
- **Memory Optimization**: Reuse memory buffers efficiently

## How It Works

\`\`\`python
# Convert ONNX model to TensorRT
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
engine = builder.build_serialized_network(network, config)

# Typical speedup: 2-5x over PyTorch inference
# INT8 quantization can achieve 10x+ speedup
\`\`\`

## Optimization Techniques

| Technique | Description | Speedup |
|-----------|-------------|---------|
| Layer Fusion | Merge Conv+BN+ReLU into one | 1.5-2x |
| FP16 Precision | Half-precision floating point | 2-3x |
| INT8 Quantization | 8-bit integer precision | 3-5x |
| Kernel Tuning | Best kernel per operation | 1.2-1.5x |
| Memory Pooling | Efficient buffer allocation | Reduced latency |

## Evolution

- **2017**: TensorRT 1.0 released for inference optimization
- **2019**: INT8 calibration for production quantization
- **2021**: TensorRT 8 with improved transformer support
- **2023**: TensorRT-LLM for large language model inference
- **2024+**: TensorRT-LLM powers NVIDIA's LLM serving stack`,

    wandb: `# Weights & Biases (W&B)

Weights & Biases is an MLOps platform for experiment tracking, model versioning, dataset management, and collaborative ML development. It is the most popular experiment tracking tool in ML.

## Key Features

- **Experiment Tracking**: Log metrics, hyperparameters, and artifacts automatically
- **Visualizations**: Interactive dashboards, comparison plots, custom charts
- **Sweeps**: Automated hyperparameter optimization
- **Artifacts**: Version datasets, models, and outputs
- **Tables**: Log and visualize structured data (predictions, examples)
- **Reports**: Collaborative documents with embedded visualizations

## How It Works

\`\`\`python
import wandb

# Initialize run
wandb.init(project="my-project", config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32,
    "architecture": "ResNet50"
})

# Training loop with logging
for epoch in range(config.epochs):
    train_loss = train(model, dataloader)
    val_acc = evaluate(model, val_loader)
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_acc
    })

# Log model artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pth")
wandb.log_artifact(artifact)

wandb.finish()
\`\`\`

## Platform Components

| Feature | Purpose |
|---------|---------|
| Runs | Individual experiment tracking |
| Sweeps | Hyperparameter search (Bayesian, grid, random) |
| Artifacts | Dataset and model versioning |
| Tables | Prediction visualization and debugging |
| Reports | Shareable research documents |

## Evolution

- **2018**: W&B founded; initial experiment tracking tool
- **2020**: Sweeps and Artifacts features added
- **2022**: Becomes de facto standard for ML experiment tracking
- **2023**: Tables and model evaluation features mature
- **2024+**: Integration with LLM evaluation and production monitoring`,

    mlflow: `# MLflow

MLflow is an open-source platform by Databricks for managing the complete ML lifecycle including experimentation, reproducibility, deployment, and model registry.

## Key Components

| Component | Purpose |
|-----------|---------|
| MLflow Tracking | Log experiments, parameters, metrics, artifacts |
| MLflow Projects | Reproducible runs with conda/docker environments |
| MLflow Models | Standard format for packaging ML models |
| MLflow Registry | Central model store with versioning and staging |
| MLflow Evaluate | Evaluate LLMs and ML models systematically |

## How It Works

\`\`\`python
import mlflow
import mlflow.sklearn

# Start experiment
mlflow.set_experiment("classification-project")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Register model for deployment
    mlflow.register_model("runs:/{run_id}/model", "ProductionClassifier")
\`\`\`

## MLflow vs W&B

| Feature | MLflow | W&B |
|---------|--------|-----|
| License | Open source (Apache 2.0) | Freemium (proprietary) |
| Hosting | Self-hosted or Databricks | Cloud-hosted or self |
| Model Registry | Built-in | Via Artifacts |
| Deployment | Built-in serving | No native serving |
| Visualization | Basic UI | Rich interactive dashboards |

## Evolution

- **2018**: MLflow released by Databricks
- **2020**: MLflow Model Registry for model lifecycle
- **2022**: MLflow Pipelines for ML workflow automation
- **2023**: MLflow 2.0 with LLM evaluation support
- **2024+**: MLflow Tracing for LLM observability; Databricks integration deepens`,

    gradio_streamlit: `# Gradio & Streamlit

Gradio and Streamlit are Python libraries for building interactive ML demos and web applications with minimal frontend code. They enable ML engineers to share models without web development expertise.

## Gradio

\`\`\`python
import gradio as gr
from transformers import pipeline

# Create interface in 3 lines
classifier = pipeline("sentiment-analysis")

def classify(text):
    result = classifier(text)[0]
    return {result['label']: result['score']}

demo = gr.Interface(fn=classify, inputs="text", outputs="label")
demo.launch()  # Opens web UI at localhost:7860

# Advanced: Chatbot interface
def chat(message, history):
    response = llm.generate(message)
    return response

gr.ChatInterface(fn=chat).launch()
\`\`\`

## Streamlit

\`\`\`python
import streamlit as st
import pandas as pd

st.title("ML Dashboard")

# Interactive widgets
uploaded_file = st.file_uploader("Upload CSV")
model_choice = st.selectbox("Model", ["Random Forest", "XGBoost", "SVM"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Train Model"):
        with st.spinner("Training..."):
            model = train_model(df, model_choice)
            st.success(f"Accuracy: {model.score:.2%}")
            st.bar_chart(feature_importances)
\`\`\`

## Comparison

| Feature | Gradio | Streamlit |
|---------|--------|-----------|
| Best for | ML model demos | Data apps & dashboards |
| API | Function-based | Script-based |
| Hosting | HuggingFace Spaces | Streamlit Cloud |
| Components | Input/output focused | Rich widget library |
| Sharing | One-line public link | Deploy to cloud |
| Chatbot UI | Built-in ChatInterface | Via st.chat_message |

## Evolution

- **2019**: Streamlit launched; rapid adoption for data apps
- **2020**: Gradio released; focused on ML demos
- **2022**: Gradio acquired by HuggingFace; integrated with Spaces
- **2023**: Both widely used for LLM demos and prototype apps
- **2024+**: Standard tools for AI application prototyping`,
  };

  Object.assign(window.AI_DOCS, content);
})();
