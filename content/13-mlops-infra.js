// 13 - MLOps & Infrastructure
(function () {
  const content = {
    gpu_tpu: `# GPU & TPU Infrastructure

GPUs and TPUs are the hardware accelerators that power modern AI. Understanding compute infrastructure is essential for training and deploying ML models efficiently.

## GPU (Graphics Processing Unit)

| Generation | Example | Memory | Use Case |
|-----------|---------|--------|----------|
| Consumer | RTX 4090 | 24GB | Small models, prototyping |
| Data Center | A100 | 40/80GB | Training, inference |
| Latest | H100 | 80GB | LLM training, HPC |
| Next Gen | B200 | 192GB | Largest models |

## TPU (Tensor Processing Unit)

- **Custom ASIC**: Designed by Google specifically for ML workloads
- **Matrix Multiply**: Optimized for large matrix operations
- **Bfloat16**: Custom number format for ML training
- **Pods**: Connect multiple TPUs for distributed training
- **Versions**: TPU v2 -> v3 -> v4 -> v5e -> v5p (increasing performance)

## Key Concepts

\`\`\`
GPU Architecture for ML:
- CUDA Cores: Parallel processing units (thousands)
- Tensor Cores: Specialized for matrix multiply (FP16, BF16, INT8)
- HBM (High Bandwidth Memory): Fast memory for large models
- NVLink: High-speed GPU-to-GPU interconnect
- CUDA: NVIDIA's parallel computing platform

Memory Hierarchy:
Registers (fastest) -> Shared Memory -> L1/L2 Cache -> HBM -> System RAM
\`\`\`

## Cloud GPU Providers

| Provider | GPUs Available | Service |
|----------|---------------|---------|
| AWS | A100, H100 | EC2 P4d/P5 instances |
| Google Cloud | A100, H100, TPU v5 | Compute Engine, Vertex AI |
| Azure | A100, H100 | ND-series VMs |
| Lambda Labs | A100, H100 | On-demand GPU cloud |
| CoreWeave | H100 | GPU-native cloud |

## Evolution

- **2012**: GPUs first used for deep learning (AlexNet on GTX 580)
- **2016**: Google announces TPU v1
- **2017**: NVIDIA V100 with Tensor Cores
- **2020**: A100 enables training of 100B+ parameter models
- **2024+**: H100/B200 and TPU v5 power LLM training at scale`,

    model_serving: `# Model Serving & Deployment

Model serving is the process of making trained ML models available for inference in production. It handles scaling, latency, versioning, and monitoring.

## Serving Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| REST API | HTTP endpoints | Web applications |
| gRPC | Binary protocol | Low-latency microservices |
| Batch | Process data in bulk | Offline predictions |
| Streaming | Real-time event processing | Live data pipelines |
| Edge | On-device inference | Mobile, IoT |
| Serverless | Auto-scaling functions | Variable workloads |

## Key Tools

\`\`\`
Serving Frameworks:
- TorchServe: PyTorch native model serving
- TF Serving: TensorFlow production serving
- Triton: NVIDIA's multi-framework inference server
- vLLM: High-throughput LLM serving
- TGI: HuggingFace Text Generation Inference
- Ollama: Local LLM serving

Deployment:
- FastAPI + Uvicorn: Python REST API
- BentoML: ML model packaging and serving
- Ray Serve: Scalable serving with Ray
- Seldon Core: Kubernetes-native ML serving
\`\`\`

## Example: FastAPI Model Server

\`\`\`python
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pt")
model.eval()

@app.post("/predict")
async def predict(data: dict):
    input_tensor = preprocess(data["input"])
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.tolist()}
\`\`\`

## LLM Serving

| Tool | Focus | Key Feature |
|------|-------|-------------|
| vLLM | LLM inference | PagedAttention, continuous batching |
| TGI | LLM inference | HuggingFace integration |
| Ollama | Local LLMs | Simple CLI for running models |
| llama.cpp | CPU inference | Quantized LLM on consumer hardware |

## Evolution

- **2017**: TF Serving established for production ML
- **2019**: TorchServe and Triton for multi-framework serving
- **2023**: vLLM and TGI optimize LLM serving
- **2024**: Ollama makes local LLM serving accessible
- **2025+**: Serving infrastructure becomes commodity`,

    docker_k8s: `# Docker & Kubernetes for ML

Containerization with Docker and orchestration with Kubernetes provide reproducible, scalable infrastructure for ML workflows. They ensure models run consistently across environments.

## Docker for ML

\`\`\`dockerfile
# Example ML model Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ ./model/
COPY serve.py .

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

## Kubernetes for ML

\`\`\`yaml
# Model serving deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    spec:
      containers:
      - name: model
        image: myregistry/ml-model:v2
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        ports:
        - containerPort: 8000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
\`\`\`

## ML-Specific K8s Tools

| Tool | Purpose |
|------|---------|
| Kubeflow | End-to-end ML platform on K8s |
| Seldon Core | Model serving and monitoring |
| KServe | Serverless model inference |
| NVIDIA GPU Operator | GPU management for K8s |
| Argo Workflows | ML pipeline orchestration |

## Evolution

- **2013**: Docker released (containerization revolution)
- **2014**: Kubernetes released by Google
- **2019**: Kubeflow 1.0 for ML on Kubernetes
- **2022**: KServe matures for serverless ML inference
- **2024+**: GPU-aware scheduling and autoscaling become standard`,

    cicd_ml: `# CI/CD for ML Pipelines

CI/CD (Continuous Integration / Continuous Deployment) for ML extends traditional software CI/CD with data validation, model training, evaluation, and deployment automation.

## ML CI/CD vs Software CI/CD

| Aspect | Software CI/CD | ML CI/CD |
|--------|---------------|----------|
| Testing | Unit/integration tests | + Model quality tests |
| Artifacts | Code binaries | + Models, datasets |
| Validation | Code linting | + Data validation, schema checks |
| Deployment | App deployment | + Model registry, A/B testing |
| Monitoring | App metrics | + Model performance, data drift |

## ML Pipeline Stages

\`\`\`
1. Data Validation
   - Schema checks, distribution analysis
   - Detect data drift from training distribution

2. Feature Engineering
   - Transform raw data into model features
   - Feature store integration

3. Model Training
   - Train with tracked hyperparameters
   - Experiment tracking (W&B, MLflow)

4. Model Evaluation
   - Test on holdout dataset
   - Compare against baseline / previous model
   - Check for bias and fairness

5. Model Registry
   - Version and stage models (dev/staging/prod)
   - Approval gates for production

6. Deployment
   - Canary or blue-green deployment
   - A/B testing with traffic splitting

7. Monitoring
   - Track prediction quality over time
   - Alert on model degradation or data drift
\`\`\`

## Tools

| Tool | Purpose |
|------|---------|
| GitHub Actions | CI/CD automation |
| DVC (Data Version Control) | Data and model versioning |
| Great Expectations | Data validation |
| MLflow Registry | Model versioning and staging |
| Evidently AI | ML monitoring and drift detection |

## Evolution

- **2017**: DVC introduced for data versioning
- **2019**: ML pipeline tools mature (Kubeflow Pipelines, Airflow)
- **2021**: Feature stores (Feast, Tecton) become standard
- **2023**: LLMOps extends MLOps for language models
- **2024+**: Automated evaluation and deployment for LLM applications`,

    distributed_training: `# Distributed Training

Distributed Training splits model training across multiple GPUs or machines to handle models too large for a single device and to accelerate training time.

## Parallelism Strategies

| Strategy | What is Split | Use Case |
|----------|--------------|----------|
| Data Parallel | Data batches across GPUs | Most common, easy to implement |
| Model Parallel | Model layers across GPUs | Very large models |
| Tensor Parallel | Individual tensors across GPUs | Transformer layers |
| Pipeline Parallel | Model stages as a pipeline | Very deep models |
| Expert Parallel | MoE experts across GPUs | Mixture of Experts models |

## How It Works

\`\`\`python
# PyTorch Data Parallel (simplest)
model = nn.DataParallel(model)  # Wraps model for multi-GPU
output = model(input)  # Automatically distributes across GPUs

# PyTorch Distributed Data Parallel (production)
import torch.distributed as dist
dist.init_process_group("nccl")
model = DistributedDataParallel(model, device_ids=[local_rank])

# DeepSpeed (Microsoft) for large models
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model, config=ds_config
)
# Supports ZeRO stages 1-3 for memory optimization

# FSDP (Fully Sharded Data Parallel - PyTorch native)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
\`\`\`

## Key Frameworks

| Framework | Provider | Key Feature |
|-----------|----------|-------------|
| DeepSpeed | Microsoft | ZeRO optimizer, 3D parallelism |
| FSDP | PyTorch | Native sharded training |
| Megatron-LM | NVIDIA | Tensor and pipeline parallelism |
| Ray Train | Anyscale | Framework-agnostic distributed training |
| Horovod | Uber | Ring-allreduce data parallelism |

## Evolution

- **2017**: Horovod simplifies distributed training
- **2019**: Megatron-LM enables training billion-parameter models
- **2020**: DeepSpeed ZeRO reduces memory by 8x
- **2022**: PyTorch FSDP becomes native distributed solution
- **2024+**: Training 1T+ parameter models across thousands of GPUs`,

    data_pipelines: `# Data Pipelines & Feature Stores

Data pipelines move, transform, and prepare data for ML models. Feature stores provide a centralized repository for computing, storing, and serving features consistently across training and inference.

## Data Pipeline Components

\`\`\`
Data Sources -> Ingestion -> Transformation -> Validation -> Storage -> Serving

- Ingestion: Kafka, Spark Streaming, Airbyte
- Transformation: Spark, dbt, Pandas, Polars
- Validation: Great Expectations, Pandera
- Storage: Data lake (S3/GCS), warehouse (BigQuery, Snowflake)
- Orchestration: Airflow, Prefect, Dagster
\`\`\`

## Feature Stores

| Store | Type | Key Feature |
|-------|------|-------------|
| Feast | Open source | Offline + online serving |
| Tecton | Managed | Real-time feature serving |
| Databricks Feature Store | Managed | Unity Catalog integration |
| Hopsworks | Open source | Python-centric feature platform |
| SageMaker Feature Store | AWS | AWS ecosystem integration |

## How Feature Stores Work

\`\`\`python
# Feast example
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo/")

# Define features in Python
# driver_stats.py:
# Entity: driver (with driver_id)
# Features: conv_rate, acc_rate, avg_daily_trips

# Get training data (offline)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_stats:conv_rate", "driver_stats:acc_rate"]
).to_df()

# Get online features (real-time inference)
features = store.get_online_features(
    features=["driver_stats:conv_rate"],
    entity_rows=[{"driver_id": 1001}]
).to_dict()
\`\`\`

## Applications

- Real-time fraud detection (streaming features)
- Recommendation systems (user behavior features)
- Churn prediction (aggregated customer features)
- Credit scoring (financial history features)

## Evolution

- **2015**: Airflow released for workflow orchestration
- **2019**: Feast open-sourced by Gojek and Google
- **2021**: Feature stores become standard MLOps component
- **2023**: dbt + Dagster for analytics engineering in ML
- **2024+**: Real-time feature computation at scale`,

    model_monitoring: `# Model Monitoring & Observability

Model monitoring tracks ML model performance in production, detecting degradation, data drift, and anomalies. It ensures models continue to perform well after deployment.

## What to Monitor

| Category | Metrics | Alert On |
|----------|---------|----------|
| Model Performance | Accuracy, F1, RMSE | Drop below threshold |
| Data Drift | Distribution changes | Significant shift detected |
| Prediction Drift | Output distribution changes | Anomalous patterns |
| Data Quality | Missing values, schema violations | Data pipeline failures |
| Operational | Latency, throughput, errors | SLA violations |

## Types of Drift

\`\`\`
Data Drift (Feature Drift):
- Input data distribution changes
- Example: Customer age distribution shifts
- Detection: KS test, PSI, Jensen-Shannon divergence

Concept Drift:
- Relationship between features and target changes
- Example: COVID changes spending patterns
- Detection: Model performance degradation over time

Prediction Drift:
- Model output distribution changes
- Even without labels, can detect anomalies
- Detection: Statistical tests on prediction distributions
\`\`\`

## Key Tools

| Tool | Type | Focus |
|------|------|-------|
| Evidently AI | Open source | Data and model monitoring dashboards |
| WhyLabs | Managed | ML observability platform |
| Arize AI | Managed | LLM and ML observability |
| NannyML | Open source | Performance estimation without labels |
| Prometheus + Grafana | Open source | Operational metrics |

## Applications

- Detecting model decay in production
- Compliance and audit trails for regulated industries
- A/B testing and model comparison
- Root cause analysis for prediction failures
- LLM quality monitoring (hallucination detection)

## Evolution

- **2019**: ML monitoring recognized as critical MLOps component
- **2021**: Evidently AI and NannyML open-source monitoring tools
- **2023**: LLM monitoring adds new dimensions (hallucination, toxicity)
- **2024**: Observability platforms unify ML and LLM monitoring
- **2025+**: Automated retraining triggered by monitoring alerts`,

    edge_ai: `# Edge AI & TinyML

Edge AI runs ML models directly on edge devices (phones, IoT sensors, embedded systems) rather than in the cloud. TinyML focuses on running ML on microcontrollers with extreme resource constraints.

## Key Concepts

- **On-Device Inference**: Process data locally without cloud connectivity
- **Model Optimization**: Compress models to fit device constraints
- **Latency**: Real-time processing without network round-trips
- **Privacy**: Data never leaves the device
- **Power Efficiency**: Optimize for battery-powered devices

## Optimization Techniques

| Technique | Description | Size Reduction |
|-----------|-------------|---------------|
| Quantization | Reduce precision (FP32 to INT8) | 4x smaller |
| Pruning | Remove unnecessary connections | 2-10x smaller |
| Knowledge Distillation | Train small model from large model | Variable |
| Architecture Search (NAS) | Find efficient architectures | Hardware-specific |
| Weight Sharing | Share weights across layers | 2-4x smaller |

## Frameworks

| Framework | Target Platform |
|-----------|----------------|
| TensorFlow Lite | Mobile (Android/iOS) |
| Core ML | Apple devices |
| ONNX Runtime Mobile | Cross-platform mobile |
| TensorFlow Lite Micro | Microcontrollers |
| Edge Impulse | IoT and embedded |
| NCNN | Mobile (Tencent) |
| MediaPipe | On-device ML pipelines |

## Applications

- **Smartphones**: On-device speech recognition, face unlock
- **IoT**: Anomaly detection in industrial sensors
- **Wearables**: Health monitoring, activity recognition
- **Automotive**: In-vehicle AI for ADAS
- **Agriculture**: Crop disease detection from drone images
- **Smart Home**: Wake word detection ("Hey Siri", "OK Google")

## Evolution

- **2017**: TensorFlow Lite for mobile deployment
- **2019**: TinyML coined; TF Lite Micro for microcontrollers
- **2020**: Apple Neural Engine in every iPhone
- **2022**: Edge AI market reaches $15B+
- **2024+**: On-device LLMs (Gemini Nano, Apple Intelligence)`,

    cloud_ai: `# Cloud AI Services

Major cloud providers offer managed AI/ML services that handle infrastructure, scaling, and deployment. They provide the fastest path from prototype to production.

## Provider Comparison

| Service | AWS | Google Cloud | Azure |
|---------|-----|-------------|-------|
| ML Platform | SageMaker | Vertex AI | Azure ML |
| LLM API | Bedrock | Vertex AI (Gemini) | Azure OpenAI |
| AutoML | SageMaker Autopilot | AutoML | Automated ML |
| GPUs | P4d/P5 (A100/H100) | A100/H100/TPU | ND-series |
| Data | S3, Redshift, Glue | BigQuery, GCS | Blob, Synapse |
| AI APIs | Rekognition, Comprehend | Vision, NLP, Speech | Cognitive Services |

## Key Services

\`\`\`
Training & Experimentation:
- SageMaker Studio / Vertex AI Workbench / Azure ML Studio
- Managed Jupyter notebooks with GPU instances
- Experiment tracking and model registry

Model Deployment:
- SageMaker Endpoints / Vertex AI Endpoints / Azure ML Endpoints
- Auto-scaling, A/B testing, monitoring
- Serverless inference options

LLM Services:
- AWS Bedrock: Access Claude, Llama, Titan models
- Vertex AI: Gemini models + model garden
- Azure OpenAI: GPT-4, DALL-E hosted by Microsoft

Pre-built AI APIs:
- Vision: Image classification, object detection, OCR
- Language: Sentiment, entities, translation, summarization
- Speech: Speech-to-text, text-to-speech
\`\`\`

## Choosing a Provider

| Factor | Recommendation |
|--------|---------------|
| Existing infra | Use your current cloud provider |
| LLM focus | Azure (OpenAI) or Google (Gemini) |
| Research/TPU | Google Cloud (TPU access) |
| Enterprise | AWS (broadest ecosystem) |
| Cost sensitive | Compare spot/preemptible instances |

## Evolution

- **2017**: AWS SageMaker launches
- **2018**: Google Cloud AI Platform (now Vertex AI)
- **2019**: Azure Machine Learning service
- **2023**: LLM-as-a-Service becomes primary cloud AI offering
- **2024+**: All providers offer managed LLM APIs and fine-tuning`,

    model_compression: `# Model Compression & Quantization

Model compression reduces the size and computational cost of ML models while maintaining accuracy. It is essential for deployment on edge devices, reducing serving costs, and improving inference speed.

## Techniques

| Technique | Description | Typical Savings |
|-----------|-------------|----------------|
| Quantization | Reduce numerical precision | 2-4x size, 2-4x speed |
| Pruning | Remove redundant weights | 2-10x size reduction |
| Knowledge Distillation | Train small student from large teacher | 3-10x smaller model |
| Low-Rank Factorization | Decompose weight matrices | 2-5x compression |
| Weight Sharing | Share weights across layers | 2-4x compression |

## Quantization Types

\`\`\`
Precision Levels:
FP32 (32-bit float) -> FP16 (16-bit) -> BF16 -> INT8 (8-bit) -> INT4 (4-bit)

Post-Training Quantization (PTQ):
- Quantize after training (no retraining needed)
- Some accuracy loss, very convenient

Quantization-Aware Training (QAT):
- Simulate quantization during training
- Better accuracy, requires retraining

Dynamic Quantization:
- Quantize weights statically, activations dynamically
- Good for NLP models (LSTM, Transformer)

# PyTorch quantization
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# GPTQ (for LLMs)
# Quantize LLMs to 4-bit with minimal quality loss
# Used by: TheBloke models on HuggingFace
\`\`\`

## LLM Quantization

| Method | Bits | Speed | Quality | Use |
|--------|------|-------|---------|-----|
| FP16/BF16 | 16 | 2x FP32 | Same | Standard training |
| GPTQ | 4 | 3-4x | ~Same | Offline quantization |
| AWQ | 4 | 3-4x | ~Same | Activation-aware |
| GGUF | 2-8 | Variable | Good | llama.cpp format |
| bitsandbytes | 4/8 | 2-4x | Good | HuggingFace integration |

## Applications

- Running LLMs on consumer GPUs (4-bit LLaMA on RTX 3090)
- Mobile deployment (quantized models on phones)
- Reducing cloud inference costs
- Real-time inference in latency-sensitive applications
- Edge AI on microcontrollers

## Evolution

- **2015**: Deep Compression paper (pruning + quantization + Huffman)
- **2019**: Knowledge distillation popularized (DistilBERT)
- **2022**: GPTQ enables 4-bit LLM quantization
- **2023**: AWQ and GGUF formats for efficient LLM deployment
- **2024+**: FP8 training on H100; 2-bit quantization research`,
  };

  Object.assign(window.AI_DOCS, content);
})();
