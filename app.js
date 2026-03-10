// ============================================================
// AIfor1 - Main Application Logic
// ============================================================

// Global content store - initialized in index.html <head> before content scripts
// This line keeps backward compatibility if someone loads app.js standalone
window.AI_DOCS = window.AI_DOCS || {};

// ---- SEO Helper Module ----
const SEO = {
  // Base info
  siteName: "AIfor1",
  baseTitle: "AIfor1 - Complete AI Knowledge Base",
  baseDesc: "Comprehensive encyclopedia covering 179+ AI topics: ML, Deep Learning, GenAI, LangChain, Frameworks, and more.",
  baseUrl: location.origin + location.pathname,

  // Section-level keywords for automatic SEO enrichment
  sectionKeywords: {
    foundations: "linear algebra, probability, optimization, information theory, game theory, AI fundamentals",
    ml: "supervised learning, unsupervised learning, regression, decision trees, SVM, clustering, ensemble methods",
    nn: "neural networks, CNN, RNN, LSTM, GRU, backpropagation, activation functions, autoencoders",
    dl: "deep learning, object detection, image classification, NLP, speech recognition, TTS",
    transformers: "transformers, LLMs, BERT, GPT, LLaMA, Claude, attention mechanism, fine-tuning, RLHF",
    genai: "generative AI, GANs, diffusion models, DALL-E, Stable Diffusion, text generation, image synthesis",
    nlp_adv: "NLP, text summarization, machine translation, question answering, sentiment analysis, dialogue systems",
    cv: "computer vision, image preprocessing, object tracking, 3D vision, OCR, medical imaging, autonomous driving",
    rl: "reinforcement learning, Q-learning, policy gradients, DQN, PPO, multi-agent RL, RLHF",
    frameworks: "TensorFlow, PyTorch, JAX, Keras, scikit-learn, HuggingFace, ONNX, MLflow",
    langchain: "LangChain, LangGraph, RAG, vector databases, AI agents, function calling, embeddings",
    languages: "Python, JavaScript, Rust, Julia, R, C++, SQL, Mojo, AI programming",
    mlops: "MLOps, GPU, TPU, model serving, Docker, Kubernetes, distributed training, edge AI, cloud AI",
    ethics: "AI ethics, bias, fairness, explainability, AI safety, data privacy, GDPR, responsible AI",
    emerging: "quantum ML, neuromorphic computing, foundation models, multimodal AI, AGI, robotics, brain-computer interface",
  },

  // Update all SEO meta tags for a given topic
  updateForTopic(topicName, sectionTitle, sectionId, subTitle, topicKey) {
    const title = `${topicName} - ${sectionTitle} | AIfor1`;
    const desc = `Learn about ${topicName} in ${sectionTitle}. Part of AIfor1 comprehensive encyclopedia covering AI, ML, Deep Learning, GenAI, and 179+ topics.`;
    const url = this.baseUrl + "#" + topicKey;
    const keywords = this.sectionKeywords[sectionId] || "";

    // <title>
    document.title = title;

    // meta description
    this._setMeta("description", desc);

    // meta keywords (section-specific + topic name)
    this._setMeta("keywords", `${topicName}, ${sectionTitle}, ${subTitle}, ${keywords}, AI, machine learning, deep learning`);

    // canonical
    const canonical = document.getElementById("canonicalLink");
    if (canonical) canonical.href = url;

    // Open Graph
    this._setMetaProperty("ogTitle", title);
    this._setMetaProperty("ogDescription", desc);
    this._setMetaProperty("ogUrl", url);

    // Twitter
    this._setMetaProperty("twTitle", title);
    this._setMetaProperty("twDescription", desc);

    // JSON-LD Structured Data (TechArticle for each topic)
    this._updateJsonLd({
      "@context": "https://schema.org",
      "@type": "TechArticle",
      "headline": topicName,
      "description": desc,
      "url": url,
      "author": { "@type": "Organization", "name": "AIfor1" },
      "publisher": { "@type": "Organization", "name": "AIfor1" },
      "mainEntityOfPage": { "@type": "WebPage", "@id": url },
      "about": {
        "@type": "Thing",
        "name": sectionTitle,
        "description": `${sectionTitle} section of AIfor1`
      },
      "isPartOf": {
        "@type": "WebSite",
        "name": "AIfor1",
        "url": this.baseUrl
      },
      "breadcrumb": {
        "@type": "BreadcrumbList",
        "itemListElement": [
          { "@type": "ListItem", "position": 1, "name": "AIfor1", "item": this.baseUrl },
          { "@type": "ListItem", "position": 2, "name": sectionTitle, "item": this.baseUrl + "#section-" + sectionId },
          { "@type": "ListItem", "position": 3, "name": subTitle },
          { "@type": "ListItem", "position": 4, "name": topicName, "item": url }
        ]
      }
    });
  },

  // Reset to homepage SEO
  resetToHome() {
    document.title = this.baseTitle;
    this._setMeta("description", this.baseDesc);
    this._setMeta("keywords", "AI, artificial intelligence, machine learning, deep learning, neural networks, transformers, LLMs, generative AI, LangChain, RAG, NLP, computer vision, reinforcement learning, frameworks, PyTorch, TensorFlow, MLOps");
    const canonical = document.getElementById("canonicalLink");
    if (canonical) canonical.href = this.baseUrl;
    this._setMetaProperty("ogTitle", this.baseTitle);
    this._setMetaProperty("ogDescription", this.baseDesc);
    this._setMetaProperty("ogUrl", this.baseUrl);
    this._setMetaProperty("twTitle", this.baseTitle);
    this._setMetaProperty("twDescription", this.baseDesc);
    this._updateJsonLd({
      "@context": "https://schema.org",
      "@type": "WebSite",
      "name": "AIfor1",
      "description": this.baseDesc,
      "url": this.baseUrl
    });
  },

  // Private helpers
  _setMeta(name, content) {
    let el = document.querySelector(`meta[name="${name}"]`);
    if (el) el.content = content;
  },
  _setMetaProperty(id, content) {
    const el = document.getElementById(id);
    if (el) el.content = content;
  },
  _updateJsonLd(data) {
    const el = document.getElementById("jsonLd");
    if (el) el.textContent = JSON.stringify(data, null, 2);
  }
};

// ---- Analytics Tracker Module ----
const Tracker = {
  STORAGE_KEY: "aidocs-analytics",
  _searchTimer: null,

  _defaultData() {
    return {
      version: 1,
      topicViews: {},
      searches: { terms: {}, noResults: {} },
      sessions: { total: 0, lastStart: null },
      firstTracked: new Date().toISOString(),
      lastUpdated: new Date().toISOString(),
    };
  },

  _getData() {
    try {
      const raw = localStorage.getItem(this.STORAGE_KEY);
      if (!raw) return this._defaultData();
      const data = JSON.parse(raw);
      if (!data.version) return this._defaultData();
      return data;
    } catch (e) {
      return this._defaultData();
    }
  },

  _saveData(data) {
    try {
      data.lastUpdated = new Date().toISOString();
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(data));
    } catch (e) { /* localStorage full or unavailable */ }
  },

  trackTopicView(key) {
    const data = this._getData();
    if (!data.topicViews[key]) {
      data.topicViews[key] = { count: 0, lastViewed: null };
    }
    data.topicViews[key].count++;
    data.topicViews[key].lastViewed = new Date().toISOString();
    this._saveData(data);
  },

  trackSearch(term, hasResults) {
    if (!term || term.length < 3) return;
    const data = this._getData();
    const bucket = hasResults ? data.searches.terms : data.searches.noResults;
    const normalized = term.toLowerCase().trim();
    if (!bucket[normalized]) {
      bucket[normalized] = { count: 0, lastSearched: null };
    }
    bucket[normalized].count++;
    bucket[normalized].lastSearched = new Date().toISOString();
    // Prune if too many unique terms (keep top 500)
    const keys = Object.keys(bucket);
    if (keys.length > 500) {
      keys.sort((a, b) => bucket[a].count - bucket[b].count);
      keys.slice(0, keys.length - 500).forEach((k) => delete bucket[k]);
    }
    this._saveData(data);
  },

  trackSearchDebounced(term, hasResults) {
    clearTimeout(this._searchTimer);
    this._searchTimer = setTimeout(() => {
      this.trackSearch(term, hasResults);
    }, 800);
  },

  trackSession() {
    const data = this._getData();
    const now = new Date();
    const last = data.sessions.lastStart ? new Date(data.sessions.lastStart) : null;
    // Only count new session if >30 min since last
    if (!last || now - last > 30 * 60 * 1000) {
      data.sessions.total++;
      data.sessions.lastStart = now.toISOString();
      if (!data.firstTracked) data.firstTracked = now.toISOString();
      this._saveData(data);
    }
  },

  getData() {
    return this._getData();
  },

  clearData() {
    localStorage.removeItem(this.STORAGE_KEY);
  },
};

// ---- Topic Structure Definition ----
const docStructure = [
  {
    title: "Foundational AI Concepts",
    icon: "1",
    id: "foundations",
    sub: [
      {
        title: "Mathematical Foundations",
        id: "math",
        items: [
          { name: "Linear Algebra", key: "linear_algebra" },
          { name: "Probability & Statistics", key: "probability_statistics" },
          { name: "Optimization", key: "optimization" },
          { name: "Information Theory", key: "information_theory" },
          { name: "Numerical Methods", key: "numerical_methods" },
          { name: "Game Theory", key: "game_theory" },
          { name: "Graph Theory", key: "graph_theory" },
          { name: "Dynamical Systems", key: "dynamical_systems" },
          { name: "Control Theory", key: "control_theory" },
        ],
      },
      {
        title: "Core AI Principles",
        id: "core_ai",
        items: [
          { name: "Intelligent Agents", key: "intelligent_agents" },
          { name: "Search & Planning", key: "search_planning" },
          { name: "Constraint Satisfaction", key: "csp" },
          { name: "Knowledge Representation", key: "knowledge_representation" },
          { name: "Logic Systems", key: "logic" },
          { name: "Inference Engines", key: "inference_engines" },
        ],
      },
    ],
  },
  {
    title: "Machine Learning",
    icon: "2",
    id: "ml",
    sub: [
      {
        title: "Learning Paradigms",
        id: "paradigms",
        items: [
          { name: "Supervised Learning", key: "supervised_learning" },
          { name: "Unsupervised Learning", key: "unsupervised_learning" },
          { name: "Semi-supervised Learning", key: "semi_supervised" },
          { name: "Self-supervised Learning", key: "self_supervised" },
          { name: "Reinforcement Learning Intro", key: "rl_intro" },
          { name: "Online Learning", key: "online_learning" },
          { name: "Transfer Learning", key: "transfer_learning" },
          { name: "Meta Learning", key: "meta_learning" },
          { name: "Active Learning", key: "active_learning" },
          { name: "Federated Learning", key: "federated_learning" },
          { name: "Continual Learning", key: "continual_learning" },
        ],
      },
      {
        title: "Classical Algorithms",
        id: "classical",
        items: [
          { name: "Linear & Logistic Regression", key: "regression" },
          { name: "Decision Trees & Random Forests", key: "decision_trees" },
          { name: "Support Vector Machines", key: "svm" },
          { name: "k-Nearest Neighbors", key: "knn" },
          { name: "Naive Bayes", key: "naive_bayes" },
          { name: "Clustering (K-Means, DBSCAN)", key: "clustering" },
          { name: "Dimensionality Reduction", key: "dim_reduction" },
          { name: "Ensemble Methods", key: "ensemble" },
        ],
      },
    ],
  },
  {
    title: "Neural Networks",
    icon: "3",
    id: "nn",
    sub: [
      {
        title: "Core Architectures",
        id: "nn_arch",
        items: [
          { name: "Perceptrons", key: "perceptrons" },
          { name: "Multi-Layer Perceptrons", key: "mlps" },
          { name: "Convolutional Neural Networks", key: "cnns" },
          { name: "Recurrent Neural Networks", key: "rnns" },
          { name: "LSTM & GRU", key: "lstm_gru" },
          { name: "Autoencoders", key: "autoencoders" },
          { name: "Graph Neural Networks", key: "gnns" },
          { name: "Attention Mechanism", key: "attention" },
        ],
      },
      {
        title: "Training & Optimization",
        id: "nn_train",
        items: [
          { name: "Backpropagation", key: "backprop" },
          { name: "Activation Functions", key: "activations" },
          { name: "Loss Functions", key: "loss_functions" },
          { name: "Optimizers (SGD, Adam)", key: "optimizers" },
          { name: "Regularization", key: "regularization" },
          { name: "Learning Rate Scheduling", key: "lr_scheduling" },
        ],
      },
    ],
  },
  {
    title: "Deep Learning Specializations",
    icon: "4",
    id: "dl",
    sub: [
      {
        title: "Vision Tasks",
        id: "dl_cv",
        items: [
          { name: "Object Detection", key: "object_detection" },
          { name: "Image Classification", key: "image_classification" },
          { name: "Semantic Segmentation", key: "segmentation" },
          { name: "Image Generation", key: "image_generation" },
          { name: "Style Transfer", key: "style_transfer" },
        ],
      },
      {
        title: "NLP Foundations",
        id: "dl_nlp",
        items: [
          { name: "Tokenization", key: "tokenization" },
          { name: "Word Embeddings", key: "word_embeddings" },
          { name: "Sequence Models", key: "sequence_models" },
          { name: "Text Classification", key: "text_classification" },
          { name: "Named Entity Recognition", key: "ner" },
        ],
      },
      {
        title: "Speech & Audio",
        id: "dl_speech",
        items: [
          { name: "Speech Recognition (ASR)", key: "asr" },
          { name: "Text-to-Speech (TTS)", key: "tts" },
          { name: "Speaker Recognition", key: "speaker_recognition" },
          { name: "Emotion Recognition", key: "emotion_recognition" },
          { name: "Music Generation", key: "music_generation" },
        ],
      },
    ],
  },
  {
    title: "Transformers & LLMs",
    icon: "5",
    id: "transformers",
    sub: [
      {
        title: "Architecture",
        id: "tx_arch",
        items: [
          { name: "Self-Attention Mechanism", key: "self_attention" },
          { name: "Multi-Head Attention", key: "multi_head_attention" },
          { name: "Positional Encoding", key: "positional_encoding" },
          { name: "Encoder-Decoder Architecture", key: "enc_dec" },
        ],
      },
      {
        title: "Evolution of LLMs",
        id: "tx_llms",
        items: [
          { name: "BERT & Variants", key: "bert" },
          { name: "GPT Series (1 to 4)", key: "gpt_series" },
          { name: "T5 & Seq2Seq Models", key: "t5" },
          { name: "LLaMA & Open Source LLMs", key: "llama" },
          { name: "Claude & Constitutional AI", key: "claude" },
          { name: "PaLM, Gemini & Google Models", key: "palm_gemini" },
          { name: "Mistral & Mixture of Experts", key: "mistral" },
        ],
      },
      {
        title: "LLM Techniques",
        id: "tx_tech",
        items: [
          { name: "Fine-tuning & LoRA", key: "finetuning" },
          { name: "RLHF", key: "rlhf_technique" },
          { name: "Prompt Engineering", key: "prompt_engineering" },
          { name: "Context Windows & Scaling", key: "context_scaling" },
        ],
      },
    ],
  },
  {
    title: "Generative AI",
    icon: "6",
    id: "genai",
    sub: [
      {
        title: "Generative Models",
        id: "gen_models",
        items: [
          { name: "GANs (Generative Adversarial Networks)", key: "gans" },
          { name: "VAEs (Variational Autoencoders)", key: "vaes" },
          { name: "Diffusion Models", key: "diffusion_models" },
          { name: "Flow-based Models", key: "flow_models" },
          { name: "Autoregressive Models", key: "autoregressive" },
        ],
      },
      {
        title: "GenAI Applications",
        id: "gen_apps",
        items: [
          { name: "Text Generation", key: "text_generation" },
          { name: "Image Synthesis", key: "image_synthesis" },
          { name: "Video Generation", key: "video_generation" },
          { name: "Code Generation", key: "code_generation" },
          { name: "Music & Audio Generation", key: "music_gen" },
        ],
      },
      {
        title: "GenAI Ecosystem",
        id: "gen_eco",
        items: [
          { name: "DALL-E & Midjourney Evolution", key: "dalle_midjourney" },
          { name: "Stable Diffusion Deep Dive", key: "stable_diffusion" },
          { name: "AI Art & Creativity", key: "ai_art" },
          { name: "Multimodal Generation", key: "multimodal_gen" },
        ],
      },
    ],
  },
  {
    title: "Advanced NLP",
    icon: "7",
    id: "nlp_adv",
    sub: [
      {
        title: "NLP Tasks",
        id: "nlp_tasks",
        items: [
          { name: "Text Summarization", key: "summarization" },
          { name: "Machine Translation", key: "machine_translation" },
          { name: "Question Answering", key: "question_answering" },
          { name: "Sentiment Analysis", key: "sentiment_analysis" },
          { name: "Text-to-SQL", key: "text_to_sql" },
        ],
      },
      {
        title: "NLP Systems",
        id: "nlp_sys",
        items: [
          { name: "Dialogue Systems", key: "dialogue_systems" },
          { name: "Information Retrieval", key: "information_retrieval" },
          { name: "Document Understanding", key: "document_understanding" },
          { name: "Multilingual NLP", key: "multilingual_nlp" },
          { name: "NLP Benchmarks & Evaluation", key: "nlp_benchmarks" },
        ],
      },
    ],
  },
  {
    title: "Computer Vision",
    icon: "8",
    id: "cv",
    sub: [
      {
        title: "CV Techniques & Tasks",
        id: "cv_tasks",
        items: [
          { name: "Image Preprocessing", key: "image_preprocessing" },
          { name: "Feature Extraction", key: "feature_extraction" },
          { name: "Object Tracking", key: "object_tracking" },
          { name: "3D Vision & Depth", key: "vision_3d" },
          { name: "Video Understanding", key: "video_understanding" },
          { name: "OCR & Document AI", key: "ocr" },
          { name: "Medical Imaging AI", key: "medical_imaging" },
          { name: "Autonomous Driving Vision", key: "autonomous_vision" },
        ],
      },
    ],
  },
  {
    title: "Reinforcement Learning",
    icon: "9",
    id: "rl",
    sub: [
      {
        title: "RL Foundations",
        id: "rl_basics",
        items: [
          { name: "Markov Decision Processes", key: "mdp" },
          { name: "Q-Learning", key: "q_learning" },
          { name: "Policy Gradients", key: "policy_gradients" },
          { name: "Value Functions", key: "value_functions" },
        ],
      },
      {
        title: "Advanced RL",
        id: "rl_advanced",
        items: [
          { name: "Deep Q-Networks (DQN)", key: "dqn" },
          { name: "PPO (Proximal Policy Optimization)", key: "ppo" },
          { name: "A3C / A2C", key: "a3c" },
          { name: "Actor-Critic Methods", key: "actor_critic" },
          { name: "RLHF for LLMs", key: "rlhf" },
          { name: "Multi-Agent RL", key: "multi_agent_rl" },
        ],
      },
    ],
  },
  {
    title: "Frameworks & Tools",
    icon: "10",
    id: "frameworks",
    sub: [
      {
        title: "ML/DL Frameworks",
        id: "fw_ml",
        items: [
          { name: "TensorFlow", key: "tensorflow" },
          { name: "PyTorch", key: "pytorch" },
          { name: "JAX", key: "jax" },
          { name: "Keras", key: "keras" },
          { name: "Scikit-learn", key: "sklearn" },
          { name: "FastAI", key: "fastai" },
        ],
      },
      {
        title: "AI Tooling",
        id: "fw_tools",
        items: [
          { name: "HuggingFace Transformers", key: "huggingface" },
          { name: "ONNX Runtime", key: "onnx" },
          { name: "TensorRT", key: "tensorrt" },
          { name: "Weights & Biases", key: "wandb" },
          { name: "MLflow", key: "mlflow" },
          { name: "Gradio & Streamlit", key: "gradio_streamlit" },
        ],
      },
    ],
  },
  {
    title: "LangChain, Agents & RAG",
    icon: "11",
    id: "langchain",
    sub: [
      {
        title: "LangChain Ecosystem",
        id: "lc_core",
        items: [
          { name: "LangChain Overview", key: "langchain_overview" },
          { name: "LangGraph", key: "langgraph" },
          { name: "Prompt Templates & Chains", key: "prompt_chains" },
          { name: "Memory Systems", key: "memory_systems" },
          { name: "LangSmith & Observability", key: "langsmith" },
        ],
      },
      {
        title: "RAG & Agents",
        id: "lc_rag",
        items: [
          { name: "RAG (Retrieval-Augmented Generation)", key: "rag" },
          { name: "Vector Databases", key: "vector_databases" },
          { name: "Embeddings & Semantic Search", key: "embeddings_search" },
          { name: "AI Agents & Tool Use", key: "ai_agents" },
          { name: "Function Calling", key: "function_calling" },
          { name: "Multi-Agent Frameworks", key: "multi_agent_frameworks" },
        ],
      },
    ],
  },
  {
    title: "Programming Languages for AI",
    icon: "12",
    id: "languages",
    sub: [
      {
        title: "Languages",
        id: "lang_all",
        items: [
          { name: "Python for AI/ML", key: "python_ai" },
          { name: "JavaScript for AI", key: "javascript_ai" },
          { name: "Rust for ML", key: "rust_ml" },
          { name: "Julia for Scientific Computing", key: "julia" },
          { name: "R for Statistical Learning", key: "r_language" },
          { name: "C++ for Performance ML", key: "cpp_ml" },
          { name: "SQL for Data Pipelines", key: "sql_data" },
          { name: "Mojo Language", key: "mojo" },
        ],
      },
    ],
  },
  {
    title: "MLOps & Infrastructure",
    icon: "13",
    id: "mlops",
    sub: [
      {
        title: "Training & Deployment",
        id: "mlops_deploy",
        items: [
          { name: "GPU & TPU Infrastructure", key: "gpu_tpu" },
          { name: "Model Serving & Deployment", key: "model_serving" },
          { name: "Docker & K8s for ML", key: "docker_k8s" },
          { name: "CI/CD for ML Pipelines", key: "cicd_ml" },
          { name: "Distributed Training", key: "distributed_training" },
        ],
      },
      {
        title: "Operations & Edge",
        id: "mlops_ops",
        items: [
          { name: "Data Pipelines & Feature Stores", key: "data_pipelines" },
          { name: "Model Monitoring", key: "model_monitoring" },
          { name: "Edge AI & TinyML", key: "edge_ai" },
          { name: "Cloud AI Services", key: "cloud_ai" },
          { name: "Model Compression & Quantization", key: "model_compression" },
        ],
      },
    ],
  },
  {
    title: "AI Ethics & Governance",
    icon: "14",
    id: "ethics",
    sub: [
      {
        title: "Ethics & Safety",
        id: "ethics_all",
        items: [
          { name: "Bias & Fairness", key: "bias_fairness" },
          { name: "Explainability (XAI)", key: "xai" },
          { name: "AI Safety & Alignment", key: "ai_safety" },
          { name: "Data Privacy & GDPR", key: "data_privacy" },
          { name: "Responsible AI", key: "responsible_ai" },
          { name: "AI Regulation Landscape", key: "ai_regulation" },
          { name: "Deepfakes & Misinformation", key: "deepfakes" },
          { name: "Environmental Impact", key: "environmental_impact" },
        ],
      },
    ],
  },
  {
    title: "Emerging Frontiers",
    icon: "15",
    id: "emerging",
    sub: [
      {
        title: "Next-Gen AI",
        id: "emerging_all",
        items: [
          { name: "Quantum Machine Learning", key: "quantum_ml" },
          { name: "Neuromorphic Computing", key: "neuromorphic" },
          { name: "Foundation Models", key: "foundation_models" },
          { name: "Multimodal AI", key: "multimodal_ai" },
          { name: "World Models", key: "world_models" },
          { name: "AI for Science", key: "ai_science" },
          { name: "Robotics & Embodied AI", key: "robotics_ai" },
          { name: "Brain-Computer Interfaces", key: "bci" },
          { name: "Artificial General Intelligence", key: "agi" },
          { name: "AI & Blockchain", key: "ai_blockchain" },
        ],
      },
    ],
  },
];

// ---- State ----
let currentTopicKey = null;
let openSections = new Set();
let openSubs = new Set();

// ---- DOM References ----
const els = {
  nav: document.getElementById("sidebarNav"),
  container: document.getElementById("docContainer"),
  search: document.getElementById("searchInput"),
  breadcrumb: document.getElementById("breadcrumb"),
  breadcrumbText: document.getElementById("breadcrumbText"),
  sidebar: document.getElementById("sidebar"),
  overlay: document.getElementById("sidebarOverlay"),
  menuToggle: document.getElementById("menuToggle"),
  themeToggle: document.getElementById("themeToggle"),
  themeToggleMobile: document.getElementById("themeToggleMobile"),
  mainContent: document.getElementById("mainContent"),
  backToTop: document.getElementById("backToTop"),
  stats: document.getElementById("topicStats"),
  welcomeGrid: document.getElementById("welcomeGrid"),
};

// ---- Count Topics ----
function countTopics() {
  let total = 0;
  docStructure.forEach((s) =>
    s.sub.forEach((sub) => (total += sub.items.length))
  );
  return total;
}

function countSectionTopics(section) {
  let total = 0;
  section.sub.forEach((sub) => (total += sub.items.length));
  return total;
}

// ---- Render Sidebar ----
function renderSidebar(filter = "") {
  els.nav.innerHTML = "";
  const term = filter.toLowerCase();
  let totalVisible = 0;

  docStructure.forEach((section) => {
    const sectionEl = document.createElement("div");
    sectionEl.className = "nav-section";

    // Filter items
    let hasVisible = false;
    const subEls = [];

    section.sub.forEach((sub) => {
      const filtered = sub.items.filter((item) =>
        item.name.toLowerCase().includes(term) ||
        item.key.toLowerCase().includes(term)
      );

      if (filtered.length === 0 && term) return;

      hasVisible = true;
      totalVisible += filtered.length;

      const subDiv = document.createElement("div");
      subDiv.className = "sub-section";

      const subBtn = document.createElement("button");
      subBtn.className = "sub-btn";
      subBtn.textContent = sub.title;
      subBtn.onclick = () => {
        openSubs.has(sub.id) ? openSubs.delete(sub.id) : openSubs.add(sub.id);
        renderSidebar(filter);
      };

      const ul = document.createElement("ul");
      ul.className = "topic-list";

      if (!openSubs.has(sub.id) && !term) {
        ul.style.display = "none";
      }

      const itemsToShow = term ? filtered : sub.items;
      itemsToShow.forEach((item) => {
        const li = document.createElement("li");
        li.className = "topic-item" + (currentTopicKey === item.key ? " active" : "");
        li.textContent = item.name;
        li.onclick = () => loadTopic(item.key, section, sub, item.name);
        ul.appendChild(li);
      });

      subDiv.appendChild(subBtn);
      subDiv.appendChild(ul);
      subEls.push(subDiv);
    });

    if (!hasVisible && term) return;

    // Section button
    const count = countSectionTopics(section);
    const btn = document.createElement("button");
    btn.className = "section-btn" + (openSections.has(section.id) ? " active" : "");
    btn.innerHTML = `<span class="arrow">&#9654;</span> ${section.icon}. ${section.title} <span class="badge">${count}</span>`;
    btn.onclick = () => {
      if (openSections.has(section.id)) {
        openSections.delete(section.id);
      } else {
        openSections.add(section.id);
      }
      renderSidebar(filter);
    };

    const contentDiv = document.createElement("div");
    if (!openSections.has(section.id) && !term) {
      contentDiv.style.display = "none";
    }
    subEls.forEach((el) => contentDiv.appendChild(el));

    sectionEl.appendChild(btn);
    sectionEl.appendChild(contentDiv);
    els.nav.appendChild(sectionEl);
  });

  if (totalVisible === 0 && term) {
    const noRes = document.createElement("div");
    noRes.className = "no-results";
    noRes.textContent = "No topics found for \"" + filter + "\"";
    els.nav.appendChild(noRes);
  }
}

// ---- Load Topic ----
function loadTopic(key, section, sub, name) {
  currentTopicKey = key;
  Tracker.trackTopicView(key);
  const content = window.AI_DOCS[key];

  if (content) {
    els.container.innerHTML = "";
    els.container.className = "markdown-body";
    els.container.innerHTML = marked.parse(content);

    // Add heading IDs for in-page anchors (SEO + accessibility)
    els.container.querySelectorAll("h1, h2, h3, h4").forEach((h) => {
      const id = h.textContent.trim().toLowerCase()
        .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      h.id = id;
    });
  } else {
    els.container.className = "";
    els.container.innerHTML = `<div class="welcome-screen"><h2>Coming Soon</h2><p>${name} documentation is being prepared.</p></div>`;
  }

  // Breadcrumb
  els.breadcrumb.style.display = "block";
  els.breadcrumbText.innerHTML =
    `${section.icon}. ${section.title} <span class="sep">/</span> ${sub.title} <span class="sep">/</span> <strong>${name}</strong>`;

  // Scroll to top
  els.mainContent.scrollTop = 0;

  // Update URL hash
  history.replaceState(null, "", "#" + key);

  // Update SEO meta tags for this topic
  SEO.updateForTopic(name, section.title, section.id, sub.title, key);

  // Re-render sidebar for active state
  renderSidebar(els.search.value);

  // Close mobile sidebar
  closeMobileSidebar();
}

// ---- Search ----
els.search.addEventListener("input", (e) => {
  const term = e.target.value.trim();
  // When searching, open all sections and subs
  if (term) {
    docStructure.forEach((s) => {
      openSections.add(s.id);
      s.sub.forEach((sub) => openSubs.add(sub.id));
    });
  }
  renderSidebar(term);

  // Track search analytics (debounced, 3+ char only)
  if (term.length >= 3) {
    const hasResults = !document.querySelector(".no-results");
    Tracker.trackSearchDebounced(term, hasResults);
  }
});

// ---- Theme Toggle ----
function setTheme(dark) {
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  localStorage.setItem("ai-docs-theme", dark ? "dark" : "light");
  const icon = dark ? "&#9788;" : "&#9790;";
  document.querySelectorAll(".theme-icon").forEach((el) => (el.innerHTML = icon));
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  setTheme(!isDark);
}

els.themeToggle.onclick = toggleTheme;
els.themeToggleMobile.onclick = toggleTheme;

// Init theme
const savedTheme = localStorage.getItem("ai-docs-theme");
if (savedTheme === "dark") setTheme(true);

// ---- Mobile Sidebar ----
function openMobileSidebar() {
  els.sidebar.classList.add("open");
  els.overlay.classList.add("active");
}

function closeMobileSidebar() {
  els.sidebar.classList.remove("open");
  els.overlay.classList.remove("active");
}

els.menuToggle.onclick = () => {
  els.sidebar.classList.contains("open") ? closeMobileSidebar() : openMobileSidebar();
};
els.overlay.onclick = closeMobileSidebar;

// ---- Back to Top ----
els.mainContent.addEventListener("scroll", () => {
  els.backToTop.classList.toggle("visible", els.mainContent.scrollTop > 300);
});

els.backToTop.onclick = () => {
  els.mainContent.scrollTo({ top: 0, behavior: "smooth" });
};

// ---- Hash Navigation ----
function loadFromHash() {
  const hash = location.hash.slice(1);
  if (!hash) return;

  for (const section of docStructure) {
    for (const sub of section.sub) {
      for (const item of sub.items) {
        if (item.key === hash) {
          openSections.add(section.id);
          openSubs.add(sub.id);
          loadTopic(item.key, section, sub, item.name);
          return;
        }
      }
    }
  }
}

// ---- Welcome Grid ----
function renderWelcomeGrid() {
  const icons = ["🧮","🤖","🧠","🔬","🤯","🎨","📝","👁","🎮","🔧","🔗","💻","🏗","⚖","🚀"];
  docStructure.forEach((section, i) => {
    const card = document.createElement("div");
    card.className = "welcome-card";
    card.innerHTML = `
      <div class="card-icon">${icons[i] || "📖"}</div>
      <h3>${section.title}</h3>
      <p>${countSectionTopics(section)} topics</p>
    `;
    card.onclick = () => {
      openSections.add(section.id);
      section.sub.forEach((s) => openSubs.add(s.id));
      renderSidebar(els.search.value);
      // Scroll sidebar to section
    };
    els.welcomeGrid.appendChild(card);
  });
}

// ---- Stats ----
function renderStats() {
  const total = countTopics();
  const sections = docStructure.length;
  els.stats.innerHTML = `<span>${total} topics</span> <span>${sections} sections</span>`;
}

// ---- Initialize ----
renderSidebar();
renderWelcomeGrid();
renderStats();
SEO.resetToHome();
Tracker.trackSession();
loadFromHash();
window.addEventListener("hashchange", loadFromHash);
