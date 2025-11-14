// ---- Define AI Documentation Structure ----
const docStructure = [
  {
    title: "1️⃣ Foundational AI Concepts",
    id: "foundational",
    count: 15,
    sub: [
      {
        title: "Mathematical Foundations",
        id: "mathFound",
        items: [
          "Linear Algebra",
          "Probability & Statistics",
          "Optimization",
          "Information Theory",
          "Numerical Methods",
          "Game Theory",
          "Graph Theory",
          "Dynamical Systems",
          "Control Theory",
        ],
      },
      {
        title: "Core AI Principles",
        id: "coreAI",
        items: [
          "Intelligent Agents",
          "Search & Planning",
          "Constraint Satisfaction Problems",
          "Knowledge Representation & Reasoning",
          "Logic",
          "Inference Engines",
        ],
      },
    ],
  },
  {
    title: "🤖 2. Machine Learning (ML)",
    id: "ml",
    count: 20,
    sub: [
      {
        title: "Learning Paradigms",
        id: "learningParadigms",
        items: [
          "Supervised Learning",
          "Unsupervised Learning",
          "Semi-supervised Learning",
          "Self-supervised Learning",
          "Reinforcement Learning",
          "Online Learning",
          "Transfer Learning",
          "Meta Learning",
          "Active Learning",
          "Federated Learning",
          "Continual Learning",
        ],
      },
      {
        title: "Classical ML Algorithms",
        id: "classicalML",
        items: [
          "Linear Regression",
          "Decision Trees",
          "SVM",
          "k-NN",
          "Naive Bayes",
          "Clustering",
          "Dimensionality Reduction",
          "Ensemble Methods",
        ],
      },
    ],
  },
  {
    title: "🧩 3. Neural Networks (NN)",
    id: "nn",
    count: 25,
    sub: [
      {
        title: "Core Architectures",
        id: "nnCore",
        items: [
          "Perceptrons",
          "MLPs",
          "Autoencoders",
          "RNNs",
          "LSTM / GRU",
          "CNNs",
          "GNNs",
          "Attention Networks",
          "Transformers",
        ],
      },
      {
        title: "Training & Optimization",
        id: "nnTrain",
        items: [
          "Backpropagation",
          "Regularization",
          "Optimizers",
          "Loss Functions",
          "Schedulers",
        ],
      },
    ],
  },
  {
    title: "🧮 4. Deep Learning Specializations",
    id: "dl",
    count: 35,
    sub: [
      {
        title: "Computer Vision (CV)",
        id: "cv",
        items: ["Object Detection", "Image Classification", "Image Generation"],
      },
      {
        title: "Natural Language Processing (NLP)",
        id: "nlp",
        items: ["Word Embeddings", "LLMs", "RAG"],
      },
      {
        title: "Speech & Audio AI",
        id: "speech",
        items: ["ASR", "TTS", "Emotion Recognition"],
      },
      {
        title: "Reinforcement Learning",
        id: "rl",
        items: ["Q-Learning", "PPO", "RLHF"],
      },
      {
        title: "Generative AI",
        id: "genAI",
        items: ["GANs", "VAEs", "Diffusion Models"],
      },
    ],
  },
  { title: "🧠 5. Transformers", id: "transformers", count: 10 },
  { title: "🧬 6. Multimodal AI", id: "multi", count: 8 },
  { title: "🌍 7. Applied AI Domains", id: "applied", count: 10 },
  { title: "🧩 8. Cognitive & Hybrid AI", id: "cognitive", count: 8 },
  { title: "🧱 9. Systems & Infrastructure", id: "systems", count: 9 },
  { title: "🧑‍🏫 10. Ethics & Governance", id: "ethics", count: 9 },
  { title: "🔮 11. Emerging Frontiers", id: "emerging", count: 12 },
  { title: "⚙ 12. Supporting Technologies", id: "support", count: 8 },
];

// ---- Render Sidebar ----
const accordion = document.getElementById("aiDocsAccordion");

docStructure.forEach((section, i) => {
  const sec = document.createElement("div");
  sec.className = "accordion-item";

  sec.innerHTML = `
    <h2 class="accordion-header">
      <button class="accordion-button collapsed" data-bs-toggle="collapse" data-bs-target="#collapse${i}">
        ${section.title} <span class="badge bg-secondary ms-2">${section.count}</span>
      </button>
    </h2>
    <div id="collapse${i}" class="accordion-collapse collapse">
      <div class="accordion-body ps-3" id="${section.id}"></div>
    </div>
  `;
  accordion.appendChild(sec);

  if (section.sub) {
    section.sub.forEach((sub) => {
      const subBtn = document.createElement("button");
      subBtn.className = "dropdown-toggle sub-section";
      subBtn.dataset.bsToggle = "collapse";
      subBtn.dataset.bsTarget = `#collapse_${sub.id}`;
      subBtn.textContent = sub.title;

      const ul = document.createElement("ul");
      ul.className = "collapse";
      ul.id = `collapse_${sub.id}`;

      sub.items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        li.onclick = () => loadDoc(section.id, sub.id, item);
        ul.appendChild(li);
      });

      document.getElementById(section.id).append(subBtn, ul);
    });
  }
});

// ---- Search ----
document.getElementById("searchInput").addEventListener("input", (e) => {
  const term = e.target.value.toLowerCase();
  document.querySelectorAll("li").forEach((li) => {
    li.style.display = li.textContent.toLowerCase().includes(term)
      ? ""
      : "none";
  });
});

// ---- Markdown Loader ----
async function loadDoc(section, sub, item) {
  const filePath = `/docs/${section}/${sub}/${item
    .toLowerCase()
    .replace(/[^\w]+/g, "_")}.md`;
  const viewer = document.getElementById("docViewer");

  try {
    const res = await fetch(filePath);
    if (!res.ok) throw new Error("Not found");
    const md = await res.text();
    viewer.innerHTML = marked.parse(md);
  } catch (err) {
    viewer.innerHTML = `<p class="text-danger">⚠️ Could not load document: <b>${item}</b></p>`;
  }
}
