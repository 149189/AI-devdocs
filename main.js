const topics = [
  { name: "Linear Algebra", file: "linear_algebra.md" },
  { name: "Probability & Statistics", file: "probability_statistics.md" },
  { name: "Optimization", file: "optimization.md" },
  { name: "Information Theory", file: "information_theory.md" },
  { name: "Numerical Methods", file: "numerical_methods.md" },
  { name: "Game Theory", file: "game_theory.md" },
  { name: "Graph Theory", file: "graph_theory.md" },
  { name: "Dynamical Systems", file: "dynamical_systems.md" },
  { name: "Control Theory", file: "control_theory.md" },
];

const foundationList = document.getElementById("foundationList");
const searchInput = document.getElementById("searchInput");
const docContainer = document.getElementById("docContainer");

// Render list items in dropdown
function renderTopics(filtered = topics) {
  foundationList.innerHTML = "";
  filtered.forEach((topic) => {
    const li = document.createElement("li");
    li.classList.add("list-group-item");
    li.textContent = topic.name;
    li.onclick = () => loadMarkdown(topic.file, li);
    foundationList.appendChild(li);
  });
}

// Load markdown and render it
async function loadMarkdown(file, element) {
  document
    .querySelectorAll(".list-group-item")
    .forEach((li) => li.classList.remove("active"));
  element.classList.add("active");

  const path = `./docs/mathematical_foundations/${file}`;
  try {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Failed to load ${file}`);
    const text = await res.text();
    const html = marked.parse(text);
    docContainer.innerHTML = html;
  } catch (err) {
    docContainer.innerHTML = `<p class="text-danger">Error: ${err.message}</p>`;
  }
}

// Search filtering
searchInput.addEventListener("input", (e) => {
  const query = e.target.value.toLowerCase();
  const filtered = topics.filter((t) => t.name.toLowerCase().includes(query));
  renderTopics(filtered);
});

// Initial render
renderTopics();
