// 01 - Foundational AI Concepts
(function () {
  const content = {

    linear_algebra: `# Linear Algebra

Linear algebra is the branch of mathematics concerned with **vectors**, **matrices**, and **linear transformations**. It provides the computational backbone for nearly every AI and machine learning algorithm, from simple regression to deep neural networks. Without linear algebra, representing and manipulating high-dimensional data would be practically impossible.

## Key Concepts

- **Vectors** -- ordered arrays of numbers representing points or directions in space. A feature vector encodes the attributes of a data sample.
- **Matrices** -- rectangular arrays used to represent datasets, weights in neural networks, and linear maps between vector spaces.
- **Matrix Multiplication** -- the core operation in neural network forward passes: \`y = Wx + b\`.
- **Eigenvalues & Eigenvectors** -- scalars and vectors satisfying \`Av = lambda v\`. They reveal the principal axes along which a transformation acts.
- **Singular Value Decomposition (SVD)** -- factors any matrix \`A\` into \`U * Sigma * V^T\`, crucial for dimensionality reduction and data compression.
- **Norms** -- measures of vector magnitude (L1, L2, L-infinity) used in regularization and distance calculations.
- **Tensor Operations** -- generalizations of matrices to higher dimensions, fundamental to frameworks like TensorFlow and PyTorch.

## How It Works

At its core, a neural network computes a series of matrix-vector products interleaved with nonlinear activations. An input vector \`x\` is multiplied by a weight matrix \`W\`, a bias \`b\` is added, and the result passes through an activation function. Stacking these operations builds deep networks. Gradient-based training then adjusts \`W\` using the chain rule -- again relying on matrix calculus.

## Applications in AI/ML

- **Principal Component Analysis (PCA)** uses eigen-decomposition to reduce feature dimensions while preserving maximum variance.
- **Word embeddings** (Word2Vec, GloVe) represent words as dense vectors; cosine similarity between vectors captures semantic relationships.
- **Convolutional layers** in CNNs are implemented as sliding matrix multiplications over image patches.
- **Attention mechanisms** in Transformers compute scaled dot-product attention entirely through matrix operations: \`softmax(QK^T / sqrt(d)) V\`.

## Evolution / Timeline

- **1850s** -- Arthur Cayley and James Sylvester formalize matrix algebra.
- **1900s** -- David Hilbert develops the theory of infinite-dimensional vector spaces (Hilbert spaces).
- **1965** -- Gene Golub and William Kahan publish the modern SVD algorithm.
- **1986** -- Backpropagation popularized by Rumelhart, Hinton, and Williams -- matrix calculus enters AI mainstream.
- **2012+** -- GPU-accelerated matrix operations enable training of massive deep networks (AlexNet and beyond).`,


    probability_statistics: `# Probability & Statistics

Probability and statistics form the mathematical language of **uncertainty** and **data-driven inference**. AI systems must reason under uncertainty -- from classifying images with varying confidence to predicting the next word in a sentence. Statistical methods provide the tools to learn patterns from data, quantify confidence, and make principled decisions.

## Key Concepts

- **Random Variables** -- quantities whose values are determined by outcomes of random phenomena (discrete or continuous).
- **Probability Distributions** -- functions describing the likelihood of outcomes. Key distributions include Gaussian (Normal), Bernoulli, Poisson, and Categorical.
- **Bayes' Theorem** -- \`P(A|B) = P(B|A) * P(A) / P(B)\`. The foundation of Bayesian inference: updating beliefs given new evidence.
- **Expectation & Variance** -- \`E[X]\` measures the central tendency; \`Var(X)\` measures spread. Both are critical for loss function analysis.
- **Maximum Likelihood Estimation (MLE)** -- finding parameter values that maximize the probability of observed data.
- **Hypothesis Testing** -- statistical framework for deciding whether observed effects are significant or due to chance.
- **Conditional Independence** -- a simplifying assumption that powers Naive Bayes and graphical models.

## How It Works

Machine learning can be viewed as statistical estimation. Given a dataset, we assume data is drawn from some distribution parameterized by theta. Training finds the theta that best explains the data -- typically via MLE or Maximum A Posteriori (MAP) estimation. Bayesian approaches maintain full posterior distributions \`P(theta|data)\` rather than point estimates, enabling principled uncertainty quantification.

## Applications in AI/ML

- **Naive Bayes classifiers** apply Bayes' theorem with conditional independence assumptions for text classification and spam detection.
- **Gaussian Mixture Models (GMMs)** use probability distributions for clustering and density estimation.
- **Variational Autoencoders (VAEs)** learn latent representations by maximizing a variational lower bound on the data likelihood.
- **Bayesian Neural Networks** place distributions over weights, providing uncertainty estimates alongside predictions.
- **A/B testing** in production ML systems uses hypothesis testing to compare model performance.

## Evolution / Timeline

- **1763** -- Thomas Bayes' theorem published posthumously, founding Bayesian probability.
- **1809** -- Gauss develops the method of least squares and the Normal distribution.
- **1900s** -- Fisher, Neyman, and Pearson formalize frequentist statistics (MLE, hypothesis testing, p-values).
- **1970s** -- Bayesian methods gain traction with advances in computational statistics (MCMC sampling).
- **2010s** -- Probabilistic programming languages (Stan, Pyro, TensorFlow Probability) bring Bayesian ML to practitioners.`,


    optimization: `# Optimization

Optimization is the mathematical discipline of finding the **best solution** from a set of feasible alternatives. In AI and machine learning, optimization is how models learn -- by minimizing a loss function that measures the gap between predictions and ground truth. Virtually every ML training procedure reduces to solving an optimization problem.

## Key Concepts

- **Objective Function** -- the function to be minimized (loss) or maximized (reward). Common losses include Mean Squared Error and Cross-Entropy.
- **Gradient Descent** -- iteratively moving parameters in the direction of steepest descent: \`theta = theta - alpha * grad(L)\`, where \`alpha\` is the learning rate.
- **Stochastic Gradient Descent (SGD)** -- computes gradients on mini-batches rather than the full dataset, trading exactness for speed.
- **Convex vs. Non-Convex** -- convex problems have a single global minimum; neural networks are non-convex with many local minima and saddle points.
- **Constraints** -- optimization subject to equality or inequality constraints, solved via Lagrange multipliers or KKT conditions.
- **Learning Rate Schedules** -- strategies like cosine annealing or warm-up that adjust the step size during training.
- **Momentum & Adaptive Methods** -- techniques like Adam, RMSProp, and AdaGrad that accelerate convergence by adapting per-parameter learning rates.

## How It Works

Training a model involves: (1) forward pass -- compute predictions and loss, (2) backward pass -- compute gradients of the loss with respect to each parameter using backpropagation, and (3) update step -- adjust parameters using an optimizer. Modern optimizers like **Adam** combine momentum (exponential moving average of gradients) with adaptive learning rates (per-parameter scaling based on second moments of gradients). The result is faster, more stable convergence across diverse architectures.

## Applications in AI/ML

- **Neural network training** is fundamentally a large-scale non-convex optimization problem solved with SGD variants.
- **Hyperparameter tuning** uses optimization strategies like Bayesian optimization, grid search, and random search.
- **Support Vector Machines (SVMs)** solve a convex quadratic optimization problem to find the maximum-margin hyperplane.
- **Reinforcement learning** optimizes cumulative reward through policy gradient or value-based methods.
- **Neural Architecture Search (NAS)** uses optimization to discover optimal network topologies.

## Evolution / Timeline

- **1847** -- Cauchy describes the gradient descent method for function minimization.
- **1951** -- Robbins and Monro introduce stochastic approximation, the precursor to SGD.
- **1986** -- Backpropagation enables efficient gradient computation for multi-layer networks.
- **2012** -- Dropout and batch normalization improve optimization stability in deep networks.
- **2014** -- Kingma and Ba publish the **Adam** optimizer, now the most widely used optimizer in deep learning.`,


    information_theory: `# Information Theory

Information theory, founded by **Claude Shannon** in 1948, is the mathematical study of **quantifying, storing, and communicating information**. It provides foundational concepts like entropy and mutual information that underpin modern AI systems -- from designing loss functions to understanding what neural networks learn, to compressing models for deployment.

## Key Concepts

- **Entropy** -- \`H(X) = -sum P(x) log P(x)\`. Measures the average uncertainty or "surprise" in a random variable. Maximum entropy means maximum uncertainty.
- **Cross-Entropy** -- \`H(P, Q) = -sum P(x) log Q(x)\`. Measures the average bits needed when using distribution Q to encode data from distribution P. The standard classification loss in deep learning.
- **KL Divergence** -- \`D_KL(P || Q) = sum P(x) log(P(x)/Q(x))\`. Measures how one distribution diverges from another. Always non-negative; zero only when P equals Q.
- **Mutual Information** -- \`I(X; Y) = H(X) - H(X|Y)\`. Quantifies how much knowing Y reduces uncertainty about X. Used for feature selection and representation learning.
- **Channel Capacity** -- the maximum rate at which information can be transmitted reliably over a noisy channel (Shannon's channel coding theorem).
- **Data Compression** -- encoding data using fewer bits by exploiting statistical redundancy (Huffman coding, arithmetic coding).

## How It Works

In classification, a model outputs a predicted probability distribution Q over classes. The true label defines a one-hot distribution P. Cross-entropy loss \`H(P, Q)\` measures how well Q matches P. Minimizing cross-entropy is equivalent to minimizing KL divergence between the true and predicted distributions (plus a constant). In generative models like VAEs, the loss includes a KL term that regularizes the learned latent distribution toward a prior, ensuring smooth and meaningful latent spaces.

## Applications in AI/ML

- **Cross-entropy loss** is the standard objective for training classification models and language models (next-token prediction).
- **KL Divergence** appears in the VAE loss (ELBO) and in policy optimization methods like PPO (constraining policy updates).
- **Mutual information** is used in representation learning (InfoNCE loss in contrastive learning) and feature selection.
- **Model compression** techniques (pruning, quantization, knowledge distillation) draw on information-theoretic principles.
- **Information Bottleneck** theory provides a framework for understanding what deep networks learn at each layer.

## Evolution / Timeline

- **1948** -- Claude Shannon publishes "A Mathematical Theory of Communication," founding information theory.
- **1951** -- Solomon Kullback and Richard Leibler introduce KL divergence.
- **1972** -- Akaike Information Criterion (AIC) connects information theory to model selection.
- **2015** -- The Information Bottleneck theory is applied to deep learning by Tishby and colleagues.
- **2020s** -- Information-theoretic tools become central to understanding LLM compression, tokenization efficiency, and self-supervised learning objectives.`,


    numerical_methods: `# Numerical Methods

Numerical methods are **computational algorithms** for solving mathematical problems that lack closed-form analytical solutions. In AI and machine learning, nearly every operation -- from training neural networks to inverting matrices to sampling from distributions -- relies on numerical approximation. The efficiency and stability of these methods directly impact model training speed and accuracy.

## Key Concepts

- **Floating-Point Arithmetic** -- computers represent real numbers with finite precision (float32, float16, bfloat16). Understanding rounding errors is critical for numerical stability.
- **Numerical Differentiation** -- approximating derivatives using finite differences: \`f'(x) approx (f(x+h) - f(x-h)) / 2h\`. Backpropagation replaces this with exact automatic differentiation.
- **Automatic Differentiation (AutoDiff)** -- computing exact gradients by applying the chain rule to computational graphs. The backbone of PyTorch and TensorFlow.
- **Iterative Solvers** -- algorithms like Conjugate Gradient and GMRES for solving large linear systems \`Ax = b\` without explicit matrix inversion.
- **Numerical Integration** -- approximating integrals using quadrature rules (trapezoidal, Simpson's) or Monte Carlo sampling.
- **Interpolation & Approximation** -- fitting functions to discrete data points using polynomials, splines, or neural networks.
- **Stability & Conditioning** -- a problem is well-conditioned if small input changes produce small output changes. Ill-conditioned problems amplify numerical errors.

## How It Works

Consider training a neural network. The forward pass evaluates a composition of functions numerically. Automatic differentiation then traverses the computational graph in reverse (reverse-mode AD), computing exact gradients without numerical approximation errors. However, numerical issues still arise: vanishing or exploding gradients, loss of precision in half-precision training, and overflow in softmax computations. Techniques like gradient clipping, mixed-precision training, and the log-sum-exp trick address these problems.

## Applications in AI/ML

- **Automatic differentiation** engines (PyTorch Autograd, TensorFlow GradientTape, JAX) are the workhorse of all gradient-based training.
- **Mixed-precision training** (FP16/BF16 with FP32 master weights) uses numerical method principles to halve memory and accelerate computation.
- **Monte Carlo sampling** methods power variational inference, MCMC for Bayesian models, and policy evaluation in RL.
- **Numerical linear algebra** (LU, QR, Cholesky decomposition) is used in Gaussian processes, PCA, and solving optimization subproblems.
- **ODE/SDE solvers** are central to Neural ODEs and diffusion models (solving the reverse diffusion process).

## Evolution / Timeline

- **1687** -- Newton's method for root-finding published, one of the earliest numerical algorithms.
- **1947** -- John von Neumann pioneers numerical computing on electronic computers.
- **1970** -- The BLAS (Basic Linear Algebra Subprograms) standard enables portable, efficient matrix computation.
- **2015** -- Theano and TensorFlow popularize automatic differentiation for deep learning at scale.
- **2020s** -- Mixed-precision and quantization-aware training become standard; bfloat16 adopted widely for LLM training.`,


    game_theory: `# Game Theory

Game theory is the mathematical study of **strategic interaction** among rational decision-makers. Originally developed for economics, it has become essential in AI for designing multi-agent systems, auction mechanisms, adversarial training, and understanding competitive or cooperative behavior among intelligent agents.

## Key Concepts

- **Players, Strategies, Payoffs** -- the basic elements. Each player chooses a strategy; the combination of all strategies determines each player's payoff.
- **Nash Equilibrium** -- a strategy profile where no player can improve their payoff by unilaterally changing their strategy. A fundamental solution concept.
- **Zero-Sum Games** -- one player's gain is another's loss. Chess, Go, and adversarial ML are examples.
- **Cooperative vs. Non-Cooperative Games** -- cooperative games allow binding agreements between players; non-cooperative games do not.
- **Minimax Theorem** -- in two-player zero-sum games, the optimal strategy minimizes the maximum possible loss.
- **Mechanism Design** -- "reverse game theory" -- designing rules of a game to achieve desired outcomes (e.g., auction design, incentive-compatible AI systems).
- **Evolutionary Game Theory** -- models strategy evolution in populations, applicable to evolutionary algorithms and population-based training.

## How It Works

In a **Generative Adversarial Network (GAN)**, two neural networks play a zero-sum game. The generator G tries to produce realistic data to fool the discriminator D, while D tries to distinguish real from generated data. Training seeks a Nash Equilibrium where G produces data indistinguishable from real data. The minimax objective is: \`min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]\`. Convergence to equilibrium is challenging and has motivated extensive research into training stability.

## Applications in AI/ML

- **GANs** are a direct application of two-player game theory, producing state-of-the-art generative models.
- **Multi-Agent Reinforcement Learning (MARL)** uses game-theoretic concepts to train agents that interact, compete, or cooperate.
- **Adversarial robustness** frames attacks and defenses as a game between an attacker (perturbation) and defender (model).
- **Mechanism design** informs the creation of AI-powered auction systems, matching markets, and incentive structures.
- **AlphaGo and AlphaZero** use self-play (a game-theoretic training paradigm) to achieve superhuman performance.

## Evolution / Timeline

- **1928** -- John von Neumann proves the minimax theorem for zero-sum games.
- **1950** -- John Nash introduces the Nash Equilibrium concept in his PhD thesis.
- **1973** -- John Maynard Smith applies game theory to biology (Evolutionary Stable Strategies).
- **2014** -- Ian Goodfellow introduces GANs, applying game theory to generative modeling.
- **2017** -- AlphaZero masters chess, shogi, and Go through pure self-play, demonstrating game-theoretic AI at its peak.`,


    graph_theory: `# Graph Theory

Graph theory is the study of **graphs** -- mathematical structures consisting of **nodes (vertices)** connected by **edges (links)**. Graphs naturally model relationships, networks, and structures that appear throughout AI: social networks, knowledge graphs, molecular structures, computational graphs, and more. Graph-based algorithms and representations are fundamental to modern AI systems.

## Key Concepts

- **Graphs** -- defined as \`G = (V, E)\` where V is a set of vertices and E is a set of edges. Edges can be directed or undirected, weighted or unweighted.
- **Adjacency Matrix** -- a matrix A where \`A[i][j] = 1\` (or the edge weight) if an edge exists from node i to node j.
- **Degree** -- the number of edges connected to a node. In directed graphs, distinguish in-degree and out-degree.
- **Paths & Connectivity** -- a path is a sequence of edges connecting two nodes. A graph is connected if a path exists between every pair of nodes.
- **Trees** -- connected acyclic graphs. Decision trees and parse trees are AI staples.
- **Graph Traversal** -- algorithms like BFS (Breadth-First Search) and DFS (Depth-First Search) for exploring graph structures.
- **Shortest Path** -- algorithms like Dijkstra's and A* for finding optimal routes in weighted graphs.
- **Graph Coloring & Cliques** -- structural properties used in constraint satisfaction and community detection.

## How It Works

A **Graph Neural Network (GNN)** operates directly on graph-structured data. In each layer, a node aggregates feature vectors from its neighbors (message passing), combines them with its own features, and applies a learned transformation. After K layers, each node's representation captures information from its K-hop neighborhood. The message-passing framework can be expressed as: \`h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({h_u^(k) : u in N(v)}))\`, where \`N(v)\` denotes the neighbors of node v.

## Applications in AI/ML

- **Graph Neural Networks (GNNs)** learn on molecular graphs for drug discovery, social networks for recommendation, and citation networks for classification.
- **Knowledge Graphs** (Wikidata, Google Knowledge Graph) represent factual relationships as directed labeled graphs, powering question-answering systems.
- **Computational Graphs** in TensorFlow and PyTorch represent neural network operations as directed acyclic graphs (DAGs) for automatic differentiation.
- **A* Search** is a graph search algorithm widely used in robotics, pathfinding, and game AI.
- **PageRank** applies graph theory (eigenvector centrality) to rank web pages -- an early triumph of graph-based AI.

## Evolution / Timeline

- **1736** -- Leonhard Euler solves the Konigsberg bridge problem, founding graph theory.
- **1959** -- Dijkstra publishes his shortest-path algorithm.
- **1998** -- Google's PageRank uses graph eigenvalue analysis to revolutionize web search.
- **2017** -- Message Passing Neural Networks (MPNNs) unify GNN architectures under a common framework.
- **2020s** -- GNNs achieve breakthroughs in protein structure prediction (AlphaFold), chip design, and combinatorial optimization.`,


    dynamical_systems: `# Dynamical Systems

A dynamical system is a mathematical model describing how a **system's state evolves over time** according to a fixed rule. The state can represent anything from the position of a robot to the activations in a neural network. Dynamical systems theory provides tools for understanding stability, chaos, convergence, and long-term behavior -- all critical for designing reliable AI systems.

## Key Concepts

- **State Space** -- the set of all possible states of a system. For a neural network, this is the space of all possible weight configurations.
- **Differential Equations** -- ODEs \`dx/dt = f(x, t)\` describe continuous-time dynamics; difference equations \`x_{t+1} = f(x_t)\` describe discrete-time dynamics.
- **Fixed Points & Equilibria** -- states where the system remains unchanged: \`f(x*) = x*\`. Training convergence corresponds to reaching a fixed point of the optimization dynamics.
- **Stability Analysis** -- determines whether perturbations from an equilibrium grow or decay. Lyapunov stability theory is a core tool.
- **Attractors** -- states or sets toward which a system evolves over time. Stable attractors, limit cycles, and strange (chaotic) attractors describe different long-term behaviors.
- **Chaos** -- deterministic systems exhibiting sensitive dependence on initial conditions, making long-term prediction fundamentally difficult.
- **Bifurcations** -- qualitative changes in system behavior as parameters are varied (e.g., a stable system becoming oscillatory).

## How It Works

**Neural ODEs** treat neural network layers as continuous-time dynamical systems. Instead of discrete layers \`h_{t+1} = f(h_t)\`, the hidden state evolves according to an ODE: \`dh/dt = f(h, t, theta)\`. The output is computed by integrating this ODE from an initial to a final time using a numerical ODE solver. Backpropagation through the ODE is done efficiently via the adjoint method, which runs a second ODE backward in time without storing intermediate states.

## Applications in AI/ML

- **Recurrent Neural Networks** are discrete-time dynamical systems; their training dynamics (vanishing/exploding gradients) are analyzed using stability theory.
- **Neural ODEs** provide memory-efficient continuous-depth networks, used for irregular time series, normalizing flows, and physics-informed modeling.
- **Optimization dynamics** -- gradient descent is a dynamical system on the loss landscape. Understanding its trajectories explains convergence and generalization.
- **Reservoir Computing** (Echo State Networks) leverages the dynamics of fixed random recurrent networks for time series prediction.
- **Robotics and control** model physical systems as dynamical systems and use AI for state estimation, prediction, and planning.

## Evolution / Timeline

- **1890** -- Henri Poincare founds qualitative theory of differential equations and discovers chaotic behavior.
- **1963** -- Edward Lorenz discovers the Lorenz attractor, establishing chaos theory.
- **1997** -- Hochreiter and Schmidhuber propose LSTM, solving the vanishing gradient problem in RNN dynamics.
- **2018** -- Chen et al. introduce Neural ODEs, bridging deep learning with continuous dynamical systems.
- **2020s** -- Dynamical systems perspectives inform understanding of diffusion models, Transformer training dynamics, and neural scaling laws.`,


    control_theory: `# Control Theory

Control theory is the engineering discipline concerned with designing systems that **regulate their behavior** to achieve desired outcomes through feedback. It provides the mathematical framework for making systems act intelligently -- adjusting their actions based on the difference between desired and actual states. Control theory deeply connects to reinforcement learning, robotics, and autonomous systems.

## Key Concepts

- **Open-Loop vs. Closed-Loop Control** -- open-loop applies a fixed input; closed-loop (feedback control) adjusts inputs based on observed output errors.
- **PID Controllers** -- Proportional-Integral-Derivative controllers are the most widely used feedback controllers. They compute control signals from the error, its integral, and its derivative.
- **State-Space Representation** -- models systems as \`dx/dt = Ax + Bu\` and \`y = Cx + Du\`, where x is the state, u is input, and y is output.
- **Stability** -- a controlled system is stable if it returns to equilibrium after perturbation. The Routh-Hurwitz criterion and Lyapunov functions assess stability.
- **Controllability & Observability** -- controllability asks whether any state can be reached via appropriate inputs; observability asks whether the internal state can be inferred from outputs.
- **Optimal Control** -- finding control inputs that minimize a cost function over time, solved via the Hamilton-Jacobi-Bellman (HJB) equation or Pontryagin's Maximum Principle.
- **Robust Control** -- designing controllers that perform well despite model uncertainties and disturbances.

## How It Works

Consider a self-driving car. The desired state is a planned trajectory (position, velocity, heading). Sensors measure the actual state. A controller computes steering and throttle commands to minimize the tracking error. In an **LQR (Linear Quadratic Regulator)**, the cost function is quadratic in state deviation and control effort: \`J = integral (x^T Q x + u^T R u) dt\`, and the optimal feedback law is linear: \`u = -Kx\`. Model Predictive Control (MPC) extends this by optimizing over a receding time horizon, handling constraints on states and inputs.

## Applications in AI/ML

- **Reinforcement learning** is closely related to optimal control. The Bellman equation in RL mirrors the HJB equation in control theory.
- **Robotics** uses control theory for trajectory tracking, balancing (inverted pendulum), and manipulation (robotic arms).
- **Autonomous vehicles** combine perception (ML) with control algorithms (MPC, PID) for safe navigation.
- **Training dynamics** -- learning rate schedules and gradient clipping can be viewed as control mechanisms regulating the optimization process.
- **RLHF** for LLMs uses a reward model as a feedback signal to control (steer) the language model's output distribution via PPO.

## Evolution / Timeline

- **1868** -- James Clerk Maxwell analyzes the stability of governors, an early control theory paper.
- **1948** -- Norbert Wiener publishes "Cybernetics," linking control, communication, and computation.
- **1960** -- Rudolf Kalman introduces state-space methods and the Kalman Filter for optimal estimation.
- **1990s** -- Robust and adaptive control methods mature; connections to RL formalized.
- **2020s** -- Control-theoretic perspectives shape RLHF design, safe RL, and autonomous system engineering.`,


    intelligent_agents: `# Intelligent Agents

An **intelligent agent** is an entity that perceives its environment through sensors and acts upon it through actuators to achieve goals. The agent concept is the central abstraction in AI -- providing a unified framework for understanding everything from simple thermostats to sophisticated AI assistants and autonomous robots.

## Key Concepts

- **Agent & Environment** -- the agent interacts with an environment via a perception-action loop. The environment can be fully or partially observable, deterministic or stochastic, static or dynamic.
- **PEAS Description** -- Performance measure, Environment, Actuators, Sensors -- the standard framework for specifying an agent's task.
- **Rationality** -- a rational agent selects actions that maximize expected performance, given its percept sequence and built-in knowledge.
- **Agent Architectures** -- simple reflex agents (condition-action rules), model-based agents (maintain internal state), goal-based agents (plan toward goals), utility-based agents (maximize a utility function), and learning agents (improve from experience).
- **Autonomy** -- the degree to which an agent relies on its own experience versus built-in knowledge.
- **Multi-Agent Systems** -- environments with multiple interacting agents, introducing cooperation, competition, and communication challenges.

## How It Works

A **utility-based agent** maintains a model of the world and evaluates possible actions by estimating the expected utility of their outcomes. Given a current state, the agent: (1) generates possible actions, (2) predicts the resulting states using its world model, (3) evaluates each predicted state using a utility function, and (4) selects the action with the highest expected utility. When the environment is uncertain, the agent must reason about probability distributions over outcomes, combining decision theory with its world model.

## Applications in AI/ML

- **Chatbots and AI assistants** (ChatGPT, Claude) are intelligent agents that perceive user messages and produce helpful responses as actions.
- **Game-playing AI** (AlphaGo, OpenAI Five) are agents that perceive board/game states and select moves to maximize win probability.
- **Autonomous vehicles** are embodied agents perceiving through cameras/LiDAR and acting through steering/acceleration.
- **Recommendation systems** are agents that perceive user behavior and act by selecting content to display.
- **LLM-based agents** (AutoGPT, LangChain agents) use language models as reasoning engines, equipped with tools (search, code execution) to accomplish complex tasks.

## Evolution / Timeline

- **1950** -- Alan Turing proposes the Turing Test, the first formalization of machine intelligence as agent behavior.
- **1995** -- Russell and Norvig structure the AI textbook around the intelligent agent paradigm.
- **1997** -- IBM's Deep Blue defeats Kasparov -- a landmark agent achievement in game playing.
- **2016** -- DeepMind's AlphaGo defeats Lee Sedol, demonstrating superhuman agents in complex domains.
- **2023+** -- LLM-powered autonomous agents (GPT-based agents, Claude tool-use) open new frontiers in agentic AI systems.`,


    search_planning: `# Search & Planning

Search and planning are fundamental AI techniques for **finding sequences of actions** that lead from an initial state to a goal state. Search algorithms systematically explore state spaces to find solutions, while planning extends search with richer representations of actions, states, and goals. Together, they enable AI to solve puzzles, navigate environments, schedule tasks, and reason about complex multi-step problems.

## Key Concepts

- **State Space** -- the set of all possible configurations of the problem. Defined by an initial state, actions (operators), a transition model, and a goal test.
- **Uninformed Search** -- algorithms that explore without domain knowledge: **BFS** (breadth-first), **DFS** (depth-first), **Uniform Cost Search** (expand lowest-cost node first), and **Iterative Deepening**.
- **Informed (Heuristic) Search** -- uses a heuristic function \`h(n)\` estimating the cost to reach the goal. **A*** search: \`f(n) = g(n) + h(n)\` (actual cost + estimated remaining cost). Optimal when \`h\` is admissible (never overestimates).
- **Local Search** -- algorithms like hill climbing, simulated annealing, and genetic algorithms that optimize by moving through neighboring states without keeping the full search tree.
- **Planning** -- uses explicit representations of actions (preconditions and effects) and goals to construct action sequences. STRIPS and PDDL are standard planning languages.
- **Adversarial Search** -- search in competitive environments. **Minimax** with **alpha-beta pruning** is classic for two-player zero-sum games.
- **Monte Carlo Tree Search (MCTS)** -- combines tree search with random simulations to evaluate promising moves; key to AlphaGo's success.

## How It Works

**A* search** maintains a priority queue of nodes ordered by \`f(n) = g(n) + h(n)\`. It expands the node with the lowest f-value, generating successors and updating costs. If the heuristic is admissible and consistent, A* is both complete and optimal -- guaranteed to find the shortest path. For large state spaces, variants like IDA* (Iterative Deepening A*) reduce memory usage, and weighted A* trades optimality for speed.

## Applications in AI/ML

- **Robot navigation** uses A* or RRT (Rapidly-exploring Random Trees) to plan collision-free paths through environments.
- **Game AI** uses minimax with alpha-beta pruning (chess engines) and MCTS (AlphaGo) for move selection.
- **Automated planning** systems solve logistics problems, satellite scheduling, and workflow optimization.
- **Theorem proving** and symbolic AI use search to find proofs in logical state spaces.
- **Code generation** and reasoning in LLMs can be enhanced with tree-of-thought search, exploring multiple reasoning paths.

## Evolution / Timeline

- **1956** -- Newell and Simon create the Logic Theorist, the first AI program using heuristic search.
- **1968** -- Hart, Nilsson, and Raphael publish the A* algorithm.
- **1997** -- Deep Blue uses alpha-beta search with custom hardware to defeat Kasparov in chess.
- **2006** -- Coulom introduces MCTS, later crucial for Go-playing programs.
- **2023+** -- Tree-of-thought prompting and beam search in LLMs bring search methods back to the frontier of AI reasoning.`,


    csp: `# Constraint Satisfaction Problems (CSPs)

A **Constraint Satisfaction Problem** is a mathematical problem defined by a set of variables, each with a domain of possible values, and a set of constraints that restrict which combinations of values are allowed. CSPs provide a powerful framework for modeling and solving a wide variety of AI problems where the goal is to find an assignment that satisfies all constraints simultaneously.

## Key Concepts

- **Variables** -- the unknowns to be assigned values (e.g., colors for nodes in a map, values for cells in Sudoku).
- **Domains** -- the set of possible values for each variable (e.g., \`{Red, Green, Blue}\` for coloring, \`{1..9}\` for Sudoku).
- **Constraints** -- conditions that must be satisfied. Can be unary (one variable), binary (two variables), or higher-order. Examples: "adjacent regions must have different colors" or "all values in a row must be distinct."
- **Constraint Graph** -- nodes are variables, edges connect variables that share a constraint. The structure of this graph drives solution strategies.
- **Backtracking Search** -- depth-first search that assigns values one variable at a time and backtracks when a constraint is violated.
- **Arc Consistency (AC-3)** -- a preprocessing technique that removes values from domains that cannot participate in any consistent assignment.
- **Constraint Propagation** -- using constraints to reduce domains before and during search, dramatically pruning the search space.

## How It Works

The standard approach combines **backtracking search** with **constraint propagation**. At each step: (1) select an unassigned variable (using heuristics like **MRV** -- Minimum Remaining Values), (2) choose a value from its domain (using heuristics like **Least Constraining Value**), (3) propagate constraints to prune neighbor domains (forward checking or AC-3), and (4) if a domain becomes empty, backtrack. Advanced techniques include **conflict-directed backjumping** (jump back to the source of failure) and **nogood learning** (record and reuse reasons for failure).

## Applications in AI/ML

- **Scheduling** -- assigning time slots, rooms, and resources to events while respecting constraints (exam timetabling, job-shop scheduling).
- **Sudoku, crossword puzzles, and configuration** problems are naturally modeled as CSPs.
- **Circuit design and verification** use constraint solving to ensure hardware correctness.
- **Natural language parsing** can be framed as constraint satisfaction over grammatical structures.
- **SAT solvers** -- Boolean Satisfiability is a special case of CSP. Modern SAT solvers are used in AI verification, planning, and formal methods. They underpin tools like Z3.

## Evolution / Timeline

- **1965** -- Golomb and Baumert describe backtracking for constraint problems.
- **1977** -- Mackworth introduces arc consistency algorithms (AC-1, AC-3).
- **1990s** -- CSP solving matures with MAC (Maintaining Arc Consistency), conflict-directed backjumping, and constraint programming languages.
- **2000s** -- SAT/SMT solvers achieve remarkable scalability, solving problems with millions of variables.
- **2020s** -- Neural constraint solving and learning-based heuristics for CSPs emerge, combining ML with combinatorial optimization.`,


    knowledge_representation: `# Knowledge Representation & Reasoning

**Knowledge Representation and Reasoning (KR&R)** is the field of AI concerned with how to formally encode information about the world so that machines can use it to solve complex tasks. The goal is to create structured representations that support automated reasoning, inference, and explanation -- moving beyond pattern recognition toward genuine understanding.

## Key Concepts

- **Ontologies** -- formal specifications of concepts, relationships, and categories in a domain. OWL (Web Ontology Language) is a standard for defining ontologies.
- **Semantic Networks** -- graph structures where nodes represent concepts and edges represent relationships (e.g., "Dog IS-A Animal").
- **Frames** -- data structures grouping related knowledge about an entity with slots for attributes and default values (Minsky, 1975).
- **Knowledge Graphs** -- large-scale directed labeled graphs encoding factual knowledge as (subject, predicate, object) triples. Examples: Wikidata, Google Knowledge Graph.
- **Rules & Production Systems** -- IF-THEN rules that encode domain knowledge for expert systems.
- **Description Logics** -- a family of formal languages for defining ontologies with well-defined inference procedures.
- **Commonsense Knowledge** -- the vast body of everyday knowledge humans use effortlessly but is extremely difficult to formalize (e.g., "water flows downhill").

## How It Works

A **knowledge graph** stores facts as triples: \`(subject, predicate, object)\` -- for example, \`(Paris, capitalOf, France)\`. Queries traverse the graph to answer questions: "What is the capital of France?" becomes a lookup for triples matching \`(?, capitalOf, France)\`. More complex reasoning chains multiple triples: \`(Eiffel Tower, locatedIn, Paris)\` and \`(Paris, capitalOf, France)\` together infer that the Eiffel Tower is in the capital of France. **Knowledge graph embeddings** (TransE, RotatE) learn vector representations of entities and relations, enabling link prediction and approximate reasoning.

## Applications in AI/ML

- **Question answering** systems use knowledge graphs to ground answers in structured facts (Google Search, Siri, Alexa).
- **Retrieval-Augmented Generation (RAG)** combines knowledge retrieval with LLMs for more accurate, grounded responses.
- **Biomedical AI** uses ontologies (SNOMED, Gene Ontology) for drug discovery, disease classification, and clinical decision support.
- **Expert systems** in domains like medicine (MYCIN), geology (PROSPECTOR), and law use rule-based KR for decision making.
- **Neuro-symbolic AI** integrates neural networks with symbolic knowledge representation to achieve both learning and reasoning.

## Evolution / Timeline

- **1969** -- Quillian introduces semantic networks for representing word meanings.
- **1975** -- Minsky proposes frame-based knowledge representation.
- **1980s** -- Expert systems boom: MYCIN, R1/XCON apply KR to real-world domains.
- **2012** -- Google launches the Knowledge Graph, bringing KR to web-scale.
- **2020s** -- Integration of knowledge graphs with LLMs (RAG, knowledge-grounded generation) becomes a major research direction.`,


    logic: `# Logic in AI

**Logic** provides the formal foundation for **reasoning** in artificial intelligence. It defines precise languages for expressing statements about the world and rigorous rules for deriving new truths from existing knowledge. Logic enables AI systems to perform deduction, verify correctness, and reason about complex scenarios in a mathematically sound way.

## Key Concepts

- **Propositional Logic** -- deals with statements (propositions) that are either true or false, combined using connectives: AND (conjunction), OR (disjunction), NOT (negation), IMPLIES (implication), IFF (biconditional).
- **Predicate Logic (First-Order Logic)** -- extends propositional logic with variables, quantifiers (FOR ALL, THERE EXISTS), predicates, and functions. Allows expressing "For all x, if x is human then x is mortal."
- **Inference Rules** -- valid patterns of reasoning: **Modus Ponens** (if P then Q; P; therefore Q), **Resolution** (the basis of automated theorem proving), **Unification** (finding substitutions that make logical expressions identical).
- **Satisfiability** -- a formula is satisfiable if there exists an assignment of truth values making it true. **SAT** is the canonical NP-complete problem.
- **Soundness & Completeness** -- an inference system is sound if it derives only true conclusions; complete if it can derive all true conclusions.
- **Modal Logic** -- extends classical logic with operators for necessity and possibility, used in reasoning about knowledge, belief, and time.
- **Non-Monotonic Reasoning** -- allows conclusions to be retracted when new information arrives, modeling common-sense reasoning more realistically than classical logic.

## How It Works

In **resolution-based theorem proving**, a query is negated and added to the knowledge base. All statements are converted to **Conjunctive Normal Form (CNF)** -- a conjunction of disjunctions (clauses). The resolution rule then repeatedly combines pairs of clauses containing complementary literals: from \`(A OR B)\` and \`(NOT A OR C)\`, derive \`(B OR C)\`. If the empty clause is derived, the original query is proven true (proof by contradiction). This is the foundation of Prolog's execution model and modern SAT solvers.

## Applications in AI/ML

- **Expert systems** use logical rules to encode domain expertise and draw conclusions (medical diagnosis, configuration).
- **Automated theorem proving** verifies mathematical proofs and software correctness (Coq, Lean, Isabelle).
- **SAT and SMT solvers** solve planning, scheduling, and hardware verification problems at industrial scale.
- **Logic programming** (Prolog) enables declarative problem-solving for NLP, databases, and symbolic AI.
- **Neuro-symbolic AI** combines neural networks with logical reasoning, enabling LLMs to perform verified multi-step reasoning.

## Evolution / Timeline

- **1879** -- Gottlob Frege publishes Begriffsschrift, creating modern predicate logic.
- **1930** -- Kurt Godel proves the completeness theorem for first-order logic.
- **1965** -- J. Alan Robinson introduces the resolution principle, enabling automated theorem proving.
- **1972** -- Prolog programming language created, bringing logic programming to AI.
- **2020s** -- SAT/SMT solvers integrated into AI verification; neuro-symbolic methods combine LLMs with formal logic for reliable reasoning.`,


    inference_engines: `# Inference Engines

An **inference engine** is the component of an AI system that applies **logical rules** to a **knowledge base** to derive new information, answer queries, and make decisions. It is the "reasoning machine" that transforms stored knowledge into actionable conclusions. Inference engines power expert systems, rule-based AI, semantic web technologies, and increasingly serve as components within modern AI architectures.

## Key Concepts

- **Knowledge Base (KB)** -- the repository of facts and rules that the inference engine reasons over. Can contain logical rules, ontological definitions, or probabilistic relationships.
- **Forward Chaining (Data-Driven)** -- starts from known facts and repeatedly applies rules to derive new facts until a goal is reached or no more rules fire. Used in production systems and monitoring.
- **Backward Chaining (Goal-Driven)** -- starts from a goal and works backward, finding rules whose conclusions match the goal and recursively trying to satisfy their premises. Used in Prolog and diagnostic systems.
- **Rule Engine** -- a software system that executes business or domain rules against data. The **Rete algorithm** efficiently matches rules to facts by building a network of pattern nodes.
- **Certainty Factors & Probabilistic Inference** -- extensions that handle uncertainty. MYCIN used certainty factors; modern systems use Bayesian networks or probabilistic logic.
- **Conflict Resolution** -- when multiple rules can fire, a strategy determines which to apply (priority, specificity, recency).
- **Explanation Facility** -- a key feature allowing the inference engine to explain its reasoning chain ("I concluded X because of Y and Z").

## How It Works

In a **forward-chaining** inference engine: (1) the engine matches rule conditions against the current working memory (fact base), (2) all matched rules form the **conflict set**, (3) a conflict resolution strategy selects one rule, (4) the selected rule fires, potentially adding new facts to working memory, and (5) the cycle repeats until the goal is satisfied or no rules match. The **Rete algorithm** optimizes step 1 by compiling rules into a discrimination network that incrementally updates matches as facts change, avoiding redundant pattern matching.

## Applications in AI/ML

- **Expert systems** -- MYCIN (medical diagnosis), DENDRAL (chemical analysis), and R1/XCON (computer configuration) all relied on inference engines.
- **Business rule engines** (Drools, IBM ODM) automate complex decision-making in finance, insurance, and compliance.
- **Semantic Web** -- OWL reasoners (Pellet, HermiT) use inference engines to derive implicit knowledge from ontologies.
- **Diagnosis systems** in healthcare and industrial maintenance use backward chaining to trace symptoms to root causes.
- **LLM-augmented reasoning** -- modern systems combine LLMs with symbolic inference engines to achieve reliable, explainable multi-step reasoning over structured knowledge.

## Evolution / Timeline

- **1965** -- Dendral becomes the first expert system, pioneering rule-based inference in AI.
- **1972** -- MYCIN system uses backward chaining with certainty factors for medical diagnosis.
- **1979** -- Charles Forgy introduces the Rete algorithm for efficient rule matching.
- **1980s** -- Expert system boom drives commercial adoption of inference engines (OPS5, CLIPS, Jess).
- **2020s** -- Hybrid architectures combine neural LLMs with symbolic inference engines for grounded, explainable AI reasoning.`,

  };

  Object.assign(window.AI_DOCS, content);
})();
