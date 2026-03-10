// 09 - Reinforcement Learning
(function () {
  const content = {
    mdp: `# Markov Decision Processes (MDP)

A Markov Decision Process is the mathematical framework underlying reinforcement learning. It formalizes sequential decision-making where outcomes are partly random and partly under agent control.

## Key Components

| Component | Symbol | Description |
|-----------|--------|-------------|
| States | S | All possible situations the agent can be in |
| Actions | A | All possible decisions the agent can make |
| Transition | T(s'|s,a) | Probability of reaching s' from s via action a |
| Reward | R(s,a,s') | Immediate reward for a transition |
| Discount | gamma | Factor (0-1) weighting future vs immediate rewards |
| Policy | pi(a|s) | Agent's strategy mapping states to actions |

## Key Concepts

- **Markov Property**: Future depends only on current state, not history
- **Return**: Cumulative discounted reward G_t = R_t + gamma*R_{t+1} + gamma^2*R_{t+2} + ...
- **Value Function V(s)**: Expected return starting from state s under policy pi
- **Action-Value Q(s,a)**: Expected return taking action a in state s, then following pi
- **Bellman Equation**: Recursive relationship defining optimal values

## How It Works

\`\`\`
# Bellman Optimality Equation
V*(s) = max_a Sum_s'[ T(s'|s,a) * (R(s,a,s') + gamma * V*(s')) ]

Q*(s,a) = Sum_s'[ T(s'|s,a) * (R(s,a,s') + gamma * max_a' Q*(s',a')) ]

# Value Iteration Algorithm
Initialize V(s) = 0 for all s
Repeat until convergence:
  For each state s:
    V(s) = max_a Sum_s'[ T(s'|s,a) * (R + gamma * V(s')) ]

# Policy Iteration
1. Policy Evaluation: Compute V^pi for current policy
2. Policy Improvement: Update policy greedily w.r.t. V^pi
3. Repeat until policy is stable
\`\`\`

## Applications

- Game playing (Chess, Go, Atari)
- Robotics control
- Resource management
- Traffic signal optimization
- Clinical trial design

## Evolution

- **1957**: Bellman publishes dynamic programming and MDPs
- **1960s**: Howard introduces policy iteration
- **1989**: Watkins introduces Q-learning for model-free MDPs
- **2013**: DQN combines deep learning with MDPs for Atari
- **2020s**: MDPs extended with partial observability, multi-agent settings`,

    q_learning: `# Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that learns the optimal action-value function Q*(s,a) without knowing the environment's transition dynamics.

## Key Concepts

- **Q-Table**: Stores Q-values for every (state, action) pair
- **Exploration vs Exploitation**: Balance trying new actions vs using known good ones
- **Epsilon-Greedy**: With probability epsilon, explore randomly; otherwise exploit best Q
- **Learning Rate (alpha)**: How much to update Q-values per step
- **Temporal Difference (TD)**: Learning from the difference between predicted and actual returns

## How It Works

\`\`\`python
# Q-Learning Algorithm
import numpy as np

Q = np.zeros((num_states, num_actions))
alpha = 0.1   # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state])         # exploit

        next_state, reward, done, _ = env.step(action)

        # Q-Learning update (off-policy)
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (
            reward + gamma * best_next - Q[state, action]
        )
        state = next_state
\`\`\`

## Properties

| Property | Description |
|----------|-------------|
| Model-free | No need to know transition probabilities |
| Off-policy | Learns optimal policy while following exploratory policy |
| Tabular | Stores values for each state-action pair |
| Convergent | Guaranteed to converge to Q* with sufficient exploration |

## Limitations

- Requires discrete state and action spaces (tabular)
- Does not scale to large/continuous state spaces
- Slow convergence in complex environments
- Solution: Deep Q-Networks (DQN) for function approximation

## Evolution

- **1989**: Watkins proposes Q-learning in his PhD thesis
- **1992**: Convergence proof published (Watkins & Dayan)
- **1995**: SARSA variant (on-policy alternative) introduced
- **2013**: DQN extends Q-learning with neural networks
- **2015**: Double DQN and Dueling DQN improve stability`,

    policy_gradients: `# Policy Gradients

Policy Gradient methods directly optimize the policy (mapping from states to actions) by computing gradients of expected return with respect to policy parameters.

## Key Concepts

- **Parameterized Policy**: pi_theta(a|s) - a neural network that outputs action probabilities
- **Objective**: Maximize expected return J(theta) = E[Sum of rewards]
- **Policy Gradient Theorem**: Gradient of J with respect to theta
- **REINFORCE**: Monte Carlo policy gradient algorithm
- **Baseline**: Subtracting a baseline to reduce variance

## How It Works

\`\`\`python
# REINFORCE Algorithm
# Policy gradient theorem: nabla J = E[Sum_t nabla log pi(a_t|s_t) * G_t]

import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# Training loop
for episode in range(num_episodes):
    states, actions, rewards = collect_episode(env, policy)
    returns = compute_returns(rewards, gamma=0.99)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

    # Compute policy gradient loss
    log_probs = torch.log(policy(states).gather(1, actions))
    loss = -(log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
\`\`\`

## Advantages vs Q-Learning

| Aspect | Policy Gradients | Q-Learning |
|--------|-----------------|------------|
| Action space | Continuous or discrete | Discrete only |
| Policy type | Stochastic | Deterministic (greedy) |
| Convergence | Local optimum | Global optimum (tabular) |
| Variance | High (needs variance reduction) | Lower |
| Sample efficiency | Lower | Higher |

## Applications

- Continuous control (robotics, locomotion)
- Game playing with large action spaces
- NLP (text generation with RL rewards)
- RLHF for language model alignment

## Evolution

- **1992**: Williams introduces REINFORCE algorithm
- **2000**: Sutton et al. prove policy gradient theorem
- **2015**: Trust Region Policy Optimization (TRPO) stabilizes training
- **2017**: PPO simplifies TRPO with clipped objective
- **2022**: RLHF uses policy gradients to align LLMs`,

    value_functions: `# Value Functions

Value Functions estimate how good it is for an agent to be in a given state (or take a given action in a state). They are central to both planning and learning in RL.

## Types

| Function | Definition | Description |
|----------|-----------|-------------|
| V^pi(s) | E[G_t \\| s_t = s, pi] | Expected return from state s following policy pi |
| Q^pi(s,a) | E[G_t \\| s_t = s, a_t = a, pi] | Expected return from taking a in state s |
| V*(s) | max_pi V^pi(s) | Optimal state-value function |
| Q*(s,a) | max_pi Q^pi(s,a) | Optimal action-value function |
| A^pi(s,a) | Q^pi(s,a) - V^pi(s) | Advantage function |

## Bellman Equations

\`\`\`
# State-value Bellman equation
V^pi(s) = Sum_a pi(a|s) * Sum_s' T(s'|s,a) * [R(s,a,s') + gamma * V^pi(s')]

# Bellman optimality (value iteration)
V*(s) = max_a Sum_s' T(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

# TD Learning (estimate value from experience)
V(s) = V(s) + alpha * (r + gamma * V(s') - V(s))
         ^current    ^--------TD target------^  ^TD error
\`\`\`

## Estimation Methods

| Method | Approach | Bias/Variance |
|--------|----------|---------------|
| Monte Carlo | Full episode returns | Unbiased, high variance |
| TD(0) | One-step bootstrap | Some bias, lower variance |
| TD(lambda) | Multi-step blend | Tunable bias-variance |
| GAE | Generalized Advantage | Practical for policy gradients |

## Key Concepts

- **Bootstrapping**: Estimating value using other value estimates
- **TD Error**: delta = r + gamma * V(s') - V(s) (prediction error)
- **Advantage Function**: How much better an action is than average
- **GAE (Generalized Advantage Estimation)**: Weighted average of multi-step advantages

## Applications

- Value iteration for solving MDPs
- Critic in Actor-Critic methods
- Advantage estimation for policy gradients
- Baseline for REINFORCE variance reduction

## Evolution

- **1957**: Bellman defines value functions and dynamic programming
- **1988**: Sutton introduces TD(lambda) learning
- **1995**: TD-Gammon uses value function approximation for backgammon
- **2015**: GAE introduced for practical advantage estimation
- **2020s**: Value functions remain core to most RL algorithms`,

    dqn: `# Deep Q-Networks (DQN)

Deep Q-Networks combine Q-learning with deep neural networks, enabling RL agents to learn directly from high-dimensional inputs like raw pixels. DQN was the breakthrough that launched deep reinforcement learning.

## Key Innovations

- **Neural Q-Function**: Replace Q-table with a neural network Q(s,a;theta)
- **Experience Replay**: Store transitions in a buffer, sample random mini-batches
- **Target Network**: Separate network for stable Q-value targets
- **Frame Stacking**: Use last 4 frames as input for temporal context

## How It Works

\`\`\`python
# DQN Architecture (Atari)
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

# DQN Training Loop
# 1. Observe state s, select action (epsilon-greedy on Q-network)
# 2. Execute action, observe reward r and next state s'
# 3. Store (s, a, r, s') in replay buffer
# 4. Sample mini-batch from buffer
# 5. Compute target: y = r + gamma * max_a' Q_target(s', a')
# 6. Update Q-network: minimize (Q(s,a) - y)^2
# 7. Periodically copy Q-network weights to target network
\`\`\`

## DQN Improvements

| Variant | Innovation | Benefit |
|---------|-----------|---------|
| Double DQN | Decouple action selection and evaluation | Reduces overestimation |
| Dueling DQN | Separate value and advantage streams | Better value estimation |
| Prioritized Replay | Sample important transitions more | Faster learning |
| Rainbow | Combines 6 improvements | State-of-the-art Atari |
| Distributional DQN | Predict return distribution | Risk-aware decisions |

## Applications

- Atari game playing (superhuman on many games)
- Robot control from visual input
- Network routing optimization
- Game AI (tested on 57 Atari 2600 games)

## Evolution

- **2013**: Mnih et al. introduce DQN (arXiv preprint)
- **2015**: DQN published in Nature (superhuman Atari performance)
- **2015**: Double DQN fixes overestimation bias
- **2016**: Dueling DQN and Prioritized Experience Replay
- **2017**: Rainbow DQN combines all improvements`,

    ppo: `# PPO (Proximal Policy Optimization)

PPO is the most widely used policy gradient algorithm. It provides stable, reliable training by constraining policy updates to stay close to the previous policy, avoiding destructively large updates.

## Key Concepts

- **Trust Region**: Limit how much the policy changes per update
- **Clipped Objective**: Simpler alternative to TRPO's KL constraint
- **Ratio**: r(theta) = pi_new(a|s) / pi_old(a|s) measures policy change
- **Clip Range (epsilon)**: Typically 0.1-0.3, limits the ratio
- **Multiple Epochs**: Reuses collected data for several gradient steps

## How It Works

\`\`\`python
# PPO Clipped Objective
# L_CLIP = E[ min(r * A, clip(r, 1-eps, 1+eps) * A) ]

def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

# PPO Training Loop:
# 1. Collect trajectories using current policy
# 2. Compute advantages (GAE)
# 3. For K epochs (typically 3-10):
#    a. Compute new log_probs
#    b. Compute clipped PPO loss
#    c. Add value function loss and entropy bonus
#    d. Gradient step
# 4. Repeat with new trajectories
\`\`\`

## PPO vs Other Methods

| Method | Stability | Sample Efficiency | Complexity |
|--------|-----------|-------------------|-----------|
| REINFORCE | Low | Low | Simple |
| TRPO | High | Medium | Complex (KL constraint) |
| PPO | High | Medium | Simple (clipping) |
| SAC | High | High | Medium (off-policy) |

## Applications

- OpenAI Five (Dota 2 at professional level)
- Robot locomotion and manipulation
- RLHF for ChatGPT/Claude alignment
- Game AI across many domains
- Autonomous vehicle decision-making

## Evolution

- **2015**: TRPO introduces trust region concept
- **2017**: Schulman introduces PPO at OpenAI (simpler than TRPO)
- **2019**: OpenAI Five uses PPO to defeat Dota 2 world champions
- **2022**: PPO is the RL backbone of RLHF for ChatGPT
- **2024+**: PPO remains the go-to algorithm for LLM alignment and robotics`,

    a3c: `# A3C / A2C (Asynchronous Advantage Actor-Critic)

A3C runs multiple agents in parallel environments, each computing gradients independently and updating a shared model. A2C is the synchronous variant that collects data in parallel but updates synchronously.

## Key Concepts

- **Asynchronous**: Multiple agents explore simultaneously in separate environment copies
- **Advantage**: Uses advantage function A(s,a) = Q(s,a) - V(s) instead of raw returns
- **Actor**: Policy network that selects actions
- **Critic**: Value network that evaluates states
- **Shared Parameters**: All workers update one global model

## How It Works

\`\`\`
A3C Architecture:
- Global shared network (policy + value head)
- N worker threads, each with own environment copy

Each Worker:
1. Copy global parameters to local network
2. Collect n-step trajectory in local environment
3. Compute advantages: A_t = R_t - V(s_t)
4. Compute gradients of policy loss + value loss
5. Apply gradients to global network (async)

A2C (Synchronous):
1. All workers collect trajectories in parallel
2. Aggregate all gradients
3. Single synchronized update to model
4. Repeat

# A2C is generally preferred: same performance, simpler debugging
\`\`\`

## A3C vs A2C

| Aspect | A3C | A2C |
|--------|-----|-----|
| Updates | Asynchronous (each worker) | Synchronous (batched) |
| Stability | Can have stale gradients | More stable updates |
| GPU Utilization | Poor (CPU-based) | Better (batched on GPU) |
| Implementation | Complex (threading) | Simpler (vectorized envs) |
| In practice | Mostly replaced | Still used |

## Applications

- Atari game playing
- 3D navigation tasks (ViZDoom, DeepMind Lab)
- Real-time strategy games
- Continuous control tasks

## Evolution

- **2016**: Mnih et al. introduce A3C at DeepMind
- **2017**: A2C shown to match A3C performance with simpler implementation
- **2017**: PPO largely supersedes A3C/A2C for most applications
- **2018**: IMPALA scales actor-learner to thousands of workers
- **2020s**: Distributed RL frameworks (RLlib, Sample Factory) build on these ideas`,

    actor_critic: `# Actor-Critic Methods

Actor-Critic methods combine policy-based (actor) and value-based (critic) approaches. The actor selects actions while the critic evaluates how good those actions are, providing lower-variance learning signals.

## Key Concepts

- **Actor**: Policy network pi(a|s; theta) that outputs action distribution
- **Critic**: Value network V(s; w) or Q(s,a; w) that evaluates states/actions
- **Advantage**: A(s,a) = Q(s,a) - V(s) measures how much better action a is vs average
- **TD Error**: delta = r + gamma*V(s') - V(s) used as advantage estimate
- **Entropy Bonus**: Encourages exploration by penalizing deterministic policies

## How It Works

\`\`\`python
# Simple Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.actor = nn.Linear(256, action_dim)   # policy head
        self.critic = nn.Linear(256, 1)            # value head

    def forward(self, state):
        features = self.shared(state)
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value

# Training:
# 1. Sample action from actor: a ~ pi(.|s)
# 2. Get reward r and next state s'
# 3. Compute TD error: delta = r + gamma*V(s') - V(s)
# 4. Update critic: minimize delta^2
# 5. Update actor: policy_loss = -log(pi(a|s)) * delta.detach()
\`\`\`

## Variants

| Algorithm | Key Idea |
|-----------|----------|
| A2C/A3C | Advantage actor-critic, parallel environments |
| PPO | Clipped policy updates for stability |
| SAC | Maximum entropy objective for exploration |
| TD3 | Twin critics to reduce overestimation |
| DDPG | Deterministic policy for continuous actions |

## Applications

- Continuous control (MuJoCo, robotics)
- Game playing (Atari, board games)
- RLHF for language model training
- Resource scheduling and optimization

## Evolution

- **1999**: Konda & Tsitsiklis formalize actor-critic convergence
- **2015**: A3C and DDPG popularize deep actor-critic
- **2017**: PPO and SAC become dominant actor-critic variants
- **2018**: TD3 fixes overestimation in continuous control
- **2022+**: Actor-critic is the backbone of RLHF (PPO for LLM alignment)`,

    rlhf: `# RLHF (Reinforcement Learning from Human Feedback)

RLHF aligns AI models with human preferences by training a reward model on human feedback, then optimizing the AI policy using reinforcement learning. It is the key technique behind ChatGPT, Claude, and other aligned LLMs.

## Key Concepts

- **Human Preference Data**: Humans rank model outputs (e.g., response A > response B)
- **Reward Model**: Neural network trained to predict human preferences
- **Policy Optimization**: Fine-tune LLM to maximize reward model scores
- **KL Penalty**: Prevent policy from diverging too far from base model
- **Constitutional AI**: Anthropic's approach using AI-generated feedback

## How It Works

\`\`\`
RLHF Pipeline (3 stages):

Stage 1: Supervised Fine-Tuning (SFT)
- Fine-tune base LLM on high-quality demonstration data
- Result: SFT model that follows instructions

Stage 2: Reward Model Training
- Collect comparison data: humans rank model outputs
- Train reward model: R(prompt, response) -> scalar score
- Loss: maximize log-probability that preferred response scores higher

Stage 3: RL Optimization (PPO)
- Generate responses from current policy
- Score with reward model
- Update policy with PPO to maximize reward
- Add KL penalty: reward - beta * KL(policy || SFT_model)
\`\`\`

## Alternatives to RLHF

| Method | Description | Advantage |
|--------|-------------|-----------|
| RLHF (PPO) | Full RL optimization | Most flexible, proven at scale |
| DPO | Direct Preference Optimization | No reward model needed, simpler |
| RLAIF | RL from AI Feedback | Scalable, less human annotation |
| Constitutional AI | Self-critique and revision | Anthropic's approach for Claude |
| KTO | Kahneman-Tversky Optimization | Works with binary feedback |

## Applications

- ChatGPT alignment (OpenAI)
- Claude alignment (Anthropic - Constitutional AI + RLHF)
- Gemini alignment (Google DeepMind)
- Code generation quality improvement
- Reducing harmful, biased, or unhelpful outputs

## Evolution

- **2017**: Deep RL from Human Preferences (Christiano et al.)
- **2020**: Learning to summarize with human feedback (OpenAI)
- **2022**: InstructGPT uses RLHF (precursor to ChatGPT)
- **2023**: DPO simplifies preference optimization (no RL needed)
- **2024+**: Hybrid approaches (DPO + online RLHF) and process reward models`,

    multi_agent_rl: `# Multi-Agent Reinforcement Learning (MARL)

Multi-Agent RL studies how multiple learning agents interact, cooperate, or compete in shared environments. It extends single-agent RL to handle the complexity of multi-agent dynamics.

## Key Concepts

- **Cooperative**: Agents share a common goal (team robotics)
- **Competitive**: Agents have opposing goals (games)
- **Mixed**: Both cooperative and competitive elements
- **Centralized Training, Decentralized Execution (CTDE)**: Train with global info, act independently
- **Nash Equilibrium**: No agent benefits from unilateral strategy change
- **Non-stationarity**: Environment changes as other agents learn simultaneously

## Types

| Setting | Description | Example |
|---------|-------------|---------|
| Fully Cooperative | Shared reward | Robot team assembly |
| Fully Competitive | Zero-sum game | Poker, Go |
| Mixed-Motive | Both cooperation and competition | Traffic, trading |
| Communication | Agents can send messages | Coordinated exploration |

## Key Algorithms

\`\`\`
Algorithms for MARL:

Independent Learning:
- Each agent runs its own RL algorithm (e.g., PPO)
- Simple but ignores other agents' adaptation

CTDE Methods:
- QMIX: Monotonic value decomposition for cooperative teams
- MAPPO: Multi-agent PPO with shared critic
- MADDPG: Multi-agent DDPG with centralized critic

Communication:
- CommNet: Continuous communication between agents
- TarMAC: Targeted multi-agent communication

Self-Play:
- Train agent against copies of itself
- Used in AlphaGo, OpenAI Five, Pluribus
\`\`\`

## Applications

- Multi-robot coordination (warehouse, search and rescue)
- Autonomous vehicle fleet coordination
- Game AI (StarCraft, Dota 2, poker)
- Network routing and resource allocation
- Financial market simulation

## Evolution

- **2000s**: Early multi-agent systems research
- **2016**: AlphaGo uses self-play for superhuman Go
- **2017**: QMIX for cooperative multi-agent tasks
- **2019**: OpenAI Five (5v5 Dota 2) and Pluribus (6-player poker)
- **2024+**: Foundation models as agents in multi-agent simulations`,
  };

  Object.assign(window.AI_DOCS, content);
})();
