# DRL-Algorithms

# 🧠 Deep Reinforcement Learning Portfolio

A curated collection of **Deep Reinforcement Learning (DRL)** algorithms implemented **from scratch** using PyTorch and Gymnasium.  

The goals of this repository are to:

- 🎯 Understand and implement core RL and DRL algorithms from scratch  
- 🧩 Build fundamental components (replay buffers, target networks, policy gradients, etc.) manually  
- 📈 Train and evaluate agents on classic control environments  
- 📚 Provide clear, educational code for others studying reinforcement learning  

This repository includes **tabular methods**, **value-based deep RL**, and **policy-gradient methods**, tested on well-known Gymnasium environments such as:

- CartPole-v1  
- FrozenLake-v1  
- MountainCar-v0  
- LunarLander-v2  
- Acrobot-v1  

---

Reinforcement Learning is best learned through **hands-on implementation**.  
By coding each algorithm step-by-step, I aim to deeply understand how RL works behind the scenes.

This repository is an evolving portfolio and a demonstration of my progress in DRL.

---

# 🚕 Taxi-v3 Q-Learning Project

This project trains an agent using **Q-learning** to solve the classic **Taxi-v3** environment from *OpenAI Gymnasium*. The agent learns optimal pickup and drop-off strategies through exploration, exploitation, and iterative updates to a Q-table.

📄 **Source Code:**  
[Q-Learning Taxi Agent](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/q_learning.ipynb)


---

## 📌 Features of This Implementation

* Q-learning with:

  * High learning rate: `alpha = 0.9`
  * Discount factor: `gamma = 0.95`
  * Epsilon-greedy action selection
  * Epsilon decay from 1.0 to 0.01
* Moving average reward plot saved automatically
* GIF video recordings (5 episodes) using the learned policy
* Organized project structure: saves results under `results/plots` and `results/videos`

---

## 📂 Files Generated

Below are examples of the **actual results** produced by the code:

### 🎯 **Training Performance (Moving Average Reward)**

The plot below shows the learning curve of the Taxi agent. The average reward increases over episodes as the agent learns an optimal policy.

**Preview:**
<img width="3000" height="1500" alt="moving_avg_reward" src="https://github.com/user-attachments/assets/1902a523-12b3-4bba-9778-8a61d7fab0c2" />


### 🎞️ **Trained Agent Performance (GIF Episodes)**

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/8a622882-da2d-4a10-8fc7-8ff52f7ab34d" width="230"/>
  <img src="https://github.com/user-attachments/assets/bd4e4b63-93eb-4cbc-a36a-9697c435b7e0" width="230"/>
  <img src="https://github.com/user-attachments/assets/29846e67-b0f1-4c86-bd67-5647ced157ad" width="230"/>
</div>

<div style="display: flex; gap: 10px; margin-top: 10px;">
  <img src="https://github.com/user-attachments/assets/e5655478-336f-4fa2-96de-13128424a440" width="230"/>
  <img src="https://github.com/user-attachments/assets/a4b50f1c-cedf-4be5-b662-7b9aa235b12a" width="230"/>
</div>



## 🧠 Q‑Learning Formula Used

The Q-table is updated using:

```
Q(s, a) = (1 - α) * Q(s, a) + α * (reward + γ * max(Q(s')))
```

Where:

* **α (alpha)** = learning rate
* **γ (gamma)** = discount factor
* **s** = current state
* **a** = chosen action
* **s'** = next state

---

## ▶️ How the Training Works

1. Initialize q-table to zeros
2. For each episode:

   * Reset the environment
   * For up to 100 steps:

     * Choose action via epsilon-greedy
     * Take action → receive reward and next state
     * Update Q-table
     * Break if the episode ends
3. Decay epsilon
4. Store total reward per episode

---

# 🎮 FrozenLake DQN Project

This project implements **Deep Q-Network (DQN)** to solve the **FrozenLake-v1** environment from *OpenAI Gymnasium*. The agent learns to navigate the 4x4 FrozenLake map using reinforcement learning, experience replay, and a target network to stabilize training.

📄 **Source Code:**
[FrozenLake DQN Agent](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/DQL-FrozenLake.ipynb)

---

## 📌 Features of This Implementation

* Deep Q-Network (DQN) with:

  * Single hidden-layer feedforward neural network
  * Adam optimizer with learning rate: `0.001`
  * Experience replay buffer (`memory_size=1000`)
  * Target network synced every `sync_rate=10` steps
  * Discount factor: `gamma=0.9`
  * Epsilon-greedy policy with linear decay
* Moving average reward plot saved automatically
* GIF video recordings (episodes) using the learned policy
* Organized project structure: saves results under `results/plots` and `results/videos`

---

## 📂 Files Generated

Below are examples of the **actual results** produced by the code:

### 🎯 **Training Performance (Moving Average Reward)**

The plot below shows the learning curve of the FrozenLake agent. The average reward increases over episodes as the agent learns an optimal policy.

<img width="1674" height="920" alt="Screenshot 2025-12-11 185216" src="https://github.com/user-attachments/assets/aee8f461-f5e8-4cc5-88fa-95b109e56f90" />



### 🎞️ **Trained Agent Performance (GIF Episodes)**

![frozenlake_episode_0](https://github.com/user-attachments/assets/38042cd4-444d-48dd-a78d-4676085ab6d8)

---

## 🧠 DQN Formula Used

The Q-values are updated using the DQN method with a target network:

```
Q(s, a) = reward + γ * max(Q_target(s'))
```

Where:

* **γ (gamma)** = discount factor
* **s** = current state
* **a** = chosen action
* **s'** = next state
* **Q_target** = target network prediction

---

## ▶️ How the Training Works

1. Initialize **Policy** and **Target** networks
2. For each episode:

   * Reset the environment
   * Choose action using epsilon-greedy policy
   * Step through the environment → store transition in memory
   * Update Policy network using sampled mini-batches from memory
   * Sync Target network every `sync_rate` steps
   * Decay epsilon gradually
3. Record total reward per episode
4. Save the trained model and learning curve plot

---

## ▶️ How Testing Works

1. Load the trained Policy network
2. Run agent for N episodes
3. Render each step and store frames
4. Save frames as GIFs for visualization

---

# 🚗⛰️ MountainCar-v0 Deep Q-Learning Project

This project trains an agent using a **Deep Q-Network (DQN)** to solve the classic **MountainCar-v0** environment from *OpenAI Gymnasium*.  
The agent learns to climb the mountain by building momentum and optimizing long-term rewards using replay memory, target networks, and gradient-based Q-value updates.

📄 **Source Code:**  
[mountaincar DQN Agent](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/DQL_mountaincar.ipynb)

---

## 📌 Features of This Implementation

✔ Fully implemented DQN with PyTorch  
✔ Replay Memory for experience replay  
✔ Target network for stable learning  
✔ Epsilon-greedy exploration with decay  
✔ Reward shaping for faster convergence  
✔ GIF recording of trained agent  
✔ Moving average reward plot  

### 🔧 Hyperparameters Used

* Learning rate: `0.001`
* Discount factor (gamma): `0.9`
* Batch size: `32`
* Replay memory size: `10,000`
* Target network sync rate: `1000`
* Epsilon: `1.0 → 0.01`

---

## 📂 Files Generated by the Code

### 🎯 Training Performance (Moving Average Reward)

This plot illustrates how the agent improves over time:

<img width="1200" height="600" alt="moving_average_rewards" src="https://github.com/user-attachments/assets/a8b333f4-eec9-4f25-ac5c-d488d83b1d82" />


---

### 🎞️ Trained Agent GIFs

Below are example GIFs produced during evaluation:

![mountain_car_episode_7](https://github.com/user-attachments/assets/d33f90d2-e794-4875-b85d-15791d2aa5d6)





---

## 🧠 DQN Learning Formula

The target Q-value is computed using:

Q_target = reward + γ * max(Q_target_network(s'))


And the loss function is:

Loss = MSE( Q_policy(s), Q_target )


---

## ▶️ How Training Works

1. Initialize policy & target networks  
2. Create replay memory  
3. For each episode:
   * Reset environment  
   * Select action using epsilon-greedy  
   * Execute action → observe reward and next state  
   * Store transition in replay memory  
   * Sample batch and optimize policy network  
   * Sync target network every 1000 steps  
4. Save:
   * Trained model → `MountainCar_dql.pt`
   * Reward plot → `moving_average_rewards.png`
   * GIFs → `results/videos/`

---

# 🏋️ A3C Multiprocessing CartPole Agent (PyTorch)

This project implements an **A3C-style reinforcement learning agent** for the **CartPole-v1** environment using **PyTorch multiprocessing**.  
Multiple workers run in parallel, each interacting with its own environment, updating a shared global network.

The implementation includes:

- Shared global Actor–Critic network  
- Multiple workers using `torch.multiprocessing`  
- Entropy regularization for exploration  
- Advantage estimation  
- GIF recording of test episodes  
- Clean code structure without external frameworks

---

## 📄 Source Code

👉 [A3C-CartPole Implementation](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/A3C-cartpole.py)

---

## 🔧 Features

### 🧠 Neural Network Architecture
- Shared **Actor–Critic network** with:
  - 2 hidden layers (`n_hidden = 32`)
  - Softmax policy head (action probabilities)
  - Value head (state value estimate)

### ⚙️ Training Setup
- Uses **three parallel workers**
- Each worker:
  - Interacts with its own environment
  - Computes returns and advantages
  - Pushes gradients to the global network
  - Receives updated parameters
- Hyperparameters:
  - `gamma = 0.9`
  - `lr = 3e-4`
  - `max_steps = 30` per rollout
  - Entropy coefficient = `0.01`

### 🎥 Testing & Visualization
- Greedy policy testing after training
- Saves GIF videos of episodes:
  - `results/videos/cartpole_episode_x.gif`

---


## ▶️ How Training Works

1. Create global shared network  
2. Spawn N workers (processes)  
3. Each worker:
   - Runs episodes independently  
   - Collects:
     - log probabilities  
     - rewards  
     - state values  
     - entropy  
   - Computes:
     - Returns  
     - Advantages  
     - Actor + Critic + Entropy loss  
   - Sends gradients → updates global net  
   - Syncs with global net  
4. After all workers finish, run test episodes

---

## ▶️ How Testing Works

- The trained global network is evaluated for `n_episodes`  
- The agent selects actions greedily  
- Each frame is captured and saved as a GIF  
- The reward for each episode is printed  

Example output:

---

## 📷 Example Test Result (GIF)

After training, the agent should balance the pole for the full 500 steps:

![cartpole_episode_4](https://github.com/user-attachments/assets/810b4b85-b715-4160-91a4-492cdbe29601)


## 🧠 Algorithm Summary

### Actor Loss  
L_actor = -log_prob(a) * advantage

### Critic Loss  
L_critic = (returns - values)^2

### Entropy Regularization  
L_entropy = -0.08 * entropy

### Total Loss  

L = L_actor + L_critic + L_entropy

---

# 🚀 CartPole-v1 Policy Gradient (REINFORCE) with PyTorch

This repository implements the **REINFORCE / Vanilla Policy Gradient** algorithm to solve **CartPole-v1** from OpenAI Gymnasium.  
The policy network outputs action probabilities and is optimized using the log-probability trick with discounted returns.

---

## 📄 Source Code

👉 [VPG-CartPole Implementation](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/A3C-cartpole.py)

---

## 🧠 Algorithm: REINFORCE

The policy is updated using the classic Monte-Carlo Policy Gradient:

loss = − Σ [ log π(aₜ | sₜ) * Gₜ ]

Where:

- `Gₜ` = discounted future return  
- `log π(aₜ | sₜ)` = log-probability of chosen action  
- The negative sign ensures **gradient ascent** on expected reward  

---

## 🏗 Policy Network Architecture

state (4-dim)
→ Linear(4 → 32) → ReLU
→ Linear(32 → 32) → ReLU
→ Linear(32 → 2) → Softmax


Output is a probability distribution over two actions:

- `0` = move left  
- `1` = move right  

---

## ▶️ Training Process Overview

1. Run episodes and record:
   - rewards
   - log-probabilities of actions taken  
2. Compute discounted returns:
Gₜ = rₜ + γ Gₜ₊₁

3. After every `count_num = 20` episodes:
- Concatenate all log-probs and returns  
- Compute policy loss  
- Backpropagate and update the network  
4. Save the moving average reward plot  
5. Evaluate the policy and save 5 GIF videos  


---


## 📊 **Test Performance Overview**

After training, the agent is evaluated for **5 episodes** using a greedy policy (`argmax`).  

---

## 📈 Training Performance

The script generates:

### 💡 Moving Average Reward (window = 100)

A plot is saved as:


This shows the agent’s improvement and stabilization over time.

---

## 🎞️ Evaluation Videos (GIF)

After training, 5 test episodes are rendered and saved in:




