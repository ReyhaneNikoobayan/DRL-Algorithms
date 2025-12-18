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

### 🎞️ Trained Agent GIF

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

This code implements the **REINFORCE / Vanilla Policy Gradient** algorithm to solve **CartPole-v1** from OpenAI Gymnasium.  
The policy network outputs action probabilities and is optimized using the log-probability trick with discounted returns.

---

## 📄 Source Code

👉 [VPG-CartPole Implementation](https://github.com/ReyhaneNikoobayan/DRL-Algorithms/blob/main/VPG-cartpole.ipynb)

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

This plot illustrates how the agent improves over time:

<img width="1200" height="600" alt="moving_average_rewards" src="https://github.com/user-attachments/assets/819a5d5d-ebc0-4224-80ad-214998533c02" />


This shows the agent’s improvement and stabilization over time.

---

## 🎞️ Evaluation Video (GIF)

Below are example GIFs produced during evaluation:

![cartpole_episode_2](https://github.com/user-attachments/assets/9401ed3f-882c-476d-a4fb-4221466c75b8)

---

# Generalized Advantage Estimation (GAE)

Generalized Advantage Estimation (GAE) is a technique used in actor-critic reinforcement learning algorithms to reduce the high variance commonly found in basic policy gradient methods such as REINFORCE. In the Advantage Actor-Critic (A2C) framework, policy updates rely on the advantage function, defined as the difference between the action-value function and the state-value function. The challenge lies in estimating this advantage accurately while maintaining a balance between bias and variance.

GAE addresses this challenge by computing the advantage as an exponentially weighted sum of temporal-difference (TD) errors, controlled by the discount factor `γ` and a smoothing parameter `λ`. When `λ = 0`, GAE reduces to the one-step TD advantage, which has low variance but may be biased. When `λ = 1`, it becomes equivalent to the Monte Carlo advantage estimate, which is unbiased but has high variance. Intermediate values of `λ` interpolate between these two extremes, allowing practitioners to tune the bias-variance trade-off.

In practice, GAE is efficiently computed using a backward recursive formulation over a trajectory, making it suitable for large-scale training. It is widely used in modern policy optimization algorithms such as PPO and TRPO, where it significantly improves training stability, sample efficiency, and convergence speed.

# 🧠 Actor–Critic with GAE for LunarLander-v3 

This repository contains my **third Deep Reinforcement Learning (DRL) implementation**, where I implement an **Actor–Critic algorithm enhanced with Generalized Advantage Estimation (GAE)** from scratch using **PyTorch**.

The agent is trained on the **LunarLander-v3** environment with a discrete action space using **on-policy learning**.

---

## 📌 Project Overview

- **Algorithm:** Actor–Critic (A2C-style) with GAE  
- **Framework:** PyTorch  
- **Environment:** LunarLander-v3 (Gymnasium)  
- **Action Space:** Discrete  
- **Training Type:** On-policy  
- **Visualization:** GIF rendering of trained policy  

The objective is to learn a policy that safely lands the spacecraft between the flags while minimizing fuel consumption and avoiding crashes.

---

## 🧠 Algorithm Explanation

### 1️⃣ Actor–Critic Framework

The Actor–Critic method combines:
- **Policy-based learning** (Actor)
- **Value-based learning** (Critic)

Both networks are trained simultaneously.

#### Actor (Policy Network)
- Outputs a probability distribution over actions
- Uses a softmax layer for discrete action selection
- Actions are sampled from a categorical distribution

#### Critic (Value Network)
- Estimates the state value function \( V(s) \)
- Used to reduce variance in policy gradient updates

---

### 2️⃣ Neural Network Architecture

Two separate streams share the same input state:

State
├── Actor Network → Softmax → Action probabilities
└── Critic Network → Value estimate


- Fully connected layers
- ReLU activation
- Separate output heads for policy and value

---

### 3️⃣ Action Selection

At each timestep:
1. The policy outputs action probabilities
2. A categorical distribution is created
3. An action is sampled
4. Log-probabilities and entropy are stored for learning

This encourages **exploration during training**.

---

## 4️⃣ Generalized Advantage Estimation (GAE)

To stabilize training and reduce variance in policy gradient updates,
Generalized Advantage Estimation (GAE) is used.

--------------------------------------------------

Temporal Difference (TD) Error

<img width="1324" height="188" alt="image" src="https://github.com/user-attachments/assets/a5468442-c6e8-474f-b91c-e6860b2729a9" />


--------------------------------------------------

Advantage Function

<img width="1370" height="344" alt="image" src="https://github.com/user-attachments/assets/5e1df6ae-82fe-4c5f-bd1b-f4f85577a72c" />
​

This formulation combines multiple-step TD errors to obtain
a low-variance and low-bias advantage estimate.

--------------------------------------------------

Hyperparameters

γ (gamma)   = 0.99    discount factor

λ (lambda)  = 0.96    GAE smoothing parameter

--------------------------------------------------


---
## 🧮 Loss Functions

The total loss used to train the Actor–Critic model consists of three components:

--------------------------------------------------

Policy Loss (Actor)
Encourages actions with positive advantage

L_policy = - E[ log π(a | s) * A ]

Where:
- π(a | s) : policy (actor network)
- A        : advantage estimate (GAE)

--------------------------------------------------

Value Loss (Critic)
Minimizes the error between predicted state value and return

L_value = 0.5 * (R - V(s))^2

Where:
- R    : estimated return
- V(s) : value predicted by the critic

--------------------------------------------------

Entropy Bonus
Encourages exploration and prevents premature convergence

L_entropy = - β * H(π)

Where:
- H(π) : entropy of the policy distribution
- β    : entropy coefficient

--------------------------------------------------

Final Loss
Combined objective optimized during training

L = L_policy + L_value + L_entropy

--------------------------------------------------


---

## ⚙️ Training Configuration

| Parameter | Value |
|----------|-------|
| Episodes | 5500 |
| Learning Rate | 1e-3 |
| Hidden Units | 256 |
| Discount Factor (γ) | 0.99 |
| GAE Lambda (λ) | 0.96 |
| Entropy Coefficient | 0.01 |
| Optimizer | Adam |

Model updates are performed **on-policy** after each episode.

---

## 📊 Results

### Training Performance
- The agent successfully learns to land the spacecraft
- Average reward steadily increases over training
- Crash frequency decreases significantly
- Stable landings achieved consistently after convergence

**Typical learning behavior:**
- Early training: frequent crashes, unstable descent
- Mid training: partial control, occasional success
- Late training: smooth descent and successful landing

The average reward over the last 100 episodes is printed during training to monitor convergence.

---

### 🎥 Policy Evaluation

After training, the policy is evaluated in render mode:

- Multiple test episodes are executed
- Each episode is saved as a **GIF**
- Videos are stored in:


<div style="display: flex; gap: 10px;">
  <img src="![LunarLander_0](https://github.com/user-attachments/assets/aed85487-43bb-48d2-9227-f05df91c5856)" width="230"/>
  <img src="![LunarLander_1](https://github.com/user-attachments/assets/4c4d2f2d-c54e-48d2-b713-ffea083215be)" width="230"/>
  <img src="![LunarLander_2](https://github.com/user-attachments/assets/36f50ab3-3828-48e8-a801-fe657e81d81e)" width="230"/>
</div>

<div style="display: flex; gap: 10px; margin-top: 10px;">
  <img src="![LunarLander_3](https://github.com/user-attachments/assets/629ea985-9b6f-49f8-8a81-f34fed64f1bb)" width="230"/>
  <img src="![LunarLander_4](https://github.com/user-attachments/assets/725a4050-8fec-413d-a795-b2ea72325ce6)" width="230"/>
</div>


--- 

### 📊 Training Results and Evaluation

Episode 5400 | Avg reward (last 100): 130.39
Episode 5400 | Steps: 482 | Failure ❌ | Total rewards: 30.60 | Current reward: -100.00 | Status: Crashed
Episode 5404 | Steps: 444 | Failure ❌ | Total rewards: 34.17 | Current reward: -100.00 | Status: Crashed
Episode 5407 | Steps: 471 | Success ✅ | Total rewards: 124.78 | Current reward: 100.00 | Status: Reached the goal
Episode 5408 | Steps: 561 | Success ✅ | Total rewards: 41.93 | Current reward: 100.00 | Status: Reached the goal
Episode 5411 | Steps: 549 | Failure ❌ | Total rewards: 35.64 | Current reward: -100.00 | Status: Crashed
Episode 5412 | Steps: 520 | Success ✅ | Total rewards: 60.48 | Current reward: 100.00 | Status: Reached the goal
Episode 5414 | Steps: 501 | Success ✅ | Total rewards: 77.84 | Current reward: 100.00 | Status: Reached the goal
Episode 5417 | Steps: 607 | Success ✅ | Total rewards: 69.63 | Current reward: 100.00 | Status: Reached the goal
Episode 5419 | Steps: 516 | Success ✅ | Total rewards: 98.60 | Current reward: 100.00 | Status: Reached the goal
Episode 5420 | Steps: 522 | Success ✅ | Total rewards: 88.31 | Current reward: 100.00 | Status: Reached the goal
Episode 5427 | Steps: 530 | Success ✅ | Total rewards: 102.97 | Current reward: 100.00 | Status: Reached the goal
Episode 5430 | Steps: 504 | Success ✅ | Total rewards: 121.89 | Current reward: 100.00 | Status: Reached the goal
Episode 5434 | Steps: 560 | Failure ❌ | Total rewards: 35.14 | Current reward: -100.00 | Status: Crashed
Episode 5441 | Steps: 930 | Success ✅ | Total rewards: 130.30 | Current reward: 100.00 | Status: Reached the goal
Episode 5443 | Steps: 754 | Success ✅ | Total rewards: 119.95 | Current reward: 100.00 | Status: Reached the goal
Episode 5444 | Steps: 539 | Success ✅ | Total rewards: 67.51 | Current reward: 100.00 | Status: Reached the goal
Episode 5445 | Steps: 435 | Failure ❌ | Total rewards: 95.84 | Current reward: -100.00 | Status: Crashed
Episode 5446 | Steps: 450 | Success ✅ | Total rewards: 117.70 | Current reward: 100.00 | Status: Reached the goal
Episode 5447 | Steps: 825 | Success ✅ | Total rewards: 137.54 | Current reward: 100.00 | Status: Reached the goal
Episode 5448 | Steps: 457 | Success ✅ | Total rewards: 112.58 | Current reward: 100.00 | Status: Reached the goal
Episode 5449 | Steps: 543 | Success ✅ | Total rewards: 93.96 | Current reward: 100.00 | Status: Reached the goal
Episode 5450 | Steps: 465 | Failure ❌ | Total rewards: 61.18 | Current reward: -100.00 | Status: Crashed
Episode 5452 | Steps: 419 | Success ✅ | Total rewards: 138.86 | Current reward: 100.00 | Status: Reached the goal
Episode 5453 | Steps: 490 | Success ✅ | Total rewards: 86.50 | Current reward: 100.00 | Status: Reached the goal
Episode 5454 | Steps: 552 | Success ✅ | Total rewards: 143.91 | Current reward: 100.00 | Status: Reached the goal
Episode 5455 | Steps: 406 | Success ✅ | Total rewards: 132.91 | Current reward: 100.00 | Status: Reached the goal
Episode 5456 | Steps: 428 | Failure ❌ | Total rewards: 68.74 | Current reward: -100.00 | Status: Crashed
Episode 5459 | Steps: 385 | Success ✅ | Total rewards: 133.98 | Current reward: 100.00 | Status: Reached the goal
Episode 5460 | Steps: 379 | Success ✅ | Total rewards: 139.78 | Current reward: 100.00 | Status: Reached the goal
Episode 5461 | Steps: 426 | Success ✅ | Total rewards: 140.84 | Current reward: 100.00 | Status: Reached the goal
Episode 5462 | Steps: 431 | Success ✅ | Total rewards: 128.58 | Current reward: 100.00 | Status: Reached the goal
Episode 5463 | Steps: 380 | Success ✅ | Total rewards: 144.25 | Current reward: 100.00 | Status: Reached the goal
Episode 5464 | Steps: 408 | Success ✅ | Total rewards: 145.76 | Current reward: 100.00 | Status: Reached the goal
Episode 5465 | Steps: 439 | Success ✅ | Total rewards: 154.40 | Current reward: 100.00 | Status: Reached the goal
Episode 5466 | Steps: 293 | Failure ❌ | Total rewards: 63.44 | Current reward: -100.00 | Status: Crashed
Episode 5467 | Steps: 419 | Success ✅ | Total rewards: 84.19 | Current reward: 100.00 | Status: Reached the goal
Episode 5469 | Steps: 437 | Success ✅ | Total rewards: 128.13 | Current reward: 100.00 | Status: Reached the goal
Episode 5470 | Steps: 442 | Success ✅ | Total rewards: 155.21 | Current reward: 100.00 | Status: Reached the goal
Episode 5472 | Steps: 456 | Success ✅ | Total rewards: 133.68 | Current reward: 100.00 | Status: Reached the goal
Episode 5473 | Steps: 412 | Success ✅ | Total rewards: 118.87 | Current reward: 100.00 | Status: Reached the goal
Episode 5474 | Steps: 438 | Success ✅ | Total rewards: 119.51 | Current reward: 100.00 | Status: Reached the goal
Episode 5475 | Steps: 388 | Success ✅ | Total rewards: 127.20 | Current reward: 100.00 | Status: Reached the goal
Episode 5476 | Steps: 453 | Success ✅ | Total rewards: 126.55 | Current reward: 100.00 | Status: Reached the goal
Episode 5478 | Steps: 407 | Success ✅ | Total rewards: 154.61 | Current reward: 100.00 | Status: Reached the goal
Episode 5479 | Steps: 463 | Success ✅ | Total rewards: 137.05 | Current reward: 100.00 | Status: Reached the goal
Episode 5481 | Steps: 338 | Failure ❌ | Total rewards: 118.25 | Current reward: -100.00 | Status: Crashed
Episode 5483 | Steps: 359 | Success ✅ | Total rewards: 137.98 | Current reward: 100.00 | Status: Reached the goal
Episode 5484 | Steps: 407 | Success ✅ | Total rewards: 129.14 | Current reward: 100.00 | Status: Reached the goal
Episode 5485 | Steps: 401 | Success ✅ | Total rewards: 141.95 | Current reward: 100.00 | Status: Reached the goal
Episode 5486 | Steps: 681 | Success ✅ | Total rewards: 144.09 | Current reward: 100.00 | Status: Reached the goal
Episode 5487 | Steps: 418 | Success ✅ | Total rewards: 133.70 | Current reward: 100.00 | Status: Reached the goal
Episode 5489 | Steps: 384 | Success ✅ | Total rewards: 121.64 | Current reward: 100.00 | Status: Reached the goal
Episode 5490 | Steps: 579 | Success ✅ | Total rewards: 109.30 | Current reward: 100.00 | Status: Reached the goal
Episode 5491 | Steps: 394 | Success ✅ | Total rewards: 138.36 | Current reward: 100.00 | Status: Reached the goal
Episode 5492 | Steps: 385 | Success ✅ | Total rewards: 165.90 | Current reward: 100.00 | Status: Reached the goal
Episode 5493 | Steps: 499 | Success ✅ | Total rewards: 34.39 | Current reward: 100.00 | Status: Reached the goal
Episode 5494 | Steps: 362 | Success ✅ | Total rewards: 162.52 | Current reward: 100.00 | Status: Reached the goal
Episode 5495 | Steps: 274 | Failure ❌ | Total rewards: 78.18 | Current reward: -100.00 | Status: Crashed
Episode 5497 | Steps: 401 | Success ✅ | Total rewards: 124.08 | Current reward: 100.00 | Status: Reached the goal
Episode 5498 | Steps: 386 | Success ✅ | Total rewards: 117.66 | Current reward: 100.00 | Status: Reached the goal
Episode 5500 | Avg reward (last 100): 152.24
Episode 5499 | Steps: 399 | Success ✅ | Total rewards: 138.79 | Current reward: 100.00 | Status: Reached the goal
reached
Saved GIF: results/videos/LunarLander_0.gif
Test Episode 1: total reward = 209.14110268153448
reached
Saved GIF: results/videos/LunarLander_1.gif
Test Episode 2: total reward = 253.14918661507846
reached
Saved GIF: results/videos/LunarLander_2.gif
Test Episode 3: total reward = 205.78820721239532
reached
Saved GIF: results/videos/LunarLander_3.gif
Test Episode 4: total reward = 195.28689352775007
reached
Saved GIF: results/videos/LunarLander_4.gif
Test Episode 5: total reward = 208.77817685209357


