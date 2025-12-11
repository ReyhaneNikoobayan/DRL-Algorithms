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

## 🌟 Why this project?

Reinforcement Learning is best learned through **hands-on implementation**.  
By coding each algorithm step-by-step, I aim to deeply understand how RL works behind the scenes.

This repository is an evolving portfolio and a demonstration of my progress in DRL.

---

# 🚕 Taxi-v3 Q-Learning Project

This project trains an agent using **Q-learning** to solve the classic **Taxi-v3** environment from *OpenAI Gymnasium*. The agent learns optimal pickup and drop-off strategies through exploration, exploitation, and iterative updates to a Q-table.

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
<img width="3000" height="1500" alt="moving_avg_reward" src="https://github.com/user-attachments/assets/11118364-0589-4407-968c-47d0881055b4" />


### 🎞️ **Trained Agent Performance (GIF Episodes)**

The following GIFs show the trained Taxi agent navigating the environment using the learned Q‑table:

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b592f6b6-49f2-4058-8878-f15d03bcfe1c" width="230"/></td>
    <td><img src="https://github.com/user-attachments/assets/2db8b1e1-bec8-4436-b8cd-2d4b6040b7ea" width="230"/></td>
    <td><img src="https://github.com/user-attachments/assets/86163ec1-886f-45a8-a169-de9e00c1fccc" width="230"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/98a376cc-90fa-439b-8831-4936c05d35d3" width="230"/></td>
    <td><img src="https://github.com/user-attachments/assets/c38512ca-d42a-49b6-81cc-fa594e1eeeec" width="230"/></td>
    <td></td>
  </tr>
</table>

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


```
pip install gymnasium numpy matplotlib imageio
```

