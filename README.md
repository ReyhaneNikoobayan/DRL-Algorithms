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

* `results/videos/taxi_episode_0.gif`

* ![taxi_episode_0](https://github.com/user-attachments/assets/ab2a3ba1-75d0-42f2-ab33-b9c6ac0d2222)




### **1. 📊 Moving Average Reward Plot**

Saved here:

```
results/plots/moving_avg_reward.png
```

This plot shows the agent’s performance improvement over time using a 100‑episode moving window.

### **2. 🎥 Episode GIFs**

Saved here:

```
results/videos/taxi_episode_0.gif
results/videos/taxi_episode_1.gif
results/videos/taxi_episode_2.gif
results/videos/taxi_episode_3.gif
results/videos/taxi_episode_4.gif
```

Each GIF visualizes the trained agent navigating the Taxi-v3 environment.

---

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

## 📈 Visualization

A plot is generated to help understand the agent's learning trend. It shows:

* Increasing reward over time (as exploration decreases)
* Stabilization once optimal policy is learned

---

## 🎞️ Policy Demonstration

After training, the agent runs 5 evaluation episodes in greedy mode (`argmax(Q[state])`).
GIFs are rendered using RGB frames from the environment.

---

## ✔️ Requirements

Make sure you have:

* Python 3.8+
* `gymnasium`
* `numpy`
* `matplotlib`
* `imageio`

Install with:

```
pip install gymnasium numpy matplotlib imageio
```

---

## 🚀 Running the Code

Simply run the script:

```
python your_script.py
```

All results will be automatically saved.

---

## 🏁 Final Notes

This implementation is a clean, reproducible example of reinforcement learning using Q-learning. It helps visualize both:

* Training progression
* Behavior of the learned agent

Feel free to modify hyperparameters and explore improved methods like SARSA or deep Q-networks (DQN)! 🚀

