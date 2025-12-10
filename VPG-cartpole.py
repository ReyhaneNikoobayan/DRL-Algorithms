
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch import tensor

class Policy(nn.Module):
    def __init__(self, n_state, n_hidden, n_action):
       super().__init__()

       self.fc1=nn.Linear(n_state,n_hidden)
       self.fc2=nn.Linear(n_hidden,n_hidden)
       self.fc3=nn.Linear(n_hidden,n_action)
       

    def forward(self,x): 

       x=self.fc1(x)
       x=F.relu(x) 
       x=self.fc2(x)
       x=F.relu(x) 
       action_prob=torch.softmax(self.fc3(x),dim=-1)

       return action_prob

def agent (number_episode, gamma, count_num, render, n_hidden):
    
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    num_state=env.observation_space.shape[0]
    num_action= env.action_space.n

    network=Policy( num_state,n_hidden,num_action)
    optimizer= optim.Adam(network.parameters(), lr=0.001)

    
    counter=0

    all_rewards=[]
    all_probs=[]
    episode_rewards_list = []

    for i in range(number_episode):

        state, info = env.reset()
        terminated=False
        truncated=False
        rewards=[]
        log_probs=[]

        while not terminated and not truncated:

            state = torch.tensor(state, dtype=torch.float32)
            probs=network(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, info = env.step(action.item())
 
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

            state = next_state

        counter+=1
        
        episode_reward = sum(rewards)
        episode_rewards_list.append(episode_reward)

    

        # ---- Print average reward every 100 episodes ----
        if (i + 1) % 100 == 0:
            avg_reward = sum(episode_rewards_list[-100:]) / 100
            print(f"Episode {i + 1}: average reward (last 100 episodes) = {avg_reward:.2f}")


        G=0
        returns = []

        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns=torch.tensor(returns, dtype=torch.float32)   
        
        all_probs.append(torch.stack(log_probs))
        all_rewards.append(returns)

        if counter >= count_num :
            
            log_probs_tensor = torch.cat(all_probs)
            returns_tensor = torch.cat(all_rewards)
            
            #returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            loss = -(log_probs_tensor * returns_tensor).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_probs = []
            all_rewards = []
            counter = 0
    env.close()

            # ---- Plot average reward per 100 episodes ----
    window = 100
    avg_rewards = []

    for i in range(0, len(episode_rewards_list), window):
       avg = sum(episode_rewards_list[i:i+window]) / len(episode_rewards_list[i:i+window])
       avg_rewards.append(avg)

    # X-axis: episode number at the end of each 100-episode block
    x = list(range(window, len(episode_rewards_list)+1, window))

    plt.plot(x, avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (per {window} episodes)")
    plt.title("REINFORCE on CartPole")
    plt.show()

    return network

def test_policy_max(network, n_episodes=5, render=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    
    for i in range(n_episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:

           

            state_tensor = torch.tensor(state, dtype=torch.float32)
            probs = network(state_tensor)
            
            # Choose max probability action
            action = torch.argmax(probs).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
        
        print(f"Test Episode {i + 1}: total reward = {total_reward}")
    
    env.close()




if __name__ == "__main__":

    train_policy=agent (number_episode=5000, gamma=0.99, count_num=20, render=False, n_hidden=32)
   
    test=test_policy_max(train_policy, n_episodes=5, render=True)

    








