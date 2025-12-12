
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time
import gymnasium as gym
import torch.nn.functional as F
import torch.distributions as dist
import imageio
import torch.multiprocessing as mp
import os

class network(nn.Module):
        def __init__(self, n_state, n_hidden, n_action):
            super().__init__()

            self.fc1=nn.Linear(n_state,n_hidden)
            self.fc2=nn.Linear(n_hidden,n_hidden)
           # self.fc3=nn.Linear(n_hidden,n_hidden)
            self.prob=nn.Linear(n_hidden, n_action)
            self.value=nn.Linear(n_hidden,1)

        def forward(self,x):

            x=self.fc1(x)
            x=F.relu(x)
            x=self.fc2(x)
            x=F.relu(x)
            action_prob=torch.softmax(self.prob(x), dim=-1)
            value=self.value(x)

            return action_prob, value

def agent(global_net, number_episode, gamma, max_steps, n_hidden, render, optimizer, worker_id):
    
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    num_state=env.observation_space.shape[0]
    num_action=env.action_space.n

    local_net=network(n_state=num_state, n_hidden= n_hidden, n_action=num_action)
    local_net.load_state_dict(global_net.state_dict())

    all_rewards = []  # store episode rewards


    for i in range(number_episode):

        terminated=False
        truncated=False
        state, info = env.reset()
        logs=[]
        rewards=[]
        values=[]
        entropies=[]
        step=0
        episode_reward = 0  # track total reward for this episode


        while not terminated and not truncated :

            state = torch.tensor(state, dtype=torch.float32)
            action_prob, value= local_net(state)

            m = dist.Categorical(action_prob)
            entropy = m.entropy()    # tensor scalar
            entropies.append(entropy)


            action = m.sample()

            log_prob = m.log_prob(action)    # tensor scalar

            logs.append(log_prob)
            values.append(value.squeeze())

            next_state, reward, terminated, truncated, info = env.step(action.item())

            episode_reward += reward  # accumulate total reward

            
            rewards.append(reward)
            step +=1
            state= next_state

            if step >= max_steps or terminated or truncated :
                returns = []

                if terminated or truncated :
                    R=0
                else :
                    with torch.no_grad():

                         _,R= local_net(torch.tensor(next_state, dtype=torch.float32))  
                         R=R.item() 

                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.append(R)

                returns.reverse() 

                returns = torch.tensor(returns, dtype=torch.float32)
                values = torch.stack(values)
                log_probs = torch.stack(logs)

                advantages = returns - values

                actor_loss = -(log_probs * advantages.detach()).sum()

                critic_loss = F.mse_loss(values, returns)

                entropies = torch.stack(entropies)
                entropy_loss = -0.01 * entropies.sum()  # encourage exploration


                loss = actor_loss + critic_loss + entropy_loss

                
                optimizer.zero_grad()
                loss.backward()

                # Push grads from local to global
                for local_param, global_param in zip(local_net.parameters(), global_net.parameters()):
                     global_param._grad = local_param.grad
                     local_param.grad = None

                optimizer.step()

                # Sync local net with global
                local_net.load_state_dict(global_net.state_dict())
                
                logs=[]
                rewards=[]
                values=[]
                entropies=[]
                step=0

        all_rewards.append(episode_reward)
        
        if (i+1) % 100 == 0:
            avg_reward = sum(all_rewards[-100:]) / 100
            print(f"Worker {worker_id} | Episode {i+1} | Avg reward last 100 episodes: {avg_reward:.2f}")


def test_agent(global_net, n_episodes=5, render=True):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    os.makedirs("results/videos", exist_ok=True)
    num_state = env.observation_space.shape[0]

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        frames = []

        while not (terminated or truncated):

            frame = env.render()
            frames.append(np.array(frame, dtype=np.uint8))
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action_prob, _ = global_net(state_tensor)
            
            # Choose the action with the highest probability (greedy)
            action = torch.argmax(action_prob).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                    frames.append(env.render())

        # Save GIF
        gif_path = f"results/videos/cartpole_episode_{ep}.gif"
        imageio.mimsave(gif_path, frames, fps=40)
        print("Saved GIF:", gif_path)

        print(f"[TEST] Episode {ep} Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")   

    env = gym.make("CartPole-v1")
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n
    n_hidden = 32

    global_net = network(n_state=num_state, n_hidden=n_hidden,n_action=num_action)

    global_net.share_memory()     # VERY IMPORTANT for multiprocessing

    optimizer = optim.Adam(global_net.parameters(), lr=(1e-4)*3)

    # create 3 workers
    processes = []
    num_workers = 3

    for rank in range(num_workers):
        p = mp.Process( target=agent,args=(global_net,3000,0.9,30,n_hidden,False, optimizer,rank))
        p.start()
        processes.append(p)

    # wait for workers to finish
    for p in processes:
        p.join()
    print("\n=== Testing Trained Policy ===")
    test_agent(global_net, n_episodes=5, render=True)    

          






