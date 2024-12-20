import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from actor_critic import Actor, Critic
from memory import ExperienceReplay
import random


class DDPG:
    def __init__(self, state_dim, action_dim, h1_dim, env):
        self.learning_rate_a = 0.0001         
        self.discount_factor_g = 0.9        
        self.network_sync_rate = 10          
        replay_memory_size = 1000  
        self.optimize_batch_size = 32 
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h1_dim = h1_dim

        self.Actor = Actor(self.state_dim, self.action_dim, self.h1_dim)
        self.Critic = Critic(self.state_dim, self.action_dim, self.h1_dim)
        self.Actor_target = Actor(self.state_dim, self.action_dim, self.h1_dim)
        self.Critic_target = Critic(self.state_dim, self.action_dim, self.h1_dim)


        self.actor_optimizer = Adam(self.Actor.parameters(), lr=self.learning_rate_a)
        self.critic_optimizer = Adam(self.Critic.parameters(), lr=self.learning_rate_a)

        self.memory = ExperienceReplay(replay_memory_size)

        self.Actor.load_state_dict(self.Actor_target.state_dict())
        self.Critic.load_state_dict(self.Critic_target.state_dict())

        
        
        
    def train(self, episodes, env):
        noise = 0.15 #higher -> more agressive exploration
        noise_decay_rate = 0.99
        reward_record = np.zeros(episodes)
        step_count = 0
        for i in range(0, episodes):
            reward_score = 0
            state = env.reset()[0]
            terminated = False
            truncated = False
            while (not terminated and not truncated):

                with torch.no_grad(): #disable weight updating when using model for prediction
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = self.Actor(state_tensor)
                    action = (action.numpy())[0]
                    action = np.clip(action + noise, -1, 1)
                        
                next_state,reward,terminated,truncated,_ = env.step(action)
                
                reward_score+=reward
                self.memory.append((state, action, next_state, reward, terminated)) 
                step_count+=1

                state = next_state
                self.optimize()
                noise *= noise_decay_rate

           
            reward_record[i] = reward_score
            if i % 10 == 0 and i != 0:
                print(f"Episode: {i}, Score: {reward_score:.2f}, Avg Score: {np.mean(reward_record[i-10:i]):.2f}")
            


        torch.save(self.Actor.state_dict(), "Actor_tensors.pt")
        torch.save(self.Critic.state_dict(), "Critic_tensors.pt")

    def optimize(self):

        if len(self.memory.exp) < self.optimize_batch_size:   
            return
        batch = self.memory.sample(self.optimize_batch_size)

        states = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
        actions = torch.stack([torch.tensor(item[1], dtype=torch.float32) for item in batch])
        next_states = torch.stack([torch.tensor(item[2], dtype=torch.float32) for item in batch])

        rewards = torch.tensor([item[3] for item in batch], dtype=torch.float32).unsqueeze(1)
        terminated = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(1)
        
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.Actor_target(next_states)
            target_q_values = self.Critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - terminated) * self.discount_factor_g * target_q_values

        current_q_values = self.Critic(states, actions)

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        critic_loss.backward()
        self.critic_optimizer.step()

        
        self.actor_optimizer.zero_grad()

        actor_loss = -self.Critic(states, self.Actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        #implement tau parameter to smooth data transfer
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.Critic_target.load_state_dict(self.Critic.state_dict())
    def test(self, env, episodes):
        self.Actor.load_state_dict(torch.load("Actor_tensors.pt"))
        self.Critic.load_state_dict(torch.load("Critic_tensors.pt"))
        self.Actor.eval()
        self.Critic.eval()

        reward_record = [0] * episodes
        for i in range(episodes):
            state = env.reset()[0]  
            terminated = False    
            truncated = False   
            while(not terminated and not truncated):    
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = self.Actor(state_tensor)
                    action = (action.numpy())[0]
                    

               
                state,reward,terminated,truncated,_ = env.step(action)
            reward_record[i] = reward #do plotting later