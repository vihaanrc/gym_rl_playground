import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from foundation import *


class FrozenLakeDQL():
    loss_fn = nn.MSELoss()
    optimizer = None #initialize later when num_states known 
    

    #adjustable hyperparameters 
    learning_rate_a = 0.001         
    discount_factor_g = 0.9        
    network_sync_rate = 10          
    replay_memory_size = 1000  
    optimize_batch_size = 32 
    num_states = 0

    

    def train(self, episodes, render=False, is_slippery=False):
        epsilon = 1
        epsilon_decay_rate = 1/episodes

        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        self.num_states = env.observation_space.n
        num_actions = env.action_space.n

        
        policy_nn = DQN(inputFeatures = self.num_states, outputFeatures=num_actions)
        
        target_nn = DQN(inputFeatures = self.num_states, outputFeatures=num_actions)

        target_nn.load_state_dict(policy_nn.state_dict()) #sync nn's
        
        self.optimizer = torch.optim.Adam(policy_nn.parameters(), lr = self.learning_rate_a)
        memory = ExperienceReplay(self.replay_memory_size)
        reward_record = np.zeros(episodes)
    
        step_count = 0
        for i in range(episodes):
            if (i%10==0):
                print("episode number: " + str(i))
            state = env.reset()[0]
            terminated = False #terminated->reaches goal or falls in hole
            truncated = False #truncated -> 100 steps for 4x4 map.

            while (not terminated and not truncated):
                if random.uniform(0,1) < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = F.one_hot(torch.tensor(state), num_classes=self.num_states).float()
                    with torch.no_grad(): #disable weight updating when using model for prediction
                        q_values = policy_nn(state_tensor)
                    
                    action = torch.argmax(q_values).item() #find action with highest value


                new_state,reward,terminated,truncated,_ = env.step(action)
                memory.append((state, action, new_state, reward, terminated)) 
                
                step_count +=1
                state = new_state
            if reward==1:
                reward_record[i] = reward
            
            if (np.sum(reward_record)>=1 and len(memory.exp) > self.optimize_batch_size):
               
                self.optimize(memory.sample(self.optimize_batch_size), policy_nn, target_nn)

                epsilon = max(epsilon-epsilon_decay_rate, 0)
                if step_count > self.network_sync_rate:
                    target_nn.load_state_dict(policy_nn.state_dict())
                    step_count=0

        env.close()

        torch.save(policy_nn.state_dict(), "frozen_lake_tensors.pt")
    def optimize(self, batch, policy_nn, target_nn):
        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in batch:

            if terminated: 
                target = torch.FloatTensor([reward])
            else: 
                with torch.no_grad():
                    new_state_tensor = F.one_hot(torch.tensor(new_state), num_classes=self.num_states).float()
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_nn(new_state_tensor).max()
                    )

            state_tensor = F.one_hot(torch.tensor(state), num_classes=self.num_states).float()
            current_q = policy_nn(state_tensor)
            current_q_list.append(current_q)

            target_q = target_nn(state_tensor).clone() 
            target_q[action] = target
            
            target_q_list.append(target_q)
                
        
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        self.num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy_nn = DQN(inputFeatures = self.num_states, outputFeatures=num_actions)
        print(policy_nn)
        policy_nn.load_state_dict(torch.load("frozen_lake_tensors.pt"))
        policy_nn.eval()

        reward_record = [0] * episodes
        for i in range(episodes):
            state = env.reset()[0]  
            terminated = False    
            truncated = False   
            while(not terminated and not truncated):    
                with torch.no_grad():
                    state_tensor = F.one_hot(torch.tensor(state), num_classes=self.num_states).float()
                    action = policy_nn(state_tensor).argmax().item()

               
                state,reward,terminated,truncated,_ = env.step(action)
            reward_record[i] = reward #do plotting later
        env.close()

 