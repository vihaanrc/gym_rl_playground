import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
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

    

    def train(self, episodes, render=False, is_slippery=True):
        epsilon = 1
        epsilon_decay_rate = 1/episodes

        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        self.optimizer = torch.optim.Adam(policy_nn.parameters(), lr = self.learning_rate_a)
        memory = ExperienceReplay(self.replay_memory_size)
        reward_record = np.zeros(episodes)
        policy_nn = DQN(inputFeatures = num_states, outputFeatures=num_actions)
        target_nn = DQN(inputFeatures = num_states, outputFeatures=num_actions)

        target_nn.load_state_dict(policy_nn.state_dict()) #sync nn's
        
    
        step_count = 0
        for i in range(episodes):
            state = env.reset()[0]
            terminated, truncated = False 
            #terminated->reaches goal or falls in hole
            #truncated -> 100 steps for 4x4 map.

            while (not terminated and not truncated):
                if random.uniform(0,1) < epsilon:
                    action = env.action_space.sample()
                else:
                    state_tensor = F.one_hot(torch.tensor(state), num_classes=num_states).float()
                    with torch.no_grad(): #disable weight updating when using model for prediction
                        q_values = policy_nn(state_tensor)
                    
                    action = torch.argmax(q_values).item() #find action with highest value


                new_state,reward,terminated,truncated,_ = env.step(action)
                memory.append((state, action, new_state, reward, terminated)) 
                
                step_count +=1
                state = new_state
            if reward==1:
                reward_record[episodes] = reward

            if (np.sum(reward_record)>1 and len(memory) > self.optimize_batch_size):
                #To-do: Optimize Policy network based on sampled mini-batch from memory

                epsilon = max(epsilon-epsilon_decay_rate, 0)
                if step_count > self.network_sync_rate:
                    target_nn.load_state_dict(policy_nn.state_dict())
                    step_count=0

        env.close()
