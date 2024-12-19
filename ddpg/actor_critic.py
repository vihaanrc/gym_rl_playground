import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

class Actor(nn.module):
    def __init__(self, num_inputs, action_dim, h1_nodes):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h1_nodes)
        self.fc3 = nn.Linear(h1_nodes, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) #constrain actions to between -1 and 1
        return x
    
class Critic(nn.module):
    def __init__(self, num_states, action_dim, h1_nodes):
        super().__init__()

        self.fc1 = nn.Linear(num_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes+action_dim, h1_nodes)
        self.fc3 = nn.Linear(h1_nodes, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(torch.cat([x, action], dim=1)))
        x = self.fc3(x)
        return x #Q(s,a)