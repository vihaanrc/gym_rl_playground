import numpy as np

import random
from torch import nn
import torch.nn.functional as F

from collections import deque

class DQN(nn.Module):
    def __init__(self, inputFeatures, outputFeatures,  l1OutNodes = None):
        super().__init__()
        if l1OutNodes is None:
            l1OutNodes = inputFeatures
        self.fc1 = nn.Linear(inputFeatures, l1OutNodes)   # first fully connected layer
        self.out = nn.Linear(l1OutNodes, outputFeatures) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x
        
        
#deque wrapper class that stores prior transitions for SGD
class ExperienceReplay:
    def __init__(self, maxLen):
        self.exp = deque(maxlen=maxLen)
        self.length = maxLen


    def append(self, experience):
        self.exp.append(experience)

    def sample(self, numElements):
        assert self.length >= numElements
        return random.sample(self.exp,numElements )
    
    def __str__(self):
        return str(self.exp)

    
    