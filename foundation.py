import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Sequential):
    def __init__(self, inputFeatures, outputFeatures,  l1OutNodes = None):
        if l1OutNodes is None:
            l1OutNodes = inputFeatures
        super().__init__(
           nn.Linear(inputFeatures, l1OutNodes),
           nn.ReLU(),
           nn.Linear(inputFeatures, outputFeatures),
           nn.ReLU())
        
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

    
    