import numpy as np
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


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


if __name__ == '__main__':
    a = ExperienceReplay(5)
    a.append(4)
    a.append(3)
    a.append(2)
    print(a)
    print(a.sample(2))