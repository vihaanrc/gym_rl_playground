from collections import deque
import random

class ExperienceReplay: #same as class in DQN
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