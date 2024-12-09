
import torch
from foundation import *
from frozenlake_dq import *

def viewPickle():
    

# Replace 'your_file.pkl' with the path to your pickle file
    with open('frozen_lake_tensors.pt', 'rb') as file:
        data = torch.load(file)
        print(data)
    

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = ("cuda")
    else:
        device = ("cpu")

    print(f"Using {device}")
    torch.autograd.set_detect_anomaly(True)
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(500, is_slippery=False)
    frozen_lake.test(10, render=True, is_slippery=False)
    viewPickle()

