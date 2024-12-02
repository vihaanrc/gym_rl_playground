
import torch
from foundation import *
from frozenlake_dq import *


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = ("cuda")
    else:
        device = ("cpu")

    print(f"Using {device}")
    torch.autograd.set_detect_anomaly(True)
    frozen_lake = FrozenLakeDQL()
   # frozen_lake.train(500, is_slippery=False)
    frozen_lake.test(10, render=True, is_slippery=False)
