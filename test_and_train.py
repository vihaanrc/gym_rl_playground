
import torch
from foundation import *
from frozenlake_dq import *


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = ("cuda")
    else:
        device = ("cpu")

    print(f"Using {device}")

    frozen_lake = FrozenLakeDQL()
    #frozen_lake.train(2000, is_slippery=False)
    