import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_device(args):
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    args.device = device

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.device != 'cpu':
        torch.cuda.manual_seed_all(seed)
