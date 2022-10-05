import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_device(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)



# x = range(10)
# loss = [3000, 2000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
# t1 = [0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.4, 0.5]
# t2 = [0.45, 0.47, 0.55, 0.45, 0.47, 0.55, 0.45, 0.47, 0.55, 0.45]

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('loss')
# ax1.plot(x, loss, color='r', linestyle='-', label='loss')

# ax2 = ax1.twinx()
# ax2.plot(x, t1, color='b', linestyle='--', label='train acc')
# ax2.plot(x, t2, color='g', linestyle='--', label='test acc')
# ax2.set_ylabel('train or test acc')
# ax2.set_ylim(0.2, 0.9)
# fig.tight_layout()
# fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
# plt.savefig('./test.jpg')
# plt.show()