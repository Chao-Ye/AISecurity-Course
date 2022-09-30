import argparse
from cProfile import label
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import get_train_dataset
from src.datasets import get_test_dataset
from src.models.LeNet import LeNet

from src.utils.utils import set_device
from src.utils.utils import set_seed


class Trainer():
    def __init__(self, args):
        self.args = args

        self.net = LeNet()
        self.net.to(args.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()


    def run_train(self, train_loader):
        args = self.args
        for epoch in tqdm(range(int(args.epochs)), desc='Epoch'):
            epoch_loss = 0
            for step, samples_batch in enumerate(train_loader):
                inputs, labels = samples_batch
                result = self.net(inputs)
                loss = self.loss_function(result, labels)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"loss epoch {epoch} : {epoch_loss}")
                


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization and dataset shuffling")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="The initial learning rate for optimizer.")
    parser.add_argument("--train_batch_size", type=int, default=24, help="Amount of train data per batch.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Amount of test data per batch.")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs to perform.")

    args = parser.parse_args()

    # 设置cpu/gpu & 固定随机种子
    set_device(args)
    set_seed(args)

    # 加载训练数据和测试数据
    train_dataset = get_train_dataset()
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    test_dataset = get_test_dataset()
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    trainer = Trainer(args)
    trainer.run_train(train_loader)


if __name__ == '__main__':
    main()
