import argparse
from cProfile import label
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import get_train_dataset
from src.datasets import get_test_dataset
from src.models.LeNet import LeNet

from src.utils.utils import set_device
from src.utils.utils import set_seed
from src.utils.metrics import cal_precision, cal_f1

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args):
        self.args = args

        self.net = LeNet()
        self.net.to(args.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

    def run_train(self, train_loader, test_loader):
        args = self.args
        for epoch in tqdm(range(int(args.epochs)), desc='Epoch'):
            epoch_loss = 0
            self.net.train()
            for step, samples_batch in enumerate(train_loader):
                inputs, labels = samples_batch

                result = self.net(inputs)
                loss = self.loss_function(result, labels)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"loss epoch {epoch+1} : {epoch_loss}")
            self.save_net(epoch+1)
            self.evaluate_net(test_loader, epoch+1)
    
    def save_net(self, epoch):
        args = self.args

        net_to_save = self.net.module if hasattr(self.net, 'module') else self.net
        net_save_dir = os.path.join(args.dir, 'result')
        if not os.path.exists(net_save_dir):
            os.mkdir(net_save_dir)
        net_name = 'pytorch_model.bin' + '.epoch' + str(epoch)
        net_save_file = os.path.join(net_save_dir, net_name)
        
        torch.save(net_to_save.state_dict(), net_save_file)
    
    def evaluate_net(self, test_loader, epoch):
        args = self.args

        precision = 0
        f1 = 0
        step = 0
        self.net.eval()
        with torch.no_grad():
            for samples_batch in test_loader:
                inputs, labels_true = samples_batch
                result = self.net(inputs)
                _, labels_pred = torch.max(result, dim=1)

                precision += cal_precision(labels_true, labels_pred)
                f1 += cal_f1(labels_true, labels_pred)

                step += 1
        precision /= step
        f1 /= step

        print(f"precision epoch {epoch} : {precision}")
        print(f"f1-score epoch {epoch} : {f1}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization and dataset shuffling")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="The initial learning rate for optimizer.")
    parser.add_argument("--train_batch_size", type=int, default=24, help="Amount of train data per batch.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Amount of test data per batch.")
    parser.add_argument("--epochs", type=int, default=30, help="Total number of training epochs to perform.")

    args = parser.parse_args()
    args.dir = os.path.dirname(os.path.abspath(__file__))

    # 设置cpu/gpu & 固定随机种子
    set_device(args)
    set_seed(args)

    # 加载训练数据和测试数据
    train_dataset = get_train_dataset(args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    test_dataset = get_test_dataset(args)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    trainer = Trainer(args)
    trainer.run_train(train_loader, test_loader)


if __name__ == '__main__':
    main()
