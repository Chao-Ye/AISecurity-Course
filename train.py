import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import get_train_dataset
from src.datasets import get_test_dataset
from src.models.LeNet import LeNet

from src.utils.utils import set_device
from src.utils.utils import set_seed
from src.utils.metrics import cal_precision, cal_f1, cal_accuracy

import warnings
warnings.filterwarnings('ignore')

class Trainer():
    def __init__(self, args):
        self.args = args

        self.net = LeNet()
        self.net.to(args.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.train_precision_history = []
        self.test_precision_history = []
        self.train_f1_history = []
        self.test_f1_history = []
        
    def run_train(self, train_loader, test_loader):
        args = self.args

        for epoch in tqdm(range(int(args.epochs)), desc='Epoch'):
            epoch_loss = 0
            step = 0
            self.net.train()
            for samples_batch in train_loader:
                inputs, labels = samples_batch

                result = self.net(inputs)
                loss = self.loss_function(result, labels)
                epoch_loss += loss.item()
                step += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss /= step
            self.loss_history.append(epoch_loss)
            print(f"loss epoch {epoch+1} : {epoch_loss}")
            self.save_net(epoch+1)
            self.evaluate_net(train_loader, epoch+1, 'train')
            self.evaluate_net(test_loader, epoch+1, 'test')
        self.result_visual()
    
    def save_net(self, epoch):
        args = self.args

        net_to_save = self.net.module if hasattr(self.net, 'module') else self.net
        net_save_dir = os.path.join(args.dir, 'result')
        if not os.path.exists(net_save_dir):
            os.mkdir(net_save_dir)
        net_name = 'pytorch_model.bin' + '.epoch' + str(epoch)
        net_save_file = os.path.join(net_save_dir, net_name)
        
        torch.save(net_to_save.state_dict(), net_save_file)
    
    def evaluate_net(self, data_loader, epoch, data_type):   # data_type提示评估的是训练集或者测试集
        args = self.args

        accuracy = 0
        precision = 0
        f1 = 0
        step = 0
        self.net.eval()
        with torch.no_grad():
            for samples_batch in data_loader:
                inputs, labels_true = samples_batch
                result = self.net(inputs)
                _, labels_pred = torch.max(result, dim=1)

                accuracy += cal_accuracy(labels_true, labels_pred)
                precision += cal_precision(labels_true, labels_pred)
                f1 += cal_f1(labels_true, labels_pred)

                step += 1
        accuracy /= step
        precision /= step
        f1 /= step

        if data_type == 'test':
            self.test_acc_history.append(accuracy)
            self.test_precision_history.append(precision)
            self.test_f1_history.append(f1)
            print(f"test accuracy epoch {epoch} : {accuracy}")
            print(f"test precision epoch {epoch} : {precision}")
            print(f"test f1-score epoch {epoch} : {f1}")
        elif data_type == 'train':
            self.train_acc_history.append(accuracy)
            self.train_precision_history.append(precision)
            self.train_f1_history.append(f1)
            print(f"train accuracy epoch {epoch} : {accuracy}")
            print(f"train precision epoch {epoch} : {precision}")
            print(f"train f1-score epoch {epoch} : {f1}")
        else:
            pass

    def result_visual(self):
        args = self.args

        image_save_dir = os.path.join(args.dir, 'result')
        if not os.path.exists(image_save_dir):
            os.mkdir(image_save_dir)
        image_save_file = os.path.join(image_save_dir, 'LeNet_result.jpg')

        x = range(1, args.epochs+1)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.plot(x, self.loss_history, color='r', linestyle='-', label='loss')
        loss_up = int(max(self.loss_history)) + 1
        ax1.set_ylim(0, loss_up)

        ax2 = ax1.twinx()
        ax2.set_ylabel('train or test acc')
        ax2.plot(x, self.train_acc_history, color='b', linestyle='--', label='train acc')
        ax2.plot(x, self.test_acc_history, color='g', linestyle='--', label='test acc')
        ax2.set_ylim(0.2, 0.9)
        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

        plt.savefig(image_save_file)
        plt.show()
        


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization and dataset shuffling")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="The initial learning rate for optimizer.")
    parser.add_argument("--train_batch_size", type=int, default=24, help="Amount of train data per batch.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Amount of test data per batch.")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs to perform.")

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
