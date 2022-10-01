import os

import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def get_train_dataset(args):
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'data'),
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    return train_dataset


def get_test_dataset(args):
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.dir, 'data'),
                                                train=False,
                                                download=True,
                                                transform=transform)
    return test_dataset