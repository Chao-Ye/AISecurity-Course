import os

import torchvision
import torchvision.transforms as transforms



dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def get_train_dataset():
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dir, 'data'),
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
    return train_dataset


def get_test_dataset():
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(dir, 'data'),
                                                train=False,
                                                download=True,
                                                transform=transform)
    return test_dataset