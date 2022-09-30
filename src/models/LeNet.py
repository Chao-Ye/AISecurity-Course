import torch.nn as nn
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.sequential = []
        self.sequential.append(
            nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=(5,5), stride=(1,1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            )
        )
        self.sequential.append(
            nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=(5,5), stride=(1,1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            )
        )
        self.conv = nn.Sequential(*self.sequential)

        self.sequential = []
        self.sequential.append(
            nn.Sequential(
                nn.Linear(16*5*5, 120),
                nn.ReLU()
            )
        )
        self.sequential.append(
            nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10)
            )
        )
        self.fc = nn.Sequential(*self.sequential)
    
    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv(x)
        x = x.reshape(batch_size, -1)

        x = self.fc(x)

        return x

if __name__ == '__main__':
    net = LeNet()

    x = torch.rand(24, 3, 32, 32)
    x = net(x)
    print(x.shape)



