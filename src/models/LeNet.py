from matplotlib.style import use
import torch.nn as nn
import torch


class LeNet(nn.Module):
    def __init__(self, use_dropout):
        super(LeNet, self).__init__()

        self.use_dropout = use_dropout

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

        self.fc1 = nn.Sequential( 
            nn.Linear(16*5*5, 120), 
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU())
        self.fc3 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(p=0.5) 
    
    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv(x)
        x = x.reshape(batch_size, -1)

        if self.use_dropout:
            x = self.dropout(self.fc1(x))
            x = self.dropout(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            
        return x

if __name__ == '__main__':
    net = LeNet(True)

    x = torch.rand(24, 3, 32, 32)
    x = net(x)
    print(x.shape)



