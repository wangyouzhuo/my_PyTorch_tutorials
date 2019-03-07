import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False


train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,)

test_data = torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data,dim = 1),volatile = True).type(torch.FloatStorage)[:2000]/255.0
test_y = test_data.test_lable[:2000]

class CNN():

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Linear(32*7*7,10)


    def forward(self):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)  # (batch_size,32*7*7)
        output = self.out(x)
        return output


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(),lr = LR)
loss_func = nn.CrossEntropyLoss()

#.........the same as others..........