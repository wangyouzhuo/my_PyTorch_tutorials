import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.unsqueeze: turn [1,2,3] into [[1,2,3]]  , torch can only process the data whhose dim >=2
x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)

y = x.pow(2) + 0.2*torch.rand(x.size())

# turn x,y into torch variable
x,y = Variable(x),Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# define the network

class Net(torch.nn.Module):

    # some step about build the network architecture
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()  # just remeber it
        self.hidden  = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    # feed forward     x -- input_data
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_features=1,n_hidden=10,n_output=1)
#print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.5)

loss_func = torch.nn.MSELoss()  # regression_problem

for t in range(1000):
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

