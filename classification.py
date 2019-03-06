import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# define two cluster of data,the label is 1 and
n_data = torch.ones(100,2)  # shape=[100,2] all value = 1
#  cluster-1
x0 = torch.normal(2*n_data,1)  # normal distribution ,mean = n_Data,std = 1
y0 = torch.zeros(100)
# cluster-2
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)

x = torch.cat((x0,x1), 0).type(torch.FloatTensor)
y = torch.cat((y0,y1), 0).type(torch.LongTensor)

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


net = Net(n_features=2,n_hidden=10,n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.005)

loss_func = torch.nn.CrossEntropyLoss()  # regression_problem

for t in range(1000):
    """
        1. forward
        2. compute the loss
        3. reset the grad in optimizer
        4. BP the grad
        5. use gradient to update the node
    """
    prediction = net(x)

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

