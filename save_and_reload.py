import torch
from torch.autograd import Variable


x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)

y = x.pow(2) + 0.2*torch.rand(x.size())

# turn x,y into torch variable
x,y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)

def save():
    net2 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),  # when we use this way too define a NN，relu is a layer not a function
        torch.nn.Linear(10, 2),
    )

    optimizer = torch.optim.SGD(net2.parameters(), lr=0.005)

    loss_func = torch.nn.CrossEntropyLoss()  # regression_problem

    for t in range(1000):
        out = net2(x)

        prediction = torch.max(F.softplus(out), 1)[1]  # [0] return max_value    [1] return max_index

        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net2,'net.pkl') # save the compute graph and the params in network

    torch.save(net2.state_dict(),'net_only_params.pkl')  # only save the params in network


# restore the compute and params in network
def restore_net():
    net2 = torch.load('net.pkl')


# restore the params in network,fistly,we must build a network with the same architecture
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(2, 10),
        torch.nn.ReLU(),  # when we use this way too define a NN，relu is a layer not a function
        torch.nn.Linear(10, 2),
    )

    net3.load_state_dict(torch.load('net_only_params.pkl'))