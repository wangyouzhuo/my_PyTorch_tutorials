import torch
import torch.utils.data as Data

#  hyper parameters

LR = 0.01

BATCH_SIZE = 32

EPOCH = 12


x = torch.unsqueeze(torch.linspace(-1,1,1000),dim = 1)
y = x.pow(x) + 0.1*torch.normal(torch.zeros(x.size()))


torch_dataset = Data.TensorDataset(data_tensor = x,target_tensor = y)

loader = Data.DataLoader(dataset=torch_dataset,shuffle=True,batch_size=BATCH_SIZE)

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


# build 4 nn
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()

nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum  = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha = 0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas = (0.9,0.99))

optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

loss_func = torch.nn.MSELoss()

losses_his = [[],[],[],[]]

for epoch in range(EPOCH)：
    print(epoch)
    for step,(batch_x,batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net,opt,l_his in zip(nets,optimizers,losses_his):
            output = net(b_x)
            loss = loss_func(output,b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data[0])