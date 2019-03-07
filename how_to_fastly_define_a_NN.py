import torch



net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),      # when we use this way too define a NN，relu is a layer not a function
    torch.nn.Linear(10,2),
)

print(net2)

optimizer = torch.optim.SGD(net2.parameters(),lr = 0.005)

loss_func = torch.nn.CrossEntropyLoss()  # regression_problem

for t in range(1000):

    out = net2(x)

    prediction = torch.max(F.softplus(out),1)[1] # [0] return max_value    [1] return max_index

    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
