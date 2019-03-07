import torch



net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),      # when we use this way too define a NN，relu is a layer not a function
    torch.nn.Linear(10,2),
)


print(net2)