import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.linspace(-1,5,200)
x = Variable(x)
x_np = x.data.numpy() # turn the torch data into numpy data,just for the plt

y_relu = F.relu(x).data.numpy()
y_sigmod = F.sigmoid(x).data.numpy()
y_tanh = F.tanh(x).data.numpy()
y_softmax = F.softplus(x).data.numpy()

