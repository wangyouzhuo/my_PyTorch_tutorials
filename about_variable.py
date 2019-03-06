import torch
from torch.autograd import Variable




tensor = torch.FloatTensor([[1,2],[3,4]])

variable = Variable(tensor,requires_grad = True)  # if True：compute the grads about this node

t_out = torch.mean(tensor*tensor) #  t_out = torch^2

v_out = torch.mean(variable*variable)


v_out.backward()

# output the grads on this node
print(variable.grad)

# ouput the value of this node
print(variable.data)

# turn the variable node into a numpy data
print(variable.data.numpy())

