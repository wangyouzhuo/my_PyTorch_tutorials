import torch
import numpy as np

# new a numpy data
np_data = np.arange(6).reshape((2,3))


# turn the numpy data into a torch tensor
torch_tensor  = torch.from_numpy(np_data)

# turn the torch_tensor into a numpy data
torch2numpy = torch_tensor.numpy()


data = [-1,-2,1,2]

# turn list into float tensor
tensor = torch.FloatTensor(data)


data_two = [[1,2],[3,4]]
tensor_data_two = torch.FloatTensor(data_two)



if __name__ == "__main__":
    print(
        # "numpy_data:\n",np_data,
        # "\ntorch_tensor:\n",torch_tensor,'\n',
        # "torch2numpy:\n",torch2numpy,

        # '\nnumpy:',data,
        # '\nfloat_tensor:',tensor,
        # '\nnumpy_abs:',np.abs(data),
        # '\ntorch_abs:',torch.abs(tensor),
        # '\nnumpy_mean:', np.mean(data),
        # '\ntorch_mean:', torch.mean(tensor)

        '\nnumpy_matmul:',np.matmul(data_two,data_two),
        '\nnumpy_dot:', np.dot(data_two, data_two),
        '\ntorch_matmul:',torch.mm(tensor_data_two,tensor_data_two),
        '\ntorch_dot:', tensor_data_two.dot(tensor_data_two),

    )
