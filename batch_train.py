import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)


torch_dataset = Data.TensorDataset(data_tensor = x,target_tensor = y)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 # use 2 threads to load the data_tensor and the target_tensor
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):   # enumerate: add the index - step
        #  training
        print(spoch,step,batch_x.numpy(),batch_y.numpy())
