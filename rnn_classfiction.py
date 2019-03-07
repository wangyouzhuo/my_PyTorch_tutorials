import torch
import torch.nn as nn

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True

#.......pre step about data...........


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True #  batch_first=True  ---> (batch_input,time_stepp,....)
                             #  batch_first=False ---> (time_stepp,batch_input,....)
        )
        self.out = nn.Linear(64,10)


    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:,])  # (batch,time_step,iinput)
        return out


rnn = RNN()
print(rnn)