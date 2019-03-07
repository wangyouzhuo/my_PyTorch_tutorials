import torch
import torch.nn as nn


# hyper parameters
TIME_STEP = 0
INPUT_SIZE = 1
LR = 0.02

# show_data

steps = np.linspace(0,np.pi*2,100,dtype=np.float32)  # 100 points from 0 to 2pi
x_np = np.sin(steps)
y_np = np.cos(steps)



class RNN(nn.Module):

    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32,1)

    def forward(self, x ,h_state):
        """

        :param x: (batch_size,time_step,input_size)
        :param h_state: (n_layers,batch,hidden_size)
        :return: (batch_size,time_step,hidden_size)
        """
        r_out,h_state = self.rnn(x,h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out())

