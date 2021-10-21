import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
    
class LSTM_module(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM_module, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State

    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State = self.forward(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
        return Hidden_State, Cell_State

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        return Hidden_State, Cell_State