import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from prediction.att_embeddings import Embeddings
from prediction.LSTM_module import LSTM_module

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')

class LSTM(nn.Module):
    def __init__(self, att_size, step_size, fea_size, pred_size=1):
        super(LSTM, self).__init__()

        self.att_size = att_size

        self.embedding = Embeddings([att_size, 100, 100, 50, 20, 10], step_size, fea_size, att_size) # output feature size is 10

        self.lstm = LSTM_module(fea_size, fea_size, fea_size) # output feature size is 1

        self.fc = nn.Linear(11, pred_size)

    def forward(self, atts, cases):
        atts = self.embedding(atts)
        cases, _ = self.lstm.loop(cases)
        cases = cases.unsqueeze(2)
        # print('merge',atts.size(), cases.size())
        outputs = self.fc(torch.cat([atts, cases], dim=2))
        return outputs.transpose(1, 2) # batch size * pred_size * n_cnt