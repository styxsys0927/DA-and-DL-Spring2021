"""
The main network structure.(pytorch style)
"""
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from prediction.LSTM_module import LSTM_module as LSTM

# AutoEncoder
class Embeddings(nn.Module):
    def __init__(self, hidden, step_size, fea_size, att_size, dropout=0.1):
        super(Embeddings, self).__init__()

        self.step_size = step_size
        self.fea_size = fea_size
        self.att_size = att_size

        d1 = OrderedDict()
        for i in range(len(hidden)-1):
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])
            #d1['enc_bn' + str(i)] = nn.BatchNorm1d(hidden[i+1])
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            d1['enc_relu'+str(i)] = nn.ReLU() #if change?
        self.encoder = nn.Sequential(d1)
        self.lstm = LSTM(input_size=fea_size*hidden[-1], hidden_size=fea_size*hidden[-1], cell_size=fea_size*hidden[-1])
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.fea_size, self.att_size)
        # print('em0', x.size())
        x = self.encoder(x)
        # print('em1', x.size())
        x = x.view(-1, self.step_size, x.size()[1]*x.size()[2])
        # print('em1', x.size())
        x, _ = self.lstm.loop(x)
        # print('em2', x.size())
        x = x.view(x.size()[0], self.fea_size, -1)
        # print('em2', x.size())
        x = self.fc(F.relu(x))
        # print('em3', x.size())
        return x # batch size * n_cnt * n_emb