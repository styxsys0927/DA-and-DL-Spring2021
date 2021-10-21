"""
The main network structure.(pytorch style)
"""
from collections import OrderedDict
import torch
from torch import nn

# AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super(AutoEncoder, self).__init__()
        d1 = OrderedDict()
        for i in range(len(hidden)-1):
            d1['enc_linear' + str(i)] = nn.Linear(hidden[i], hidden[i + 1])
            #d1['enc_bn' + str(i)] = nn.BatchNorm1d(hidden[i+1])
            d1['enc_drop' + str(i)] = nn.Dropout(dropout)
            d1['enc_relu'+str(i)] = nn.ReLU() #if change?
        self.encoder = nn.Sequential(d1)
        d2 = OrderedDict()
        for i in range(len(hidden) - 1, 0, -1):
            d2['dec_linear' + str(i)] = nn.Linear(hidden[i], hidden[i - 1])
            #d2['dec_bn' + str(i)] = nn.BatchNorm1d(hidden[i - 1])
            d2['dec_drop' + str(i)] = nn.Dropout(dropout)
            d2['dec_relu' + str(i)] = nn.Sigmoid()
        self.decoder = nn.Sequential(d2)

    def forward(self, x):
        x = self.decoder(self.encoder(x))
        return x