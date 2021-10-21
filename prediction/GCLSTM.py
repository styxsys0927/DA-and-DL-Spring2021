import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from prediction.modules import FilterLinear
from prediction.att_embeddings import Embeddings
import math
import numpy as np

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')

class GCLSTM_module(nn.Module):

    def __init__(self, K, A, feature_size, Clamp_A=True):
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: the dimension of features
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(GCLSTM_module, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.K = K
        self.A_list = []  # Adjacency Matrix List

        # normalization
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A = torch.FloatTensor(A)
        A_temp = torch.eye(feature_size, feature_size)
        for i in range(K):
            A_temp = torch.matmul(A_temp, torch.Tensor(A))
            if Clamp_A:
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)
            # self.A_list.append(torch.mul(A_temp, torch.Tensor(FFR)))
            self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList(
            [FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])

        hidden_size = self.feature_size
        input_size = self.feature_size * K

        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

    def forward(self, input, Hidden_State, Cell_State):

        x = input

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        combined = torch.cat((gc, Hidden_State), 1)
        # print(gc.size(), Hidden_State.size(), combined.size())
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).to(DEVICE), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        return Hidden_State, Cell_State, gc

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        for i in range(time_step):
            Hidden_State, Cell_State, gc = self.forward(torch.squeeze(inputs[:, i:i + 1, :]), Hidden_State, Cell_State)
        return Hidden_State, Cell_State

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        return Hidden_State, Cell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        Hidden_State = Variable(Hidden_State_data.to(DEVICE), requires_grad=True)
        Cell_State = Variable(Cell_State_data.to(DEVICE), requires_grad=True)
        return Hidden_State, Cell_State
        
class GCLSTM(nn.Module):
    def __init__(self, att_size, step_size, fea_size, K, A, pred_size=1, Clamp_A=True):
        super(GCLSTM, self).__init__()

        self.att_size = att_size

        self.embedding = Embeddings([att_size, 100, 100, 50, 20, 10], step_size, fea_size, att_size) # output feature size is 10

        self.gclstm = GCLSTM_module(K, A, fea_size, Clamp_A) # output feature size is 1

        self.fc = nn.Linear(11, pred_size)

    def forward(self, atts, cases):
        atts = self.embedding(atts)
        cases, _ = self.gclstm.loop(cases)
        cases = cases.unsqueeze(2)
        # print('merge',atts.size(), cases.size())
        outputs = self.fc(torch.cat([atts, cases], dim=2))
        return outputs.transpose(1, 2) # batch size * pred_size * n_cnt