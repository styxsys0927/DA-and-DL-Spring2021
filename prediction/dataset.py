import numpy as np
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, att_seq, case_seq, case_label):
        self.att_seq = att_seq
        self.case_seq = case_seq
        self.case_label = case_label

    def __getitem__(self, index):
        # n_cnt * n_att * n_time, n_cnt * n_time, n_cnt * 1
        return self.att_seq[index], self.case_seq[index], self.case_label[index]

    def __len__(self):
        return self.att_seq.shape[0]
