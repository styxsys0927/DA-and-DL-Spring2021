"""

"""
import numpy as np
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, att_list, n_time, n_cnt):
        self.tid = [a[0] for a in att_list]
        self.cid = [a[1] for a in att_list]
        self.features = [a[2] for a in att_list]
        self.masks = [a[3] for a in att_list]
        self.n_time = n_time
        self.n_cnt = n_cnt

    def __getitem__(self, index):
        return self.tid[index], self.cid[index], self.features[index], self.masks[index]

    def __len__(self):
        return len(self.tid)
