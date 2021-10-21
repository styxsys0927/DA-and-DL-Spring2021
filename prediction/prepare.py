import torch.utils.data as utils
import numpy as np
import torch
import pandas as pd
from prediction.dataset import Dataset as pDataset
from sklearn.preprocessing import MinMaxScaler

def scaler(inputs, i):
    sclr = MinMaxScaler()
    sclr.fit(inputs[:, :, i])
    res = sclr.transform(inputs[:, :, i])
    return res, sclr

def extract_label(cases, attributes, seq_len, pred_len, max_case):
    """ Turn speed matrix into sequence-and-label pair

    Args:
        cases: raw speed matrix
        seq_len: length of input sequence
        pred_len: length of predicted sequence

    Returns: speed_sequences, speed_labels

    """
    time_len = cases.shape[0]
    # normalize cases
    cases = cases / max_case

    # normalize attributes
    for i in range(attributes.shape[2]):
        attributes[:, :, i], sclr = scaler(attributes, i)

    att_sequences, case_sequences, case_labels = [], [], []
    for i in range(time_len - seq_len - pred_len):
        att_sequences.append(attributes[i:i + seq_len])
        case_sequences.append(cases[i:i + seq_len])
        case_labels.append(cases[i + seq_len:i + seq_len + pred_len])
    att_sequences, case_sequences, case_labels = np.array(att_sequences), np.array(case_sequences), np.array(case_labels)

    print('att', att_sequences.shape, 'case', case_sequences.shape, 'label', case_labels.shape)

    return att_sequences, case_sequences, case_labels

def PrepareDataset(attributes, cases, BATCH_SIZE=40, seq_len=10, pred_len=1, train_propotion=0.7, valid_propotion=0.1):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        cases: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    max_case = np.abs(cases).max()

    att_sequences, case_sequences, case_labels = extract_label(cases, attributes, seq_len, pred_len, max_case)

    # shuffle and split the dataset to training and testing datasets
    sample_size = case_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_dataset = pDataset(att_sequences[:train_index], case_sequences[:train_index], case_labels[:train_index])
    valid_dataset = pDataset(att_sequences[train_index:valid_index], case_sequences[train_index:valid_index], case_labels[train_index:valid_index])
    test_dataset = pDataset(att_sequences[valid_index:], case_sequences[valid_index:], case_labels[valid_index:])

    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_case
