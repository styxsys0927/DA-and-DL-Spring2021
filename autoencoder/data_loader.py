import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path_prefix = '../data/'

def scaler(inputs, i):
    sclr = MinMaxScaler()
    sclr.fit(inputs[:, :, i])
    res = sclr.transform(inputs[:, :, i])
    return res, sclr

def load_data(dataset='inputs', train_ratio=0.9):
    """
    Attributes and mark missing data, split into train and test
    :param dataset:
    :param train_ratio:
    :return: train list, test list, n_timestep, n_county
    """
    fname = path_prefix+dataset+'.npy'
    records = []

    if not os.path.exists(fname):
        print('[Error] File %s not found!' % fname)
        sys.exit(-1)

    inputs = np.load(fname)
    n_nan = np.isnan(inputs).sum()
    if n_nan != 0:
        print('nan:', n_nan)
        inputs = np.nan_to_num(inputs)
        print('filled',np.isnan(inputs).sum())
        np.save(fname, inputs, allow_pickle=True)
    print(inputs.shape)

    scalers = []
    for i in range(inputs.shape[2]):
        inputs[:, :, i], sclr = scaler(inputs, i)
        scalers.append(sclr)

    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            mask = np.ones(inputs.shape[2])

            if not np.any(inputs[i, j, :40]): # if all set of features is zero, mask it
                mask[:40] = 0
            if not np.any(inputs[i, j, 40:80]):
                mask[40:80] = 0
            if not np.any(inputs[i, j, 80:]):
                mask[80:] = 0

            records.append((i, j, inputs[i, j], mask)) # day id, county id, features

    np.random.shuffle(records)
    train_list = records[0:int(len(records)*train_ratio)]
    test_list = records[int(len(records)*train_ratio):]
    return train_list, test_list, inputs.shape[0], inputs.shape[1], scalers