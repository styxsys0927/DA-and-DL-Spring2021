import autoencoder.dataset as ds
import autoencoder.model as model
from datetime import datetime
from autoencoder.data_loader import load_data
import torch.utils.data as utils
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

# parameters
rank = 100
batch_size = 32
user_based = False

if __name__ == '__main__':
    start = datetime.now()
    train_list, test_list, n_time, n_cnt, scalers = load_data('inputs', 0.9)

    train_dataset = ds.Dataset(train_list, n_time, n_cnt)
    test_dataset = ds.Dataset(test_list, n_time, n_cnt)

    mod = model.Model(hidden=[train_list[0][2].shape[0], 100, 100, 50, 20, 10], # n_neurons of each layer
                      learning_rate=0.001,
                      batch_size=batch_size,
                      n_time=n_time,
                      n_cnt=n_cnt)

    outputs = mod.run(train_dataset, test_dataset, num_epoch=100)
    end = datetime.now()
    print("Total time: %s" % str(end-start))

    print('output:', outputs.shape)
    for i, s in enumerate(scalers):
        outputs[:, :, i] = s.inverse_transform(outputs[:, :, i])
    np.save('../data/inputs_filled.npy', outputs, allow_pickle=True)