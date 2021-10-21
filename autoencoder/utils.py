import numpy as np

def fill_original():
    """
    Fill existing values back to autoencoder results
    :return:
    """
    filled = np.load('../data/inputs_filled.npy')
    inputs = np.load('../data/inputs.npy')

    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            mask = np.ones(inputs.shape[2])

            if np.any(inputs[i, j, :40]): # if all set of features is zero, mask it
                filled[i, j, :40] = inputs[i, j, :40]
            if np.any(inputs[i, j, 40:80]):
                filled[i, j, 40:80] = inputs[i, j, 40:80]
            if np.any(inputs[i, j, 80:]):
                filled[i, j, 80:] = inputs[i, j, 80:]

    np.save('../data/inputs_final.npy', filled, allow_pickle=True)