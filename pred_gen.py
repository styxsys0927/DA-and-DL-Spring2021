########## generate predicted result for the whole dataset ##########
import torch.utils.data as utils
import numpy as np
import torch
import pandas as pd
from prediction.prepare import *
from prediction.train_dkfn import *
from prediction.DKFN import *
from prediction.GCLSTM import *
from prediction.RNN import *
from prediction.neDKFN import *
from prediction.dataset import Dataset as pDataset

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')

def test(model_name, model, dataloader, max_speed, pred_size=1):
    model.eval()
    atts, cases, labels = next(iter(dataloader))
    [batch_size, step_size, fea_size, att_size] = atts.size()
    [_, pred_size, _] = labels.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    tested_batch = 0

    MAEs = []
    MAPEs = []
    MSEs = []
    MSPEs = []
    RMSEs = []
    R2s = []
    VARs = []

    predictions = np.array([], dtype=np.float64).reshape(0, pred_size, fea_size)
    ground_truths = np.array([], dtype=np.float64).reshape(0, pred_size, fea_size)

    for data in dataloader:
        atts, cases, labels = data

        if atts.shape[0] != batch_size:
            continue

        atts, cases, labels = Variable(atts.float().to(DEVICE)), Variable(cases.float().to(DEVICE)), Variable(labels.float().to(DEVICE))

        pred = model(atts, cases)

        MAE = torch.mean(torch.abs(pred - labels))
        MAPE = torch.mean(torch.abs(pred - labels) / labels)
        MSE = torch.mean((labels - pred)**2)
        MSPE = torch.mean(((pred - labels) / labels)**2)
        RMSE = math.sqrt(torch.mean((labels - pred)**2))
        R2 = 1-((labels-pred)**2).sum()/(((labels)-(labels).mean())**2).sum()
        VAR = 1-(torch.var(labels-pred))/torch.var(labels)

        MAEs.append(MAE.item())
        MAPEs.append(MAPE.item())
        MSEs.append(MSE.item())
        MSPEs.append(MSPE.item())
        RMSEs.append(RMSE)
        R2s.append(R2.item())
        VARs.append(VAR.item())

        predictions = np.concatenate((predictions, pred.cpu().data.numpy()*max_speed), axis=0)
        ground_truths = np.concatenate((ground_truths, labels.cpu().data.numpy()*max_speed), axis=0)

        tested_batch += 1

    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    MSPEs = np.array(MSPEs)
    RMSEs = np.array(RMSEs)
    R2s = np.array(R2s)
    VARs = np.array(VARs)

    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    MSPE_ = np.mean(MSPEs) * 100
    RMSE_ = np.mean(RMSEs) * max_speed
    R2_ = np.mean(R2s)
    VAR_ = np.mean(VARs)

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, MSPE: {}, RMSE: {}, R2: {}, VAR: {}'.format(MAE_, std_MAE_,
                                                                                                        MAPE_, MSE_,
                                                                                                        MSPE_, RMSE_,
                                                                                                        R2_, VAR_))
    print('preds', predictions.shape, 'truth', ground_truths.shape)
    return predictions, ground_truths


attributes = np.load('data/inputs_final.npy')
cases = np.load('data/labels.npy')
A = np.load('data/adj_mat_full.npy')
max_case = np.abs(cases).max()

seq_len, pred_len = 10, 1
BATCH_SIZE = 4
K = 3
pred_size = 1

# load data
print('Loading data...')
att_sequences, case_sequences, case_labels = extract_label(cases, attributes, seq_len, pred_len, max_case)
dataset = pDataset(att_sequences, case_sequences, case_labels)
dataloader = utils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# DKFN
# print('Generating DKFN prediction results...')
# dkfn = DKFN(att_sequences.shape[3], att_sequences.shape[1], att_sequences.shape[2], K, torch.Tensor(A), pred_size)
# dkfn.to(DEVICE)
#
# dkfn.load_state_dict(torch.load('models/dkfn'))
# predictions, ground_truths = test('dkfn', dkfn, dataloader, max_case, pred_size=pred_size)
# np.save('results/preds_dkfn.npy', predictions, allow_pickle=True)
# np.save('results/truth.npy', ground_truths, allow_pickle=True)
#
# # GCLSTM
# print('Generating GCLSTM prediction results...')
# gclstm = GCLSTM(att_sequences.shape[3], att_sequences.shape[1], att_sequences.shape[2], K, torch.Tensor(A), pred_size)
# gclstm.to(DEVICE)
#
# gclstm.load_state_dict(torch.load('models/gclstm'))
# predictions, ground_truths = test('gclstm', gclstm, dataloader, max_case, pred_size=pred_size)
# np.save('results/preds_gclstm.npy', predictions, allow_pickle=True)
#
# # LSTM
# print('Generating LSTM prediction results...')
# lstm = LSTM(att_sequences.shape[3], att_sequences.shape[1], att_sequences.shape[2], pred_size)
# lstm.to(DEVICE)
#
# lstm.load_state_dict(torch.load('models/lstm'))
# predictions, ground_truths = test('lstm', lstm, dataloader, max_case, pred_size=pred_size)
# np.save('results/preds_lstm.npy', predictions, allow_pickle=True)

# neDKFN
print('Generating neDKFN prediction results...')
nedkfn = neDKFN(att_sequences.shape[3], att_sequences.shape[1], att_sequences.shape[2], K, torch.Tensor(A), pred_size)
nedkfn.to(DEVICE)

nedkfn.load_state_dict(torch.load('models/nedkfn'))
predictions, ground_truths = test('nedkfn', nedkfn, dataloader, max_case, pred_size=pred_size)
np.save('results/preds_nedkfn.npy', predictions, allow_pickle=True)