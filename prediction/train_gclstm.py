import torch
import numpy as np
from torch.autograd import Variable
import time
from prediction.GCLSTM import *

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')

def TrainGCLSTM(train_dataloader, valid_dataloader, A, K, num_epochs=1):
    torch.autograd.set_detect_anomaly(True)
    atts, cases, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size, att_size] = atts.size()
    [_, pred_size, _] = labels.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    gclstm = GCLSTM(att_size, step_size, fea_size, K, torch.Tensor(A), pred_size)
    gclstm.to(DEVICE)
    learning_rate = 1e-4
    optimizer = torch.optim.RMSprop(gclstm.parameters(), lr=learning_rate)

    ###################### print model parameter states ######################
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in gclstm.state_dict():
        print(param_tensor, '\t', gclstm.state_dict()[param_tensor].size())
        total_param += np.prod(gclstm.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])
    ###########################################################################

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []

    losses_epoch = []

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        # validation data loader iterator init
        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            gclstm.train()
            atts, cases, labels = data

            if atts.shape[0] != batch_size:
                continue

            atts, cases, labels = Variable(atts.float().to(DEVICE)), Variable(cases.float().to(DEVICE)), Variable(
                labels.float().to(DEVICE))

            gclstm.zero_grad()           
            pred = gclstm(atts, cases)
            loss_train = loss_MSE(pred, labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses_train.append(loss_train.data)

            # validation
            try:
                gclstm.eval()
                atts_val, cases_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                atts_val, cases_val, labels_val = next(valid_dataloader_iter)

            atts_val, cases_val, labels_val = Variable(atts_val.float().to(DEVICE)), Variable(cases_val.float().to(DEVICE)), Variable(labels_val.float().to(DEVICE))

            pred = gclstm(atts_val, cases_val)
            loss_valid = loss_MSE(pred, labels_val)
            losses_valid.append(loss_valid.data)

            # output
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format( \
                    trained_number * batch_size, \
                    loss_interval_train, \
                    loss_interval_valid, \
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

        loss_epoch = loss_valid.cpu().data.numpy()
        print('Validation loss:', loss_epoch)
        losses_epoch.append(loss_epoch)
    np.save('loss_gclstm.npy', np.array(losses_epoch), allow_pickle=True)

    return gclstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def TestGCLSTM(gclstm, test_dataloader, max_speed):
    gclstm.eval()
    atts, cases, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size, att_size] = atts.size()
    [_, pred_size, _] = labels.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    MSPEs = []
    RMSEs = []
    R2s = []
    VARs = []
    for data in test_dataloader:
        atts, cases, labels = data

        if atts.shape[0] != batch_size:
            continue
            
        atts, cases, labels = Variable(atts.float().to(DEVICE)), Variable(cases.float().to(DEVICE)), Variable(
                labels.float().to(DEVICE))

        pred = gclstm(atts, cases)
        
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()

        loss_mse = loss_MSE(pred, labels)
        loss_l1 = loss_L1(pred, labels)
        MAE = torch.mean(torch.abs(pred - labels))
        MAPE = torch.mean(torch.abs(pred - labels) / labels)
        MSE = torch.mean((labels - pred)**2)
        MSPE = torch.mean(((pred - labels) / labels)**2)
        RMSE = math.sqrt(torch.mean((labels - pred)**2))
        R2 = 1-((labels-pred)**2).sum()/(((labels)-(labels).mean())**2).sum()
        VAR = 1-(torch.var(labels-pred))/torch.var(labels)

        losses_mse.append(loss_mse.item())
        losses_l1.append(loss_l1.item())
        MAEs.append(MAE.item())
        MAPEs.append(MAPE.item())
        MSEs.append(MSE.item())
        MSPEs.append(MSPE.item())
        RMSEs.append(RMSE)
        R2s.append(R2.item())
        VARs.append(VAR.item())

        tested_batch += 1

        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_l1.data[0]], decimals=8), \
                np.around([loss_mse.data[0]], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    MSPEs = np.array(MSPEs)
    RMSEs = np.array(RMSEs)
    R2s = np.array(R2s)
    VARs = np.array(VARs)

    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    MSPE_ = np.mean(MSPEs)  * 100
    RMSE_ = np.mean(RMSEs) * max_speed
    R2_ = np.mean(R2s)
    VAR_ = np.mean(VARs)
    results = [MAE_, std_MAE_, MAPE_, MSE_, MSPE_, RMSE_, R2_, VAR_]

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, MSPE: {}, RMSE: {}, R2: {}, VAR: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, MSPE_, RMSE_, R2_, VAR_))
    return results