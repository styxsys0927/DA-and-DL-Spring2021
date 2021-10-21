import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DKFN import *
from sklearn.preprocessing import MinMaxScaler


###################### fill nan with values ##########################
def fill_nan0(df, ind, col):
    """find the closest not nan along time axis (0 means along index) """
    if not np.isnan(df[col][ind]):
        return

    i = j = ind
    while i > 0 and np.isnan(df[col][i]):
        i -= 1
    while j < len(df.index)-1 and np.isnan(df[col][j]):
        j += 1

    if i == 0 and j == len(df.index)-1: # fill by columns if all nan in this row
        for k in range(i, j):
            fill_nan1(df, k, col)
    else:
        if np.isnan(df[col][i]):
            df[col][i] = df[col][j]
        if np.isnan(df[col][j]):
            df[col][j] = df[col][i]

        for k in range(i+1, j):
            df[col][k] = (df[col][j]-df[col][i])/(j-i)*(k-i)+df[col][i]

def fill_nan1(df, ind, col):
    """find the closest not nan along detector axis (1 means along column) """
    i = j = col
    while i > 0 and np.isnan(df[i][ind]):
        i -= 1
    while j < len(df.columns)-1 and np.isnan(df[j][ind]):
        j += 1
    if np.isnan(df[i][ind]):
        df[i][ind] = df[j][ind]
    if np.isnan(df[j][ind]):
        df[j][ind] = df[i][ind]

    for k in range(i+1, j):
        df[k][ind] = (df[j][ind]-df[j][ind])/(j-i)*(k-i)+df[i][ind]

# speed_matrix = pd.DataFrame(np.load('PeMS08_Dataset/occupancy.npy'))
# print(len(speed_matrix), len(speed_matrix.columns))
#
# # speed_matrix = speed_matrix.replace(0, np.nan)
# print(speed_matrix.isna().sum().sum()/(len(speed_matrix)*len(speed_matrix.columns)))
#
# nan_lst = np.where(np.asanyarray(np.isnan(speed_matrix)))
# for ind, col in zip(nan_lst[0], nan_lst[1]):
#     # print(ind, col)
#     fill_nan0(speed_matrix, ind, col)
# print(speed_matrix.isna().sum().sum())
# np.save('PeMS08_Dataset/occupancy1.npy', speed_matrix, allow_pickle=True)


###################### plot original data ##########################
# speed_matrix = np.load('PeMS08_Dataset/speed1.npy')
accidents = np.load('PeMS08_Dataset/accidents.npy')
dur_dt = accidents[7699:8600, :].sum(axis=0)
dur_dt = dur_dt[dur_dt>0]
print('accident frequency', dur_dt.shape, dur_dt.sum())
# for i in range(dur_dt.shape[0]):
#     condition = dur_dt[i]
#     if condition[0]:
#         condition = np.concatenate([[False], condition])
#     print(dur_dt[i])

# print('timesteps', accidents.shape)
# ind = 1
# x = [i for i in range(accidents.shape[0]-1)]
# st, ed = 8400, 8600
# ac = np.ma.masked_where(accidents < 0.001, speed_matrix)
# nc = np.ma.masked_where(accidents > 0.001, speed_matrix)
# print(ac[:, 0].shape)
# plt.plot(x[st:ed], ac[st:ed, ind], label='accident')
# plt.plot(x[st:ed], nc[st:ed, ind], label='normal')
# plt.legend()
# plt.show()

# dataframe: timestamp * # detector
# speed_origin = pd.read_pickle('./METR_LA_Dataset/la_speed')
# speed_noisy = pd.read_pickle('./METR_LA_Dataset/speed_matrix_new')

# adjacent matrix of 0 and 1
# A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
# adj_matrix = pd.read_pickle('PeMS08_Dataset/pems08_adj_mat_clean.pkl')
# A = np.array(adj_matrix[2])
# for i in range(A.shape[0]):
#     A[i, i] = 1.
# np.save('PeMS08_Dataset/pems08_adj_mat_clean.npy', A, allow_pickle=True)
# print(np.rint(A[:10][:10]).astype(int))
# print(np.isnan(A).any())


# print(speed_noisy.shape)
# print(speed_matrix.describe())

# low_t = np.mean(speed_origin, axis=0)-np.std(speed_origin, axis=0)
# high_t = np.mean(speed_origin, axis=0)+np.std(speed_origin, axis=0)
# print(low_t, high_t)

# plot speed distribution
# plt.hist(speed_matrix[:][0], bins=100)
# plt.xlabel('speed')
# plt.ylabel('count')
# plt.show()

###################### plot prediction result against label ##########################
# # model_lst = ['dkfn_pems_9', 'dkfn_pems_29', 'dkfn_pems_59', 'dkfn_pems_99', 'dkfn_pems']
dkfn = np.load('PeMS08_Dataset/compare/preds_origin_dkfn_pems.npy')
mdkfn1 = np.load('PeMS08_Dataset/compare/output_epoch_69_test.npz')['prediction']
mdkfn2 = np.load('PeMS08_Dataset/compare/output_epoch_79_test.npz')['prediction']
# print(mdkfn2.shape)

# align the start timeslot of different prediction result
speed_origin0 = np.load('PeMS08_Dataset/compare/truth_origin.npy')
speed_origin1 = np.load('PeMS08_Dataset/compare/output_epoch_69_test.npz')['data_target_tensor']
speed_origin2 = np.load('PeMS08_Dataset/compare/output_epoch_79_test.npz')['data_target_tensor']
#
# start = speed_origin1[403, 0]
# print(speed_origin0.shape, speed_origin1.shape, speed_origin2.shape)
# for i in range(7680, 8600):
#     _s = np.abs(speed_origin0[i, :, 0]-start).sum()
#     if _s < 1.:
#         print(i, speed_origin0[i, :, 0])
# print(speed_origin0[7699, :, 0:2], speed_origin1[403, 0:2], speed_origin2[0, 0:2])

# compute mse for anomaly score

# plot prediction against origin
# # for i, label in enumerate(model_lst):
# #     speed_train = np.load('PeMS08_Dataset/preds_origin_'+label+'.npy')
# #     plt.plot(x[st:ed], speed_train[st:ed, ind], label=label)
# # plt.plot(x[st:ed], speed_origin[695:1295, ind, 0], label='origin')
# for i in range(2):
#     # plt.plot(x[st:ed], speed_origin[st-10-i:ed-10-i, i, ind], label='origin')
#     plt.plot(x[st:ed], ac[st:ed, ind], label='accident')
#     plt.plot(x[st:ed], nc[st:ed, ind], label='normal')
#     plt.plot(x[st:ed], dkfn[st-10+i:ed-10+i, i, ind], c='r', label='dkfn_'+str(i))
#     plt.plot(x[st:ed], mdkfn1[1083+11-i:1283+11-i, ind, i], c='g', label='mdkfn1_'+str(i))
#     plt.plot(x[st:ed], mdkfn2[680+11-i:880+11-i, ind, i], c='b', label='mdkfn2_'+str(i))
#     plt.xlabel('timestamp')
#     plt.ylabel('speed')
#     plt.legend()
#     plt.show()
###################### anomaly score ##########################
# speed_origin = np.load('PeMS08_Dataset/truth_origin.npy')
# speed_train = np.load('PeMS08_Dataset/preds_origin_dkfn_pems.npy')
# delta = np.square(speed_origin - speed_train)
# scaler = MinMaxScaler()
# scaler.fit(delta)
# print(scaler.data_max_)
# delta = scaler.transform(delta)
# ac = np.ma.masked_where(accidents[:-33] < 0.001, delta)
# nc = np.ma.masked_where(accidents[:-33] > 0.001, delta)
# print(ac[:, 0].shape)
# plt.plot(x[st:ed], ac[st:ed, ind], label='accident')
# plt.plot(x[st:ed], nc[st:ed, ind], label='normal')
# plt.legend()
# plt.show()

