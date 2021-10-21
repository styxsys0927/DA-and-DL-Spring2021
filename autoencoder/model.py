import torch
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim, nn
import torch.nn.functional as F
import numpy as np

import autoencoder.networks as nets

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

# the main model to do main process:
# including training, testing, evaluating
class Model:
    def __init__(self, hidden, learning_rate, batch_size, n_time, n_cnt):
        self.batch_size = batch_size
        self.net = nets.AutoEncoder(hidden)
        self.net.to(DEVICE)
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=0.04) # for fine-tone
        self.feature_size = hidden[0] # number of features
        self.res = np.zeros((n_time, n_cnt, hidden[0]))

    def fill_res(self, tids, cids, outputs):
        for i in range(tids.shape[0]):
            self.res[tids[i], cids[i]] = outputs[i]

    def run(self, trainset, testset, num_epoch):
        torch.autograd.set_detect_anomaly(True) # for debug
        train_loader = DataLoader(trainset, self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(testset, self.batch_size, shuffle=False, pin_memory=True)
        for epoch in range(1, num_epoch + 1):
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            trn_tid, trn_cid, out_train = self.train(train_loader, epoch)
            tst_tid, tst_cid, out_test = self.test(test_loader)

        torch.save(self.net.state_dict(), '../models/autoencoder')
        self.fill_res(trn_tid, trn_cid, out_train)
        self.fill_res(tst_tid, tst_cid, out_test)
        return self.res

    def train(self, train_loader, epoch):
        self.net.train()
        tids, cids, outputs = [], [], []
        for bid, (tid, cid, feature, mask) in enumerate(train_loader):
            feature = Variable(feature.float()).to(DEVICE)
            mask = Variable(mask).to(DEVICE)
            self.opt.zero_grad()
            output = self.net(feature)
            loss = F.mse_loss(output * mask, feature * mask) # masked parts do not count

            loss.backward()
            self.opt.step()

            tids.append(tid.cpu().data.numpy())
            cids.append(cid.cpu().data.numpy())
            outputs.append(output.cpu().data.numpy())

        tids, cids, outputs = np.concatenate(tids), np.concatenate(cids), np.concatenate(outputs)
        print("Epoch {}, train end. loss: {}".format(epoch, loss.cpu().data))
        return tids, cids, outputs

    def test(self, test_loader):
        self.net.eval()

        rmse = []
        tids, cids, outputs = [], [], []
        for bid, (tid, cid, feature, mask) in enumerate(test_loader):
            features = Variable(feature.float()).to(DEVICE)
            masks = Variable(mask).to(DEVICE)
            output = self.net(features)
            rmse.append(np.sum(((output * masks).cpu().data.numpy() - (features * masks).cpu().data.numpy())**2)**0.5)  # masked parts do not count

            tids.append(tid.cpu().data.numpy())
            cids.append(cid.cpu().data.numpy())
            outputs.append(output.cpu().data.numpy())

        tids, cids, outputs = np.concatenate(tids), np.concatenate(cids), np.concatenate(outputs)
        rmse = math.sqrt(sum(rmse) / len(test_loader))

        print(" Test RMSE = %f" % rmse)
        return tids, cids, outputs