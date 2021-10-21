import argparse
from prediction.prepare import *
from prediction.train_dkfn import *
from prediction.train_rnn import *
from prediction.train_gclstm import *
from prediction.train_nedkfn import *

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

parser = argparse.ArgumentParser(description='covid case prediction')

# model
parser.add_argument('-model', type=str, default='lstm', help='choose model to train and test [options: lstm, gclstm, dkfn, nedkfn]')
args = parser.parse_args()

# load data
print("\nLoading data...")
attributes = np.load('data/inputs_final.npy')
cases = np.load('data/labels.npy')
A = np.load('data/adj_mat_full.npy')

print("\nPreparing data...")
train_dataloader, valid_dataloader, test_dataloader, max_case = PrepareDataset(attributes, cases, BATCH_SIZE=4, pred_len=1)

# model
if args.model == 'dkfn':
    print("\nTraining dkfn model...")
    dkfn, dkfn_loss = TrainDKFN(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    torch.save(dkfn.state_dict(), 'models/dkfn')
    print("\nTesting dkfn model...")
    results = TestDKFN(dkfn, test_dataloader, max_case)

if args.model == 'nedkfn':
    print("\nTraining nedkfn model...")
    nedkfn, nedkfn_loss = TrainneDKFN(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    torch.save(nedkfn.state_dict(), 'models/nedkfn')
    print("\nTesting nedkfn model...")
    results = TestDKFN(nedkfn, test_dataloader, max_case)

elif args.model == 'lstm':
    print("\nTraining lstm model...")
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs=100)
    torch.save(lstm.state_dict(), 'models/lstm')
    print("\nTesting lstm model...")
    results = TestLSTM(lstm, test_dataloader, max_case)

elif args.model == 'gclstm':
    print("\nTraining gclstm model...")
    gclstm, gclstm_loss = TrainGCLSTM(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    torch.save(gclstm.state_dict(), 'models/gclstm')
    print("\nTesting gclstm model...")
    results = TestGCLSTM(gclstm, test_dataloader, max_case)


