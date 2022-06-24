from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
#1500
parser.add_argument('--lr', type=float, default=0.05, help='learning rate.')
#0.01
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=16, help='hidden dimensions.')
# 64
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
#0.5
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
adj, features, labels,idx_train,idx_val,idx_test = load_datamat()
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

model = GCNII(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    # model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        # spe_val, sen_val = calculate_index(labels[idx_val], output[idx_val])
        spe_test, sen_test = calculate_index(labels[idx_test], output[idx_test])
        # spe = (spe_val + spe_test) / 2
        # sen = (sen_val + sen_test) / 2
        return loss_test.item(),acc_test.item(), sen_test, spe_test

def train_gcnii():
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train()
        loss_val, acc_val = validate()
        loss_test, acc_test, sen_test, spe_test = test()
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100),
                  '| test',
                  'acc:{:.2f}'.format(acc_test * 100),
                  'sen:{:.2f}'.format(sen_test * 100),
                  'spe:{:.2f}'.format(spe_test * 100),
                  )

        # if loss_val < best:
        #     best = loss_val
        #     best_epoch = epoch
        #     acc = acc_val
        #     torch.save(model.state_dict(), checkpt_file)
        #     bad_counter = 0
        # else:
        #     bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        acc = test()[1]
        sen = test()[2]
        spe = test()[3]

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val", "acc.:{:.4f}".format(acc * 100), "sen.:{:.4f}".format(sen * 100), "spe.:{:.4f}".format(spe * 100))

    return acc, sen, spe

if __name__ == '__main__':
    acclist = []
    senlist = []
    spelist = []
    for epoch in range(10):
        # np.random.seed(epoch*10)
        # torch.manual_seed(epoch*10)
        # if args.cuda:
        #     torch.cuda.manual_seed(72)
        measures = train_gcnii()
        acclist.append(measures[0])
        senlist.append(measures[1])
        spelist.append(measures[2])

    accmean = np.mean(acclist)
    # accmean = torch.mean(torch.stack(acclist))
    senmean = np.mean(senlist)
    spemean = np.mean(spelist)
    accmax = max(acclist, key = abs)
    accmaxi = acclist.index(accmax)
    spemax = spelist[accmaxi]
    senmax = senlist[accmaxi]
    # accstd = torch.std(torch.stack(acclist))
    accstd = np.std(acclist, ddof=1)
    spestd = np.std(spelist, ddof=1)
    senstd = np.std(senlist, ddof=1)
    print("all_acc={:.4f}, all_sen={:.4f}, all_spe={:.4f}".format(accmean, senmean, spemean))
    print(
        "acc={:.2f}+-{:.2f}  sen={:.2f}+-{:.2f}  spe={:.2f}+={:.2f}".format(accmean * 100, accstd * 100, senmean * 100,
                                                                            senstd * 100, spemean * 100, spestd * 100))
    print("best_acc={:.4f}, sen={:.4f}, spe={:.4f}".format(accmax, senmax, spemax))





