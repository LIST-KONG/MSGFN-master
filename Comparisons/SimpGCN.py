import scipy.io as sio
import argparse
from deeprobust.graph.defense import GAT, GCN, RGCN, GCNSVD, GCNJaccard, ProGNN, SimPGCN, ChebNet
from deeprobust.graph.data import Dataset
from utils import noise_adj_fea, preprocess
import numpy as np
import scipy.sparse as sp
import torch
import json
from sklearn.neighbors import kneighbors_graph
from numpy import mat
from scipy.sparse import lil_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='polblogs',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
# parser.add_argument('--ptb_rate', type=float, default=0.05,  help='perturbation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def readfile():
    filena = "/data/features_final.mat"
    features = sio.loadmat(filena)["features"]
    features = mat(features)
    labels = sio.loadmat(filena)["labels"]
    labels = labels.argmax(axis=1)
    sample_num = features.shape[0]

    K_neighbour = 0.1
    # adj = construct_graph(features, metric_type='static_KNN', knn_size=int(K_neighbour * sample_num))
    adj = kneighbors_graph(features, int(K_neighbour * sample_num), metric='cosine', include_self=True)
    adj = lil_matrix(adj)
    all_idx = np.arange(len(labels))
    idx_train = all_idx[:int(0.6 * len(labels))]
    # idx_val = all_idx[int(0.4*len(labels)):int(0.6*len(labels))]
    idx_val = None
    idx_test = all_idx[int(0.6 * len(labels)):]

    return adj, features, labels, idx_train, idx_val, idx_test


from sklearn.metrics import confusion_matrix


def train(adj, features, labels, idx_train, idx_val, idx_test):
    # 读入数据
    # adj, features, labels = data.adj, data.features, data.labels
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    acclist = {}

    # #GCN
    gcnmodel = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max() + 1, device=device)
    gcnmodel = gcnmodel.to(device)
    print('==================')
    print('=== train with GCN ===')
    gcnmodel.fit(features, adj, labels, idx_train, train_iters=500, verbose=True)
    gcnmodel.eval()
    # You can use the inner function of model to test
    gcn_acc = gcnmodel.test(idx_test)
    # acclist['gcn'] = gcn_acc
    output = gcnmodel.predict()
    output = output.max(1)[1].type_as(torch.LongTensor(labels))
    preds = output.cpu().detach().numpy()
    # preds = output.max(1)[1].type_as(labels)
    matrix = confusion_matrix(labels[idx_test], preds[idx_test])
    svdspe = matrix[0][0] * 1.0 / (matrix[0][0] + matrix[0][1])
    svdsen = matrix[1][1] * 1.0 / (matrix[1][0] + matrix[1][1])
    # print(svdspe, svdsen)
    acclist['gcnsvd'] = gcn_acc
    acclist['svdsen'] = svdsen
    acclist['svdspe'] = svdspe

    # SimP-GCN
    simp = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nhid=32, nclass=labels.max() + 1,
                   dropout=0.5, lr=0.01, weight_decay=5e-2, lambda_=1, gamma=0.01, device=device)
    simp = simp.to(device)
    print('=== train with simP ===')
    simp.fit(features, adj, labels, idx_train, idx_val, train_iters=100, verbose=True)
    simpacc = simp.test(idx_test)
    output = simp.predict()
    output = output.max(1)[1].type_as(torch.LongTensor(labels))
    preds = output.cpu().detach().numpy()
    # preds = output.max(1)[1].type_as(labels)
    matrix = confusion_matrix(labels[idx_test], preds[idx_test])
    simpspe = matrix[0][0] * 1.0 / (matrix[0][0] + matrix[0][1])
    simpsen = matrix[1][1] * 1.0 / (matrix[1][0] + matrix[1][1])
    acclist['simp'] = simpacc
    acclist['simpsen'] = simpsen
    acclist['simpspe'] = simpspe
    with open('filemsgcn.txt', 'a') as file:
        file.write(json.dumps(acclist))
    return acclist


def train_10_split():
    gcnsvdacclist = []
    svdspelist = []
    svdsenlist = []

    simpacclist = []
    simpspelist = []
    simpsenlist = []

    for i in range(1):
        # data = readdata(dataseed = i*10+i)
        adj, features, labels, idx_train, idx_val, idx_test = readfile()
        acc = train(adj, features, labels, idx_train, idx_val, idx_test)

        simpacclist.append(acc['simp'])
        simpspelist.append(acc['simpspe'])
        simpsenlist.append(acc['simpsen'])
        gcnsvdacclist.append(acc['gcnsvd'])
        svdsenlist.append(acc['svdsen'])
        svdspelist.append(acc['svdspe'])

    simpacc = np.mean(simpacclist)
    simpsen = np.mean(simpsenlist)
    simpspe = np.mean(simpspelist)
    gcnsvdacc = np.mean(gcnsvdacclist)
    svdsen = np.mean(svdsenlist)
    svdspe = np.mean(svdspelist)

    print(acc)
    print(
        'simpacc: {:.4f}'.format(simpacc),
        'simpsen: {:.4f}'.format(simpsen),
        'simpspe: {:.4f}'.format(simpspe),
        'gcnsvd: {:.4f}'.format(gcnsvdacc),
        'svdsen: {:.4f}'.format(svdsen),
        'svdspe: {:.4f}'.format(svdspe), )


train_10_split()


