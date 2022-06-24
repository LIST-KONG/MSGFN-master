import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

# # ====================================== GCN =====================================
def coef_to_adj(coef, threshold):
    """convert Coef matrix to adjacency csr_matrix"""

    coef = 0.5 * (coef + coef.T)
    samples_number = coef.shape[0]
    neighbor_number = round(threshold*samples_number)
    adj = np.zeros((samples_number, samples_number), dtype=np.int)
    for i in range(samples_number):
        top_index = np.argsort(np.abs(coef[i]))[::-1][0: neighbor_number]
        adj[i][top_index] = 1
    return adj

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
#
#
# def adj_softmax(x):
#     """ soft-max function """
#     # x -= np.max(x, axis=1, keepdims=True)  # avoid numerical overflow
#     # x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
#     x = x / np.sum(x, axis=1, keepdims=True)
#     return x


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0], dtype="float32"))
    # a = adj_normalized.toarray()
    #GAT
    # return adj_normalized
    #GCN
    return sparse_to_tuple(adj_normalized)


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    # print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features'][i]: features[i] for i in range(len(features))})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['num_features_nonzero'][i]: features[i][2] for i in range(len(features))})
    return feed_dict


# ========================================= GAT =======================================
def adj_to_bias(adj, sizes, nhood=1):
    """
     Prepare adjacency matrix by expanding up to a given neighbourhood.
     This will insert loops on every node.
     Finally, the matrix is converted to bias vectors.
     Expected shape: [graph, nodes, nodes]
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


# ========================================= KNN graph construction =======================================
def knn_graph_construction(datasets, k_prop):
    samples_num = datasets.shape[0]

    neighbour_num = round(samples_num * k_prop)
    distance = cdist(datasets, datasets, metric='euclidean')
    dis = tf.placeholder(dtype=tf.float32, shape=distance.shape)
    indices = tf.nn.top_k(dis * -1, neighbour_num)
    with tf.Session() as sess:
        index = sess.run(indices, feed_dict={dis: distance}).indices
    adj = np.zeros((samples_num, samples_num), dtype=np.int)
    for i in range(samples_num):
        adj[i][index[i]] = 1
    return adj


def calculate_index(labels, predicts):
    """
    :param labels:
    :param predicts:
    :return:
    """
    # predicts = np.argmax(predicts, axis=1)
    # labels = np.argmax(labels, axis=1)
    predicts = np.argmax(predicts, axis=1)
    labels = np.argmax(labels, axis=1)
    correct = np.equal(predicts,labels)
    acc = np.mean(correct)
    matrix = confusion_matrix(labels, predicts)
    spe = matrix[0][0] * 1.0 / (matrix[0][0] + matrix[0][1])
    sen = matrix[1][1] * 1.0 / (matrix[1][0] + matrix[1][1])
    return acc, spe, sen

def t_test(hc, mdd):
    # 两类样本t检验
    J = hc.shape[1]
    pvalues = np.zeros(J)
    for j in range(J):
        var_hc = hc[:,j]
        var_mdd = mdd[:,j]
        res = ttest_ind(var_hc, var_mdd)
        pvalues[j] = res.pvalue
    return pvalues


def load_data(semi_supervised_prop = 0.6, seed = 1, feature_selection = 'false'):
    if feature_selection == 'true':
        hc_data = sio.loadmat("./data/Features_MSGCN/HC_Feature.mat")['HC_Feature']
        mdd_data = sio.loadmat("./data/Features_MSGCN/MDD_Feature.mat")['MDD_Feature']
        #set random dataset
        np.random.seed(seed)
        indices_hc = np.random.permutation(hc_data.shape[0])
        indices_mdd = np.random.permutation(mdd_data.shape[0])
        hc_data_in = hc_data[indices_hc]
        mdd_data_in = mdd_data[indices_mdd]
        hc_train = hc_data_in[:int(hc_data_in.shape[0] * semi_supervised_prop), :]
        mdd_train = mdd_data_in[:int(mdd_data_in.shape[0] * semi_supervised_prop), :]

        # calculate P and FDR
        pvalues = t_test(hc_train, mdd_train)
        fdrrvaluesstats = fdrcorrection(pvalues)
        fdrindex = np.where(fdrrvaluesstats[0] == True)[0]


        hc_data_in = np.stack((hc_data_in[:, index] for index in fdrindex), axis=1)
        hc_data_out = np.concatenate((hc_data_in, np.ones((hc_data_in.shape[0], 1), dtype=np.int32)), axis=1)
        hc_data_out = np.concatenate((hc_data_out, np.zeros((hc_data_in.shape[0], 1), dtype=np.int32)), axis=1)
        mdd_data_in = np.stack((mdd_data_in[:, index] for index in fdrindex), axis=1)
        mdd_data_out = np.concatenate((mdd_data_in, np.zeros((mdd_data_in.shape[0], 1), dtype=np.int32)), axis=1)
        mdd_data_out = np.concatenate((mdd_data_out, np.ones((mdd_data_in.shape[0], 1), dtype=np.int32)), axis=1)

        hc_train_out = hc_data_out[:int(hc_data_in.shape[0] * semi_supervised_prop), :]
        hc_test_out = hc_data_out[int(hc_data_in.shape[0] * semi_supervised_prop):, :]
        mdd_train_out = mdd_data_out[:int(mdd_data_in.shape[0] * semi_supervised_prop), :]
        mdd_test_out = mdd_data_out[int(mdd_data_in.shape[0] * semi_supervised_prop):, :]
        train_out = np.concatenate((hc_train_out, mdd_train_out))
        test_out = np.concatenate((hc_test_out, mdd_test_out))
        np.random.shuffle(train_out)
        np.random.shuffle(test_out)
        x_out = np.concatenate((train_out, test_out))
    else:
        features = sio.loadmat('./data/features_final.mat')["features"]
        labels = sio.loadmat('./data/features_final.mat')["labels"]
        x_out = np.concatenate((features,labels), axis=1)

    return x_out