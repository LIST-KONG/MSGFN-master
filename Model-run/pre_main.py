# encoding=utf-8
# _author = "Steven gao, Suzy Niu"

from utils import *
from dsc.dsc import ResDSC
from gcn.multi_gcn import MGCN
import json
import tensorflow as tf
import numpy as np

with open("./configs.json", "r", encoding="utf-8") as fp:
    configs = json.load(fp)
restore_dscmodel_path = "./pretrained-model/dscpretrain/dscmodel.ckpt"
restore_msgcnmodel_path = "./pretrained-model/gcnpretrain/mgcn.ckpt"\

def train_dsc_model(data):
    #load the pretrained dsc model
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)
    model.restore(restore_dscmodel_path)

    # train the dsc network
    train_step = 2000
    display_step = 100
    is_training = True
    batch_x = np.reshape(data, [-1, data.shape[1]])
    for i in range(train_step):
        cost, coefs, latents = model.partial_fit(batch_x, is_training)
        if i % display_step == 0:
            print("loss: {}".format(cost))
    model.sess.close()
    return coefs, latents


def train_gcn(labels, all_mask, test_mask, coefs, latents):
    supports = list()
    features = list()
    num_supports = len(configs["n_hidden"]) + 1
    for coef in coefs:
        adj = coef_to_adj(coef, configs["adj_threshold"])
        supports.append(preprocess_adj(adj))
    for latent in latents:
        features.append(preprocess_features(latent))

    # Define placeholders
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': [tf.placeholder(tf.float32) for _ in range(num_supports)],
            'labels': tf.placeholder(tf.float32, shape=(None, labels.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        input_dim = list()
        for latent in latents:
            input_dim.append(latent.shape[1])
        model = MGCN(placeholders, samples_num, configs, input_dim=input_dim, logging=False)
        y = model.outputs

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, restore_msgcnmodel_path)
            feed_dict = construct_feed_dict(features, supports, labels, all_mask, placeholders)
            y_pred = sess.run(y, feed_dict=feed_dict)
            acc, spe, sen = calculate_index(labels[test_mask], y_pred[test_mask])
    sess.close()
    return acc, sen, spe




if __name__ == '__main__':
    #load data
    print("load data ... ")
    supervised_prop = 0.6
    data = load_data(semi_supervised_prop = supervised_prop, seed = 1)
    x = data[:, :-2]
    y = data[:, -2:]

    #split the dataset
    samples_num = x.shape[0]
    supervise_num = int(samples_num * supervised_prop)
    test_mask = range(supervise_num, samples_num)
    test_mask = sample_mask(test_mask, samples_num)
    all_mask = range(samples_num)
    all_mask = sample_mask(all_mask, samples_num)

    #preprocess the features
    x = preprocess_features(x)

    #train dsc model
    coefs, latents = train_dsc_model(x)
    acclist = []
    senlist = []
    spelist = []
    acc, sen, spe = train_gcn(y, all_mask,test_mask, coefs, latents)
    print("accuracy={:.4f}, sensitivity={:.4f}, specificity={:.4f}".format(acc, sen, spe))

