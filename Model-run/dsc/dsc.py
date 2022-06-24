# encoding=utf-8

# _Author = "Steven Gao"

from utils import *
import tensorflow as tf
import json


class ResDSC(object):
    """
    semi-supervised deep subspace clustering with self-supervised GCN
    """
    def __init__(self, configs, features_shape, noise=False):
        self.n_input = configs["n_input"]
        self.n_hidden = configs["n_hidden"]
        self.layer_num = len(self.n_hidden)
        self.reg = configs["reg"]
        self.pre_model_path = configs["pre_trained_model_path"]
        self.model_path = configs["model_path"]
        self.log_path = configs["log_path"]
        self.batch_size = configs["batch_size"]
        self.lr = configs["learning_rate"]
        self.network_type = configs["network_type"]
        self.activate_type = configs["activate_type"]
        self.regularized_constant1 = configs["regularized_constant1"]
        self.regularized_constant2 = configs["regularized_constant2"]

        self.labels_num = configs["labels_num"]

        self.coef_size = features_shape[0]
        self.each_layer_reconstruct = list()
        self.each_layer_output = list()
        self.iter = 0
        weights = self._initialize_weights()

        # feed_dictionary
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.is_training = tf.placeholder(tf.bool)

        if noise:
            self.x = tf.add(self.x, tf.random_normal(shape=tf.shape(self.x),
                                                     mean=0,
                                                     stddev=0.2,
                                                     dtype=tf.float32))

        # encoder
        representations = dict()
        representations["layer_e_0"] = self.x
        for i in range(self.layer_num):
            representations["layer_e_{}".format(i + 1)] = self.encoder(weights["encoder_w{}".format(i)],
                                                                       representations["layer_e_{}".format(i)])

        self.z = representations["layer_e_{}".format(self.layer_num)]

        # decoder
        representations["layer_d_{}".format(self.layer_num)] = self.z
        for i in range(self.layer_num):
            representations["layer_d_{}".format(self.layer_num - i - 1)] = self.decoder(
                weights["decoder_w{}".format(i)],
                representations["layer_d_{}".format(self.layer_num - i)])

        self.x_r = representations["layer_d_0"]

        # reconstruct loss
        self.reconst_cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x_r, self.x), 2.0))

        # regularization loss
        self.reg_losses = tf.reduce_sum(tf.pow(weights["Coef_0"], 2.0))
        for i in range(1, self.layer_num + 1):
            self.reg_losses += tf.reduce_sum(tf.pow(weights["Coef_{}".format(i)], 2.0))

        # self expressive loss
        self.selfexpress_losses = tf.reduce_sum(
            tf.pow(tf.subtract(representations["layer_e_0"], tf.matmul(weights["Coef_0"],
                                                                       representations["layer_e_0"])), 2.0))
        # 变换是否使用强制共享C
        for i in range(1, self.layer_num+1):
            self.selfexpress_losses += tf.reduce_sum(
                tf.pow(tf.subtract(representations["layer_e_{}".format(i)],
                                   tf.matmul(weights["Coef_{}".format(i)], representations["layer_e_{}".format(i)])),
                2.0))
            # self.selfexpress_losses += tf.reduce_sum(
            #     tf.pow(tf.subtract(representations["layer_e_{}".format(i)],
            #                        tf.matmul(weights["Coef_0"], representations["layer_e_{}".format(i)])), 2.0))
        self.coef_list = [v for v in tf.trainable_variables() if v.name.startswith("Coef")]
        # self.coef_list = weights["Coef_0"]

        self.ae_variables = [v for v in tf.trainable_variables() if not v.name.startswith("Coef")]
        self.latent_representations = list()
        self.coefs = list()
        for i in range(self.layer_num+1):
            self.latent_representations.append(representations["layer_e_{}".format(i)])
            self.coefs.append(weights["Coef_{}".format(i)])
            # # 共享C矩阵
            # self.coefs.append(weights["Coef_0"])

        # ResDSC loss
        # self.losses = self.reconst_cost + self.regularized_constant1 * self.reg_losses + self.regularized_constant2 * self.selfexpress_losses
        self.losses = self.regularized_constant1 * self.reg_losses + self.regularized_constant2 * self.selfexpress_losses

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.reconst_cost,
                                                                                       var_list=self.ae_variables)
            # res-dsc optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.losses, var_list=self.coef_list)

        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not v.name.startswith("Coef")])
        # self.summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()
        for index in range(self.layer_num):
            decoder_index = self.layer_num - index - 1
            # encoder
            if index == 0:
                all_weights["encoder_w{}".format(index)] = tf.Variable(
                    tf.truncated_normal([self.n_input, self.n_hidden[index]]), name="encoder_w{}".format(index))
            else:
                all_weights["encoder_w{}".format(index)] = tf.Variable(
                    tf.truncated_normal([self.n_hidden[index - 1], self.n_hidden[index]]),
                    name="encoder_w{}".format(index))
            # decoder
            if decoder_index == 0:
                all_weights["decoder_w{}".format(index)] = tf.Variable(
                    tf.truncated_normal([self.n_hidden[decoder_index], self.n_input]), name="decoder_w{}".format(index))
            else:
                all_weights["decoder_w{}".format(index)] = tf.Variable(
                    tf.truncated_normal([self.n_hidden[decoder_index], self.n_hidden[decoder_index - 1]]),
                    name="decoder_w{}".format(index))

        # self_expression
        for index in range(self.layer_num + 1):
            all_weights['Coef_{}'.format(index)] = tf.Variable(
                1.0e-4 * tf.ones([self.coef_size, self.coef_size], tf.float32), name='Coef_{}'.format(index))

        return all_weights

    def encoder(self, weight, input_data):
        output = None
        activate_type = "tf.nn.{}".format(self.activate_type)
        output = tf.matmul(input_data, weight)
        output = tf.layers.batch_normalization(output, training=self.is_training)
        output = eval("{}(output)".format(activate_type))
        return output

    def decoder(self, weight, input_data):
        output = None
        activate_type = "tf.nn.{}".format(self.activate_type)
        output = tf.matmul(input_data, weight)
        output = tf.layers.batch_normalization(output, training=self.is_training)
        output = eval("{}(output)".format(activate_type))
        return output

    def initialization(self):
        self.sess.run(self.init)

    def ae_partial_fit(self, X, is_training):
        ae_cost, _ = self.sess.run((self.reconst_cost, self.ae_optimizer),
                                   feed_dict={self.x: X, self.is_training: is_training})
        self.iter = self.iter + 1
        return ae_cost

    def partial_fit(self, X, is_training):
        coef_loss, coefs, latents, _ = self.sess.run(
            (self.losses, self.coefs, self.latent_representations, self.optimizer),
            feed_dict={self.x: X, self.is_training: is_training})
        return coef_loss, coefs, latents

    def reconstruct(self, X, is_training):
        return self.sess.run(self.x_r, feed_dict={self.x: X, self.is_training: is_training})

    def transform(self, X, is_training):
        return self.sess.run(self.z, feed_dict={self.x: X, self.is_training: is_training})

    def save_model(self, samples_set, iteration, latent_dim):
        dims = ""
        for i in latent_dim:
            dims += str(i) + '_'
        dims = dims[:-1]
        model_path = "{}{}_{}_iteration_{}.ckpt".format(self.model_path, samples_set, dims, iteration)
        # self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        self.saver = tf.train.Saver(tf.global_variables())
        save_path = self.saver.save(self.sess, model_path)
        print("model saved in file: %s" % save_path)

    def restore(self, model_path):
        self.saver.restore(self.sess, model_path)
        print("model restored")


# ===========================================================================================
def pre_train_model(data):
    with open("../configs.json", "r", encoding="utf-8") as fp:
        configs = json.load(fp)
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)

    iteration = 0
    pretrained_step = 5000
    is_training = True
    # pretrain the network
    batch_x = np.reshape(data, [-1, data.shape[1]])

    while iteration < pretrained_step:
        print("======================  epoch: {}   ====================".format(iteration))
        cost = model.ae_partial_fit(batch_x, is_training)
        print("ae loss: {}".format(cost))
        iteration += 1
    model.save_model(5000, configs["n_hidden"])


def train_model(data, labels):
    """
    使用谱聚类直接聚类
    """
    with open("../configs.json", "r", encoding="utf-8") as fp:
        configs = json.load(fp)
    configs["n_input"] = data.shape[1]
    tf.reset_default_graph()
    model = ResDSC(configs, data.shape)
    # model.restore("../pretrain_model/hcmdd_latent_256_32_iteration_5000.ckpt")

    iteration = 0
    pretrained_step = 5000
    train_step = 1000
    display_step = 50
    is_training = True
    accuracy = 0
    # pretrain the network
    batch_x = np.reshape(data, [-1, data.shape[1]])

    # while iteration < pretrained_step:
    #     print("======================  epoch: {}   ====================".format(iteration))
    #     cost = model.ae_partial_fit(batch_x, is_training)
    #     print("ae loss: {}".format(cost))
    #     iteration += 1
    # model.save_model(5000)

    for i in range(train_step):
        cost, coefs, latents = model.partial_fit(batch_x, is_training)
        if i % display_step == 0:
            # print("======================  epoch: {}   ====================".format(i))
            predict_index, _ = post_proC(coefs[0], configs["num_cluster"], 8, 3.5)
            idx_test = range(int(218*0.6),218)
            err, sen, spe = err_rate(labels[idx_test], predict_index[idx_test])
            acc = 1 - err
            print("cost: {}-------accuracy: {:.5}, sensitivity: {}, specificity: {}".format(cost, acc, sen, spe))
            if acc > accuracy:
                accuracy = acc
                sensitivity = sen
                specificity = spe
    print("final accuracy : {}, sensitivity:{}, specificity:{}".format(accuracy, sensitivity, specificity))
    model.sess.close()



