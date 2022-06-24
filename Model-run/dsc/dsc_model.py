# encoding=utf-8

# _Author = "Steven Gao"

import tensorflow as tf

class Model(object):
    def __init__(self, param_dict, model_name=None, noise=False):
        self.n_input = param_dict["n_input"]
        self.n_hidden = param_dict["n_hidden"]
        self.layer_num = len(self.n_hidden)
        self.reg = param_dict["reg"]
        self.restore_path = param_dict["restore_path"]
        self.log_path = param_dict["log_path"]
        self.batch_size = param_dict["batch_size"]
        self.lr = param_dict["learning_rate"]
        self.network_type = param_dict["network_type"]
        self.activate_type = param_dict["activate_type"]
        self.iter = 0
        self.noise = noise

        self.vars = {}
        self.x = None
        self.is_training = None
        self.x_r = None
        self.z = None

        if not model_name:
            model_name = self.__class__.__name__.lower()
        self.name = model_name

    def _initialize_weights(self):
        raise NotImplemented

    def encoder(self):
        pass

    def decoder(self):
        pass

    def reconstruct(self, data, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        return sess.run(self.x_r, feed_dict={self.x: data})

    def transform(self, data, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        return sess.run(self.z, feed_dict={self.x: data})

    def _loss(self):
        raise NotImplemented

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "{}_iter_{}_model.ckpt".format(self.name, self.iter))
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "{}_iter_{}_model.ckpt".format(self.name, self.iter)
        saver.restore(sess, save_path)
        print("Model restored from file: {}".format(save_path))
