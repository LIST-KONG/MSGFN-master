from gcn.gcn_layers import *
from gcn.gcn_inits import masked_softmax_cross_entropy, masked_accuracy
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.epsilon = 0

        self.loss = 0
        self.accuracy = 0
        # self.accuracy_mask = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs[0])
        for index in range(len(self.layers) - 1):
            hidden = self.layers[index](self.activations[-1])
            hidden = (1 - self.epsilon) * hidden + self.epsilon * self.inputs[index+1]
            self.activations.append(hidden)
        self.activations.append(self.layers[-1](self.activations[-1]))
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, configs, input_dim,  **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.epsilon = configs["epsilon"]
        self.gcn_hidden = configs["gcn_hidden"]
        self.weight_decay = configs["weight_decay"]
        self.dropout = placeholders["dropout"]
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=configs["gcn_learning_rate"])

        self._build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers) - 1):
            for var in self.layers[i].vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy_mask, self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                                            self.placeholders['labels_mask'])
        # self.accuracy_mask = masked_accuracy(self.outputs, self.placeholders['labels'],
        #                                      self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.gcn_hidden[0],
                                            support=self.placeholders["support"][0],
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging))
        for i in range(len(self.gcn_hidden) - 1):
            self.layers.append(GraphConvolution(input_dim=self.gcn_hidden[i],
                                                output_dim=self.gcn_hidden[i+1],
                                                support=self.placeholders["support"][1+1],
                                                act=tf.nn.relu,
                                                dropout=self.dropout,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.gcn_hidden[-1],
                                            output_dim=self.output_dim,
                                            support=self.placeholders["support"][-1],
                                            act=lambda x: x,
                                            dropout=self.dropout,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
