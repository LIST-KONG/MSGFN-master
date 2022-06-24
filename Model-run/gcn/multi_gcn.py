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
        self.dsc_layers_nums = None
        self.outputs = 0

        self.epsilon = 0

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.saver = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        #四层
        # configs["gcn_hidden"] = [[256, 32, 16, 16], [32, 16, 16], [32, 16, 16]]
        hidden0256 = self.layers[0][0](self.inputs[0])

        hidden032 = self.layers[0][1](hidden0256)
        hidden132=self.layers[1][0](self.inputs[1])
        hidden232=self.layers[2][0](self.inputs[2])

        common32 = (hidden032 + hidden132 + hidden232) / 6
        hidden016 = self.layers[0][2](hidden032/2+common32)
        hidden116=self.layers[1][1](hidden132/2+common32)
        hidden216=self.layers[2][1](hidden232/2+common32)

        common16 = (hidden016 + hidden116 + hidden216) / 6
        hidden08 = self.layers[0][3](hidden016/2+common16)
        hidden18=self.layers[1][2](hidden116/2+common16)
        hidden28=self.layers[2][2](hidden216/2+common16)

        common8 =  (hidden08 + hidden18 + hidden28) / 6
        hidden02=self.layers[0][4](hidden08/2+common8)
        hidden12 = self.layers[1][3](hidden18/2+common8)
        hidden22 = self.layers[2][3](hidden28/2+common8)
        self.outputs=(hidden02+hidden12+hidden22)/3

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

    def save(self, sess=None, epo=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(tf.global_variables())#self.vars
        tf.add_to_collection('output', self.outputs)
        save_path = saver.save(sess, "./gcnmodel/%s.ckpt" % self.name, global_step=epo)
                # saver.save(sess, 'my-model', global_step=step)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        # saver = tf.train.Saver(self.vars)
        saver = tf.train.Saver()
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MGCN(Model):
    def __init__(self, placeholders, samples_num, configs, input_dim,  **kwargs):
        super(MGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.dsc_layers_nums = len(self.inputs)
        self.input_dim = input_dim
        self.epsilon = configs["epsilon"]
        self.gcn_hidden = configs["gcn_hidden"]
        self.weight_decay = configs["weight_decay"]
        self.dropout = placeholders["dropout"]
        self.supports = placeholders["support"]
        self.learning_supports = placeholders["support"]
        # self.learning_supports = []
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.samples_num = samples_num
        self.placeholders = placeholders

        self.learning_neighbour_weights = {}

        self.optimizer = tf.train.AdamOptimizer(learning_rate=configs["gcn_learning_rate"])
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(self.dsc_layers_nums):
            for j in range(len(self.layers[i]) - 1):
                for var in self.layers[i][j].vars.values():
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
        for i in range(self.dsc_layers_nums):
            each_layer = list()
            each_layer.append(GraphConvolution(input_dim=self.input_dim[i],
                                               output_dim=self.gcn_hidden[i][0],
                                               support=self.learning_supports[i],
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               logging=self.logging))
            for j in range(len(self.gcn_hidden[i]) - 1):
                each_layer.append(GraphConvolution(input_dim=self.gcn_hidden[i][j],
                                                   output_dim=self.gcn_hidden[i][j + 1],
                                                   support=self.learning_supports[i],
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging))

            each_layer.append(GraphConvolution(input_dim=self.gcn_hidden[i][-1],
                                               output_dim=self.output_dim,
                                               support=self.learning_supports[i],
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging))
            self.layers.append(each_layer)


    def predict(self):
        return tf.nn.softmax(self.outputs)


