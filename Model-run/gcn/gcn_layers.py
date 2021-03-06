from gcn.gcn_inits import *
import tensorflow as tf

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, support, num_features_nonzero=0, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.support = support
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero

        with tf.variable_scope(self.name + '_vars'):
            # for i in range(len(self.support)):
            #     # self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
            #     #                                         name='weights_' + str(i))
            #     self.vars['weights_' + str(i)] = glorot([input_dim, output_dim])
            self.vars['weights_0'] = glorot([input_dim, output_dim], name='weights_0')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # # convolve
        # supports = list()
        # for i in range(len(self.support)):
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        # output = tf.add_n(supports)
        if not self.featureless:
            pre_sup = dot(x, self.vars['weights_0'],
                          sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars['weights_0']
        # # sparse input
        output = dot(self.support, pre_sup, sparse=True)
        # output = dot(self.support, pre_sup, sparse=False)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class GraphAttentionLayer(keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self,
                 input_dim,
                 output_dim,
                 adj,
                 nodes_num,
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 coef_dropout=0.0,
                 **kwargs):
        """
        :param input_dim: ???????????????
        :param output_dim: ???????????????????????????input_dim
        :param adj: ???????????????tuple??????????????????[coords, values, shape]??? ????????????sp.coo_matrix??????
        :param nodes_num: ?????????
        :param dropout_rate: ?????????????????????????????????0.5
        :param activation: ????????????
        :param use_bias: ???????????????True
        :param kernel_initializer: ?????????????????????
        :param bias_initializer: ?????????????????????
        :param kernel_regularizer: ???????????????
        :param bias_regularizer: ???????????????
        :param activity_regularizer: ???????????????
        :param kernel_constraint: ????????????
        :param bias_constraint: ????????????
        :param coef_dropout: ??????????????????????????????0.0
        :param kwargs:
        """
        super(GraphAttentionLayer, self).__init__()
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.support = [tf.SparseTensor(indices=adj[0][0], values=adj[0][1], dense_shape=adj[0][2])]
        self.support = adj
        self.dropout_rate = dropout_rate
        self.coef_drop = coef_dropout
        self.nodes_num = nodes_num
        self.kernel = None
        self.mapping = None
        self.bias = None

    def build(self, input_shape):
        """
        ???????????????
        """
        self.kernel = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      name = '1',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.nodes_num, self.output_dim),
                                        name='2',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True)
        print('[GAT LAYER]: GAT W & b built.')

    def call(self, inputs, training=True):
        # ????????????????????????????????????
        # inputs = tf.nn.l2_normalize(inputs, 1)
        raw_shape = inputs.shape
        inputs = tf.reshape(inputs, shape=(1, self.nodes_num, self.input_dim))  # (1, nodes_num, input_dim)
        mapped_inputs = keras.layers.Conv1D(self.output_dim, 1, use_bias=False)(inputs)  # (1, nodes_num, output_dim)
        # mapped_inputs = tf.nn.l2_normalize(mapped_inputs)

        sa_1 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)
        sa_2 = keras.layers.Conv1D(1, 1)(mapped_inputs)  # (1, nodes_num, 1)

        con_sa_1 = tf.reshape(sa_1, shape=(self.nodes_num, 1))  # (nodes_num, 1)
        con_sa_2 = tf.reshape(sa_2, shape=(self.nodes_num, 1))  # (nodes_num, 1)

        # con_sa_1 = tf.cast(self.support[0], dtype=tf.float32) * con_sa_1  # (nodes_num, nodes_num) W_hi
        con_sa_1 = self.support * con_sa_1
        con_sa_2 = self.support * tf.transpose(con_sa_2, [1, 0])
        # con_sa_2 = tf.cast(self.support[0], dtype=tf.float32) * tf.transpose(con_sa_2, [1, 0])  # (nodes_num, nodes_num) W_hj

        weights = tf.sparse.add(con_sa_1, con_sa_2)  # concatenation
        weights_act = tf.SparseTensor(indices=weights.indices,
                                      values=tf.nn.leaky_relu(weights.values),
                                      dense_shape=weights.dense_shape)  # ????????????????????????
        attention = tf.sparse.softmax(weights_act)  # ?????????????????????
        inputs = tf.reshape(inputs, shape=(self.nodes_num, self.input_dim))
        if self.coef_drop > 0.0:
            attention = tf.SparseTensor(indices=attention.indices,
                                        values=tf.nn.dropout(attention.values, self.coef_dropout),
                                        dense_shape=attention.dense_shape)
        # if training and self.dropout_rate > 0.0:
        #     inputs = tf.nn.dropout(inputs, self.dropout_rate)
        # if not training:
        #     print("[GAT LAYER]: GAT not training now.")

        attention = tf.sparse.reshape(attention, shape=[self.nodes_num, self.nodes_num])
        value = tf.matmul(inputs, self.kernel)
        # value = tf.sparse.sparse_dense_matmul(attention, value)
        value = tf.sparse_tensor_dense_matmul(attention, value)

        if self.use_bias:
            ret = tf.add(value, self.bias)
        else:
            ret = tf.reshape(value, (self.nodes_num, self.output_dim))
        return self.activation(ret)