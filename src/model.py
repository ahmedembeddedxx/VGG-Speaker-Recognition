from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from pathlib import Path

weight_decay = 1e-4

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        super(ModelMGPU, self).__init__()
        self.__dict__.update(ser_model.__dict__)
        self._smodel = ser_model

    def __getattr__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        return super(ModelMGPU, self).__getattr__(attrname)

class VladPooling(Layer):
    '''
    This layer follows the NetVlad, GhostVlad
    '''
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]],
                                       name='centers',
                                       initializer='orthogonal')
        super(VladPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # Softmax normalization to get soft-assignment
        max_cluster_score = K.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = K.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)

        A = K.expand_dims(A, -1)
        feat_broadcast = K.expand_dims(feat, -2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = tf.multiply(A, feat_res)
        cluster_res = K.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = K.l2_normalize(cluster_res, -1)
        outputs = K.reshape(cluster_l2, [-1, self.k_centers * num_features])
        return outputs

def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

def vggvox_resnet2d_icassp(input_dim=(257, 250, 1), num_class=8631, mode='train', args=None):
    if args is None:
        raise ValueError("The 'args' parameter is required but not provided.")
    
    if not hasattr(args, 'loss'):
        raise AttributeError("The 'args' object must have an attribute 'loss'.")

    net = args.net
    loss = args.loss
    vlad_clusters = args.vlad_cluster
    ghost_clusters = args.ghost_cluster
    bottleneck_dim = args.bottleneck_dim
    aggregation = args.aggregation_mode
    mgpu = len(tf.config.list_physical_devices('GPU'))

    if net == 'resnet34s':
        inputs, x = backbone.resnet_2D_v1(input_dim=input_dim, mode=mode)
    else:
        inputs, x = backbone.resnet_2D_v2(input_dim=input_dim, mode=mode)
    
    # Fully Connected Block 1
    x_fc = Conv2D(bottleneck_dim, (7, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer='orthogonal',
                   use_bias=True, trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   bias_regularizer=l2(weight_decay),
                   name='x_fc')(x)

    # Feature Aggregation
    if aggregation == 'avg':
        if mode == 'train':
            x = AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
            x = tf.reshape(x, (-1, bottleneck_dim))
        else:
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = tf.reshape(x, (1, bottleneck_dim))

    elif aggregation == 'vlad':
        x_k_center = Conv2D(vlad_clusters, (7, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='vlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

    elif aggregation == 'gvlad':
        x_k_center = Conv2D(vlad_clusters + ghost_clusters, (7, 1),
                            strides=(1, 1),
                            kernel_initializer='orthogonal',
                            use_bias=True, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='gvlad_center_assignment')(x)
        x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

    else:
        raise ValueError('==> unknown aggregation mode')

    # Fully Connected Block 2
    x = Dense(bottleneck_dim, activation='relu',
              kernel_initializer='orthogonal',
              use_bias=True, trainable=True,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay),
              name='fc6')(x)

    # Softmax Vs AMSoftmax
    if loss == 'softmax':
        y = Dense(num_class, activation='softmax',
                  kernel_initializer='orthogonal',
                  use_bias=False, trainable=True,
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay),
                  name='prediction')(x)
        trnloss = 'categorical_crossentropy'

    elif loss == 'amsoftmax':
        x_l2 = Lambda(lambda x: K.l2_normalize(x, 1))(x)
        y = Dense(num_class,
                  kernel_initializer='orthogonal',
                  use_bias=False, trainable=True,
                  kernel_constraint=tf.keras.constraints.UnitNorm(),
                  kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay),
                  name='prediction')(x_l2)
        trnloss = amsoftmax_loss

    else:
        raise ValueError('==> unknown loss.')

    if mode == 'eval':
        y = Lambda(lambda x: K.l2_normalize(x, 1))(x)

    model = Model(inputs, y, name='vggvox_resnet2D_{}_{}'.format(loss, aggregation))

    if mode == 'train':
        if mgpu > 1:
            model = ModelMGPU(model, gpus=mgpu)
        # Set up optimizer
        if args.optimizer == 'adam':
            opt = Adam(learning_rate=1e-3)
        elif args.optimizer == 'sgd':
            opt = SGD(learning_rate=0.1, momentum=0.9, decay=0.0, nesterov=True)
        else:
            raise ValueError('==> unknown optimizer type')
        model.compile(optimizer=opt, loss=trnloss, metrics=['acc'])
    return model
