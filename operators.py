
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

def set_fc_vars(in_dim, out_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b', shape=[out_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.))

def set_conv_vars(in_chl, k_h, k_w, out_chl, name):
    with tf.variable_scope(name):
        k = tf.get_variable(name='filter', shape=[k_h, k_w, in_chl, out_chl], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='bias', shape=[out_chl], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.))

def set_deconv_vars(in_chl, k_h, k_w, out_chl, name):
    with tf.variable_scope(name):
        k = tf.get_variable(name='filter', shape=[k_h, k_w, out_chl, in_chl], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='bias', shape=[out_chl], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.))

def set_fc_bn_vars(shape, name):
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [shape[-1]], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [shape[-1]], dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        ave_mean = tf.get_variable('ave_mean', shape[-1], trainable=False,
                                   initializer=tf.constant_initializer(0))
        ave_var = tf.get_variable('ave_var', shape[-1], trainable=False,
                                  initializer=tf.constant_initializer(1))

def set_conv_bn_vars(shape, name):
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', [shape[-1]], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [shape[-1]], dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        ave_mean = tf.get_variable('ave_mean', shape[-1], trainable=False,
                                   initializer=tf.constant_initializer(0))
        ave_var = tf.get_variable('ave_var', shape[-1], trainable=False,
                                  initializer=tf.constant_initializer(1))

def calc_fc(input, name):
    with tf.variable_scope(name):
        W = tf.get_variable('W')
        b = tf.get_variable('b')
        return tf.matmul(input, W) + b

def calc_conv2d(input, stride, padding, name):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter')
        bias = tf.get_variable('bias')
        return tf.nn.conv2d(input, filter, [1,stride,stride,1], padding=padding) + bias

def calc_deconv2d(input, stride, out_shape, padding, name):
    with tf.variable_scope(name):
        filter = tf.get_variable('filter')
        bias = tf.get_variable('bias')
        return tf.nn.conv2d_transpose(input, filter, out_shape, [1,stride,stride,1], padding=padding) + bias

def calc_conv_bn(input, train=True, name='bn'):
    tiny = 1e-6
    decay = 0.999
    with tf.variable_scope(name):
        scale = tf.get_variable('scale')
        beta = tf.get_variable('beta')
        ave_mean = tf.get_variable('ave_mean')
        ave_var = tf.get_variable('ave_var')

        if train:
            batch_mean, batch_var = tf.nn.moments(input, [0,1,2])
            train_mean = tf.assign(ave_mean, ave_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(ave_var, ave_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(input, ave_mean, ave_var, beta, scale, 0.001)

def calc_fc_bn(input, train=True, name='bn'):
    tiny = 1e-6
    decay = 0.999
    with tf.variable_scope(name):
        scale = tf.get_variable('scale')
        beta = tf.get_variable('beta')
        ave_mean = tf.get_variable('ave_mean')
        ave_var = tf.get_variable('ave_var')

        if train:
            batch_mean, batch_var = tf.nn.moments(input, [0])
            train_mean = tf.assign(ave_mean, ave_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(ave_var, ave_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(input, batch_mean, batch_var, beta, scale, 0.001)
        else:
            return tf.nn.batch_normalization(input, ave_mean, ave_var, beta, scale, 0.001)

def calc_relu(input, name='relu'):
    return tf.nn.relu(input, name=name)

def calc_tanh(input, name='tanh'):
    return tf.nn.tanh(input, name=name)

def calc_lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name)

def calc_sigmoid(input, name='sigmoid'):
    return tf.nn.sigmoid(input, name=name)

def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    shape = [x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]]
    return tf.concat([x, y * tf.ones(shape=shape)], axis=3)


