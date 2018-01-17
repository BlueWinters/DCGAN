
import tensorflow as tf
import tensorflow.contrib.slim as slim
import operators as op


class Discriminator(object):
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size

        # init variables
        if dataset == 'mnist':
            self.initialize_vars(kernel=5, x_chl=1, y_dim=10)
        if dataset == 'cifar10':
            self.initialize_vars(kernel=5, x_chl=3, y_dim=10)

    def initialize_vars(self, kernel, x_chl, y_dim=10):
        ks = kernel
        x_chl = x_chl
        with tf.variable_scope('discriminator'):
            op.set_conv_vars(x_chl, ks, ks, 64, 'conv1')
            op.set_conv_bn_vars([64], 'bn-conv1')
            op.set_conv_vars(64, ks, ks, 128, 'conv2')
            op.set_conv_bn_vars([128], 'bn-conv2')
            op.set_conv_vars(128, ks, ks, 256, 'conv3')
            op.set_conv_bn_vars([256], 'bn-conv3')
            op.set_fc_vars(256*4*4, 1, 'fc')

    def discriminator_small(self, x, train=True):
        # batch_size
        n = self.batch_size
        y_dim = 10
        # discriminator
        with tf.variable_scope('discriminator', reuse=True):
            # conv1
            body = op.calc_conv2d(x, 2, 'SAME', name='conv1')
            body = op.calc_conv_bn(body, train=train, name='bn-conv1')
            body = op.calc_lrelu(body, name='lrelu1')
            # conv2
            body = op.calc_conv2d(body, 2, 'SAME', name='conv2')
            body = op.calc_conv_bn(body, train=train, name='bn-conv2')
            body = op.calc_lrelu(body, name='lrelu2')
            # conv3
            body = op.calc_conv2d(body, 2, 'SAME', name='conv3')
            body = op.calc_conv_bn(body, train=train, name='bn-conv3')
            body = op.calc_lrelu(body, name='lrelu3')
            # flatten
            body = slim.flatten(body)
            # fc
            body = op.calc_fc(body, 'fc')
        return body

    def discriminator_mnist(self, x, train=True):
        return self.discriminator_small(x, train=train)

    def discriminator_cifar10(self, x, train=True):
        return self.discriminator_small(x, train=train)

if __name__ == '__main__':
    pass