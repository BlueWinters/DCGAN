
import tensorflow as tf
import tensorflow.contrib.slim as slim
import operators as op


class Generator(object):
    def __init__(self, batch_size, z_dim, dataset):
        self.batch_size = batch_size
        self.z_dim = z_dim

        # init variables
        if dataset == 'mnist':
            self.initialize_vars(kernel=5, x_chl=1, y_dim=10)
        if dataset == 'cifar10':
            self.initialize_vars(kernel=5, x_chl=3, y_dim=10)

    def initialize_vars(self, kernel, x_chl, y_dim=10):
        ks = kernel
        y_dim = y_dim
        x_chl = x_chl
        in_dim = self.z_dim + y_dim

        # generator
        with tf.variable_scope('generator'):
            op.set_fc_vars(in_dim, 256*4*4, 'fc')
            op.set_fc_bn_vars([256*4*4], name='bn-fc')
            op.set_deconv_vars(256, ks, ks, 128, 'deconv1')
            op.set_conv_bn_vars([128], name='bn-deconv1')
            op.set_deconv_vars(128, ks, ks, 64, 'deconv2')
            op.set_conv_bn_vars([64], name='bn-deconv2')
            op.set_deconv_vars(64, ks, ks, x_chl, 'deconv3')

    def generate_on_mnist(self, z, y, train=True):
        # batch_size
        ns = self.batch_size
        x_chl = 1
        # generate
        with tf.variable_scope('generator', reuse=True):
            # concat
            z_concat = tf.concat([z, y], axis=1)
            # fc
            body = op.calc_fc(z_concat, 'fc')
            body = op.calc_fc_bn(body, name='bn-fc')
            body = op.calc_relu(body, 'relu-fc') #
            # reshape
            body = tf.reshape(body, tf.stack([ns, 4, 4, 256]), 'reshape')
            # deconv1
            body = op.calc_deconv2d(body, 2, [ns, 7, 7, 128], 'SAME', 'deconv1')
            body = op.calc_conv_bn(body, train=train, name='bn-deconv1')
            body = op.calc_relu(body, 'relu1')
            # deconv2
            body = op.calc_deconv2d(body, 2, [ns, 14, 14, 64], 'SAME', 'deconv2')
            body = op.calc_conv_bn(body, train=train, name='bn-deconv2')
            body = op.calc_relu(body, 'relu1')
            # deconv4
            body = op.calc_deconv2d(body, 2, [ns, 28, 28, x_chl], 'SAME', 'deconv3')
            body = op.calc_tanh(body, 'tanh')
        # return
        return body

    def generate_on_cifar10(self, z, y, train=True):
        # batch_size
        ns = self.batch_size
        x_chl = 3
        # decoder
        with tf.variable_scope('generator', reuse=True):
            # concat
            z_concat = tf.concat([z, y], axis=1)
            # fc
            body = op.calc_fc(z_concat, 'fc')
            body = op.calc_fc_bn(body, name='bn-fc')
            body = op.calc_relu(body, 'relu-fc') #
            # reshape
            body = tf.reshape(body, tf.stack([ns, 4, 4, 256]), 'reshape')
            # deconv1
            body = op.calc_deconv2d(body, 2, [ns, 8, 8, 128], 'SAME', 'deconv1')
            body = op.calc_conv_bn(body, train=train, name='bn-deconv1')
            body = op.calc_relu(body, 'relu1')
            # deconv2
            body = op.calc_deconv2d(body, 2, [ns, 16, 16, 64], 'SAME', 'deconv2')
            body = op.calc_conv_bn(body, train=train, name='bn-deconv2')
            body = op.calc_relu(body, 'relu1')
            # deconv4
            body = op.calc_deconv2d(body, 2, [ns, 32, 32, x_chl], 'SAME', 'deconv3')
            body = op.calc_tanh(body, 'tanh')
        # return
        return body






if __name__ == '__main__':
    pass
