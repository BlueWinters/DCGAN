
import tensorflow as tf
import numpy as np
import sampler as spl
import helper.tools as tools
import scipy.io as sio

from datetime import datetime
from generator import Generator
from discriminator import Discriminator
from helper.cifar10 import Cifar10
from helper.mnist import Mnist



def ave_loss(ave_lost_list, step_loss_list, div):
    assert len(ave_lost_list) == len(step_loss_list)
    for n in range(len(ave_lost_list)):
        ave_lost_list[n] += step_loss_list[n] / div

def train():
    data_set = 'cifar10'
    prior = 'uniform'
    x_dim = 32
    x_chl = 3
    y_dim = 10
    z_dim = 64
    batch_size = 100
    num_epochs = 500*50
    step_epochs = 100 #int(num_epochs/100)
    learn_rate = 0.0005

    root_path = 'save/{}/{}'.format(data_set, prior)
    save_path = tools.make_save_directory(root_path)

    z = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='z')
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, y_dim], name='y')
    x_real = tf.placeholder(dtype=tf.float32, shape=[batch_size, x_dim, x_dim, x_chl], name='x')

    generator = Generator(batch_size=batch_size, z_dim=z_dim, dataset=data_set)
    discriminator = Discriminator(batch_size=batch_size, dataset=data_set)

    x_fake = generator.generate_on_cifar10(z, y, train=True)
    d_out_real = discriminator.discriminator_cifar10(x_real)
    d_out_fake = discriminator.discriminator_cifar10(x_fake)


    # discriminator loss
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_out_real), logits=d_out_real))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(d_out_fake), logits=d_out_fake))
    with tf.control_dependencies([D_loss_fake, D_loss_real]):
        D_loss = D_loss_fake + D_loss_real
    # generator loss
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(d_out_fake), logits=d_out_fake))

    # optimizers
    all_variables = tf.trainable_variables()
    g_var = [var for var in all_variables if 'generator' in var.name]
    d_var = [var for var in all_variables if 'discriminator' in var.name]
    optimizer = tf.train.AdamOptimizer(learn_rate)
    G_solver = optimizer.minimize(G_loss, var_list=g_var)
    D_solver = optimizer.minimize(D_loss, var_list=d_var)


    # read data
    train_data = Cifar10(train=True)
    file = open('{}/train.txt'.format(save_path), 'w')
    sess = tf.Session()

    # train the model
    sess.run(tf.global_variables_initializer())
    ave_loss_list = [0, 0, 0]
    cur_time = datetime.now()


    # for save sample images
    save_step = int(num_epochs/100)
    z_sample = spl.uniform(100, z_dim)
    y_sample = spl.onehot_categorical(100, y_dim)

    # training process
    for epochs in range(1,num_epochs+1):
        batch_x, batch_y = train_data.next_batch(batch_size)
        s_z_real = spl.uniform(batch_size, z_dim)

        for _ in range(1):
            sess.run(D_solver, feed_dict={z:s_z_real, x_real:batch_x, y:batch_y})
        for _ in range(2):
            sess.run(G_solver, feed_dict={z:s_z_real, y:batch_y})


        loss_list = sess.run([D_loss_fake, D_loss_real, G_loss],
                             feed_dict={z:s_z_real, x_real:batch_x, y:batch_y})
        ave_loss(ave_loss_list, loss_list, step_epochs)

        if epochs % save_step == 0:
            iter_counter = int(epochs / save_step)
            x_sample = sess.run(x_fake, feed_dict={z:z_sample, y:y_sample})
            tools.save_grid_images(x_sample, '{}/images/{}.png'.format(save_path, iter_counter), size=x_dim, chl=x_chl)

        # record information
        if epochs % step_epochs == 0:
            time_use = (datetime.now() - cur_time).seconds
            iter_counter = int(epochs/step_epochs)
            liner = "Epoch {:d}/{:d}, loss_dis_faker {:9f}, loss_dis_real {:9f}, loss_encoder {:9f} time_use {:f}" \
                .format(epochs, num_epochs, ave_loss_list[0], ave_loss_list[1], ave_loss_list[2], time_use)
            print(liner), file.writelines(liner + '\n')
            ave_loss_list = [0, 0, 0] # reset to 0
            cur_time = datetime.now()

    # save model
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(var_list=vars)
    saver.save(sess, save_path='{}/model'.format(save_path))

    # close all
    file.close()
    sess.close()



if __name__ == '__main__':
    train()
