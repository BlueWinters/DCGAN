
import os as os
import numpy as np
import shutil as sh
import matplotlib.pyplot as plt
import scipy
import imageio
from PIL import Image, ImageSequence
from datetime import datetime



def make_save_directory(root, verbose=False):
    if verbose == True:
        time_str = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        save_path = '{}/{}'.format(root, time_str)
        os.mkdir(save_path)
        os.mkdir('{}/images'.format(save_path)) # for save images
        return save_path
    else:
        if os.path.exists(root) == False:
            os.mkdir(root)
            os.mkdir('{}/images'.format(root))  # for save images
            os.mkdir('{}/visual2d'.format(root))  # for save images
        return root

def save_visual2d(z, labels, save_path_and_name, y_dim=10):
    color_list = plt.get_cmap('hsv', y_dim + 1)
    plt.cla()
    for n in range(y_dim):
        index = np.where(labels[:, n] == 1)[0]
        point = z[index.tolist(), :]
        x = point[:, 0]
        y = point[:, 1]
        plt.scatter(x, y, color=color_list(n), edgecolors='face')
    plt.savefig(save_path_and_name)

def save_grid_images(images, save_path_and_name, nx=10, ny=10, size=32, chl=3):
    plt.cla()
    if chl == 3:
        stack_images = np.zeros([ny*size, nx*size, chl])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size, :] = np.reshape(images[j*ny+i,:,:,:], [size,size,chl])
        scipy.misc.imsave(save_path_and_name, stack_images)
    else:
        stack_images = np.zeros([ny*size, nx*size])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size] = np.reshape(images[j*ny+i,:,:,:], [size,size])
        scipy.misc.imsave(save_path_and_name, stack_images)

def resave_to_gif(src_path, out_path_and_name):
    file_list = os.listdir(src_path)
    assert len(file_list) == 100

    name_list = list(range(1,101))
    frames = []
    for name in name_list:
        file_name = '{}/{}.png'.format(src_path, name)
        frames.append(imageio.imread(file_name))
    # save
    imageio.mimsave(out_path_and_name, frames, 'GIF', duration=0.1)



if __name__ == '__main__':
    resave_to_gif(src_path='../save/cifar10/uniform2/images', out_path_and_name='../save/cifar10/uniform2/cifar10.gif')

