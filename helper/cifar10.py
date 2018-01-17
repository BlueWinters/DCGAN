
import numpy as np
import scipy.io as sio
import helper.iterator as iter


class Cifar10:
    def __init__(self, data_path='E:/dataset/cifar10/mat', **kwargs):
        self.data_path = data_path
        self.package = None

        if kwargs['train'] == True:
            self.load_train_data()
        else:
            self.load_test_data()

    def preprocess(self, images, labels, **kwargs):
        if kwargs['data_dim'] == 2:
            batch_size = images.shape[0]
            images = np.reshape(images, [batch_size, -1])  # shape: [*,?,?,?] --> [*,?]
        if kwargs['one_hot'] == False:
            images, labels = labels = np.argmax(labels)  # [*,?] --> [?]
        if kwargs['value_norm'] == True:
            images = (images - 127.5) / 127.5
        return images, labels

    def load_train_data(self, data_dim=4, one_hot=True, value_norm=True):
        print('Extract train data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_train.mat'.format(self.data_path))
        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        images, labels = self.preprocess(images, labels,
                                         data_dim=data_dim, one_hot=one_hot,
                                         value_norm=value_norm)
        # package into iterator
        assert self.package == None
        self.package = iter.Iterator(images, labels)

    def load_test_data(self, data_dim=4, one_hot=True, value_norm=True):
        print('Extract test data: [cifar10] from {}'.format(self.data_path))
        data = sio.loadmat('{}/cifar10_test.mat'.format(self.data_path))
        images = data['images'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        images, labels = self.preprocess(images, labels,
                                         data_dim=data_dim, one_hot=one_hot,
                                         value_norm=value_norm)
        # package into iterator
        assert self.package == None
        self.package = iter.Iterator(images, labels)

    def next_batch(self, batch_size, flip=True, shuffle=True):
        return self.package.next_batch(batch_size=batch_size, shuffle=shuffle)

    @property
    def num_examples(self):
        return self.package.num_examples

    @property
    def images(self):
        return self.package.images

    @property
    def labels(self):
        return self.package.labels
