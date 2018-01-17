
import numpy as np


def normalize_images(images, labels, reshape=False, norm=True, one_hot=True, norm2=True):
    batch_size = images.shape[0]
    if reshape == True:
        images = np.reshape(images, [batch_size, -1])  # shape: [*,?,?,?] --> [*,?]
    if norm == True:
        images = images / 255.
    if one_hot == False:
        labels = np.argmax(labels)  # [*,?] --> [?]
    # if norm: # data  normalize
    #     mean = np.mean(images, axis=(0, 1, 2), dtype=np.float32)
    #     std = np.mean(images, axis=(0, 1, 2), dtype=np.float32)
    #     images = (images - mean) / std
    return images, labels

def images_normalize(images, labels):
    images = images / 255.
    return images, labels

def labels_to_dense(images, labels):
    assert len(labels.shape) == 2
    labels = np.argmax(labels)  # [*,?] --> [?]
    return images, labels

def labels_to_one_hot(images, labels, **kwargs):
    num_classes = kwargs['num_classes']
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return images, labels_one_hot

def images_normalize_distribution(images, labels):
    if len(images.shape) == 4:
        mean = np.mean(images, axis=(0,1,2), dtype=np.float32)
        std = np.mean(images, axis=(0,1,2), dtype=np.float32)
    else: # 2
        mean = np.mean(images, axis=(0), dtype=np.float32)
        std = np.mean(images, axis=(0), dtype=np.float32)
    # assert std == 0 ?
    images = (images - mean) / std
    return images, labels

def images_vertical_flip(images):
    assert len(images.shape) == 4
    re_range = range(images.shape[1]-1, -1, -1)
    flip = images[:,re_range,:,:]
    return flip

def images_horizontal_flip(images):
    assert len(images.shape) == 4
    re_range = range(images.shape[2]-1, -1, -1)
    flip = images[:,:,re_range,:]
    return flip


