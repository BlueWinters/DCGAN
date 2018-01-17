
import numpy as np
import random
from math import sin, cos, sqrt
import matplotlib.pyplot as plt



def onehot_categorical(batch_size, n_labels):
    y = np.zeros((batch_size, n_labels), dtype=np.float32)
    indices = np.random.randint(0, n_labels, batch_size)
    for b in range(batch_size):
        y[b, indices[b]] = 1
    return y

def uniform(batch_size, dim):
    return np.random.uniform(-1, 1, [batch_size, dim]).astype(np.float32)

def gaussian_mixture(batch_size, labels, n_labels=10, n_dim=2):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    # one-hot --> number code
    label_indices = np.argmax(labels, axis=1)

    def sample(x, y, label, n_labels):
        shift = 4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 1.5
    y_var = 0.3
    dim = int(n_dim/2)
    x = np.random.normal(0, x_var, [batch_size,dim])
    y = np.random.normal(0, y_var, [batch_size,dim])
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(dim):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
    return z / 10



if __name__ == '__main__':
    pass