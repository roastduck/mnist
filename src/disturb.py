import numpy
import tensorflow.contrib.keras as keras

def disturb(img):
    mat = numpy.reshape(img, (1, 28, 28))

    mat = keras.preprocessing.image.random_zoom(mat, (0.8, 1.2), 1, 2, 0, 'constant', 0.0)
    mat = keras.preprocessing.image.random_rotation(mat, 20.0, 1, 2, 0, 'constant', 0.0)
    mat = keras.preprocessing.image.random_shift(mat, 0.15, 0.15, 1, 2, 0, 'constant', 0.0)

    return numpy.reshape(mat, (784,))

def disturbBatch(batch):
    for i in range(batch.shape[0]):
        batch[i] = disturb(batch[i])

