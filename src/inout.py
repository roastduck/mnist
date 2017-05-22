import random
import numpy # Should be imported after random
from tensorflow.examples.tutorials.mnist import input_data as tfData

def oneHot(size, key):
    return numpy.array([i == key for i in range(size)])

def TrainDataGetter(batchSize):
    f = open('../data/train.csv')
    next(f) # The first line is title
    lines = map(lambda row: row.strip().split(','), f)
    dataset = list(map(lambda row: (oneHot(10, int(row[0])), numpy.array(tuple(map(float, row[1:])))), lines))
    del f

    while True:
        batch = random.sample(dataset, batchSize)
        labels = list(map(lambda d: d[0], batch))
        images = list(map(lambda d: d[1], batch))
        yield (numpy.array(images), numpy.array(labels))

def TensorflowTrainDataGetter(batchSize):
    dataset = tfData.read_data_sets('MNIST_data', one_hot=True)
    while True:
        yield dataset.train.next_batch(batchSize)

