import itertools
import numpy # Should be imported after itertools
from tensorflow.examples.tutorials.mnist import input_data as tfData

def oneHot(size, key):
    return numpy.array([i == key for i in range(size)])

def TrainDataGetter(batchSize):
    f = open('../data/train.csv')
    next(f) # The first line is title
    lines = map(lambda row: row.strip().split(','), f)
    dataset = map(lambda row: (oneHot(10, int(row[0])), numpy.array(tuple(map(float, row[1:])))), lines)

    labels = []
    images = []
    for i, data in zip(itertools.count(), dataset):
        labels.append(data[0])
        images.append(data[1])
        if (i + 1) % batchSize == 0:
            assert len(images) == len(labels)
            yield (numpy.array(images), numpy.array(labels))
            labels = []
            images = []
    if labels:
        assert len(images) == len(labels)
        yield (numpy.array(images), numpy.array(labels))

def TensorflowTrainDataGetter(batchSize):
    dataset = tfData.read_data_sets('MNIST_data', one_hot=True)
    for i in range(20000):
        yield dataset.train.next_batch(batchSize)

