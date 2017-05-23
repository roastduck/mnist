import random
import numpy # Should be imported after random

def oneHot(size, key):
    return numpy.array([i == key for i in range(size)])

def DataGetter(trainSize, validateSize):
    ''' Get a random sample from train.csv each time
        @return : (training getter, validating getter) '''

    f = open('../data/train.csv')
    next(f) # The first line is title
    lines = map(lambda row: row.strip().split(','), f)
    dataset = list(map(lambda row: (oneHot(10, int(row[0])), numpy.array(tuple(map(float, row[1:])))), lines))
    del f

    def subGetter(dataset, batchSize):
        while True:
            batch = random.sample(dataset, batchSize)
            labels = list(map(lambda d: d[0], batch))
            images = list(map(lambda d: d[1], batch))
            yield (numpy.array(images), numpy.array(labels))

    splitter = int(len(dataset) * 0.67)
    return subGetter(dataset[:splitter], trainSize), subGetter(dataset[splitter:], validateSize)

