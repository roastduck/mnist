import random
import numpy # Should be imported after random

def oneHot(size, key):
    return numpy.array([i == key for i in range(size)])

def DataGetter(trainSize, validateSize):
    ''' Get a random sample from train.csv each time
        @param trainSize, validateSize: batch size for training set and validation set correspondingly.
                                        `trainSize` means sampled size and `validateSize` means valid size
        @return : (training getter, validating getter) '''

    f = open('../data/train.csv')
    next(f) # The first line is header
    lines = map(lambda row: row.strip().split(','), f)
    dataset = list(map(lambda row: (oneHot(10, int(row[0])), numpy.array(tuple(map(float, row[1:])))), lines))
    del f

    def subGetter(dataset, batchSize): # batchSize = None means whole set
        while True:
            batch = dataset if batchSize is None else random.sample(dataset, batchSize)
            labels = list(map(lambda d: d[0], batch))
            images = list(map(lambda d: d[1], batch))
            yield (numpy.array(images), numpy.array(labels))

    return subGetter(dataset[:-validateSize], trainSize), subGetter(dataset[-validateSize:], None)

def TestGetter():
    f = open('../data/test.csv')
    next(f) # The first line is header
    lines = map(lambda row: row.strip().split(','), f)
    dataset = list(map(lambda row: numpy.array(tuple(map(float, row))), lines))
    del f

    for batch in [dataset[i * 500 : (i + 1) * 500] for i in range(len(dataset) // 500)]:
        yield numpy.array(batch)

