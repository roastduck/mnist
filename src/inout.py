import random
import itertools
import numpy # Should be imported after random and itertools

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
    dataset = list(map(lambda row: (numpy.array(tuple(map(float, row[1:]))), oneHot(10, int(row[0]))), lines))
    del f
    trainset = dataset[validateSize:]
    validateset = dataset[:validateSize]
    random.shuffle(trainset)

    def subGetter(dataset, batchSize): # batchSize = None means whole set
        if batchSize is None:
            batchSize = len(dataset)
        assert batchSize <= len(dataset)
        pos = 0
        while True:
            _pos = (pos + batchSize) % len(dataset)
            if pos < _pos:
                batch = itertools.islice(dataset, pos, _pos)
            elif pos > _pos:
                batch = itertools.chain(itertools.islice(dataset, pos, len(dataset)), itertools.islice(dataset, 0, _pos))
            else:
                batch = dataset
            pos = _pos

            yield list(map(numpy.array, zip(*batch)))

    return subGetter(trainset, trainSize), subGetter(validateset, None)

def TestGetter():
    f = open('../data/test.csv')
    next(f) # The first line is header
    lines = map(lambda row: row.strip().split(','), f)
    dataset = list(map(lambda row: numpy.array(tuple(map(float, row))), lines))
    del f

    for batch in [dataset[i * 500 : (i + 1) * 500] for i in range(len(dataset) // 500)]:
        yield numpy.array(batch)

