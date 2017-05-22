import itertools
import numpy # Should be imported after itertools

def TrainDataGetter(batchSize):
    lines = map(lambda row: row.strip().split(','), open('../data/train.csv'))
    dataset = map(lambda row: (int(row[0]), numpy.array(tuple(map(float, row[1:])))), lines)

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

