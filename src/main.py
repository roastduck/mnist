import shutil
import tensorflow as tf

import inout

def weightVar(shape):
    ''' Generate variables as weight '''

    with tf.name_scope('weight'):
        ret = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
        # tf.summary.histogram('histogram', ret)
        return ret

def biasVar(shape):
    ''' Generate variables as bias '''

    with tf.name_scope('bias'):
        ret = tf.Variable(tf.constant(0.1, shape = shape))
        # tf.summary.histogram('histogram', ret)
        return ret

def conv2d(x, W):
    ''' 2D convolutional operator '''

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2x2(x):
    ''' 2*2 max-pooling operator '''

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convLayer(x, height, width, inChannels, outChannels, name):
    ''' Convolutional layer '''

    with tf.name_scope(name):
        weight = weightVar([height, width, inChannels, outChannels])
        bias = biasVar([outChannels])
        conv = tf.nn.relu(conv2d(x, weight) + bias)
        return maxPool2x2(conv)

def denseLayer(x, inChannels, outChannels, name):
    ''' Densely connected layer '''

    with tf.name_scope(name):
        weight = weightVar([inChannels, outChannels])
        bias = biasVar([outChannels])
        return tf.nn.relu(tf.matmul(x, weight) + bias)

def outLayer(x, inChannels, outChannels):
    ''' Output layer '''

    with tf.name_scope('output'):
        weight = weightVar([inChannels, outChannels])
        bias = biasVar([outChannels])
        return tf.matmul(x, weight) + bias

def run():
    x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x')
    _y = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_true')
    xImage = tf.reshape(x, [-1,28,28,1])

    conv1 = convLayer(xImage, 5, 5, 1, 32, 'conv1')
    conv2 = convLayer(conv1, 5, 5, 32, 64, 'conv2')
    conv2Flat = tf.reshape(conv2, [-1, 7 * 7 * 64])
    dense1 = denseLayer(conv2Flat, 7 * 7 * 64, 1024, 'dense1')

    keepProb = tf.placeholder(tf.float32, name = 'keepProb')
    drop = tf.nn.dropout(dense1, keepProb)

    y = outLayer(drop, 1024, 10)
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(entropy)
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
    trainAccuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    validateAccuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # Call either this two
    trainSummary = tf.summary.scalar('trainAccuracy', trainAccuracy)
    validateSummary = tf.summary.scalar('validateAccuracy', validateAccuracy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    try:
        shutil.rmtree('logs')
    except FileNotFoundError:
        pass
    summaryWriter = tf.summary.FileWriter('logs', sess.graph)
    # summaries = tf.summary.merge_all()

    trainData, validateData = inout.DataGetter(50, 500)
    for i, trainBatch in zip(range(200000), trainData):
        if i % 100 == 0:
            validateBatch = next(validateData)
            accuT, summT = sess.run((trainAccuracy, trainSummary), feed_dict = {x: trainBatch[0], _y: trainBatch[1], keepProb: 1.0})
            accuV, summV = sess.run((validateAccuracy, validateSummary), feed_dict = {x: validateBatch[0], _y: validateBatch[1], keepProb: 1.0})
            print("Step %d, training accuracy %g, validating accuracy %g"%(i, accuT, accuV))
            summaryWriter.add_summary(summT, i)
            summaryWriter.add_summary(summV, i)

        sess.run(optimizer, feed_dict = {x: trainBatch[0], _y: trainBatch[1], keepProb: 0.5})

if __name__ == '__main__':
    run()

