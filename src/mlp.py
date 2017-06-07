import sys
import json
import tensorflow as tf

import inout
import disturb

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

def denseLayer(x, inChannels, outChannels, name):
    ''' Densely connected layer '''

    with tf.name_scope(name):
        weight = weightVar([inChannels, outChannels])
        bias = biasVar([outChannels])
        return tf.nn.elu(tf.matmul(x, weight) + bias)

def outLayer(x, inChannels, outChannels):
    ''' Output layer '''

    with tf.name_scope('output'):
        weight = weightVar([inChannels, outChannels])
        bias = biasVar([outChannels])
        return tf.matmul(x, weight) + bias

def run(action, expId = None, runId = None, stepId = None):
    if action == 'train':
        with open("experiments/%s/%s/conf.json"%(expId, runId)) as f:
            conf = json.load(f)
        assert "startEpisode" in conf
        assert "endEpisode" in conf
        assert "learningRate" in conf
        assert "fromCheckpoint" in conf

    x = tf.placeholder(tf.float32, shape=[None, 784], name = 'x')
    _y = tf.placeholder(tf.float32, shape=[None, 10], name = 'y_true')

    keepProb = tf.placeholder(tf.float32, name = 'keepProb')
    dense1 = tf.nn.dropout(denseLayer(x, 784, 1024, 'dense1'), keepProb)
    dense2 = tf.nn.dropout(denseLayer(dense1, 1024, 1024, 'dense2'), keepProb)

    y = outLayer(dense2, 1024, 10)
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_y, logits=y))
    if action == 'train':
        optimizer = tf.train.AdamOptimizer(conf["learningRate"]).minimize(entropy)
    output = tf.argmax(y, 1)
    _output = tf.argmax(_y, 1)
    correct = tf.equal(output, _output)
    trainError = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
    validateError = 1 - tf.reduce_mean(tf.cast(correct, tf.float32)) # Call either this two
    trainSummary = tf.summary.scalar('trainError', trainError)
    validateSummary = tf.summary.scalar('validateError', validateError)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if action == 'train':
        summaryWriter = tf.summary.FileWriter('experiments/%s/%s/logs'%(expId, runId), sess.graph)
    # summaries = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = None)

    if action == 'train':
        if action == 'train' and conf["fromCheckpoint"] is not None:
            saver.restore(sess, "experiments/%s/%s/checkpoint%s"%(expId, runId - 1, conf["fromCheckpoint"]))

        trainData, validateData = inout.DataGetter(50, 5000)
        for i, trainBatch in zip(range(conf["startEpisode"], conf["endEpisode"]), trainData):
            if (i + 1) % 100 == 0:
                validateBatch = next(validateData)
                errT, summT = sess.run((trainError, trainSummary), feed_dict = {x: trainBatch[0], _y: trainBatch[1], keepProb: 1.0})
                errV, summV = sess.run((validateError, validateSummary), feed_dict = {x: validateBatch[0], _y: validateBatch[1], keepProb: 1.0})
                print("Step %d, training error %g, validating error %g"%(i, errT, errV))
                summaryWriter.add_summary(summT, i)
                summaryWriter.add_summary(summV, i)
            if (i + 1) % 1000 == 0:
                saver.save(sess, "experiments/%s/%s/checkpoint%s"%(expId, runId, i))

            #disturb.disturbBatch(trainBatch[0])
            sess.run(optimizer, feed_dict = {x: trainBatch[0], _y: trainBatch[1], keepProb: 0.75})
    else:
        if action == 'test':
            saver.restore(sess, "experiments/%s/%s/checkpoint%s"%(expId, runId, stepId))
        else:
            saver.restore(sess, "../data/mlp_final")
        imgId = 0
        with open('../data/submission.csv', 'w') as f:
            f.write('ImageId,Label\n')
            for batch in inout.TestGetter():
                for res in sess.run(output, feed_dict = {x: batch, keepProb: 1}):
                    imgId += 1
                    f.write('%d,%d\n'%(imgId, res))

if __name__ == '__main__':
    if not (len(sys.argv) == 4 and sys.argv[1] == 'train') and not (len(sys.argv) == 5 and sys.argv[1] == 'test') and not (len(sys.argv) == 2 and sys.argv[1] == 'run'):
        print("Usage:")
        print("Train a network and save to checkpoint:")
        print(" python3 mlp.py train <experimentID> <runID>")
        print("Test a trained checkpoint:")
        print(" python3 mlp.py test <experimentID> <runID> <stepID>")
        print("Run the final network:")
        print(" python3 mlp.py run")
        exit(0)

    if sys.argv[1] == 'train':
        run('train', sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'test':
        run('test', sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        run('run')

