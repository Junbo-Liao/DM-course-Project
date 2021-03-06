import struct

import numpy
import scipy
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

features_num = 784

def loadImageSet(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = numpy.reshape(imgs, [imgNum, width * height])
    return imgs

def loadlabels(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    labelNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = numpy.reshape(labels, [labelNum])
    return labels

def loadFashion():
    file1 = 'train-images-idx3-ubyte'
    file2 = 'train-labels-idx1-ubyte'
    imgs = loadImageSet(file1)
    label = loadlabels(file2)
    file3 = 't10k-images-idx3-ubyte'
    file4 = 't10k-labels-idx1-ubyte'
    testimgs = loadImageSet(file3)
    testlabel = loadlabels(file4)
    return imgs,label

traindat,trainlab = loadFashion()
traindat = traindat/255.0
print(numpy.shape(traindat))
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1
examples_to_show = 10
n_input = 784

X = tf.placeholder("float", [None, n_input])

def get_batch(vec,lab, i):
    batches = []
    results = []
    texts = vec[i * batch_size:i * batch_size + batch_size]
    #print(numpy.shape(texts))
    categories = lab[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        features = numpy.zeros((features_num),dtype=float)
        for i in range(features_num):
            features[i] = text[i]
        batches.append(features)
    for category in categories:
        y = numpy.zeros((10), dtype=int)
        y[category] = 1
        results.append(y)
    return numpy.array(batches), numpy.array(results)

n_hidden_1 = 400
n_hidden_2 = 200
n_hidden_3 = 100
#n_hidden_4 = 20
#n_hidden_4 = 10
n_hidden_4 = 5

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], )),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


def compute_pur(res,trueres,labsize):
    s = len(res)
    matchtable = numpy.mat(numpy.zeros((labsize, labsize)))
    for i in range(s):
        matchtable[res[i], trueres[i]] += 1
    P = []
    M = []
    G = []
    for i in range(labsize):
        P.append(float(numpy.max(matchtable[i, :])))
        M.append(float(numpy.sum(matchtable[i, :])))
        sofi = 0.0;
        for j in range(labsize):
            sofi += pow(matchtable[i, j], 2)
        if M[i] != 0:
            G.append(float((1 - (sofi / (pow(M[i], 2)))) * M[i]))
        else:
            G.append(0)
    Purity = (numpy.sum(P)) / (numpy.sum(M))
    Gini = (numpy.sum(G)) / (numpy.sum(M))
    return Purity,Gini

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    total_batch = int(len(traindat) / batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = get_batch(traindat,trainlab,i)  # max(x) = 1, min(x) = 0
            pic, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1))

    encoder_result = sess.run(encoder_op, feed_dict={X: traindat})

    cl = KMeans(n_clusters=10,init='random').fit(encoder_result)
    Purity ,Gini = compute_pur(cl.labels_,trainlab,10)
    print("Purity: " + str(Purity) + "\tGini: " + str(Gini))