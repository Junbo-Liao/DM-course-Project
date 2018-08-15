import numpy
import scipy
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

features_num = 1000

newsgroups = fetch_20newsgroups(subset='all')
count_vec = CountVectorizer(analyzer='word', stop_words='english')
vec = count_vec.fit_transform(newsgroups.data)
lab = newsgroups.target
newvec = SelectKBest(chi2, k=features_num).fit_transform(vec, lab).todense()
print(numpy.shape(newvec))

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1
examples_to_show = 10
n_input = 1000

X = tf.placeholder("float", [None, n_input])
n_hidden_1 = 500
#n_hidden_2 = 200
n_hidden_2 = 100

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

def get_batch(vec,lab, i):
    batches = []
    results = []
    texts = vec[i * batch_size:i * batch_size + batch_size]
    categories = lab[i * batch_size:i * batch_size + batch_size]
    for text in texts:
        features = numpy.zeros((features_num),dtype=float)
        for i in range(features_num):
            features[i] = text[0,i]
        batches.append(features)
    for category in categories:
        y = numpy.zeros((20), dtype=int)
        y[category] = 1
        results.append(y)
    return numpy.array(batches), numpy.array(results)

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

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

    total_batch = int(len(newvec) / batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = get_batch(newvec,lab,i)  # max(x) = 1, min(x) = 0
            pic, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    encoder_result = sess.run(encoder_op, feed_dict={X: newvec})

    cl = KMeans(n_clusters=20,init='random').fit(encoder_result)
    Purity ,Gini = compute_pur(cl.labels_,lab,20)
    print("Purity: " + str(Purity) + "\tGini: " + str(Gini))