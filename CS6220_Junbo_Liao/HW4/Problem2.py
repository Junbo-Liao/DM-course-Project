import numpy
import scipy
import tensorflow as tf
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

# def main1():
#     mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
#     learning_rate = 0.1
#     num_steps = 1000
#
#     input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'images': mnist.train.images}, y=mnist.train.labels,
#         batch_size=128, num_epochs=None, shuffle=True)
#
#     def neural_net(x_dict):
#         x = x_dict['images']
#         layer_1 = tf.layers.dense(x, 256)
#         layer_2 = tf.layers.dense(layer_1, 256)
#         out_layer = tf.layers.dense(layer_2, 10)
#         return out_layer
#
#
#     def model_fn(features, labels, mode):
#         logits = neural_net(features)
#         pred_classes = tf.argmax(logits, axis=1)
#         pred_probas = tf.nn.softmax(logits)
#
#         if mode == tf.estimator.ModeKeys.PREDICT:
#             return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
#         loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#         train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
#
#         # Evaluate the accuracy of the model
#         acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
#
#         # TF Estimators requires to return a EstimatorSpec, that specify
#         # the different ops for training, evaluating, ...
#         estim_specs = tf.estimator.EstimatorSpec(
#             mode=mode,
#             predictions=pred_classes,
#             loss=loss_op,
#             train_op=train_op,
#             eval_metric_ops={'accuracy': acc_op})
#
#         return estim_specs
#
#     model = tf.estimator.Estimator(model_fn)
#     model.train(input_fn, steps=num_steps)
#     input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={'images': mnist.test.images}, y=mnist.test.labels,
#         batch_size=128, shuffle=False)
#     print(model.evaluate(input_fn))
#
def main2():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def add_layer(inputs, in_size, out_size, activation_function=None, ):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs

    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result

    xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])

    prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))  # loss
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    print(compute_accuracy(mnist.test.images, mnist.test.labels))

features_num = 200

def main3():
    newsgroups = fetch_20newsgroups(subset='all')
    count_vec = CountVectorizer(analyzer='word', stop_words='english')
    vec = count_vec.fit_transform(newsgroups.data)
    lab = newsgroups.target
    newvec = SelectKBest(chi2, k=features_num).fit_transform(vec, lab).todense()
    print(numpy.shape(newvec))
    #print(newvec)

    def add_layer(inputs, in_size, out_size, activation_function=None, ):
        weights = {
            'h1': tf.Variable(tf.random_normal([features_num, 100])),
            'h2': tf.Variable(tf.random_normal([100, 100])),
            'out': tf.Variable(tf.random_normal([100, 20]))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([1,100])+0.1,),
            'b2': tf.Variable(tf.zeros([1,100])+0.1,),
            'out': tf.Variable(tf.zeros([1,20])+0.1,)
        }
        layer_1_multiplication = tf.matmul(inputs, weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
        layer_1 = tf.nn.relu(layer_1_addition)

        layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
        layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
        layer_2 = tf.nn.relu(layer_2_addition)

        out_layer_multiplication = tf.matmul(layer_2, weights['out'])
        out_layer_addition = out_layer_multiplication + biases['out']
        return out_layer_addition

    def compute_accuracy(v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict={xs: v_xs})
        v_ys = numpy.array(v_ys)
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
        return result

    xs = tf.placeholder(tf.float32, [None, features_num])
    ys = tf.placeholder(tf.float32, [None, 20])

    prediction = add_layer(xs, features_num, 20, activation_function=tf.nn.softmax)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    batchsize = 100

    def get_batch(vec,lab, i):
        batches = []
        results = []
        texts = vec[i * batchsize:i * batchsize + batchsize]
        categories = lab[i * batchsize:i * batchsize + batchsize]
        for text in texts:
            features = numpy.zeros((features_num),dtype=float)
            for i in range(features_num):
                features[i] = text[0,i]
            batches.append(features)
        for category in categories:
            y = numpy.zeros((20), dtype=int)
            y[category] = 1
            results.append(y)
        return batches, results

    def getall(vec,lab):
        batches = []
        results = []
        texts = vec[0:8000]
        categories = lab[0:8000]
        for text in texts:
            features = numpy.zeros((features_num), dtype=float)
            for i in range(features_num):
                features[i] = text[0, i]
            batches.append(features)
        for category in categories:
            y = numpy.zeros((20), dtype=int)
            y[category] = 1
            results.append(y)
        return batches, results

    for i in range(1000):
        print(i)
        batch_xs, batch_ys = get_batch(newvec,lab,i)
        if(len(batch_xs)==0):
            break
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    all_xs, all_ys = getall(newvec,lab)
    print(compute_accuracy(all_xs,all_ys))


#main1()
main2()
#main3()