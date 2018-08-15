from sets import Set
from time import time

import numpy
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def computDist(veca,vecb):
    return numpy.linalg.norm(veca-vecb)

def initialcent(dataSet,k):
    n = dataSet.shape[1]
    centroids = numpy.mat(numpy.zeros((k, n)))
    for j in range(n):
        print(j)
        minJ = min(dataSet[:, j])[0,0]
        rangeJ = max(dataSet[:, j])[0,0] - minJ
        centroids[:, j] = minJ + rangeJ * numpy.random.rand(k, 1)
    return centroids

def MyKmean(dataSet,k):
    m = dataSet.shape[0]
    clusterAssment = numpy.mat(numpy.zeros((m, 2)))
    centroids = initialcent(dataSet, k)
    clusterChanged = True
    itcount = 0
    while clusterChanged:
        itcount = itcount+1
        print(itcount)
        clusterChanged = False
        for i in range(m):
            minDist = numpy.inf
            minIndex = -1
            for j in range(k):
                distJI = computDist(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist
        for cent in range(k):
            index_all = clusterAssment[:, 0]
            value = numpy.nonzero(index_all == cent)
            ptsInClust = dataSet[value[0]]
            centroids[cent, :] = numpy.mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment,itcount

def main(k):
    newsgroups = fetch_20newsgroups(subset='all')
    tfidf_filter_vec=TfidfVectorizer(analyzer='word',stop_words='english',min_df=0.05,max_df=0.95)
    total_vec = tfidf_filter_vec.fit_transform(newsgroups.data)
    total_labels = newsgroups.target
    print('Parse finish')
    print(total_vec.shape)
    numpy.set_printoptions(threshold='nan')
    print(total_labels.shape)
    labset = Set()
    for i in range(total_labels.shape[0]):
        labset.add(total_labels[i])
    s = len(labset)
    start = time()
    myCentroids, clustAssing, itcount = MyKmean(total_vec, k)
    #itcount = 88
    #clustAssing = numpy.mat(numpy.zeros((60000,2)))
    end = time()
    print("Time: " + str(end - start))
    print("iteration: " + str(itcount))
    matchtable = numpy.mat(numpy.zeros((k, s)))
    for i in range(total_labels.shape[0]):
        matchtable[int(clustAssing[i, 0]), int(total_labels[i])] += 1
    P = []
    M = []
    G = []
    for i in range(k):
        P.append(float(numpy.max(matchtable[i, :])))
        M.append(float(numpy.sum(matchtable[i, :])))
        sofi = 0.0;
        for j in range(s):
            sofi += pow(matchtable[i, j], 2)
        if M[i]!=0:
            G.append(float((1 - (sofi / (pow(M[i], 2)))) * M[i]))
        else:
            G.append(0)
    Purity = (numpy.sum(P)) / (numpy.sum(M))
    Gini = (numpy.sum(G)) / (numpy.sum(M))
    print("Purity: " + str(Purity))
    print("Gini: " + str(Gini))

    #classifier = KMeans(n_clusters=k)
    #classifier.fit(total_vec)
    #numpy.set_printoptions(threshold='nan')
    #centroids = classifier.cluster_centers_
    #labs = classifier.labels_
    #print(centroids.shape)

main(20)
#main(10)
#main(5)
