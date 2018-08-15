import random
import struct
from sets import Set
from time import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy

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
    return imgs, head

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
    return labels, head

def computDist(veca,vecb):
    return numpy.linalg.norm(veca-vecb)

def initialcent(dataSet,k):
    n = dataSet.shape[1]
    centroids = numpy.mat(numpy.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
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
    file1 = 'FASHION/train-images-idx3-ubyte'
    file2 = 'FASHION/train-labels-idx1-ubyte'
    imgs, data_head = loadImageSet(file1)
    label, labels_head = loadlabels(file2)
    file3 = 'FASHION/t10k-images-idx3-ubyte'
    file4 = 'FASHION/t10k-labels-idx1-ubyte'
    testimgs, ti_head = loadImageSet(file3)
    testlabel, tl_head = loadlabels(file4)
    ds = []
    ds.extend(imgs)
    ds.extend(testimgs)
    lb = []
    lb.extend(label)
    lb.extend(testlabel)
    labels = numpy.asarray(lb)
    print('Parse finish')

    numpy.set_printoptions(threshold='nan')
    print(labels.shape)
    labset = Set()
    for i in range(labels.shape[0]):
        labset.add(labels[i])
    s = len(labset)

    start = time()
    myCentroids, clustAssing,itcount = MyKmean(numpy.asarray(ds), 10)
    #itcount = 88
    #clustAssing = numpy.mat(numpy.zeros((60000,2)))
    end = time()
    print("Time: " + str(end-start))
    print("iteration: " + str(itcount))
    matchtable = numpy.mat(numpy.zeros((k,s)))
    for i in range(labels.shape[0]):
        matchtable[int(clustAssing[i,0]),int(labels[i])] += 1
    P = []
    M = []
    G = []
    for i in range(k):
        P.append(float(numpy.max(matchtable[i,:])))
        M.append(float(numpy.sum(matchtable[i,:])))
        sofi = 0.0;
        for j in range(s):
            sofi += pow(matchtable[i,j],2)
        G.append(float((1-(sofi/(pow(M[i],2))))*M[i]))
    Purity = (numpy.sum(P))/(numpy.sum(M))
    Gini = (numpy.sum(G))/(numpy.sum(M))
    print("Purity: " + str(Purity))
    print("Gini: " + str(Gini))

    #print(itcount)
    #print(myCentroids)

    #classifier = KMeans(n_clusters=k)
    #classifier.fit(imgs)
    #centroids = classifier.cluster_centers_
    #labs = classifier.labels_
    #print(centroids.shape)
    #pca = PCA(n_components=2)
    #newData = pca.fit_transform(imgs)
    #mark = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo', 'r+', 'g+']
    #for i in xrange(1000):
    #    markIndex = int(labs[i])
    #    plt.plot(newData[i,0], newData[i,1], mark[markIndex])
    #    print(i)
    #plt.show()
main(10)
#main(5)
#main(20)

