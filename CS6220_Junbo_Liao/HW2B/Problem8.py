from time import time

import numpy
import struct

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering


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

def main():
    file1 = 'train-images-idx3-ubyte'
    file2 = 'train-labels-idx1-ubyte'
    imgs, data_head = loadImageSet(file1)
    label, labels_head = loadlabels(file2)
    file3 = 't10k-images-idx3-ubyte'
    file4 = 't10k-labels-idx1-ubyte'
    testimgs, ti_head = loadImageSet(file3)
    testlabel, tl_head = loadlabels(file4)
    ds = []
    ds.extend(imgs)
    ds.extend(testimgs)
    lb = []
    lb.extend(label)
    lb.extend(testlabel)
    labels = numpy.asarray(lb)
    #labels = testlabel
    #labels = numpy.asarray([lb[i] for i in range(1000)])
    labset = {}
    numpy.set_printoptions(threshold='nan')
    for i in range(labels.shape[0]):
        labset[labels[i]] = 0
    for i in range(labels.shape[0]):
        labset[labels[i]] += 1
    s = len(labset)
    data = numpy.asarray(ds)
    #data = testimgs
    #data = numpy.asarray([ds[i] for i in range(1000)])
    t0 = time()
    #result = hierarchy.fclusterdata(data, criterion='distance', t=10)
    cls = AgglomerativeClustering(n_clusters=10,linkage='ward')
    cls.fit(data)
    t1 = time()
    res = cls.labels_
    print(len(res))
    #print(cls.labels_)
    pur, gini, noise = getPur(res, labels)
    print("Time Cost: " + str(t1 - t0))
    print("Purity: " + str(pur))
    print("Gini: " + str(gini))
    print("Noise: " + str(noise))


def getPur(reslabels, labels):
    labset = set()
    for i in range(labels.shape[0]):
        labset.add(labels[i])
    s = len(labset)
    reslab = set()
    for i in range(len(reslabels)):
        reslab.add(reslabels[i])
    k = len(reslab)
    matchtable = numpy.mat(numpy.zeros((k, s)))
    noisecount = 0
    for i in range(labels.shape[0]):
        if reslabels[i] != -1:
            matchtable[int(reslabels[i]), int(labels[i])] += 1
        else:
            noisecount += 1
    P = []
    M = []
    G = []
    for i in range(k):
        P.append(float(numpy.max(matchtable[i, :])))
        M.append(float(numpy.sum(matchtable[i, :])))
        sofi = 0.0;
        for j in range(s):
            sofi += pow(matchtable[i, j], 2)
        if M[i] != 0:
            G.append(float((1 - (sofi / (pow(M[i], 2)))) * M[i]))
        else:
            G.append(0)
    Purity = (numpy.sum(P)) / (numpy.sum(M))
    Gini = (numpy.sum(G)) / (numpy.sum(M))
    Noise = float(noisecount) / labels.shape[0]
    return Purity, Gini, Noise


main()