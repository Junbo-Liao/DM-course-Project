import random
from time import time

import numpy
import struct

from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


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

def getDist(veca,vecb):
    return cosine_distances(veca,vecb)

def getNeib(pt, pts, e):
    res = []
    for i in range(len(pts)):
        if getDist(pt,pts[i])<e:
            res.append(i)
    return res

def myDBSCAN(dataSet,e,minPts):
    coreObjs = {}
    C = {}
    n = len(dataSet)
    for i in range(n):
        neibor = getNeib(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0
    notAccess = range(n)
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        randNum = random.randint(0, len(cores)-1)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q] if val in notAccess]
                queue.extend(delte)
                notAccess = [val for val in notAccess if val not in delte]
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C

def getkdist(data,dataset,k):
    res = []
    for i in range(len(dataset)):
        res.append(getDist(data, dataset[i]))
    res.sort()
    return res[k-1]

def main():
    newsgroups = fetch_20newsgroups(subset='all')
    tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    ds = tfidf_filter_vec.fit_transform(newsgroups.data)
    print(ds.shape)
    lb = newsgroups.target
    #labels = numpy.asarray(lb)
    labels = numpy.asarray([lb[i] for i in range(1000)])
    # data = numpy.asarray(ds)
    data = csr_matrix((1000,ds.shape[1]))
    for i in range(1000):
        data[i] = ds[i]
    print(data.shape)
    #data = numpy.asarray(data.todense())
    labset = {}
    numpy.set_printoptions(threshold='nan')
    for i in range(labels.shape[0]):
        labset[labels[i]] = 0
    for i in range(labels.shape[0]):
        labset[labels[i]] +=1
    s = len(labset)
    print("Parse Finish")
    # n = len(data)
    # single = 0
    # toomuch = 0
    # for i in range(n):
    #     #print(i)
    #     neibor = getNeib(data[i], data, 3.8)
    #     if len(neibor) <2:
    #         single += 1
    #     if len(neibor) > n/10:
    #         toomuch += 1
    # #print("res")
    # print(single)
    # print(toomuch)
    t0 = time()
    classifier = DBSCAN(eps=3.66,min_samples=2,metric='precomputed')#3.8,4
    classifier.fit(data)
    t1 = time()
    res = classifier.labels_
    #print(res)

    # classifier = myDBSCAN(data,3.66,2)
    # t1 = time()
    # print(classifier)
    # res = numpy.ones(len(data))
    # res = -res
    # count = 0
    # for i in classifier.keys():
    #     for j in classifier[i]:
    #         count += 1
    #         res[j] = i-1
    # print(count)

    #print(labset)

    # table = []
    # for i in range(len(data)):
    #     table.append(getkdist(data[i],data,2))
    # table.sort()
    # plt.plot(table,'ro')
    # plt.show()

    pur,gini,noise = getPur(res,labels)
    print("Time Cost: " + str(t1-t0))
    print("Purity: " + str(pur))
    print("Gini: " + str(gini))
    print("Noise: " + str(noise))

def getPur(reslabels,labels):
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
    Noise = float(noisecount)/labels.shape[0]
    return Purity,Gini,Noise

main()