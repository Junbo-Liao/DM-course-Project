import random
from time import time

import numpy
import struct
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def getDist(veca,vecb):
    veca = numpy.asarray(veca)
    vecb = numpy.asarray(vecb)
    return numpy.linalg.norm(veca-vecb)

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
        #print(i)
        res.append(getDist(data, dataset[i]))
    res.sort()
    return res[k-1]

def loadData(filename):
    f = open(filename)
    next(f)
    lines = []
    for e in f:
        line = e.split(";")
        if ("" not in line) & ("?" not in line):
            lines.append(parseLine(line))
    return lines

def parseLine(line):
    res = []
    res.append(float(line[2]))
    res.append(float(line[3]))
    res.append(float(line[4]))
    res.append(float(line[5]))
    res.append(float(line[6]))
    res.append(float(line[7]))
    res.append(float(line[8]))
    return res

def main():
    dataset = loadData("household_power_consumption.txt")
    print(len(dataset))
    data = []
    for i in range(10000):
        j = random.randint(0,len(dataset)-1)
        data.append(numpy.asarray(dataset[j]))
    t0 = time()
    classifier = DBSCAN(eps=2.7,min_samples=4)#3.8,4
    classifier.fit(data)
    t1 = time()
    res = classifier.labels_
    print(res)
    # classifier = myDBSCAN(data,6.6,16)
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
    #     print(i)
    #     table.append(getkdist(data[i],data,4))
    # table.sort()
    # plt.plot(table,'ro')
    # plt.show()

    # pur,gini,noise = getPur(res,labels)
    # print("Time Cost: " + str(t1-t0))
    # print("Purity: " + str(pur))
    # print("Gini: " + str(gini))
    # print("Noise: " + str(noise))

main()