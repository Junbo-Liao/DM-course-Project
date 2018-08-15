import numpy
import scipy
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

def getDist(veca,vecb):
    return numpy.linalg.norm(veca-vecb)

def getNeib(pt, pts, e):
    res = []
    for i in range(len(pts)):
        if getDist(pt,pts[i])<e:
            res.append(i)
    return res

def DBSCAN(dataSet,e,minPts):
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


def DBSCAN_circle():
    dataset = make_circles(n_samples=500)
    res = DBSCAN(dataset[0],0.15,3)
    showRes(dataset[0],res)
    showLab(dataset)

def DBSCAN_blob():
    dataset = make_blobs(n_samples=500,cluster_std=0.4)
    res = DBSCAN(dataset[0],0.3,10)
    showRes(dataset[0],res)
    showLab(dataset)

def DBSCAN_moon():
    dataset = make_moons(n_samples=500)
    res = DBSCAN(dataset[0],0.15,3)
    showRes(dataset[0],res)
    showLab(dataset)

def showLab(dataset):
    plt.figure(1)
    mark = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
    for i in range(len(dataset[1])):
        plt.plot(dataset[0][i][0], dataset[0][i][1], mark[dataset[1][i]])
    plt.show()

def showRes(dataset,res):
    print(len(dataset))
    print(res)
    plt.figure(2)
    mark = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
    for i in range(len(res)):
        for j in range(len(res[i + 1])):
            index = res[i + 1][j]
            plt.plot(dataset[index][0], dataset[index][1], mark[i])
    plt.show()

#DBSCAN_circle()
#DBSCAN_blob()
DBSCAN_moon()