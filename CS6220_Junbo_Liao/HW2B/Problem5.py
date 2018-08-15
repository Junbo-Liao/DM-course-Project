import random
import matplotlib.pyplot as plt
import numpy


def getDist(veca,vecb):
    return numpy.linalg.norm(veca-vecb)

def getNeib(pt, pts, e):
    res = []
    for i in range(len(pts)):
        if getDist(pt,pts[i])<e:
            res.append(i)
    return res

def loadData(filename):
    f = open(filename)
    next(f)
    lines = []
    for e in f:
        line = e.replace("\"\n","").split("\",\"")
        lines.append(parseLine(line))
    return lines

def parseLine(line):
    res = []
    #res.append(int(line[1]))
    res.append(float(line[2]))
    res.append(float(line[3]))
    neib = line[5].split(",")
    neibs = []
    for i in range(len(neib)):
        neibs.append(int(neib[i]))
    res.append(neibs)
    return res

def DBSCAN(dataSet, minPts):
    coreObjs = {}
    C = {}
    n = len(dataSet)
    for i in range(n):
        data = dataSet[i]
        if len(data[2])>=minPts:
            coreObjs[i] = data[2]
    oldCoreObjs = coreObjs.copy()
    k = 0
    notAccess = range(n)
    while len(coreObjs)>0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        randNum = random.randint(0,len(cores)-1)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys() :
                delte = [val for val in oldCoreObjs[q] if val in notAccess]
                queue.extend(delte)
                notAccess = [val for val in notAccess if val not in delte]
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C

def main():
    dataSet = loadData("dbscan.csv")
    res = DBSCAN(dataSet,3)
    print(res)
    mark = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
    for i in range(len(res)):
        for j in range(len(res[i+1])):
            index = res[i+1][j]
            plt.plot(dataSet[index][0], dataSet[index][1], mark[i])
    plt.show()

main()