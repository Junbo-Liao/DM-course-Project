import struct
from collections import Counter

import numpy
import operator
import math


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

def computeClosestK(trainSet,testEntry,k):
    distance = []
    indexes = []
    length = len(trainSet)
    for i in range(length):
        dist = numpy.linalg.norm(testEntry-trainSet[i][0])
        distance.append((trainSet[i][1],dist))
        indexes.append((i,dist))
    distance.sort(key=operator.itemgetter(1))
    indexes.sort(key=operator.itemgetter(1))
    neighbors = []
    indexlist = []
    for j in range(k):
        neighbors.append(distance[j][0])
        indexlist.append(indexes[j][0])
    return neighbors,indexlist

def decideClass(neighbors):
    label_count = Counter(neighbors)
    top = label_count.most_common(1)
    return top[0][0]

def main():
    file1= 'train-images.idx3-ubyte'
    file2= 'train-labels.idx1-ubyte'
    imgs,data_head = loadImageSet(file1)
    labels,labels_head = loadlabels(file2)
    trainset = [];
    length = len(labels)
    for x in range(length):
        trainset.append((imgs[x],labels[x]))

    file3= 't10k-images.idx3-ubyte'
    file4= 't10k-labels.idx1-ubyte'
    testimgs,ti_head = loadImageSet(file3)
    testlabel,tl_head = loadlabels(file4)
    print('Parse finish')

    result = []
    for i in range(len(testimgs)):
        neibor,idx = computeClosestK(trainset,testimgs[i],5)
        lab = decideClass(neibor)
        result.append(lab)
        print(i)
    correct = 0
    for i in range(len(testlabel)):
        if result[i] == testlabel[i]:
            correct += 1
    accuracy = (correct/float(len(testlabel)))*100.0
    print('Accuracy: ' + repr(accuracy) + '%')

def test():
    file1 = 'train-images.idx3-ubyte'
    file2 = 'train-labels.idx1-ubyte'
    imgs, data_head = loadImageSet(file1)
    labels, labels_head = loadlabels(file2)
    trainset = [];
    length = len(labels)
    for x in range(length):
        trainset.append((imgs[x], labels[x]))

    file3 = 't10k-images.idx3-ubyte'
    file4 = 't10k-labels.idx1-ubyte'
    testimgs, ti_head = loadImageSet(file3)
    testlabel, tl_head = loadlabels(file4)
    print('Parse finish')

    print(testimgs.shape)
    neibor,idx = computeClosestK(trainset, testimgs[2], 5)
    print(neibor)
    print(idx)

main()
#test()

