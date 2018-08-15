import heapq

from numpy import *
import struct

from sklearn.tree import tree


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
    imgs = reshape(imgs, [imgNum, width * height])
    return imgs

def loadlabels(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    labelNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = reshape(labels, [labelNum])
    return labels

def loadMINST():
    file1 = 'train-images-idx3-ubyte'
    file2 = 'train-labels-idx1-ubyte'
    imgs = loadImageSet(file1)
    label = loadlabels(file2)
    file3 = 't10k-images-idx3-ubyte'
    file4 = 't10k-labels-idx1-ubyte'
    testimgs = loadImageSet(file3)
    testlabel = loadlabels(file4)
    return imgs,label,testimgs,testlabel

def GraphdecisionTree(trainvec, trainlab,testvec,testlab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(trainvec, trainlab)
    DecTree = mode.feature_importances_
    res = mode.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))

def getAc(res,lab):
    length = shape(res)[0]
    acc = 0.0
    for i in range(length):
        if res[i] == lab[i]:
            acc += 1.0
    return acc/length

def PCA(data,N):
    meanValues=mean(data,axis=0)
    removedVals=data-meanValues
    covMat=cov(removedVals,rowvar=0)
    speVals,speVects=linalg.eig(mat(covMat))
    speValsIndex=heapq.nlargest(N,xrange(len(speVals)),speVals.__getitem__)
    remainVects=speVects[:,speValsIndex]
    return remainVects

def minusMean(data):
    meanValues = mean(data, axis=0)
    removedVals = data - meanValues
    return removedVals

def main():
    trainvec, trainlab,testvec,testlab = loadMINST()
    cv1 = PCA(trainvec,5)
    newtrainvec1 = minusMean(trainvec) * cv1
    newtestvec1 = minusMean(testvec) * cv1
    GraphdecisionTree(newtrainvec1, trainlab, newtestvec1, testlab)
    cv2 = PCA(trainvec, 20)
    newtrainvec2 = minusMean(trainvec) * cv2
    newtestvec2 = minusMean(testvec) * cv2
    GraphdecisionTree(newtrainvec2, trainlab, newtestvec2, testlab)


main()