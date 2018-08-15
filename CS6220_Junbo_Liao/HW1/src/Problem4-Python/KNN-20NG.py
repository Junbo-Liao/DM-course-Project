import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import struct
from collections import Counter

import numpy
import operator
import math
import os


def computeClosestK(trainSet,testEntry,k):
    distance = []
    indexes = []
    length = len(trainSet)
    for i in range(length):
        dist = numpy.dot(testEntry,trainSet[i][0].T)
        distance.append((trainSet[i][1],dist))
        indexes.append((i, dist))
    distance.sort(key=operator.itemgetter(1),reverse=True)
    indexes.sort(key=operator.itemgetter(1),reverse=True)
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
    newsgroups = fetch_20newsgroups(subset='all')
    tfidf_filter_vec=TfidfVectorizer(analyzer='word',stop_words='english')
    total_vec = tfidf_filter_vec.fit_transform(newsgroups.data)
    total_labels = newsgroups.target
    trainset = []
    testimgs = []
    testlabel = []
    length = len(total_labels)
    for x in range(length):
        if(random.random()<0.8):
            trainset.append((total_vec[x],total_labels[x]))
        else:
            testimgs.append(total_vec[x])
            testlabel.append(total_labels[x])

    print('Parse finish')

    result = []
    for i in range(len(testimgs)):
    #for i in range(50):
        neibor,idx = computeClosestK(trainset,testimgs[i],5)
        lab = decideClass(neibor)
        result.append(lab)
        print(i)
    correct = 0
    for i in range(len(testlabel)):
    #for i in range(50):
        if result[i] == testlabel[i]:
            correct += 1
    accuracy = (correct/float(len(testlabel)))*100.0
    #accuracy = (correct / float(50)) * 100.0
    print('Accuracy: ' + repr(accuracy) + '%')

def test():
    newsgroups = fetch_20newsgroups(subset='all')
    tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    total_vec = tfidf_filter_vec.fit_transform(newsgroups.data)
    total_labels = newsgroups.target
    trainset = []
    testimgs = []
    testlabel = []
    length = len(total_labels)
    for x in range(length):
        if (random.random() < 0.8):
            trainset.append((total_vec[x], total_labels[x]))
        else:
            testimgs.append(total_vec[x])
            testlabel.append(total_labels[x])
    print('Parse finish')
    neibor,idx = computeClosestK(trainset, total_vec[10], 5)
    print(neibor)
    print(idx)

main()
#test()