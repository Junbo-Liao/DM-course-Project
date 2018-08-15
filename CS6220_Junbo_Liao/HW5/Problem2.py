import heapq
import os
from sets import Set

import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy import *

def load20NG():
    newsgroups = fetch_20newsgroups(subset='all')
    #tfidf_filter_vec1 = CountVectorizer(analyzer='word', stop_words='english')
    tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
    vec = tfidf_filter_vec1.fit_transform(newsgroups.data)
    featuresname = tfidf_filter_vec1.get_feature_names()
    lab = newsgroups.target
    return vec,featuresname,lab

def loadDUG():
    doc = []
    path1 = "DUC2001"
    files = os.listdir(path1)
    docid = []
    for file in files:
        if not file.startswith('.'):
            docno = str(file).lower()
            p = path1 + "/" + file
            f = open(p)
            content = f.read().splitlines()
            doctext = ""
            for i in range(len(content)):
                while content[i].startswith("<TEXT>"):
                    i += 1
                    while not content[i].__contains__("</TEXT>"):
                        doctext += content[i] + "\n"
                        i += 1
            doc.append(doctext)
            docid.append(docno)
    path2 = "annotations.txt"
    f = open(path2)
    lab = {}
    for line in f:
        first = line.split(";")[0]
        mesg = first.split("@")
        lab[mesg[0].lower()] = mesg[1]
    tfidf_filter_vec1 = CountVectorizer(analyzer='word', stop_words='english')
    # tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
    vec = tfidf_filter_vec1.fit_transform(doc)
    featuresname = tfidf_filter_vec1.get_feature_names()
    return docid,vec, featuresname,lab

def lda(data,k,featuresname):
    lda = LatentDirichletAllocation(n_components=k, learning_method='batch')
    lda.fit(data)
    print(shape(lda.components_))
    topic_words = lda.components_
    temp = []
    for i in range(shape(topic_words)[0]):
        topw = {}
        s = numpy.sum(topic_words[i])
        top20w = heapq.nlargest(20, range(len(topic_words[i])), topic_words[i].__getitem__)
        for j in range(len(top20w)):
            topw[featuresname[top20w[j]]] = topic_words[i][top20w[j]]/s
        temp.append(topw)
    X_new = lda.transform(data)
    return temp,X_new

def nmf(data,k,featuresname):
    lda = NMF(n_components=k, random_state=1, alpha=.1, l1_ratio=.5)
    lda.fit(data)
    print(shape(lda.components_))
    topic_words = lda.components_
    temp = []
    for i in range(shape(topic_words)[0]):
        topw = {}
        s = numpy.sum(topic_words[i])
        top20w = heapq.nlargest(20, range(len(topic_words[i])), topic_words[i].__getitem__)
        for j in range(len(top20w)):
            topw[featuresname[top20w[j]]] = topic_words[i][top20w[j]]/s
        temp.append(topw)
    X_new = lda.transform(data)
    return temp,X_new

def getPur(reslabels,labels,k):
    labset = set()
    for i in range(labels.shape[0]):
        labset.add(labels[i])
    s = len(labset)
    reslab = set()
    matchtable = numpy.mat(numpy.zeros((k, s)))
    for i in range(labels.shape[0]):
        result = heapq.nlargest(1, range(len(reslabels[i])), reslabels[i].__getitem__)
        matchtable[result[0], int(labels[i])] += 1
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
    return Purity,Gini

# dugid,dugv,dugf,dugl= loadDUG()
# ls = Set()
# for i in dugl.keys():
#     ls.add(dugl[i])
# print(ls.__sizeof__())
# print(len(dugl.keys()))
# print(shape(dugid))
# print(shape(dugv))
# print(shape(dugf))
# print(dugf)
# for i in range(len(dugf)):
#     print dugf[i],
# k = 50
# top,res = nmf(dugv,k,dugf)
# for i in range(k):
#      print("Topic"+str(i))
#      for j in top[i].keys():
#          print (str(j) + ":" + str(top[i][j]))
# v,f,l = load20NG()
# k = 50
# top,res = nmf(v,k,f)
# for i in range(k):
#     print("Topic"+str(i))
#     for j in top[i].keys():
#         print (str(j) + ":" + str(top[i][j]))
#
# pur,gini = getPur(res,l,k)
# print "Purity: " + str(pur),
# print("Gini: " + str(gini))