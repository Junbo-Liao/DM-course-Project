import os
import string

import numpy
from elasticsearch import Elasticsearch
from numpy import *
import scipy
from sklearn.feature_extraction.text import CountVectorizer

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
                        doctext += content[i]
                        i += 1
            doc.append(doctext)
            docid.append(docno)

    return docid,doc

def computeKL(px,py):
    px = asarray(px)
    py = asarray(py)
    sumx = sum(px)
    sumy = float(sum(py))
    px /= sumx
    py = asarray(py/sumy)
    KL = 0.0
    for i in range(shape(px)[1]):
        if py[0,i] != 0 and px[0,i] != 0:
            delta = px[0,i] * log(px[0,i] / py[0,i])
            KL += delta
        # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))
    return KL

def computeLDAKL(px,py):
    px = asarray(px)
    py = asarray(py)
    sumx = sum(px)
    sumy = float(sum(py))
    px /= sumx
    py = asarray(py/sumy)
    KL = 0.0
    for i in range(shape(px)[1]):
        if py[0][i] != 0 and px[0][i] != 0:
            delta = px[0][i] * log(px[0][i] / py[0][i])
            KL += delta
        # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))
    return KL

def KLsummary(doc,k):
    fulltext = []
    fulltext.append(doc)
    subtext = doc.split(".")
    cls = CountVectorizer(analyzer='word', stop_words='english').fit(fulltext)
    Full = cls.transform(fulltext).todense()
    Sub = cls.transform(subtext).todense()

    Sum = []
    temp = mat(zeros((shape(Full))))
    newsum = []
    for i in range(k):
        for j in range(shape(Sub)[0]):
            s = temp + Sub[j]
            v = computeKL(s,Full)
            newsum.append(v)
        minindex = newsum.index(min(newsum))
        temp = numpy.add(temp,Sub[minindex])
        Sum.append(subtext[minindex])
        Sub = numpy.delete(Sub,minindex,axis=0)
        #Sub = Sub[0:minindex] + Sub[minindex+1:]
        newsum = []
    return Sum

def wordcount(text,words):
    res = []
    for i in range(len(text)):
        wc = zeros((1,len(words)))
        wordsintext = text[i].split(" ")
        for j in range(len(wordsintext)):
            if wordsintext[j] in words:
                index = words.index(wordsintext[j])
                wc[0,index] += 1
        res.append(wc)
    return res

def LDAsummary(doc,words,k):
    fulltext = []
    fulltext.append(doc)
    subtext = doc.split(".")
    Full = wordcount(fulltext,words)[0]
    Sub = wordcount(subtext,words)

    Sum = []
    temp = mat(zeros((shape(Full))))
    newsum = []
    for i in range(k):
        for j in range(shape(Sub)[0]):
            s = temp + Sub[j]
            v = computeLDAKL(s, Full)
            newsum.append(v)
        minindex = newsum.index(min(newsum))
        temp = numpy.add(temp, Sub[minindex])
        Sum.append(subtext[minindex])
        Sub = numpy.delete(Sub, minindex, axis=0)
        newsum = []
    return Sum

es = Elasticsearch()
docid,doc = loadDUG()
f = open('DUGtopicWord')
for line in f:
    words = line.split(",")
uniqueW = []
for i in range(len(words)):
    if words[i] not in uniqueW:
        uniqueW.append(words[i])
print(len(uniqueW))

path2 = "Summaries"
files = os.listdir(path2)
doc_summer = {}
for file in files:
    if not file.startswith('.'):
        docno = str(file).lower().replace(".txt","")
        f = open(path2+"/"+file)
        content = f.read()
        doc_summer[docno] = content

# for i in range(len(docid)):
#     KLS = KLsummary(doc[i], 4)
#     LDAS = LDAsummary(doc[i],uniqueW,4)
#     print(i)
#     body = {
#      'doc_id': docid[i],
#      'gold_summary': "",
#      'KL_summary': "",
#      'LDA_summary': "",
#      'doc_text': doc[i]
#     }
#     for j in range(4):
#         body['KL_summary'] += KLS[j]
#     for j in range(4):
#         body['LDA_summary'] += LDAS[j]
#     if docid[i] in doc_summer:
#         body['gold_summary'] = doc_summer[docid[i]]
#     res = es.index(index="duc2001sum", doc_type='sum', id=docid[i], body=body)

path3 = 'MyKLres/'
os.makedirs(path3)
for i in range(len(docid)):
    print(i)
    f = open(path3+docid[i]+'.txt','w')
    KLS = KLsummary(doc[i], 4)
    KL_summary = ""
    for j in range(4):
        KL_summary += KLS[j]
    f.write(KL_summary)
    f.close()