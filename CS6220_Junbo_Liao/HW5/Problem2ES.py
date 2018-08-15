import heapq
from datetime import datetime
import numpy
from numpy import *
import scipy
from elasticsearch import Elasticsearch
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from pyquery import PyQuery as pq
import os

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

    tfidf_filter_vec1 = CountVectorizer(analyzer='word', stop_words='english')
    # tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
    vec = tfidf_filter_vec1.fit_transform(doc)
    featuresname = tfidf_filter_vec1.get_feature_names()
    return docid,vec, featuresname

def lda(data,k,featuresname):
    lda = LatentDirichletAllocation(n_components=k, learning_method='batch')
    lda.fit(data)
    print(shape(lda.components_))
    topic_words = lda.components_
    temp = []
    for i in range(shape(topic_words)[0]):
        topw = {}
        s = numpy.sum(topic_words[i])
        top20w = heapq.nlargest(10, range(len(topic_words[i])), topic_words[i].__getitem__)
        for j in range(len(top20w)):
            topw[featuresname[top20w[j]]] = topic_words[i][top20w[j]]/s
        temp.append(topw)
    X_new = lda.transform(data)
    return temp,X_new

def load20NG():
    newsgroups = fetch_20newsgroups(subset='all')
    tfidf_filter_vec1 = CountVectorizer(analyzer='word', stop_words='english')
    #tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
    vec = tfidf_filter_vec1.fit_transform(newsgroups.data)
    featuresname = tfidf_filter_vec1.get_feature_names()
    lab = newsgroups.target
    data = newsgroups.data
    return vec,featuresname,lab,data

def ducES():
    es = Elasticsearch()
    dugid,dugv,dugf= loadDUG()
    k = 20
    top,res = lda(dugv,k,dugf)
    # for i in range(k):
    #     body = {
    #         'topic_id': i,
    #         'top10_words': "",
    #     }
    #     for j in top[i].keys():
    #        body['top10_words'] += str(j) + ":" + str(top[i][j]) + "\n"
    #     esres = es.index(index="ductopic", doc_type='k20', id=i, body=body)

    print("topic word finished")
    doc = {}
    doc_summer = {}
    path1 = "DUC2001"
    files = os.listdir(path1)
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
            doc[docno] = doctext
    path2 = "Summaries"
    files = os.listdir(path2)
    for file in files:
        if not file.startswith('.'):
            docno = str(file).lower().replace(".txt","")
            f = open(path2+"/"+file)
            content = f.read()
            doc_summer[docno] = content
    print(shape(res))
    print(shape(dugid))

    topic = {}
    for j in range(len(dugid)):
        aray = res[j]
        top5 = heapq.nlargest(5, range(len(aray)), aray.__getitem__)
        top5topic = ""
        for i in range(5):
            top5topic += str(top5[i]) + "\t" + str(aray[top5[i]]) + "\n"
        topic[dugid[j]] = top5topic
    print(len(topic.keys()))
    for j in doc:
        body = {
         'doc_id': j,
         'gold_summary': "",
         'doc_text': doc[j],
         'doc_topic' : topic[j]
        }
        if j in doc_summer:
            body['gold_summary'] = doc_summer[j]
        esres = es.index(index="duc2001", doc_type='text', id=j, body=body)

def NGES():
    es = Elasticsearch()
    v, f, l ,d= load20NG()
    print("load finished")
    k = 20
    top,res = lda(v,k,f)
    print("LDA finished")
    # for i in range(k):
    #     body = {
    #         'topic_id': i,
    #         'top10_words': "",
    #     }
    #     for j in top[i].keys():
    #        body['top10_words'] += str(j) + ":" + str(top[i][j]) + "\n"
    #     esres = es.index(index="20ngtopic", doc_type='k20', id=i, body=body)
    print("topic word finished")

    topic = {}
    for j in range(len(res)):
        aray = res[j]
        top5 = heapq.nlargest(5, range(len(aray)), aray.__getitem__)
        top5topic = ""
        for i in range(5):
            top5topic += str(top5[i]) + "\t" + str(aray[top5[i]]) + "\n"
        topic[j] = top5topic
    print(len(topic.keys()))

    for j in range(len(d)):
        body = {
         'doc_id': j,
         'doc_text': d[j],
         'doc_label' : l[j],
         'doc_topic' : topic[j]
        }
        esres = es.index(index="20ng", doc_type='text', id=j, body=body)

NGES()