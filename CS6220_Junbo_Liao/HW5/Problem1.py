from datetime import datetime
from numpy import *
import scipy
from elasticsearch import Elasticsearch
from sklearn.datasets import fetch_20newsgroups
es = Elasticsearch()
from pyquery import PyQuery as pq
import os

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
print(len(doc))

path2 = "Summaries"
files = os.listdir(path2)

for file in files:
    if not file.startswith('.'):
        docno = str(file).lower().replace(".txt","")
        f = open(path2+"/"+file)
        content = f.read()
        doc_summer[docno] = content
print(len(doc_summer))

for no in doc_summer:
    if not no in doc:
        doc[no] = ""
# newsgroups = fetch_20newsgroups(subset='all')
# text = newsgroups.data
# labs = newsgroups.target
# print(len(text))
# print(len(labs))
for i in doc:
     print(i)
     body = {
         'doc_id': i,
         'gold_summary': "",
         'doc_text': doc[i],
     }
     if i in doc_summer:
         body['gold_summary'] = doc_summer[i]
     res = es.index(index="duc2001", doc_type='text', id=i, body=body)


# res = es.get(index="test-index", doc_type='tweet', id=1)
#
# es.indices.refresh(index="test-index")
#
# res = es.search(index="test-index", body={"query": {"match_all": {}}})
# print("Got %d Hits:" % res['hits']['total'])
# for hit in res['hits']['hits']:
#     print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])