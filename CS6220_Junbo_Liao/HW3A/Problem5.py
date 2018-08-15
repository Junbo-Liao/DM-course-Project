import heapq
from sets import Set

from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SelectKBest
from numpy import *
from sklearn.tree import tree


def MaildecisionTree(vec,lab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(vec, lab)
    res = mode.predict(vec)
    print("Accuracy: " + str(getAc(res, lab)))

def getAc(res,lab):
    length = shape(res)[0]
    acc = 0.0
    for i in range(length):
        if res[i] == lab[i]:
            acc += 1.0
    return acc/length

newsgroups = fetch_20newsgroups(subset='all')
tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
vec = tfidf_filter_vec1.fit_transform(newsgroups.data)
lab = newsgroups.target

lsvc = LinearSVC(penalty="l1", dual=False).fit(vec, lab)

# weight = lsvc.coef_
# featureset = Set()
# for i in range(shape(weight)[0]):
#     top200f = heapq.nlargest(200, xrange(shape(weight)[1]), weight[i].__getitem__)
#     for i in range(200):
#         featureset.add(top200f[i])
# print(len(featureset))
# newvec = csr_matrix(matrix(vec[:,k] for k in featureset))

model = SelectFromModel(lsvc, prefit=True,threshold=5.0)
newvec = model.transform(vec)

# def MyFunc(x,y):
#     lsvc = LinearSVC(penalty="l1", dual=False).fit(x, y)
#     weight = lsvc.coef_
#     res = []
#     for i in range(shape(weight)[1]):
#         s = sum(weight[:, i])
#         res.append(s)
#     return res
# newvec = SelectKBest(MyFunc, k=200).fit(vec, lab)
print(shape(newvec))
MaildecisionTree(newvec,lab)