import heapq
from random import randint
from sets import Set

import numpy
from numpy import *
import scipy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import chi2


def loaddata():
    mesg = loadtxt("mnist_noisy_SAMPLE5000_K20_F31.txt")
    F_num = shape(mesg)[1]
    vec = []
    lab = []
    for i in range(shape(mesg)[0]):
        lab.append(int(mesg[i][0]))
        vec.append(mesg[i][1:F_num])
    return vec,lab

vec,lab = loaddata()
print(shape(vec))
print(shape(lab))

labset = Set()
for i in range(shape(lab)[0]):
    labset.add(lab[i])
labsize = len(labset)

def compute_pur(res,predictlabsize,trueres,trulabsize):
    s = len(res)
    matchtable = numpy.mat(numpy.zeros((predictlabsize, trulabsize)))
    for i in range(s):
        matchtable[int(res[i]), int(trueres[i])] += 1
    P = []
    M = []
    for i in range(predictlabsize):
        P.append(float(numpy.max(matchtable[i, :])))
        M.append(float(numpy.sum(matchtable[i, :])))
    Purity = (numpy.sum(P)) / (numpy.sum(M))
    return Purity

D30 = PCA(n_components=30)
D30.fit(vec)
D30_vec = D30.transform(vec)

train_vec = []
train_lab = []

for i in range(500):
    g = randint(0,5000)
    train_vec.append(D30_vec[g])
    train_lab.append(lab[g])

cls = KNeighborsClassifier(n_neighbors=10).fit(train_vec,train_lab)
predict_lab = cls.predict(D30_vec)

predictsize = Set()
for i in range(shape(predict_lab)[0]):
    predictsize.add(predict_lab[i])
ps = len(predictsize)

purity1 = compute_pur(predict_lab,ps,lab,labsize)
print(purity1)
purity2 = compute_pur(lab,labsize,predict_lab,ps)
print(purity2)
taskres = (2*purity2*purity1)/(purity2+purity1)
print(taskres)