import heapq
from sets import Set

import numpy
from numpy import *
import scipy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier


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

#print(shape(D30_vec))
print(labset)

# classifier = KMeans(n_clusters=20,tol=0.0,algorithm='full',init='random')
# res = classifier.fit_transform(D30_vec)
# predict_lab = []
# for i in range(len(res)):
#     s = heapq.nlargest(1, range(len(res[i])), res[i].__getitem__)
#     predict_lab.append(s[0])

# classifier = DBSCAN(eps=3.66,min_samples=2)#3.8,4
# classifier.fit(D30_vec)
# predict_lab = classifier.labels_

# cls = AgglomerativeClustering(n_clusters=20,linkage='ward')
# cls.fit(D30_vec)
# predict_lab = cls.labels_

gaussianmodel = GaussianMixture(n_components=20,covariance_type='diag')
gaussianmodel.fit(D30_vec)
predict_lab = gaussianmodel.predict(D30_vec)

purity1 = compute_pur(predict_lab,20,lab,labsize)
print(purity1)
purity2 = compute_pur(lab,labsize,predict_lab,20)
print(purity2)
taskres = (2*purity2*purity1)/(purity2+purity1)
print(taskres)