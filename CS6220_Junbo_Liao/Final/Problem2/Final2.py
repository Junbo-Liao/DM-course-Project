import heapq
import string

from numpy import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.mixture import GaussianMixture

import soft_clustering_measure
delset = string.punctuation

def load20NG(path):
    f = open(path)
    components = {}
    for line in f:
        #[id, 'lab', 'context']
        idindec = line.find(",")
        id = int(line[1:idindec])
        #'lab','context'
        rest = line[idindec+2:-1]
        labindex = rest.find("\'",1)
        lab = rest[1:labindex]
        #'context'
        text = rest[labindex+2:].translate(None,delset)
        components[id] = (str(lab),str(text))
    # for i in components:
    #     print(i)
    #     print(components[i])
    return components

def LDA(data,k):
    lda = LatentDirichletAllocation(n_components=k, learning_method='batch')
    lda.fit(data)
    return lda

def main():
    sampledata = load20NG("sample_dataset.txt")
    fulldata = load20NG("whole_dataset.txt")
    print("LoadFinished")
    ids = []
    data = []
    lab = []
    fulld = []
    for i in fulldata:
        fulld.append(fulldata[i][1])
    for i in sampledata:
        ids.append(i)
        data.append(sampledata[i][1])
        lab.append(sampledata[i][0])
    countvec = CountVectorizer(analyzer='word',stop_words='english')
    #countvec = TfidfVectorizer(analyzer='word',stop_words='english')
    countvec.fit(fulld)
    ldavec = countvec.transform(fulld)
    samplevec = countvec.transform(data)

    labmap = {}
    size = 0
    for i in range(len(lab)):
        if not lab[i] in labmap:
            labmap[lab[i]] = size
            lab[i] = size
            size += 1
        else:
            lab[i] = labmap[lab[i]]
    return ids,ldavec,samplevec,lab,size

ids,fullvec,samplevec,lab,size = main()
print(shape(fullvec))
print("TransFinished")


def TRY(K,T):

    ldamod = LDA(fullvec, T)
    vec = ldamod.transform(samplevec)
    print("LDAFinished")
    print(shape(vec))
    print(shape(lab))

    print("GaussianFinished")

    gaussianmodel = GaussianMixture(n_components=K, covariance_type='full')
    gaussianmodel.fit(vec)
    predict_lab = gaussianmodel.predict_proba(vec)
    print(soft_clustering_measure.v_measure(predict_lab, lab, K, size))
    return gaussianmodel
# CountVec
# TRY(10,5)     #(5.983655887742292, 1.7857961203556776, 2.750667475321601)
# TRY(10,10)    #(5.793340041551218, 1.6995679196273088, 2.628131810273599)
# TRY(10,20)    #(5.748902452262401, 1.8232317539059497, 2.768461629360497)
# TRY(20,5)     #(5.235139446837407, 2.2455859124731052, 3.1429987940992543)
# TRY(20,10)    #(5.1348565424758,   2.2731852401408212, 3.151299748332046)
# TRY(20,20)    #(5.088095772420887, 2.295808719881115,  3.1639885521521296)
# TRY(20,50)    #(5.068137170508772, 2.401508904759608,  3.258836208001375)
# TRY(40,5)     #(4.1690248312629645, 2.4604240380841094, 3.094546503748569)
# TRY(5,5)      (6.575484133870569, 1.2729455906331604, 2.132970244571595)

# TFIDF
# TRY(5,5)      #(5.840987150986896,  0.952019081632131, 1.6371930285021086)
# TRY(10,5)     #(5.1030655727101175, 1.174788697550675, 1.909895801844016)
# TRY(10,10)    #(6.165744462159868,  2.224094603521796, 3.269001676343832)
# TRY(20,5)     #(4.166416004327891,  1.732128450963391, 2.446965604598185)
# TRY(20,10)    #(4.580493301284961,  2.444482138265016, 3.1877503788542567)
# TRY(20,20)    #(5.324556429726739,  2.872336891888586, 3.731638138004408)
# TRY(20,50)    #(5.242555823754616,  2.784430936089831, 3.637114413310217)
# TRY(5,20)     #(6.591393434966861,  1.5248595217296592,2.4767461276531044)
# TRY(5,40)     #(6.573369275511941,  1.5122864462036505,2.4588772026368204)
# TRY(40,40)    #(4.561416153355922,  3.43262711889801,  3.917327003453732)

# TRY(30,5)     #(4.484020987850006,  2.4179103279908167, 3.1417179225085636)
# TRY(40,5)     #(3.8035354685609546, 2.5960676483465877, 3.0858899212616686)
# TRY(50,5)     #(3.881707188951549,  2.8866244180995704, 3.3110170736513544)
# TRY(80,5)     #(3.78764968210173,   3.4538527761824662, 3.613058048407635)

# I pick TFIDF k=5,t=5 as result:
res = TRY(5,5).means_
# if T>5
# sort each res[i] with index and reverse
# out put res[i] top5
print(res)
# [[0.15488484 0.13454477 0.17230884 0.45012352 0.08813803]
#  [0.03252341 0.03252441 0.03253928 0.7453911  0.15702181]
#  [0.0322308  0.03223902 0.13425605 0.76904089 0.03223324]
#  [0.16980546 0.03463754 0.03463374 0.72630095 0.0346223 ]
#  [0.04080745 0.17073193 0.04082432 0.7068209  0.04081541]]