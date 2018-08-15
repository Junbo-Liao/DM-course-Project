import heapq
import struct

from numpy import *
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree

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

def load20NG():
    newsgroups = fetch_20newsgroups(subset='train')
    tfidf_filter_vec1 = TfidfVectorizer(analyzer='word', stop_words='english')
    vec = tfidf_filter_vec1.fit_transform(newsgroups.data)
    lab = newsgroups.target
    featuresname = tfidf_filter_vec1.get_feature_names()
    classname = newsgroups.target_names
    return vec,lab,featuresname,classname

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

def getAc(res,lab):
    length = shape(res)[0]
    acc = 0.0
    for i in range(length):
        if res[i] == lab[i]:
            acc += 1.0
    return acc/length

def MailL2Reg(vec,lab,featuresname,classname):
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(vec, lab)
    weight1 = classifier.coef_
    res = classifier.predict(vec)
    print("Accuracy: " + str(getAc(res, lab)))
    for i in range(shape(weight1)[0]):
        print(classname[i]+":")
        top30f = heapq.nlargest(30,xrange(shape(weight1)[1]),weight1[i].__getitem__)
        for i in range(30):
            print featuresname[top30f[i]],
        print("")

def MaildecisionTree(vec,lab,featuresname):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(vec, lab)
    DecTree = mode.feature_importances_
    res = mode.predict(vec)
    print("Accuracy: " + str(getAc(res,lab)))
    top30f = heapq.nlargest(30,xrange(len(DecTree)),DecTree.__getitem__)
    for i in range(30):
        print featuresname[top30f[i]],
    print("")

def GraphL2Reg(trainvec, trainlab,testvec,testlab):
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(trainvec, trainlab)
    print("CLASSIFIER CREATED")
    weight1 = classifier.coef_
    res = classifier.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))
    for i in range(shape(weight1)[0]):
        print("Classes"+str(i)+":")
        top30f = heapq.nlargest(30,xrange(shape(weight1)[1]),weight1[i].__getitem__)
        print(top30f)
        #showTop30(top30f,i)

def GraphdecisionTree(trainvec, trainlab,testvec,testlab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(trainvec, trainlab)
    DecTree = mode.feature_importances_
    res = mode.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))
    top30f = heapq.nlargest(30, xrange(len(DecTree)), DecTree.__getitem__)
    #print(top30f)
    showTop30(top30f,1)

def showTop30(top30,id):
    plt.figure(id)
    for i in range(len(top30)):
        x = top30[i]%28
        y = 28-(top30[i]/28)
        plt.plot(x, y, 'bo')
    plt.show()

def loadSPAM():
    vec = []
    lab = []
    file = open("spambase.data")
    while 1:
        features = []
        line = file.readline()
        if not line:
            break
        s = line.replace(' ','').split(",")
        for i in range(len(s)):
            if len(s[i]) != 0 :
                v = float(s[i])
                if i == (len(s)-1):
                    v = int(v)
                    lab.append(v)
                else:
                    features.append(v)
        vec.append(features)
    featuresname = ["make","address","all","3d","our","over","remove","internet",
                    "order","mail","receive","will","people","report","addresses",
                    "free","business","email","you","credit","your","font","000",
                    "money","hp","hpl","george","650","lab","labs","telnet","857",
                    "data","415","85","technology","1999","parts","pm","direct",
                    "cs","meeting","original","project","re","edu","table",
                    "conference",";","(","[","!","$","#","capital_run_length_average",
                    "capital_run_length_longest","capital_run_length_total"]
    classname = ["NON-SPAM","SPAM"]
    return vec,lab,featuresname,classname

def main():
    vec,lab,featuresname,classname= loadSPAM()
    #vec,lab,featuresname,classname = load20NG()
    #trainvec, trainlab,testvec,testlab = loadMINST()
    MaildecisionTree(vec,lab,featuresname)
    MailL2Reg(vec, lab, featuresname, classname)
    #GraphL2Reg(trainvec,trainlab,testvec,testlab)
    #GraphdecisionTree(trainvec, trainlab, testvec, testlab)


main()