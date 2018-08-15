from numpy import *
import struct
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

def MailL2Reg(vec,lab):
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(vec, lab)
    weight1 = classifier.coef_
    res = classifier.predict(vec)
    print("Accuracy: " + str(getAc(res, lab)))

def MaildecisionTree(vec,lab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(vec, lab)
    DecTree = mode.feature_importances_
    res = mode.predict(vec)
    print("Accuracy: " + str(getAc(res,lab)))

def GraphL2Reg(trainvec, trainlab,testvec,testlab):
    classifier = LogisticRegression(penalty='l2')
    classifier.fit(trainvec, trainlab)
    weight1 = classifier.coef_
    res = classifier.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))

def GraphdecisionTree(trainvec, trainlab,testvec,testlab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(trainvec, trainlab)
    DecTree = mode.feature_importances_
    res = mode.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))

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
    trainvec, trainlab, testvec, testlab = loadMINST()
    vec, lab, featuresname, classname = loadSPAM()
    D5 = PCA(n_components=5)
    D20 = PCA(n_components=20)
    D5.fit(trainvec)
    D20.fit(trainvec)
    D5.fit(vec)
    D20.fit(vec)
    # D5_train = D5.transform(trainvec)
    # D20_train = D20.transform(trainvec)
    # D5_test = D5.transform(testvec)
    # D20_test = D20.transform(testvec)
    D5_vec = D5.transform(vec)
    D20_vec = D20.transform(vec)
    #MaildecisionTree(vec, lab, featuresname)
    #MailL2Reg(vec, lab, featuresname, classname)
    print("D5")
    MailL2Reg(D5_vec,lab)
    MaildecisionTree(D5_vec, lab)
    print("D20")
    MailL2Reg(D20_vec, lab)
    MaildecisionTree(D20_vec, lab)
    # print("D5")
    # GraphL2Reg(D5_train,trainlab,D5_test,testlab)
    # GraphdecisionTree(D5_train, trainlab, D5_test, testlab)
    # print("D20")
    # GraphL2Reg(D20_train, trainlab, D20_test, testlab)
    # GraphdecisionTree(D20_train, trainlab, D20_test, testlab)

main()
