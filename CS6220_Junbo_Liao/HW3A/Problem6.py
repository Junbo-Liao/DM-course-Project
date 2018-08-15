import struct
from numpy import *
import matplotlib.pyplot as plt
from sklearn.tree import tree


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

def switch(ele):
    res = zeros((28,28))
    for i in range(len(ele)):
        x = i%28
        y = 27-(i/28)
        res[x,y] = ele[i]
    return res

def generRandom():
    res = zeros((100,4))
    i = 0
    while i<100:
        x = random.randint(27)
        y = random.randint(27)
        wide1 = random.randint(1,28-x)
        wide2 = random.randint(1,28-y)
        area = wide1*wide2
        if 130<=area and area<=170:
            res[i][0] = x;
            res[i][1] = y;
            res[i][2] = wide1;
            res[i][3] = wide2;
            i += 1
    return res

def getAc(res,lab):
    length = shape(res)[0]
    acc = 0.0
    for i in range(length):
        if res[i] == lab[i]:
            acc += 1.0
    return acc/length

def GraphdecisionTree(trainvec,trainlab,testvec,testlab):
    mode = tree.DecisionTreeClassifier(criterion='gini')
    mode.fit(trainvec, trainlab)
    res = mode.predict(testvec)
    print("Accuracy: " + str(getAc(res, testlab)))

def generNewFeatures(oldvec,rec_100):
    newvec = zeros((shape(oldvec)[0],200))
    for i in range(shape(oldvec)[0]):
        print(i)
        oldpic = switch(oldvec[i])
        for j in range(100):
            x = int(rec_100[j][0])
            y = int(rec_100[j][1])
            wide1 = int(rec_100[j][2])
            wide2 = int(rec_100[j][3])
            v_diff = getblack(oldpic,x,y,wide1,wide2/2) - getblack(oldpic,x,y+wide2/2+1,wide1,wide2/2)
            h_diff = getblack(oldpic,x,y,wide1/2,wide2) - getblack(oldpic,x+wide1/2+1,y,wide1/2,wide2)
            newvec[i,2*j] = v_diff
            newvec[i,2*j+1] = h_diff

    return newvec

def getblack(vec,x,y,wide1,wide2):
    res = 0;
    for i in range(x,x+wide1):
        for j in range(y,y+wide2):
            res += vec[i,j]
            #if vec[i,j] == 0:
            #    res += 1
    return res

def main():
    train_vec,train_lab,test_vec,test_lab = loadMINST()
    rec_100 = generRandom()
    newtrain_vec = generNewFeatures(train_vec,rec_100)
    #print(shape(newtrain_vec))
    newtest_vec = generNewFeatures(test_vec,rec_100)
    print(newtest_vec)
    GraphdecisionTree(newtrain_vec,train_lab,newtest_vec,test_lab)

main()