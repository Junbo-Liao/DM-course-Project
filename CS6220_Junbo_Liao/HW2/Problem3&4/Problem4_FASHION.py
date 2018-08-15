import struct
from random import random
from time import time
from numpy.linalg import cholesky
import sys
import numpy
import scipy.stats


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
    imgs = numpy.reshape(imgs, [imgNum, width * height])
    return imgs, head

def loadlabels(filename):
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    labelNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(labelNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = numpy.reshape(labels, [labelNum])
    return labels, head

def pd(x, means, covs_k):
    #print(covs_k)
    #covs = numpy.array(numpy.mat(covs_k))
    #for i in range(covs.shape[0]):
        #covs[i,i] += 0.001
    norm = scipy.stats.multivariate_normal(mean=means,cov=covs_k)
    return norm.logpdf(x)

def log_sum_exp(ary,m):
    tmp = 0.0
    for i in range(len(ary)):
        tmp += pow(numpy.e,ary[i]-m)
    return (m + numpy.log(tmp))

def emloop(data, r, mean, cov):
    data = data/255.0
    #for i in range(data.shape[1]):
        #max_ = max(data[:,i])
        #min_ = min(data[:,i])
        #if max_ != min_:
        #    data[:,i] = (data[:,i]-min_)/(max_-min_)
        #elif max_ == 0:
        #    data[:,i] = numpy.zeros(data.shape[0])
        #else:
        #    data[:,i] = numpy.ones(data.shape[0])
    n = len(r)
    dataNum, dim = data.shape
    gamaArray = numpy.mat(numpy.zeros((dataNum, n)))
    prob = numpy.array(numpy.zeros((dataNum,n)))
    diff = 10.0
    itcount = 0
    while (diff > 0.0001):
        itcount += 1
        print(itcount)
        change = numpy.mat(numpy.array(gamaArray))
        #print(change)
        for k in range(n):
            tmp1 = pd(data, mean[k], cov[k])
            prob[:,k] = tmp1
        #print(prob)
        gamaArray = numpy.asarray(gamaArray)
        for k in range(n):
            tmp2 = numpy.log(r[k])
            for i in range(dataNum):
                gamaArray[i,k] = float(tmp2 + prob[i,k])
        for i in range(dataNum):
            tmp3 = float(log_sum_exp(gamaArray[i,:],max(gamaArray[i,:])))
            for j in range(n):
                tmp4 = gamaArray[i,j]-tmp3
                gamaArray[i,j] = tmp4
        #print (gamaArray)
        print("E Step")
        diff = numpy.linalg.norm(change - gamaArray)
        mean = numpy.zeros((n,dim))
        cov = []
        print(diff)
        for k in range(n):
            NK = float(log_sum_exp(gamaArray[:,k],max(gamaArray[:,k])))
            for j in range(dim):
                tmp5 = []
                for i in range(dataNum):
                    if data[i,j] != 0:
                        tmp5.append(gamaArray[i,k] + numpy.log(data[i,j]))
                    else:
                        tmp5.append(-sys.maxint-1)
                lse = log_sum_exp(tmp5,max(tmp5))
                mean[k,j] = pow(numpy.e,(lse - NK))
            cov_k = numpy.mat(numpy.zeros((dim, dim)))
            for x in range(dataNum):
                square = numpy.mat(data[x,:] - mean[k])
                #print(square.shape)
                newcov = square.T*square
                #print(newcov.shape)
                cov_k += (pow(numpy.e,gamaArray[x, k]))* newcov
            #print(cov_k.shape)
            cov_k = cov_k/(pow(numpy.e,NK))
            mu = numpy.trace(cov_k)/cov_k.shape[0]
            print(mu)
            tmp6 = numpy.identity(cov_k.shape[0])*(0.1 * mu)
            #print(tmp6.shape)
            covk = cov_k*0.9 + tmp6
            #print(numpy.linalg.norm(covk - cov_k))
            #for y in range(cov_k.shape[0]):
            #cov_k = cov_k + numpy.identity(cov_k.shape[0])*0.1
            cov.append(covk)
            r[k] = (pow(numpy.e,NK))/dataNum
            #print (r[k])
            #for j in range(data.shape[1]):
            #    cov[k, j, j] += 0.0001
        #cov = cholesky(numpy.array(cov))
        #covs_k = numpy.mat(covs_k)
        #covs = numpy.array(covs_k)
        # for y in range(covs.shape[0]):
        #    covs[y, y] += 0.1
        #covs = numpy.cov(covs_k) + 0.0001 * numpy.identity(covs_k.shape[1])
        #print(cov[0])
        print("M Step")

    return [r, mean, cov, itcount, gamaArray]


def main():
    start = time()
    file1 = 'FASHION/train-images-idx3-ubyte'
    file2 = 'FASHION/train-labels-idx1-ubyte'
    imgs, data_head = loadImageSet(file1)
    label, labels_head = loadlabels(file2)
    file3 = 'FASHION/t10k-images-idx3-ubyte'
    file4 = 'FASHION/t10k-labels-idx1-ubyte'
    testimgs, ti_head = loadImageSet(file3)
    testlabel, tl_head = loadlabels(file4)
    ds = []
    ds.extend(imgs)
    ds.extend(testimgs)
    lb = []
    lb.extend(label)
    lb.extend(testlabel)
    labels = numpy.asarray(lb)
    #labels = testlabel
    #labels = numpy.asarray([lb[i] for i in range(1000)])
    labset = set()
    numpy.set_printoptions(threshold='nan')
    for i in range(labels.shape[0]):
        labset.add(labels[i])
    s = len(labset)
    data = numpy.asarray(ds)
    #data = testimgs
    #data = numpy.asarray([ds[i] for i in range(1000)])
    print(data.shape)
    weight = []
    for i in range(s):
        weight.append(1.0/s)
    print(weight)
    mean = numpy.random.rand(s,data.shape[1])
    cov = numpy.array([numpy.ones((data.shape[1],data.shape[1]))]*s)
    for x in range(s):
        cov[x] = cov[x] + numpy.identity(data.shape[1])
    #cov = cholesky(cov)
    print(numpy.asarray(cov).shape)
    res = emloop(data, weight, mean, cov)
    end = time()
    print("Tol = 0.001")
    print("Time Cost:")
    print(end - start)
    print("Weight:")
    print(res[0])
    print("Mean:")
    print(res[1]*255.0)
    #print("Cov:")
    #print(res[2])
    print("Iteration")
    print(res[3])
    print("Likelihood")
    likelihood = numpy.zeros(res[4].shape)
    for i in range(data.shape[0]):
        for j in range(s):
            likelihood[i,j] = pow(numpy.e,res[4][i,j])
    print(likelihood)


main()






