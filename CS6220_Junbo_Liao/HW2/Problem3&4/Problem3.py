from time import time

import numpy
from scipy.stats import norm


def getData(filename):
    a = numpy.loadtxt(filename)
    return a


def prob(x, mean, cov):
    n = len(x)
    s=(x - mean) * (cov.I)
    sum = 0.0
    for i in range(n):
        sum += s[0,i]*((x - mean)[0,i])
    expOn = float(-0.5 * sum)
    divBy = pow(2 * numpy.pi, n / 2) * pow(numpy.linalg.det(cov), 0.5)
    return pow(numpy.e, expOn) / divBy

def emloop(data,r,mean,cov):
    n = len(r)
    dataNum,dim= data.shape
    gamaArray = numpy.mat(numpy.zeros((dataNum,n)))
    diff = 10.0
    itcount = 0
    while (diff>0.0001):
        itcount += 1
        for j in range(dataNum):
            for k in range(n):
                gamaArray[j, k] = r[k] * prob(data[j, :], mean[k], cov[k])
            sumAlphaMulP = numpy.sum(gamaArray[j])
            for k in range(n):
                gamaArray[j, k] /= sumAlphaMulP
        sumGamma = numpy.sum(gamaArray, axis=0)
        diff = 0.0
        for k in range(n):
            change1 = numpy.mat(mean[k])
            change2 = numpy.mat(cov[k])
            mean[k] = numpy.mat(numpy.zeros((1, dim)))
            cov[k] = numpy.mat(numpy.zeros((dim, dim)))
            for j in range(dataNum):
                mean[k] += gamaArray[j, k] * data[j, :]
            mean[k] /= sumGamma[0, k]
            for j in range(dataNum):
                cov[k] += gamaArray[j, k] * (data[j, :] - mean[k]).T * (data[j, :] - mean[k])
            cov[k] /= sumGamma[0, k]

            for i in range(dim):
                diff += abs(change1[0,i]-mean[k][0,i])
                for j in range(dim):
                    diff += abs(change2[i,j]-cov[k][i,j])
        #print(itcount)

    return [r, mean, cov, itcount]


def main2(filename):
    start = time()
    data = getData(filename)
    print(data.shape)
    #mean_1 [3,3]); cov_1 = [[1,0],[0,3]]; n1=2000 points
    #mean_2 =[7,4]; cov_2 = [[1,0.5],[0.5,1]]; ; n2=4000 points
    cov = [numpy.mat([[0.1, 0], [0, 0.1]]) for x in range(2)]
    mean = numpy.mat([[2.5,3.5],[6.5,4.5]])
    res = emloop(data,[0.33,0.67],mean,cov)
    end = time()
    print("Tol = 0.0001")
    print("Time Cost:")
    print(end - start)
    print("Cluster1 :")
    print("Mean :")
    print(res[1][0])
    print("Covariance :")
    print(res[2][0])
    print("Cluster2 :")
    print("Mean :")
    print(res[1][1])
    print("Covariance :")
    print(res[2][1])
    print("Iteration")
    print(res[3])
    
def main3(filename):
    start = time()
    data = getData(filename)
    print(data.shape)
    # mean_1 [3,3]); cov_1 = [[1,0],[0,3]]; n1=2000 points
    # mean_2 =[7,4]; cov_2 = [[1,0.5],[0.5,1]]; ; n2=4000 points
    cov = [numpy.mat([[0.1, 0], [0, 0.1]]) for x in range(3)]
    mean = numpy.mat([[2.5, 3.5], [6.5, 4.5],[6.0,6.0]])
    res = emloop(data, [0.2, 0.3,0.5], mean, cov)
    end = time()
    print("Tol = 0.0001")
    print("Time Cost:")
    print(end - start)
    print("Cluster1 :")
    print("Mean :")
    print(res[1][0])
    print("Covariance :")
    print(res[2][0])
    print("Cluster2 :")
    print("Mean :")
    print(res[1][1])
    print("Covariance :")
    print(res[2][1])
    print("Cluster3 :")
    print("Mean :")
    print(res[1][2])
    print("Covariance :")
    print(res[2][2])
    print("Iteration")
    print(res[3])

main2('2gaussian.txt')
main3('3gaussian.txt')






