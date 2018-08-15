from random import *
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D


def conti_sampling(min, max, sample_size):
    res = []
    accept = 0
    while accept<sample_size:
        x = uniform(min,max)
        y = uniform(0,1)
        if y<0.5:
            accept+=1
            res.append(x)
    for i in range(len(res)):
        plt.plot(res[i],1,'ro')
    plt.show()
    return res

def gauss(mu,sigma,x):
    pdf = numpy.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * numpy.sqrt(2 * numpy.pi))
    return pdf

def gauss_sampling(mu, sigma, sample_size):
    xmin = mu-(5*sigma)
    xmax = mu+(5*sigma)
    ymax = gauss(mu,sigma,mu)
    xres = []
    yres = []
    accept = 0
    while accept < sample_size:
        x = uniform(xmin, xmax)
        y = uniform(0, 1)
        if y < gauss(mu,sigma,x)/ymax:
            accept += 1
            xres.append(x)
            yres.append(y)
    for i in range(len(xres)):
        plt.plot(xres[i], yres[i], 'ro')
    plt.show()
    return xres,yres

def d2gauss_sampling(mu, sigma, sample_size):
    xres = []
    yres = []
    zres = []
    xmu = mu[0]
    ymu = mu[1]
    xsigma = sigma[0]
    ysigma = sigma[1]
    xmin = xmu - (5 * xsigma)
    xmax = xmu + (5 * xsigma)
    ymin = ymu - (5 * ysigma)
    ymax = ymu + (5 * ysigma)
    zmax = gauss(xmu,xsigma,xmu)*gauss(ymu,ysigma,ymu)
    accept = 0
    while accept < sample_size:
        x = uniform(xmin, xmax)
        y = uniform(ymin, ymax)
        z = uniform(0,1)
        if z < (gauss(xmu,xsigma,x)*gauss(ymu,ysigma,y))/zmax:
            accept += 1
            xres.append(x)
            yres.append(y)
            zres.append(z)
    #ax = plt.subplot(111, projection='3d')
    for i in range(len(xres)):
        # if zres[i]<zmax/3:
        #     color = 'y'
        # elif zres[i]<2*zmax/3:
        #     color = 'r'
        # else:
        #     color = 'b'
        # ax.scatter(xres[i],yres[i],zres[i], c=color)
        plt.plot(xres[i],yres[i],'r+')

    plt.show()
    return xres, yres

def prob(x,sum):
    return 1.0/((x/50.0+1.0)*sum)

def Steven_sampling(N,M):
    sum = 0
    groupcount = 60
    for i in range(1,M+1):
        sum += 1.0 / (i / 50.0 + 1.0)
    groupprob = []
    probsum = 0.0
    for i in range(1,M+1):
        probsum += prob(i,sum)
        if i%groupcount==0:
            v = probsum
            groupprob.append(v)
            probsum = 0.0
    grouptime = []
    for i in range(len(groupprob)):
        grouptime.append(0)
    for i in range(N):
        pro = random()
        s = 0.0
        for j in range(len(groupprob)):
            s+=groupprob[j]
            if pro<s:
                grouptime[j] += 1
                #print(str(pro) + " " + str(j))
                break
    #print(numpy.sum(groupprob))
    #print(groupprob)
    print(grouptime)
    res = []
    for i in range(len(grouptime)):
        temp = SelectK(grouptime[i],i)
        res.extend(temp)
    return res
    #print numpy.sum(group)
    # for i in range(len(groupprob)):
    #     pro = groupprob[i]
    #     groupprob[i] = N*pro
    # print(groupprob)
    # for i in range(len(groupprob)):
    #     plt.plot(i+1,groupprob[i],'b+')
    # plt.show()

    #print(numpy.sum(group))
# def Steven_sampling(N,M):
#     sum = 0
#     groupcount = 6
#     for i in range(M+1):
#         sum += 1.0 / (i / 50.0 + 1.0)
#     spindex = sample(groupcount,M)
#     groupprob = []
#     group = []
#     start = 0
#     end = 1
#     for i in range(groupcount):
#         s = spindex[start]
#         e = spindex[end]
#         start = end
#         end += 1
#         lenth = e-s
#         probsum = 0.0
#         for j in range(s,e):
#             probsum += prob(j,sum)
#         groupprob.append(probsum)
#         v = probsum/lenth
#         for k in range(s,e):
#             group.append(v)
#     #print numpy.sum(group)
#     for i in range(len(groupprob)):
#         pro = groupprob[i]
#         groupprob[i] = N*pro
#     print(groupprob)
#     # for i in range(len(group)):
#     #     plt.plot(i,group[i],'bo')
#     # plt.show()
#
#     #print(numpy.sum(group))
# def sample(N,M):
#     delta = 5
#     res = []
#     res.append(0)
#     bef = 0
#     for i in range(N-1):
#         bef += M/N
#         s = randint(bef - delta, bef + delta)
#         res.append(s)
#     res.append(M+1)
#     return res

def SelectK(k,i):
    res = []
    start = i*50+1
    if k == 0:
        return res
    initial = []
    for i in range(50):
        initial.append(start+i)
    for i in range(k):
        s = randint(i,50)
        res.append(initial[s])
        initial[s] = initial[i]
        initial[i] = res[i]
    return res

# conti_sampling(2,5,2000)
# gauss_sampling(0,1,2000)
# mu = [0,0]
# sigma = [1,1]
# d2gauss_sampling(mu,sigma,2000)
s = Steven_sampling(20,300)
print(s)
print(len(s))
