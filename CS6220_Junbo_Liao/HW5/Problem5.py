from math import sqrt
from random import uniform

import matplotlib.pyplot as plt
import numpy

rho = 0.5

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
    return xres,yres

def ygivenx(x,m1,m2,sigma1,sigma2):
    m = m2+rho*sigma2/sigma1*(x-m1)
    s = sqrt(1-pow(rho,2))*sigma2
    l1,l2 = gauss_sampling(m,s,1)
    return l1[0]

def xgiveny(y,m1,m2,sigma1,sigma2):
    m = m1+rho*sigma1/sigma2*(y-m2)
    s = sqrt(1-pow(rho,2))*sigma1
    l1, l2 = gauss_sampling(m,s,1)
    return l1[0]


def gibbs_sampling(mu,sigma,N):
    K = 20
    xres = []
    yres = []
    m1 = mu[0]
    m2 = mu[1]
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    y = m2
    for i in range(N):
        print()
        for j in range(K):
            x = xgiveny(y,m1,m2,sigma1,sigma2)
            y = ygivenx(x,m1,m2,sigma1,sigma2)
            xres.append(x)
            yres.append(y)
    for i in range(len(xres)):
        plt.plot(xres[i],yres[i],'r+')
    plt.show()

mu = [10,-5]
sigma = [5,2]
gibbs_sampling(mu,sigma,100)


