from math import sqrt

import matplotlib.pyplot as plt
import numpy


def loadu1(path):
    f = open(path)
    currentid = '-1'
    trainset = {}
    for line in f:
        m = line.replace("\n","").split("\t")
        m0 = int(m[0])
        m1 = int(m[1])
        m2 = int(m[2])
        if currentid == '-1':
            currentid = m0
            trainset[currentid] = {}
        if currentid != m0:
            #trainset.append(user)
            currentid = m0
            trainset[currentid] = {}
            trainset[currentid][m1] = m2
        else:
            trainset[currentid][m1] = m2
    return trainset

trainset = loadu1("ml-100k/u1.base")

def similarity_score(person1, person2):
    both_viewed = {}
    for item in trainset[person1]:
        if item in trainset[person2]:
            both_viewed[item] = 1
        if len(both_viewed) == 0:
            return 0
        sum_of_eclidean_distance = []
        for item in trainset[person1]:
            if item in trainset[person2]:
                sum_of_eclidean_distance.append(pow(trainset[person1][item] - trainset[person2][item], 2))
        sum_of_eclidean_distance = float(sum(sum_of_eclidean_distance))
        return 1 / (1 + sqrt(sum_of_eclidean_distance))

def most_similar_users(person):
    scores = [(similarity_score(person, other_person), other_person) for other_person in trainset if
              other_person != person]
    scores.sort()
    scores.reverse()
    # res = scores[0:10]
    # sum = 0
    # for i in range(len(res)):
    #     sum += res[i][0]
    # for i in range(len(res)):
    #     res[i] = (res[i][0]/sum,res[i][1])
    return scores

def predict(person,movie):
    scores = most_similar_users(person)
    count = 0
    res = []
    for i in range(len(scores)):
        if count == 10:
            break
        if movie in trainset[scores[i][1]]:
            count += 1
            res.append((scores[i][0],scores[i][1],trainset[scores[i][1]][movie]))
    sum = 0
    for i in range(len(res)):
        sum += res[i][0]
    rate = 0
    for i in range(len(res)):
        rate += res[i][2] * (res[i][0]/sum)
    return int(round(rate))
def RES(num):
    sum = 0.0
    f = open("ml-100k/u1.test")
    size = 0
    for line in f:
        if size<num:
            m = line.replace("\n", "").split("\t")
            m0 = int(m[0])
            m1 = int(m[1])
            m2 = int(m[2])
            sum += pow(predict(m0,m1) - m2,2)
            size += 1
    sum /= num
    return sum

for i in range(1,50):
    k = RES(i)
    plt.plot(i,k,'b+')
    print(i,k)
plt.show()

# for i in trainset.keys():
#     for j in trainset[i].keys():
#         print(str(i)+" "+str(j)+" "+str(trainset[i][j]))