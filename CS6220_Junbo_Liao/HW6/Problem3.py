from sets import Set
import igraph as ig
from numpy import *
import matplotlib.pyplot as plt

def cal_Q(partition, G):
    els = g.get_edgelist()
    m = len(G.get_edgelist())
    a = {}
    e = {}
    for community in partition:
        t = 0.0
        for node in partition[community]:
            t += len([x for x in G.neighbors(vertex = int(node))])
        a[community] =t / (2*m)
    for community in partition:
        t = 0.0
        for i in range(len(partition[community])):
            for j in range(len(partition[community])):
                if (int(partition[community][i]), int(partition[community][j])) in els:
                    t += 1.0
        e[community] = t / m
    q = 0.0
    for community in partition:
        q += (e[community] - pow(a[community],2))
    return q

def getpartition():
    res = {}
    p = open("Flickr_sampled_edges/community_membership_2K.csv")
    for line in p:
        c = line.split(",")
        if c[1] in res:
            res[c[1]].append(c[0])
        else:
            res[c[1]] = []
            res[c[1]].append(c[0])
    return res

def get_partition(G):
    partition = G.clusters()
    s = str(partition).replace("\n", "").replace(" ", "").split("[")
    m = {}
    for i in range(1, len(s)):
        temp = s[i].split("]")
        id = temp[0]
        context = temp[1].split(",")
        m[id] = context
    return m


path ="Flickr_sampled_edges/edges_sampled_map_2K.csv"
f = open(path)
g = ig.Graph()
edgelist = []
ver = []

for line in f:
    edge = line.split(",")
    e0 = int(edge[0])
    e1 = int(edge[1])
    ver.append(e0)
    ver.append(e1)
    edgelist.append((e0,e1))
size = max(ver)+1
g.add_vertices(size)
g.add_edges(edgelist)
ittimes = len(g.get_edgelist())
betw = g.edge_betweenness()
lenbef = len(g.clusters())
# print(g.components(mode='WEAK'))
for i in range(ittimes):
    betw = g.edge_betweenness()
    id = betw.index(max(betw))
    g.delete_edges(id)
    lennow = len(g.clusters())
    if lennow > lenbef and len(g.get_edgelist()) != 0:
        lenbef = lennow
        partition = get_partition(g)
        print(len(partition))
        # partition = getpartition()
        modi = cal_Q(partition,g)
        print(i,modi)
        plt.plot(i,modi,'r+')
plt.show



