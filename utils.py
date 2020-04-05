# encoding: utf8
import  pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

def f(x):
    return int(x[0]), int(x[1]), float(x[2])

class Graph(defaultdict):
    """Reference: https://github.com/phanein/deepwalk"""
    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None, type=None):
        if type:
            degrees = []
            for n in nodes:
                if self.has_key(n):
                    degrees.append(len(self[n]))
                else:
                    degrees.append(0)
            return degrees
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph"
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def check_edges(self):
        for x in self.keys():
            self[x] = list(set(self[x]))

    # sample K negative nodes for a given node
    def negative_sampling(self, K):
        return [self.index2node[self.node_sampling.sampling()] for _ in range(K)]

    def init_neg(self):
        # init negative_table
        self.node_negative_distribution = np.power(
            np.array([self.degree(node) for node in self.nodes()], dtype=np.float32), 0.75)
        self.node_negative_distribution /= sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        # map node to id
        self.node2index = dict([(node,i)  for i, node in enumerate(self.nodes())])
        self.index2node = dict([(i, node) for i, node in enumerate(self.nodes())])



class AliasSampling:
    """Reference: https://en.wikipedia.org/wiki/Alias_method"""

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res



def gen_batches(layers, batch_size, K=1):
    """
    :param K: number of negative sampling
    """
    node_samples = []
    for layid in layers:
        g = layers[layid]  # one graph
        edges = g.edges
        for a, b, w in edges:
            u_i = [a] * (K + 1)
            u_j = [b] + g.negative_sampling(K)
            label = [1] + [-1] * K
            this_layer = [layid] * (K + 1)

            node_samples.append([u_i, u_j, label, this_layer])

    # extend
    random.shuffle(node_samples)
    num_left = len(node_samples) % batch_size
    node_samples += node_samples[:num_left]

    # split samples into batches
    samples = []
    for i in range(len(node_samples) / batch_size):
        batch = node_samples[i * batch_size: (i + 1) * batch_size]
        u_i, u_j, label, this_layer = zip(*batch)

        u_i = np.concatenate(u_i)
        u_j = np.concatenate(u_j)

        label = np.concatenate(label)
        this_layer = np.concatenate(this_layer)
        samples.append([u_i, u_j, label, this_layer])

    random.shuffle(samples)

    return samples


def load_edgelist(edgelist, undirected=True):
    """Reference: https://github.com/phanein/deepwalk"""
    G = Graph()
    edges = []
    for e in edgelist:
        a, b, w = e
        a = int(a)
        b = int(b)
        G[a].append(b)
        edges.append([a, b, w])
        if undirected:
            G[b].append(a)
            edges.append([b, a, w])
    G.edges = edges
    G.check_edges()
    return G

def readfile(graph_path=None):
    # load network structures
    graphs = pd.read_csv(graph_path, sep=' ', header=None)
    graphs.columns = ["layerID", "n1", "n2", "weight"]
    layers = {}
    node_count = Counter()
    for layerID, graph in graphs.groupby(['layerID']):
        edges = graph[['n1', 'n2', 'weight']].values
        edges = map(f, edges)
        for a, b, _ in edges:
            node_count.update([a, b])

    # map node to id
    node2id = dict([ [n, id] for id,n in enumerate(node_count.keys())])
    id2node = dict([ [id, n] for id,n in enumerate(node_count.keys())])

    for layerID, graph in graphs.groupby(['layerID']):
        edges = graph[['n1', 'n2', 'weight']].values
        edges = map(f, edges)
        new_edges = []
        for a, b, w in edges:
            a = node2id[a]
            b = node2id[b]
            new_edges.append([a, b, w])
        layers[layerID] = load_edgelist(new_edges)

    num_nodes = max((id2node.keys()))+1
    return layers, num_nodes, id2node

def get_alignment_emb(inter_vectors, layers, id2node):
    # generate node2vec for each layer
    node2vec = {}
    for layerid in layers:
        tmp = {}
        for node_id in layers[layerid].keys():
            tmp[id2node[node_id]] = inter_vectors[node_id]
        node2vec[layerid] = tmp

    return node2vec


def get_intra_emb(inter_vectors, W, layers_embedding, layers, id2node):
    # generate node2vec for each layer
    node2vec = {}
    for layerid in layers:
        tmp = {}
        this_layer_embedding = np.dot(inter_vectors, W) + layers_embedding[layerid]
        for node in layers[layerid].keys():
            tmp[id2node[node]] = this_layer_embedding[node]
        node2vec[layerid] = tmp

    return node2vec
