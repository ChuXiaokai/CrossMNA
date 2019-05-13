# encoding: utf8
import pickle
import time
from utils import *


import heapq
def ALIGN(record, K=[]):
    record = heapq.nlargest(100, record, key=lambda x: x[2])
    ret = []
    flag = 0
    for k in K:
        if flag == 1:
            ret.append(1)
            continue
        for tmp in record[:k]:
            if tmp[0] == tmp[1]:
                flag = 1
        ret.append(flag)
    return ret


def eval_emb(network_path, embeddings):
    networks = pickle.load(open(network_path, 'rb'))

    layerids = embeddings.keys()

    # for accelerate
    need_compare = dict([ [i,{}]  for i in layerids])
    cur_layer_nodes = dict([ [i,{}] for i in layerids])

    a1 = time.clock()
    for i in layerids:
        need_compare_node = sorted(networks[i]['newid2node'].keys())

        # the comparing nodes in layer_i and their embedding
        true_corresponding_node = [networks[i]['newid2node'][nodeid] for nodeid in need_compare_node] 
        corresponding_embeddings = [embeddings[i][nodeid]   for nodeid in need_compare_node]
        corresponding_embeddings = np.array(corresponding_embeddings)
        need_compare[i]['true_node'] = true_corresponding_node
        need_compare[i]['corresponding_emb'] = corresponding_embeddings

        # all the nodes' embedding in layer_i
        total_true_node = []
        for nodeid in sorted(embeddings[i].keys()):
            if nodeid in need_compare_node:
                total_true_node.append(networks[i]['newid2node'][nodeid])
            else:
                total_true_node.append(nodeid)

        cur_layer_nodes[i]['total_true_node'] = total_true_node

    a2 = time.clock()
    print a2-a1

    precision_in_each_layer = []
    for i in layerids:
        for j in layerids:
            if i != j: 
                i_need_compare_emb = need_compare[i]['corresponding_emb']
                j_total_node = cur_layer_nodes[j]['total_true_node']
                nodes_in_layerj = set(list(embeddings[j].keys()))

                true_node_in_j = []
                for nodeid in nodes_in_layerj:
                    if nodeid in networks[j]['newid2node'].keys(): 
                        true_node_in_j.append(networks[j]['newid2node'][nodeid])
                    else:
                        true_node_in_j.append(nodeid)
                j_embed = np.array([embeddings[j][nodeid] for nodeid in nodes_in_layerj]) 

                score_between_two_layers = []

                # X
                X = []
                True_nodes = []
                for index, emb in enumerate(i_need_compare_emb):
                    if not need_compare[i]['true_node'][index] in j_total_node:
                        continue
                    X.append(emb)
                    True_nodes.append(need_compare[i]['true_node'][index])

                if X != []:
                    Answers = np.dot(X, j_embed.T)

                # find alignment for each node
                for index in range(len(X)):
                    true_node_i = True_nodes[index]
                    answer = Answers[index]
                    record = np.array([[true_node_i for _ in range(len(nodes_in_layerj))], true_node_in_j, answer]).T
                    precision_i_j = list(ALIGN(record, K=[1, 5, 10, 30, 50, 100]))
                    score_between_two_layers.append(precision_i_j)

                if len(score_between_two_layers) != 0:
                    precision_in_each_layer.append(np.mean(np.array(score_between_two_layers), axis=0))
                    print('There are {0} unknown anchor nodes from G{1} to G{2}; P_{1}_{2}: {3}'
                          .format(len(score_between_two_layers), i, j, np.mean(np.array(score_between_two_layers), axis=0)))

    print('Average Precision: {}'.format(np.mean(np.array(precision_in_each_layer), axis=0)))
    return np.mean(np.array(precision_in_each_layer), axis=0)

if __name__ == '__main__':
    start_time = time.clock()

    # load embeding vectors
    inter_vectors = pickle.load(open('emb/node2vec.pk', 'rb'))
    layers, num_nodes, id2node = readfile(graph_path= 'node_matching/Twitter/new_network0.5.txt')
    node2vec = get_alignment_emb(inter_vectors, layers, id2node)

    eval_emb('node_matching/Twitter/networks0.5.pk', node2vec)
    end_time = time.clock()
    print('time for alignment {0} s'.format(end_time - start_time))