# encoding: utf8
"""split dataset for network alignment"""
from collections import Counter
import random
import pickle
import pandas as pd
def f(x):
    return int(x[0]), int(x[1])

def readfile(graph_path=None, num_nodes=None):

    graphs = pd.read_csv(graph_path, sep=' ', header=None)
    graphs.columns = ["layerID", "n1", "n2", "weight"]
    layers = {}

    for layerID, graph in graphs.groupby(['layerID']):
        edges = graph[['n1', 'n2']].values
        edges = map(f, edges)
        layers[layerID] = edges
    return layers


def split_dataset(path, p, dir):
    """
    this function is used to split the data in to training set and test set for alignment
    we keep p% anchor links and generate a special format for our algorithm
    you can use re_generate() to transform this special format
    :param path:
    :param p: keep p% anchor links
    :param dir:
    :return: dump two files
    """
    networks = readfile(path)
    node_count = Counter()
    each_layer_nodes = {}
    for layerid in networks:
        this_layer_node_counter = Counter()
        edges = networks[layerid]
        for e in edges:
            this_layer_node_counter.update(e)
            node_count.update(e)
        each_layer_nodes[layerid] = this_layer_node_counter.keys()  # nodes in each layer
    next_id = max(node_count.keys())+1  # number of nodes in a multiplex network
    num_nodes = next_id-1

    # count how many anchor links between each two layer
    anchor_links = []  # [layer1, layer2, node_id]
    for i in sorted(networks.keys()):
        for j in sorted(networks.keys()):
            if i < j:
                anchor_node_i_j = list(set(each_layer_nodes[i])&set(each_layer_nodes[j]))  # this is an anchor link
                anchor_links += [[i, j, anchor_node_i_j[_]] for _ in range(len(anchor_node_i_j))]
    print('there are totally {0} anchor links; {1} nodes'.format(len(anchor_links), num_nodes))

    # keep p% anchor links
    random.shuffle(anchor_links)
    keep_anchor_links = anchor_links[:int(len(anchor_links)*p)]
    keep_anchor_nodes_in_each_layer = dict([[i, []] for i in networks.keys()])  # anchor nodes keep in each layer

    """
    对于任意一层的节点v，如果它在测试集中，那么它将与任何一层对应点的anchor link都将会被移除
    """
    for e in keep_anchor_links:
        layeri, layerj, n = e
        keep_anchor_nodes_in_each_layer[layeri].append(n)
        keep_anchor_nodes_in_each_layer[layerj].append(n)

    # generate the new network
    new_network = {}
    for layerid in networks:
        newid2node = {}
        node2newid = {}

        edges = networks[layerid]
        keep_nodes = keep_anchor_nodes_in_each_layer[layerid]  # the anchor nodes to keep

        # get the nodes in this layer
        this_layer_counter = Counter()
        for a, b in edges:
            this_layer_counter.update([a, b])
        this_layer_nodes = this_layer_counter.keys()

        """
        For fair comparison, 
        instead of removing anchor links from training set, we directly remove the anchor nodes.
        This will avoid data leakage caused by global consistent.
        """
        for n in this_layer_nodes:
            if not n in keep_nodes:
                newid2node[next_id] = n
                node2newid[n] = next_id
                next_id += 1

        # transform the edges
        for i in range(len(edges)):
            a, b = edges[i]
            if a in node2newid.keys():
                a = node2newid[a]
            if b in node2newid.keys():
                b = node2newid[b]
            edges[i] = [a,b]
        new_network[layerid] = {'edges':edges, 'newid2node':newid2node}

    # dump the new networks packages
    pickle.dump(new_network, open(dir+'networks'+str(p)+'.pk','wb'))

    # dump the edgelists
    fout = open(dir+'new_network'+str(p)+'.txt','wb')
    for layerid in new_network:
        edges = new_network[layerid]['edges']
        for e in edges:
            e = map(str,e)
            fout.writelines(str(layerid) + ' ' + ' '.join(e) +  ' 1\n')


def transfer(input_dir, output_dir, p):
    """
    this function transforms our special data format
    to the input format
    >> [train.txt, test.txt, groundtruth.txt]
    for method 'IONE': https://github.com/ColaLL/IONE
    :param input_dir: where 'network_p.pk' and 'multiplex.edges' locate
    :param output_dir: where to genrate the new data
    :param p:
    """
    import pickle
    p = str(p)
    pk_path = input_dir+'/networks'+p+'.pk'
    network_path = input_dir+'/multiplex.edges'
    pk_info = pickle.load(open(pk_path, 'rb'))
    buf = open(network_path,'rb').readlines()

    layers = {}
    from collections import Counter
    for line in buf:
        l, a, b, c = line.strip().split()
        l = int(l)
        if l in layers.keys():
            layers[l].update([a,b])
        else:
            layers[l] = Counter()
            layers[l].update([a, b])

    for i in layers.keys():
        for j in layers.keys():
            if i < j:
                fout = open(output_dir + '/training' + str(i) + '_' + str(j) + '_'+ p + '.txt', 'wb')

                # i-> j
                i_j_test = []
                newid2node = pk_info[i]['newid2node']
                for nodeid in newid2node:
                    node = str(newid2node[nodeid])
                    if node in layers[j].keys():
                        i_j_test.append(str(node))
                # j->i
                j_i_test = []
                newid2node = pk_info[j]['newid2node']
                for nodeid in newid2node:
                    node = str(newid2node[nodeid])
                    if node in layers[i].keys():
                        j_i_test.append(str(node))
                unknown_nodes = set(set(i_j_test) | set(j_i_test))

                fouti_j = open(output_dir + '/test' + str(i) + '_' + str(j) + '_' + p + '.txt', 'wb')
                foutj_i = open(output_dir + '/test' + str(j) + '_' + str(i) + '_' + p + '.txt', 'wb')
                for n in i_j_test:
                    fouti_j.writelines(str(n)+'\n')

                for n in j_i_test:
                    foutj_i.writelines(str(n)+'\n')

                training_nodes = list(set(layers[i].keys()) & set(layers[j].keys())-set(unknown_nodes))
                for n in training_nodes:
                    fout.writelines(str(n)+'\n')


    for i in layers.keys():
        for j in layers.keys():
            if i != j:
                fout1 = open(output_dir + '/groundtruth' + str(i) + '_' + str(j) + '_'+  p + '.txt', 'wb')
                common_nodes = list(set(layers[i].keys())& set(layers[j].keys()))
                for node in common_nodes:
                    fout1.writelines(node+'\n')



if __name__ == '__main__':
    dataset = 'Twitter'
    p = 0.5
    new_network = split_dataset(
        path= dataset + '/network.txt',
        p=p,
        dir=dataset + '/'
    )

    for p in [0.5]:
        print(p)
        transfer(
            input_dir=dataset,
            output_dir='tmp/' + dataset+'_new',
            p=p,
        )