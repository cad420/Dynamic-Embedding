from multiprocessing import Pool
from module import negative_sampling
import random
import numpy as np
def sampling(thre_ID, hp, G, unigram, ix):
    node_set = list(G.nodes())
    edge_set = list(G.edges())
    node_num = len(node_set)
    edge_num = len(edge_set)
    black = [0 for i in range(hp.node_num)]
    connection_0_set = []
    connection_1_set = []
    unconnection_0_set = []
    unconnection_1_set = []
    subgraph_set = []
    neighbor_set = []
    negative_set = []
    sub_target = hp.batch_size//2
    inx = (ix * hp.batch_size) % node_num
    imx = (ix * hp.batch_size_con) % edge_num
    i = 0
    # connection sampling
    while len(connection_0_set) < hp.batch_size_con:
        # print((imx + i) % edge_num)
        edge = edge_set[(imx + i) % edge_num]
        connection_0_set.append(edge[0])
        connection_1_set.append(edge[1])
        i += 1
    # unconnection sampling
    a = [i for i in range(hp.node_num)]
    random.shuffle(a)
    b = [i for i in range(hp.node_num)]
    random.shuffle(b)
    # print(len(label))
    for i in a:
        if len(unconnection_0_set) >= hp.batch_size_con:
            break
        for j in b:
            if len(unconnection_0_set) >= hp.batch_size_con:
                break
            if i == j:
                continue
            # if 0!=1:
            if not G.has_edge(i, j):
                unconnection_0_set.append(i)
                unconnection_1_set.append(j)
    # subgraph sampling
    # each node with clock-wise
    i = 0
    while len(subgraph_set) < sub_target:
        idx = node_set[(inx+i)%node_num]
        if black[idx]:
            i += 1
            continue
        seta = 2 * hp.k
        tem_vis = [0 for j in range(hp.node_num)]
        tem_node_set = set()
        sub_node_set = []
        tem_node_set.add(idx)
        tem_vis[idx] = 1
        while len(sub_node_set) < hp.k:
            choose_node = random.sample(tem_node_set, 1)
            tem_node_set.remove(choose_node[0])
            sub_node_set.append(choose_node[0])
            if len(tem_node_set) < seta:
                for j in G.neighbors(choose_node[0]):
                    if not tem_vis[j]:
                        tem_vis[j] = 1
                        tem_node_set.add(j)
            if (len(tem_node_set) <= 0):
                break
        if len(sub_node_set) == hp.k:
            subgraph_set.append(sub_node_set)

            tem_nei_set = []
            for i in range(hp.k):
                tem_nei_set.append(sub_node_set[:i] + sub_node_set[i + 1:])
            neighbor_set.append(tem_nei_set)

            tem_neg_set = []
            for i in range(hp.k):
                neg_set = negative_sampling(unigram, hp.ns, sub_node_set)
                tem_neg_set.append(neg_set)
            negative_set.append(tem_neg_set)
        else:
            black[idx] = 1
        i += 1
        # print("\r游走第%d个图中（指定）  %.4f" % (thre_ID, i / hp.batch_size), end=" ")

    # random nodes with unigram distribution
    while len(subgraph_set) < hp.batch_size:
        neg_it = negative_sampling(unigram, 1, [-1])[0]
        idx = node_set[neg_it]
        if black[idx]:
            continue
        seta = 5 * hp.k
        tem_vis = [0 for j in range(hp.node_num)]
        tem_node_set = set()
        sub_node_set = []
        tem_node_set.add(idx)
        tem_vis[idx] = 1
        while len(sub_node_set) < hp.k:
            choose_node = random.sample(tem_node_set, 1)
            tem_node_set.remove(choose_node[0])
            sub_node_set.append(choose_node[0])
            if len(tem_node_set) < seta:
                for j in G.neighbors(choose_node[0]):
                    if not tem_vis[j]:
                        tem_vis[j] = 1
                        tem_node_set.add(j)
            if (len(tem_node_set) <= 0):
                break
        if len(sub_node_set) == hp.k:
            subgraph_set.append(sub_node_set)

            tem_nei_set = []
            for i in range(hp.k):
                tem_nei_set.append(sub_node_set[:i]+sub_node_set[i+1:])
            neighbor_set.append(tem_nei_set)

            tem_neg_set = []
            for i in range(hp.k):
                neg_set = negative_sampling(unigram, hp.ns, sub_node_set)
                tem_neg_set.append(neg_set)
            negative_set.append(tem_neg_set)
        else:
            black[idx] = 1
        # print("\r游走%d个图中（随机）  %.4f" % (thre_ID, i / hp.batch_size), end=" ")
    return [[np.array(connection_0_set).reshape(hp.batch_size_con, 1), np.array(connection_1_set).reshape(hp.batch_size_con, 1)], [np.array(unconnection_0_set).reshape(hp.batch_size_con, 1), np.array(unconnection_1_set).reshape(hp.batch_size_con, 1)],
            np.array(subgraph_set), np.array(neighbor_set), np.array(negative_set)]

def train_data(hp, G_list, unigram, idx):
    results = []
    pool = Pool(processes=hp.T)
    for t in range(hp.T):
        results.append(
            pool.apply_async(sampling, (t, hp, G_list[t], unigram[t], idx)))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    connection_set = [[results[i][0][0] for i in range(hp.T)], [results[i][0][1] for i in range(hp.T)]]
    unconnection_set = [[results[i][1][0] for i in range(hp.T)], [results[i][1][1] for i in range(hp.T)]]
    subgraph_set = [results[i][2] for i in range(hp.T)]
    neighbor_set = [results[i][3] for i in range(hp.T)]
    negative_set = [results[i][4] for i in range(hp.T)]
    # print(subgraph_set[0].shape)
    return connection_set, unconnection_set, subgraph_set, neighbor_set, negative_set

def eval_data(hp, G, G_0):
    data = np.zeros((2, hp.eval_size, 2)).astype('int32')
    edges = list(G.edges())
    for i in range(hp.eval_size):
        data[0][i][0] = edges[i][0]
        data[0][i][1] = edges[i][1]
    cou = 0
    a = list(G_0.nodes())
    random.shuffle(a)
    b = list(G_0.nodes())
    random.shuffle(b)
    # print(len(label))
    for i in a:
        if cou >= hp.eval_size:
            break
        for j in b:
            if cou >= hp.eval_size:
                break
            if i == j:
                continue
            # if 0!=1:
            if not G_0.has_edge(i, j):
                data[1][cou][0] = i
                data[1][cou][1] = j
                cou += 1
    return data