from multiprocessing import Pool
from module import negative_sampling
import random
import numpy as np
import networkx as nx
def sampling(thre_ID, hp, G, degree, ix):
    node_set = list(G.nodes())
    edge_set = list(G.edges())
    node_num = len(node_set)
    edge_num = len(edge_set)
    black = [0 for i in range(hp.node_num)]
    connection_0_set = []
    connection_1_set = []
    weight_connection = []
    unconnection_0_set = []
    unconnection_1_set = []
    subgraph_set = []
    neighbor_set = []
    weight_subgraph = []

    sub_target = hp.batch_size
    inx = (ix * hp.batch_size) % node_num
    imx = (ix * hp.batch_size_con) % edge_num
    i = 0
    # connection sampling
    while len(connection_0_set) < hp.batch_size_con:
        # print((imx + i) % edge_num)
        edge = edge_set[(imx + i) % edge_num]
        connection_0_set.append(edge[0])
        connection_1_set.append(edge[1])
        weight_connection.append(1)
        i += 1
    # unconnection sampling
    a = [i for i in range(hp.node_num)]
    random.shuffle(a)
    b = [i for i in range(hp.node_num)]
    random.shuffle(b)
    # print(len(label))
    neg_num = hp.ns*hp.batch_size
    for i in a:
        if len(unconnection_0_set) >= neg_num:
            break
        for j in b:
            if len(unconnection_0_set) >= neg_num:
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
        sub_node_set = []
        sub_node_set.append(idx)
        P = 1/hp.k
        while len(sub_node_set) < hp.k:
            choose_node = random.sample(sub_node_set, 1)
            neig = list(G.adj[choose_node[0]])
            wei_arr = np.zeros((len(neig)))
            all_wei = 0
            for i, node in enumerate(neig):
                all_wei += degree[node]
                wei_arr[i] = all_wei
            wei_arr /= all_wei
            tar = random.random()
            L, R = 0, len(wei_arr)
            while L < R:
                mid = (L+R)//2
                if wei_arr[mid] < tar:
                    L = mid + 1
                else:
                    R  = mid
            if L<0 or L>= len(neig):
                L = 0
            sub_node_set.append(neig[L])
            if G.has_edge(sub_node_set[0], sub_node_set[-1]):
                P *= (1/degree[sub_node_set[-1]])

        if len(sub_node_set) == hp.k:
            subgraph_set.append(sub_node_set[0])
            neighbor_set.append(sub_node_set[1:])
            weight_subgraph.append(P)

        else:
            black[idx] = 1
        i += 1
        # print("\r游走第%d个图中（指定）  %.4f" % (thre_ID, i / hp.batch_size), end=" ")

    return [[np.array(connection_0_set).reshape(hp.batch_size_con, 1), np.array(connection_1_set).reshape(hp.batch_size_con, 1), np.array(weight_connection).reshape(hp.batch_size_con, 1)],
            [np.array(unconnection_0_set).reshape(neg_num, 1), np.array(unconnection_1_set).reshape(neg_num, 1)],
            [np.array(subgraph_set).reshape(hp.batch_size, 1), np.array(neighbor_set), np.array(weight_subgraph).reshape(hp.batch_size, 1)]]

def train_data(hp, G_list, unigram, idx):
    results = []
    degree = [nx.degree(G) for G in G_list]
    pool = Pool(processes=hp.T)
    for t in range(hp.T):
        results.append(
            pool.apply_async(sampling, (t, hp, G_list[t], degree[t], idx)))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    connection_set = [[results[i][0][0] for i in range(hp.T)], [results[i][0][1] for i in range(hp.T)], [results[i][0][2] for i in range(hp.T)]]
    unconnection_set = [[results[i][1][0] for i in range(hp.T)], [results[i][1][1] for i in range(hp.T)]]
    subgraph_set = [[results[i][2][0] for i in range(hp.T)], [results[i][2][1] for i in range(hp.T)], [results[i][2][2] for i in range(hp.T)]]
    # print(subgraph_set[0].shape)
    return connection_set, unconnection_set, subgraph_set

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