from multiprocessing import Pool
from module import negative_sampling
import random
import numpy as np
def sampling(thre_ID, hp, G, unigram, inx):
    node_set = list(G.nodes())
    node_num = len(node_set)
    black = [0 for i in range(hp.node_num)]
    subgraph_set = []
    neighbor_set = []
    negative_set = []
    target = hp.batch_size//2
    # each node with clock-wise
    i = 0
    while len(subgraph_set) < target:
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
                neg_set = negative_sampling(unigram, hp.ns, sub_node_set[i])
                tem_neg_set.append(neg_set)
            negative_set.append(tem_neg_set)
        else:
            black[idx] = 1
        i += 1
        # print("\r游走第%d个图中（指定）  %.4f" % (thre_ID, i / hp.batch_size), end=" ")

    # random nodes with unigram distribution
    while len(subgraph_set) < hp.batch_size:
        neg_it = negative_sampling(unigram, 1, -1)[0]
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
                neg_set = negative_sampling(unigram, hp.ns, sub_node_set[i])
                tem_neg_set.append(neg_set)
            negative_set.append(tem_neg_set)
        else:
            black[idx] = 1
        # print("\r游走%d个图中（随机）  %.4f" % (thre_ID, i / hp.batch_size), end=" ")
    return [np.array(subgraph_set), np.array(neighbor_set), np.array(negative_set)]

def train_data(hp, G_list, unigram, idx):
    results = []
    pool = Pool(processes=hp.T)
    for t in range(hp.T):
        inx = idx % len(G_list[t].nodes())
        results.append(
            pool.apply_async(sampling, (t, hp, G_list[t], unigram[t], inx)))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    subgraph_set = [results[i][0] for i in range(hp.T)]
    neighbor_set = [results[i][1] for i in range(hp.T)]
    negative_set = [results[i][2] for i in range(hp.T)]
    # print(subgraph_set[0].shape)
    return subgraph_set, neighbor_set, negative_set

