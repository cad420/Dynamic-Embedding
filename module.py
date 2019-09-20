import tensorflow as tf
import numpy as np
import random
import glob
import networkx as nx

from multiprocessing import Pool

def get_initialization(hp):
    alpha = tf.get_variable('context_embedding',
                                 dtype=tf.float32,
                                 shape=(hp.node_num, hp.dim),
                                 initializer=tf.contrib.layers.xavier_initializer())
    y = []
    for t in range(hp.T):
        y.append(tf.get_variable('dynamic_embedding_'+str(t),
                                     dtype=tf.float32,
                                     shape=(hp.node_num, hp.dim),
                                     initializer=tf.contrib.layers.xavier_initializer()))
    return alpha, y

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def negative_sampling(unigram, ns, node):
    neg_set = []
    for i in range(ns):
        neg = random.random()
        L = 0
        R = len(unigram) - 1
        while(L < R):
            mid = (L+R)//2
            if unigram[mid] < neg:
                L = mid + 1
            else:
                R = mid
        if L-1 == node:
            L = (L + 1)%(len(unigram)-1)+1
        neg_set.append(L-1)
    return neg_set

def Graphs(hp):
    G_files = glob.glob('../data/'+hp.dataset+'/*.inp')

    results = []
    pool = Pool(processes=hp.T)
    for t in range(hp.T):
        results.append(
            pool.apply_async(read_one_graph, (t, hp, G_files[t])))
    pool.close()
    pool.join()
    print("网络读取完成")
    G_list = []
    degree = np.zeros((hp.node_num+1))
    results = [res.get() for res in results]
    unigram_set = []
    for res in results:
        G_list.append(res[0])
        node_set = list(res[0].nodes())
        degree += res[1]
        degree_G = np.zeros((len(node_set)))
        for i, v in enumerate(list(node_set)):
            degree_G[i] += res[1][v]
        degree_G = degree_G**(3/4)
        degree_G = degree_G/np.sum(degree_G)
        unigram_tem = np.zeros((len(node_set) + 1))
        for i in range(1, len(node_set) + 1):
            unigram_tem[i] = unigram_tem[i - 1] + degree_G[i - 1]
        unigram_set.append(unigram_tem)
    degree = degree**(3/4)
    degree = degree/np.sum(degree)
    unigram = np.zeros((hp.node_num + 1))
    for i in range(1, hp.node_num + 1):
        unigram[i] = unigram[i-1] + degree[i - 1]
    # print(unigram)
    # print(unigram_set)
    return G_list, unigram, unigram_set

def read_one_graph(threadID, hp, G_file_name):
    G = nx.Graph()
    cnt = 0
    G_file = open(G_file_name, 'r')
    degree = np.zeros((hp.node_num+1))
    for line in G_file:
        tem = line[:-1].split(' ')
        if len(tem) < 2:
            break
        x = int(tem[0]) - 1
        y = int(tem[1]) - 1
        degree[x] += 1
        degree[y] += 1
        G.add_edge(x, y)
        cnt += 1
        print("\r读取第%d个图  %.4f" % (threadID, cnt), end=" ")
    return [G, degree]
