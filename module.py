import tensorflow as tf
import numpy as np
import random
import glob
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool

def emb_value_cos(x, y):
    a_b = np.dot(x, y)
    a = np.fabs(sum(x ** 2) ** 0.5)
    b = np.fabs(sum(y ** 2) ** 0.5)
    # print(str(x)+"    "+str(y))
    a_dev_b = a_b / (a * b)

    return a_dev_b.reshape(1)

def emb_value_weight_2(x, y):
    a_b = np.dot(x, y)

    return a_b.reshape(1)

def get_initialization(hp):
    alpha_init = (np.random.randn(hp.node_num, hp.dim) / hp.dim).astype('float32')
    y_init = (np.random.randn(hp.node_num, hp.dim) / hp.dim).astype('float32')
    alpha = tf.Variable(alpha_init, name='context_embedding', trainable=True)
    y = []
    for t in range(hp.T):
        y.append(tf.Variable(y_init + 0.001 * tf.random_normal([hp.node_num, hp.dim]) / hp.dim,
                             name='emb_'+str(t), trainable=True))
    return alpha, y

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def accuracy(data, emb):
    pre = []
    label = []
    for i in range(data.shape[1]):
        pre.append(emb_value_weight_2(emb[data[0][i][0]], emb[data[0][i][1]]))
        pre.append(emb_value_weight_2(emb[data[1][i][0]], emb[data[1][i][1]]))
        label.append(1)
        label.append(0)
    cls = LogisticRegression(C=10000)
    cls.fit(pre, label)
    rel_pre = cls.predict(pre)

    AUC = roc_auc_score(label, rel_pre)
    return AUC

def negative_sampling(unigram, ns, node):
    neg_set = []
    while len(neg_set) < ns:
        neg = random.random()
        L = 0
        R = len(unigram) - 1
        while(L < R):
            mid = (L+R)//2
            if unigram[mid] < neg:
                L = mid + 1
            else:
                R = mid
        if L-1 in node:
            continue
        neg_set.append(L-1)
    return neg_set

def Graphs(hp):
    G_files = glob.glob('../data/'+hp.dataset+'/*.inp')
    T = len(G_files)
    results = []
    pool = Pool(processes=T)
    for t in range(T):
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
        x = int(tem[0])
        y = int(tem[1])
        degree[x] += 1
        degree[y] += 1
        G.add_edge(x, y)
        cnt += 1
        print("\r读取第%d个图  %.4f" % (threadID, cnt), end=" ")
    return [G, degree]
