import tensorflow as tf
import numpy as np
import os
import sys
from args import parse_args
arg = parse_args()
from module import get_initialization, negative_sampling, noam_scheme

def save_emb(y, alpha):
    file_path = '../Embeddings/subgraphBernoulli/' + arg.dataset + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    for t in range(arg.T):
        file_nam = file_path+str(t).zfill(4) + '.emb'
        file = open(file_nam, 'w')
        for v in range(arg.node_num):
            file.write(str(v))
            for it in y[t][v]:
                file.write(' ' + str(it))
            file.write('\n')
        file.close()
    file_nam = file_path+'alpha.con'
    file = open(file_nam, 'w')
    for v in range(arg.node_num):
        file.write(str(v))
        for it in alpha[v]:
            file.write(' ' + str(it))
        file.write('\n')
    file.close()
    print('保存完成')
    return 1

class subgraphBernoulli:

    def __init__(self, args):
        self.hp = args['hp']
        self.alpha, self.y = get_initialization(self.hp)
        self.unigram = args['unigram']
        self.nodes = args['nodes']

    def train(self, xs, xn, xg):
        L_pos = 0
        L_neg = 0
        L_reg = 0
        L_smooth = 0

        for t in range(self.hp.T):
            tar_emb = tf.nn.embedding_lookup(self.y[t], xs[t]) # (batch, k, dim)
            nei_emb = tf.squeeze(tf.reduce_sum(tf.nn.embedding_lookup(self.alpha, xn[t]), 2)) # (batch, k, dim)
            neg_emb = tf.nn.embedding_lookup(self.alpha, xg[t]) # (batch, k, ns, dim)
            neg_emb = tf.squeeze(tf.reduce_sum(neg_emb, 2))

            pos_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('nij,nij->nij', tar_emb, nei_emb), -1)))
            neg_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('nij, nij->nij', neg_emb, nei_emb), -1))/self.hp.ns)
            L_pos -= tf.reduce_sum(tf.log(pos_v))
            L_neg -= tf.reduce_sum(tf.log(1-neg_v))
        for t in range(self.hp.T - 1):
            L_smooth += tf.reduce_sum((self.y[t + 1] - self.y[t]) ** 2)
        L_reg = tf.reduce_sum(self.alpha**2) + tf.reduce_sum(self.y[0]**2)
        loss = (L_pos + L_neg) + self.hp.lam*L_smooth + self.hp.lam*L_reg
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def save_embeddings(self):
        flg = tf.py_func(save_emb, [self.y, self.alpha], tf.int32)
        return flg

