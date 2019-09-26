import tensorflow as tf
import numpy as np
import os
import sys
from args import parse_args
arg = parse_args()
from module import get_initialization, noam_scheme

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
    return 1

class subgraphBernoulli:

    def __init__(self, args):
        self.hp = args['hp']
        self.alpha, self.y = get_initialization(self.hp)
        self.unigram = args['unigram']
        self.nodes = args['nodes']

    def train(self, xc_0, xc_1, xuc_0, xuc_1, xs, xn, xg):
        L_pos = 0
        L_neg = 0
        L_smooth = 0
        L_con = 0
        L_reg = 0
        L_ucon = 0

        for t in range(self.hp.T):
            tar_emb = tf.nn.embedding_lookup(self.y[t], xs[t]) # (batch, k, dim)
            nei_emb = tf.squeeze(tf.reduce_sum(tf.nn.embedding_lookup(self.alpha, xn[t]), 2)) # (batch, k, dim)
            neg_emb = tf.nn.embedding_lookup(self.alpha, xg[t]) # (batch, k, ns, dim)
            neg_emb = tf.squeeze(tf.reduce_sum(neg_emb, 2))
            con_0_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xc_0[t]))
            con_1_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xc_1[t]))
            ucon_0_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xuc_0[t]))
            ucon_1_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xuc_1[t]))

            pos_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('nij,nij->nij', tar_emb, nei_emb), -1)))
            neg_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('nij, nij->nij', neg_emb, nei_emb), -1)))
            con_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', con_0_emb, con_1_emb), -1)))
            ucon_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', ucon_0_emb, ucon_1_emb), -1)))

            # 用ground truth来算loss
            L_con -= tf.reduce_sum(tf.log(con_v + 1e-15))  # connection
            L_ucon -= tf.reduce_sum(tf.log(1 - ucon_v + 1e-15))  # unconnection
            L_pos -= tf.reduce_sum(tf.log(pos_v+1e-15))
            L_neg -= tf.reduce_sum(tf.log(1-neg_v+1e-15))

            # 用cross entropy 算loss
            # L_con -= tf.reduce_sum(tf.multiply(con_v, tf.log(con_v + 1e-15)))  # connection
            # L_ucon -= tf.reduce_sum(tf.multiply(1 - ucon_v, tf.log(1 - ucon_v + 1e-15)))  # unconnection
            # L_pos -= tf.reduce_sum(tf.multiply(pos_v, tf.log(pos_v + 1e-15)))
            # L_neg -= tf.reduce_sum(tf.multiply((1 - neg_v), tf.log(1 - neg_v + 1e-15)))
        for t in range(self.hp.T - 1):
            L_smooth += tf.reduce_sum((self.y[t + 1] - self.y[t]) ** 2)
        L_reg = tf.reduce_sum(self.alpha**2) + tf.reduce_sum(self.y[0]**2)
        loss = (L_con + L_ucon) + (L_pos + L_neg) + self.hp.lam * (L_smooth + L_reg)
        # loss = (L_con + L_ucon) + 0.0001*(L_pos + L_neg) + 0.0001*(L_smooth + L_reg)
        global_step = tf.train.get_or_create_global_step()
        # lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        lr = self.hp.lr
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def save_embeddings(self):
        flg = tf.py_func(save_emb, [self.y, self.alpha], tf.int32)
        return flg

