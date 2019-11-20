import tensorflow as tf
import numpy as np
import os
import sys
from args import parse_args
arg = parse_args()
from module import get_initialization, noam_scheme, accuracy

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
    file.write(str(arg.node_num)+' '+str(arg.dim)+'\n')
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

    def train(self, xc_0, xc_1, wc, xuc_0, xuc_1, xs, xn, ws):
        L_pos = 0
        L_smooth = 0
        L_con = 0
        L_ucon = 0

        # self.alpha = tf.divide(self.alpha, tf.reduce_sum(self.alpha, -1))
        # for t in range(self.hp.T):
        #     self.y[t] = tf.divide(self.y[t], tf.reduce_sum(self.y[t], -1))

        for t in range(self.hp.T):
            tar_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xs[t])) # (batch, dim)
            nei_emb = tf.squeeze(tf.reduce_sum(tf.nn.embedding_lookup(self.alpha, xn[t]), 1)) # (batch, dim)
            # neg_emb = tf.nn.embedding_lookup(self.alpha, xg[t]) # (batch, k, ns, dim)
            # neg_emb = tf.squeeze(tf.reduce_sum(neg_emb, 2))
            con_0_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xc_0[t]))# (batch,  dim)
            con_1_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xc_1[t]))# (batch,  dim)
            ucon_0_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xuc_0[t]))# (batch,  dim)
            ucon_1_emb = tf.squeeze(tf.nn.embedding_lookup(self.y[t], xuc_1[t]))# (batch,  dim)

            pos_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', tar_emb, nei_emb), -1)))
            con_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', con_0_emb, con_1_emb), -1)))
            ucon_v = tf.sigmoid(tf.squeeze(tf.reduce_sum(tf.einsum('ni,ni->ni', ucon_0_emb, ucon_1_emb), -1)))

            # 用ground truth来算loss
            L_con -= tf.reduce_sum(wc*tf.log(con_v + 1e-15))  # connection
            L_ucon -= tf.reduce_sum(tf.log(1 - ucon_v + 1e-15))  # unconnection
            L_pos -= tf.reduce_sum(ws*tf.log(pos_v+1e-15))

        for t in range(self.hp.T - 1):
            L_smooth += tf.reduce_sum((self.y[t + 1] - self.y[t]) ** 2)

        loss = (L_con + L_ucon) + L_pos + L_smooth
        global_step = tf.train.get_or_create_global_step()
        # lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        lr = self.hp.lr
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def eval(self, evl):
        AUC = tf.py_func(accuracy, [evl, self.y[-1]], tf.double, stateful=True)
        return AUC

    def save_embeddings(self):
        flg = tf.py_func(save_emb, [self.y, self.alpha], tf.int32)
        return flg

