# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys
sys.path.append("/")
import tensorflow as tf
from model import subgraphBernoulli
from tqdm import tqdm
from module import Graphs
from load_data import train_data, eval_data

from args import parse_args
import math
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ""

def main():
    time_start = time.time()
    hp = parse_args()
    print("开始读取数据")
    G_list, unigram, unigram_set = Graphs(hp)
    print("读取数据完成")
    arg = {}
    arg['hp'] = hp
    arg['unigram'] = unigram
    arg['nodes'] = [i for i in range(hp.node_num)]
    thres = hp.node_num // (hp.batch_size // 2)
    print("构建模型")
    m = subgraphBernoulli(arg)
    xc_0 = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size_con, 1), name='xc_0_'+str(_)) for _ in range(hp.T)]
    xc_1 = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size_con, 1), name='xc_1_' + str(_)) for _ in range(hp.T)]
    xuc_0 = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size_con, 1), name='xuc_0_' + str(_)) for _ in range(hp.T)]
    xuc_1 = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size_con, 1), name='xuc_1_' + str(_)) for _ in range(hp.T)]
    xs = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k), name='xs_'+str(_)) for _ in range(hp.T)]
    xn = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k, hp.k-1), name='xn_'+str(_)) for _ in range(hp.T)]
    xg = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k, hp.ns), name='xg_'+str(_)) for _ in range(hp.T)]
    evl = tf.placeholder(dtype=tf.int32, shape=(2, hp.eval_size, 2), name='eval')
    loss, train_op, global_step = m.train(xc_0, xc_1, xuc_0, xuc_1, xs, xn, xg)
    accuracy = m.eval(evl)
    # loss, train_op, global_step = m.train(xs, xn, xg)
    save = m.save_embeddings()
    evl_data = eval_data(hp, G_list[-1], G_list[-2])
    print("开始训练")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_steps = hp.epochs * (hp.node_num//hp.batch_size)
        _gs = sess.run(global_step)
        idx = 0
        last_loss = 0
        last_AUC = 0
        for i in tqdm(range(_gs, total_steps+1)):
            da_xc, da_xuc, da_xs, da_xn, da_xg = train_data(hp, G_list[:-1], unigram_set, idx)
            xc_0_dict = {i: j for i, j in zip(xc_0, da_xc[0])}
            xc_1_dict = {i: j for i, j in zip(xc_1, da_xc[1])}
            xuc_0_dict = {i: j for i, j in zip(xuc_0, da_xuc[0])}
            xuc_1_dict = {i: j for i, j in zip(xuc_1, da_xuc[1])}
            xs_dict = {i: j for i, j in zip(xs, da_xs)}
            xn_dict = {i: j for i, j in zip(xn, da_xn)}
            xg_dict = {i: j for i, j in zip(xg, da_xg)}
            data = {**xc_0_dict, **xc_1_dict}
            data = {**data, **xuc_0_dict}
            data = {**data, **xuc_1_dict}
            data = {**data, **xs_dict}
            data = {**data, **xn_dict}
            data = {**data, **xg_dict}
            # print(data)
            _loss, _, _gs = sess.run([loss, train_op, global_step], feed_dict=data)
            _auc = sess.run([accuracy], feed_dict={evl: evl_data})
            epoch = math.ceil(_gs / hp.batch_size)
            print("   Epoch : %02d   loss : %.2f  AUC : %.3f" % (epoch, _loss, _auc[0]))
            idx = (idx + 1) % thres
            if _gs % 10 == 0 and _gs > thres:
                # if _loss < last_loss:
                #     last_loss = _loss
                #     sess.run([save])
                #     print('保存成功！')
                if _auc[0] > last_AUC:
                    last_AUC = _auc[0]
                    sess.run([save])
                    print('保存成功！')
                else:
                    break
    time_end = time.time()
    all_time = int(time_end - time_start)
    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - hours * 3600 - 60 * minute)

if __name__ == '__main__':
    main()