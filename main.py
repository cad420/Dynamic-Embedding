# -*- coding: utf-8 -*-
#/usr/bin/python3
import sys
sys.path.append("/")
import tensorflow as tf
from model import subgraphBernoulli
from tqdm import tqdm
from module import Graphs
from load_data import train_data

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

    print("构建模型")
    m = subgraphBernoulli(arg)
    xs = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k), name='xs_'+str(_)) for _ in range(hp.T)]
    xn = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k, hp.k-1), name='xn_'+str(_)) for _ in range(hp.T)]
    xg = [tf.placeholder(dtype=tf.int32, shape=(hp.batch_size, hp.k, hp.ns), name='xg_'+str(_)) for _ in range(hp.T)]
    loss, train_op, global_step = m.train(xs, xn, xg)
    save = m.save_embeddings()
    print("开始训练")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_steps = hp.epochs * (hp.node_num//hp.batch_size)
        _gs = sess.run(global_step)
        idx = 0
        last_loss = 999999999999999999999
        for i in tqdm(range(_gs, total_steps+1)):
            da_xs, da_xn, da_xg = train_data(hp, G_list, unigram_set, idx)
            xs_dict = {i: j for i, j in zip(xs, da_xs)}
            xn_dict = {i: j for i, j in zip(xn, da_xn)}
            xg_dict = {i: j for i, j in zip(xg, da_xg)}
            data = {**xs_dict, **xn_dict}
            data = {**data, **xg_dict}
            # print(data)
            _loss, _, _gs = sess.run([loss, train_op, global_step], feed_dict=data)
            epoch = math.ceil(_gs / hp.batch_size)
            print("   Epoch : %02d   loss : %.2f" % (epoch, _loss))
            idx = idx+hp.batch_size//2
            if _gs % 20 == 0:
                if _loss < last_loss:
                    last_loss = _loss
                    sess.run([save])
                else:
                    break
    time_end = time.time()
    all_time = int(time_end - time_start)
    hours = int(all_time / 3600)
    minute = int((all_time - 3600 * hours) / 60)
    print('totally cost  :  ', hours, 'h', minute, 'm', all_time - -hours * 3600 - 60 * minute)

if __name__ == '__main__':
    main()