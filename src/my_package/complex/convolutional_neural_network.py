#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/05/17 11:50:20

@author: Changzhi Sun
"""
import os
import sys
import getopt
from collections import Counter

import tensorflow as tf
import numpy as np


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 100, 1], strides=[1, 1, 100, 1],
                          padding="SAME")


def to_index(datas, lex):
    index = []
    k = 0
    word_idx2phrase_idx = {}
    phrase_idx2word_idx = {}
    for i, data in enumerate(datas):
        phrase_idx2word_idx[i] = k
        for e in data:
            index.append([lex[e]])
            word_idx2phrase_idx[k] = i
            k += 1
    return index, word_idx2phrase_idx, phrase_idx2word_idx


def get_complex_data(near_dir):
    f = open(os.path.join(near_dir, "complex_data.ann"), "r", encoding="utf8")
    phrases = []
    ph_postags = []
    labels = []
    for line in f:
        label, ph, ph_postag = line.strip().split('\t')
        phrases.append(ph.split(' '))
        ph_postags.append(ph_postag.split(' '))
        labels.append(int(label))
    f.close()
    return phrases, ph_postags, labels

def build_dataset(datas):
    counter = Counter()
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            counter[datas[i][j]] += 1
    dictionary = {}
    for word in counter.keys():
        dictionary[word] = len(dictionary)
    reverse_dictionary = {value: key for key, value in dictionary.items()}
    return dictionary, reverse_dictionary


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def usage():
    '''print help information'''
    print("convolutional_neural_network.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:", ["help", "domain="])
    except getopt.GetoptError:
        print("命令行参数输入错误！")
        usage()
        sys.exit(1)
    for op, value in opts:
        if op in ("-h", "--help"):
            usage()
            sys.exit()
        if op in ("-d", "--domain"):
            domain = value
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
    near_dir = os.path.join(domain_dir, "near")
    phrases, ph_postags, labels = get_complex_data(near_dir)
    word_lex, word_reverse_lex = build_dataset(phrases)
    postag_lex, postag_reverse_lex = build_dataset(ph_postags)
    index_words, word_idx2phrase_idx, phrase_idx2word_idx = to_index(phrases, word_lex)
    index_postags, word_idx2phrase_idx, phrase_idx2word_idx = to_index(ph_postags, postag_lex)

    r = int(len(labels) * 0.6)
    rr = phrase_idx2word_idx[r]
    labels = np.reshape(labels, [len(labels), 1])
    train_index_words = index_words[:rr]
    test_index_words = index_words[rr:]
    train_index_postags = index_postags[:rr]
    test_index_postags = index_postags[rr:]
    train_labels, test_labels = labels[:r], labels[r:]

    batch_size = 20
    embedding_size = 10
    learning_rate = 0.1
    training_epochs = 25
    window_size = 2

    word_embeddings = tf.Variable(
        tf.random_uniform([len(word_lex), embedding_size], -1.0, 1.0))
    postag_embeddings = tf.Variable(
        tf.random_uniform([len(postag_lex), embedding_size], -1.0, 1.0))

    word = tf.placeholder(tf.int32, [None, 1])
    postag = tf.placeholder(tf.int32, [None, 1])
    y = tf.placeholder("float", [None, 1])
    x_conv1 = tf.placeholder("float", [None, 1])

    word_embed = tf.nn.embedding_lookup(word_embeddings, word)
    #  word_embed = tf.expand_dims(word_embed, -1)
    postag_embed = tf.nn.embedding_lookup(postag_embeddings, postag)

    #  word_embed = tf.reshape(word_embed, [-1])
    #  postag_embed = tf.reshape(postag_embed, [-1])

    #  merge_embed = tf.reshape(tf.concat(0, [word_embed, postag_embed]), [1, window_size, -1, embedding_size])

    #  W_conv1 = weight_variable([window_size, window_size, embedding_size, 1])
    #  b_conv1 = bias_variable([1])

    #  h_conv1 = tf.nn.relu(conv2d(merge_embed, W_conv1) + b_conv1)
    #  h_pool1 = max_pool_2x2(h_conv1)
    #  h_pool1 = tf.reshape(h_pool1, [-1, 1])

    #  W_soft = weight_variable([1, 1])
    #  b_soft = bias_variable([1])
    #  h_soft = tf.nn.softmax(tf.matmul(x_conv1, W_soft) + b_soft)
    #  cost = tf.nn.softmax_cross_entropy_with_logits(h_soft, y)
    #  #  cost = -tf.reduce_sum(y * tf.log(h_out))
    #  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print(len(train_index_words))
    print(sess.run(word_embed, feed_dict={word: train_index_words, postag: train_index_postags})[:2])

    #  for i in range(len(train_labels)):
        #  b = phrase_idx2word_idx[i]
        #  e = phrase_idx2word_idx[i+1]
        #  batch_word_index = train_index_words[b: e]
        #  batch_postag_index = train_index_postags[b: e]
        #  #  print(sess.run(h_conv1, feed_dict={word: batch_word_index, postag: batch_postag_index}))
        #  t = sess.run(h_pool1, feed_dict={word: batch_word_index, postag: batch_postag_index})
        #  if i == 0:
            #  x_pool = t
        #  else:
            #  x_pool = tf.concat(0, [x_pool, t])
    #  print(sess.run(x_pool).shape)
    #  #  print(sess.run(b_out).shape)
    #  #  train_labels = np.reshape(train_labels, [len(train_labels), 1])
    #  #  print(sum(sess.run(h_out, feed_dict={x_conv1: x_train, y: train_labels})))
    #  #  for epoch in range(10):
        #  #  sess.run(optimizer, feed_dict={x_conv1: x_train, y: train_labels})
        #  #  print(sess.run(W_soft))
    sess.close()
