#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/08/26 19:23:13

@author: Changzhi Sun
"""
import getopt
import sys
import os
import math
import numpy as np
import re
from sklearn.decomposition import NMF
from itertools import chain
from scipy.sparse import csr_matrix, coo_matrix

from my_package.sentence import Sentence
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import load_json_file
from my_package.scripts import save_json_file
from my_package.static import Static

def navie_knn(data, query, k):
    num = data.shape[0]
    diff = np.tile(query, (num, 1)) - data
    square_diff = diff ** 2
    square_diff = np.sum(square_diff, axis=1)
    sorted_dist_indices = np.argsort(square_diff)
    if k > len(sorted_dist_indices):
        k = len(sorted_dist_indices)
    return sorted_dist_indices[0:k]


def build_graph(data, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    num = data.shape[0]
    affinity_matrix = np.zeros((num, num), np.float32)
    if kernel_type == "rbf":
        if rbf_sigma == None:
            raise ValueError("You should input a sigma of rbf kernel!")
        for i in range(num):
            row_sum = 0.0
            if i % 100 == 0:
                print("row number: ", i)
            for j in range(num):
                diff = data[i] - data[j]
                affinity_matrix[i][j] = np.exp(
                    sum(diff**2) / (-2.0 * rbf_sigma**2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i] /= row_sum
    elif kernel_type == "knn":
        if knn_num_neighbors == None:
            raise ValueError("You should input a k of knn kernel!")
        for i in range(num):
            k_neighbors = navie_knn(data, data[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError("Not support kernel type! You can use knn or rbf!")
    return affinity_matrix


def get_vocab(filename):
    vocab = {}
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            vocab[line.split(' ')[0]] = len(vocab)
    return vocab


def usage():
    '''print help information'''
    print("dump_nearest.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hd:",
            ["help", "domain="])
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data", "domains", domain)
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")

    ####### HYPER PARAMETER #######
    rbf_sigma = 1.5
    kernel_type = "rbf"
    max_iter = 2000
    max_step = 1000

    ####### LABEL PROPAGATION  ######
    #  print("####### LABEL PROPAGATION  ######")
    #  word_embeddings = load_pickle_file(
        #  os.path.join(multi_opin_expr_dir, "word2vec", "embeddings.pickle"))
    word_embeddings = load_pickle_file(
        os.path.join(multi_opin_expr_dir, "fine_tune_embeddings.pickle"))
    #  context_embeddings= load_pickle_file(
        #  os.path.join(multi_opin_expr_dir, "context_embeddings.pickle"))
    print("word embedding shape: ", word_embeddings.shape)
    #  print("context embedding shape: ", context_embeddings.shape)
    #  vocab = load_pickle_file(os.path.join(multi_opin_expr_dir,
                                          #  "dictionary.pickle"))
    #  reverse_vocab = load_pickle_file(os.path.join(multi_opin_expr_dir,
                                                  #  "reverse_dictionary.pickle"))
    vocab = get_vocab(os.path.join(multi_opin_expr_dir, "word2vec", "vocab.txt"))
    reverse_vocab = {value : key for key, value in vocab.items()}
    print("vocab size: ", len(vocab))

    #  print("######  DUMP EMBEDDINGS  ######")
    #  f = open(os.path.join(multi_opin_expr_dir, "word_embeddings"),
            #  "w", encoding="utf8")
    #  print(word_embeddings.shape)
    #  for i in range(len(reverse_vocab)):
        #  print(reverse_vocab[i], "\t", word_embeddings[i], file=f)
    #  f.close()

    #  print("######  TEST ######")
    num_nearest = 10
    f = open(os.path.join(multi_opin_expr_dir, "10_nearest_general"),
             "w", encoding="utf8")
    for word in sorted(Static.opinwd):
        if word not in vocab:
            continue
        print("### %s ###" % word, file=f)
        idx_good = vocab[word]
        dist = []
        for i in range(len(word_embeddings)):
            dist.append(sum((word_embeddings[i] - word_embeddings[idx_good]) ** 2))
        dist = np.array(dist)
        idx_sort = dist.argsort()
        k = 0
        for i in range(len(word_embeddings)):
            j = idx_sort[i]
            phrases = reverse_vocab[j].split('^')
            mark = False
            for token in phrases:
                if token in Static.opinwd:
                    mark = True
                    break
            if not mark and len(phrases) > 1:
                print(reverse_vocab[idx_sort[i]], dist[idx_sort[i]], file=f)
                k += 1
            if k == num_nearest:
                break
        print(file=f)
    f.close()

    #  label_node = set()
    #  for e in vocab:
        #  if e in Static.opinwd:
            #  label_node.add(vocab[e])
    #  vocab_size = len(vocab)
    #  num_label = len(label_node)
    #  num_unlabel = vocab_size - num_label
    #  score_label = np.zeros(vocab_size)
    #  for e in label_node:
        #  score_label[e] = 1.0

    #  #  print("building graph ......")
    #  #  affinity_matrix = build_graph(word_embeddings, kernel_type, rbf_sigma)
    #  #  print("builded graph")
    #  #  save_pickle_file(
        #  #  os.path.join(multi_opin_expr_dir, "affinity_matrix.pickle"),
        #  #  affinity_matrix)
    #  affinity_matrix = load_pickle_file(
        #  os.path.join(multi_opin_expr_dir, "affinity_matrix.pickle"))

    #  iter_num = 0
    #  label_node_list = list(label_node)
    #  while iter_num < max_iter:
        #  if iter_num % 100 == 0:
            #  print("iter num: ", iter_num)
        #  np.random.shuffle(label_node_list)
        #  seed = label_node_list[0]
        #  for i in range(max_step):
            #  j = np.argmax(
                #  np.random.multinomial(1, affinity_matrix[seed], 1)[0])
            #  if j not in label_node:
                #  score_label[j] += score_label[seed] * affinity_matrix[seed][j]
            #  seed = j
        #  iter_num += 1
    #  for i in range(vocab_size):
        #  if i not in label_node:
            #  print("%s\t%f" % (reverse_vocab[i], score_label[i]))
