#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/08/21 19:18:46

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
from scipy.sparse import coo_matrix

from my_package.sentence import Sentence
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import load_json_file
from my_package.scripts import save_json_file
from my_package.static import Static

def calcu_ppmi(N, id_w, id_c, w_c, c_w):
    if id_w not in w_c or id_c not in w_c[id_w]:
        return 0
    up =  w_c[id_w][id_c] * N
    down =  len(w_c[id_w]) * len(c_w[id_c])
    return max(0, math.log2(up/down))


def build_token_matrix(filename, window=2):
    w_c = {}
    c_w = {}
    vocab = {}
    N = 0
    f = open(filename, "r", encoding="utf8")
    for line in f:
        text, _ = line.strip().split('\t')
        tokens = text.split(' ')
        n = len(tokens)
        if n <= 2 * window:
            continue
        for i in range(window, n - window):
            if tokens[i] in Static.stopwords:
                continue
            if re.search(r"^\W*$", tokens[i]):
                continue
            if Sentence.is_weak_opinwd(tokens[i]):
                continue
            if tokens[i] not in vocab:
                vocab[tokens[i]] = len(vocab)
            id_w = vocab[tokens[i]]
            for j in chain(range(i-window, i), range(i+1, i+window+1)):
                if tokens[j] not in vocab:
                    continue
                id_c = vocab[tokens[j]]
                if id_w not in w_c:
                    w_c[id_w] = {}
                if id_c not in w_c[id_w]:
                    w_c[id_w][id_c] = 0
                w_c[id_w][id_c] += 1
                if id_c not in c_w:
                    c_w[id_c] = {}
                if id_w not in c_w[id_c]:
                    c_w[id_c][id_w] = 0
                c_w[id_c][id_w] += 1
                N += 1
    f.close()
    vocab_size = len(vocab)
    print("N=", N)
    print("vocab size: ", vocab_size)
    token_matrix = np.zeros((vocab_size, vocab_size))
    f = open(os.path.join(multi_opin_expr_dir, "matrix_item"),
             "w", encoding="utf8")
    c = []
    r = []
    data = []
    for i in range(vocab_size):
        if i % 100 == 0:
            print("i= ", i)
        for j in range(vocab_size):
            ppmi = calcu_ppmi(N, i, j, w_c, c_w)
            if ppmi > 0:
                r.append(i)
                c.append(j)
                data.append(ppmi)
                print("%d %d %f" % (i, j, ppmi), file=f)
    f.close()
    token_matrix = coo_matrix((data, (r, c)))
    reverse_vocab = {value : key for key, value in vocab.items()}
    save_pickle_file(os.path.join(multi_opin_expr_dir, "token_matrix.pickle"),
                     token_matrix)
    save_pickle_file(os.path.join(multi_opin_expr_dir, "vocab.pickle"), vocab)
    save_pickle_file(
        os.path.join(multi_opin_expr_dir, "reverse_vocab.pickle"),
        reverse_vocab)


def usage():
    '''print help information'''
    print("nonnegative_matrix_factorization.py 用法:")
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

    ###  Build Token-Token Matrix  ###
    print("######  TOKEN-TOKEN MATRIX  ######")
    build_token_matrix(
        os.path.join(multi_opin_expr_dir, "replace_text"), window=2)

    ###  NMF  ###
    print("######  NON-NEGATIVE MATRIX FACTORIZATION  ######")
    token_matrix = load_pickle_file(
        os.path.join(multi_opin_expr_dir, "token_matrix.pickle"))
    print("token martix shape: ", token_matrix.shape)
    model = NMF(n_components=200)
    print("fit and transform")
    word_embeddings = model.fit_transform(token_matrix)
    print("fit end")
    context_embeddings = model.components_
    save_pickle_file(
        os.path.join(multi_opin_expr_dir, "word_embeddings.pickle"),
        word_embeddings)
    save_pickle_file(
        os.path.join(multi_opin_expr_dir, "context_embeddings.pickle"),
        context_embeddings)
