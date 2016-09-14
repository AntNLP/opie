#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/09/08 14:51:06

@author: Changzhi Sun
"""

from collections import Counter
import getopt
import sys
import os
import math
import numpy as np
import re
from itertools import chain
from scipy.sparse import csr_matrix, coo_matrix
from tensorflow.contrib import learn

from my_package.sentence import Sentence
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.static import Static


def get_vocab(filename):
    vocab = learn.preprocessing.CategoricalVocabulary(unknown_token="UNK")
    with open(filename, "r", encoding="utf8") as f:
        i = 0
        for line in f:
            word, ct = line.strip().split(' ')
            vocab.add(word, int(ct))
            if i == 0:
                vocab._freq["UNK"] = int(ct)
                i = 1
    vocab.freeze()
    return vocab


def generate_train_text(multi_opin_expr_dir, max_document_length):
    fin1 = open(os.path.join(multi_opin_expr_dir, "clean_replace_sentences"),
                "r", encoding="utf8")
    fin2 = open(os.path.join(multi_opin_expr_dir, "score"),
                "r", encoding="utf8")
    fout = open(os.path.join(multi_opin_expr_dir, "train_text_raw"),
                "w", encoding="utf8")
    review_id = 1
    line_str = ""
    for line1, line2 in zip(fin1, fin2):
        s, t = line1[:-1].split('\t')
        t = int(t)
        if review_id < t:
            if score < 3.0:
                print("%s\t0" % line_str.strip(), file=fout)
            elif score > 3.0:
                print("%s\t1" % line_str.strip(), file=fout)
            line_str = ""
            review_id += 1
        score = float(line2.strip())
        line_str += " %s" % s
    fin1.close()
    fin2.close()
    fout.close()
    f = open(os.path.join(multi_opin_expr_dir, "train_text_raw"),
                "r", encoding="utf8")
    lines = f.readlines()
    f.close()

    shuffle_indices = np.random.permutation(np.arange(len(lines)))
    f = open(os.path.join(multi_opin_expr_dir, "train_text"),
                "w", encoding="utf8")
    for i in shuffle_indices:
        text, _ = lines[shuffle_indices[i]].split('\t')
        if len(text.split(' ')) > max_document_length:
            continue
        if text.strip() == "":
            continue
        print(lines[shuffle_indices[i]], end="", file=f)
    f.close()

def load_data_and_labels(domain_dir, max_document_length=200, shuffle_data=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")
    if shuffle_data:
        generate_train_text(multi_opin_expr_dir, max_document_length)

    # Load data from files
    f = open(os.path.join(multi_opin_expr_dir, "train_text"),
             "r", encoding="utf8")
    x_text = []
    y = []
    for line in f:
        text, label = line.strip().split('\t')
        x_text.append(text)
        if int(label) == 0:
            y.append([1, 0])
        else:
            y.append([0, 1])
    vocab = get_vocab(os.path.join(multi_opin_expr_dir,
                                   "word2vec", "vocab.txt"))
    return x_text, np.array(y), vocab


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def tokenizer(iterator):
    for value in iterator:
        yield value.split(' ')




def usage():
    '''print help information'''
    print("data_helpers.py 用法:")
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
    generate_train_text(multi_opin_expr_dir, max_document_length=200)

    #  general_list = []
    #  vocab = get_vocab(os.path.join(multi_opin_expr_dir, "word2vec", "vocab.txt"))
    #  for i in range(len(vocab._mapping)):
        #  if vocab.reverse(i) in Static.opinwd:
            #  general_list.append((vocab.reverse(i), vocab._freq[vocab.reverse(i)]))
            #  #  print(vocab.reverse(i), vocab._freq[vocab.reverse(i)])
    #  for w, wc in sorted(general_list, key=lambda x: x[1], reverse=True):
        #  print(w, wc)
    #  reverse_vocab = {value : key for key, value in vocab.items()}
    #  print("vocab size: ", len(vocab))
    #  word_embeddings = load_pickle_file(
        #  os.path.join(multi_opin_expr_dir, "word2vec", "embeddings.pickle"))
    #  print("word embedding shape: ", word_embeddings.shape)

    #  i = 1
    #  filename = os.path.join(domain_dir, "pickles",
                            #  "without_parse_sentences",
                            #  "without_parse_sentences_%d.pickle" % i)
    #  f = open(os.path.join(multi_opin_expr_dir, "score"),
             #  "w", encoding="utf8")
    #  while os.path.join(filename + ".bz2"):
        #  print("pickle index: % d  loading" % i)
        #  sentences = load_pickle_file(filename)
        #  print("pickle index: % d  loaded" % i)
        #  for sentence in sentences:
            #  print(sentence.score, file=f)
        #  i += 1
        #  filename = os.path.join(domain_dir, "pickles",
                                #  "without_parse_sentences",
                                #  "without_parse_sentences_%d.pickle" % i)
        #  if i == 21:
            #  break
    #  f.close()
