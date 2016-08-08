#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/06/22 16:30:09

@author: Changzhi Sun
"""
import os
from collections import Counter
from itertools import chain
import sys

import numpy as np

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import remove
from my_package.scripts import mkdir
from my_package.static import Static
from normalization import parse
from normalization import handle_normalize

table_opinwd = set(Static.opinwd.keys())

def generate_train_datas(domain_dir,
                         start_index,
                         end_index,
                         test=False,
                         complex_opinwd=None):
    train_dir = os.path.join(domain_dir, "relation", "train")
    pickle_dir = os.path.join(domain_dir, "pickles")
    i = start_index
    filename = os.path.join(pickle_dir, "bootstrapping",
                            "bootstrapping_sentences_%d.pickle" % i)
    X, y = [], []
    while os.path.exists(filename + ".bz2") and i < end_index:
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            sentence.generate_candidate_relation(table_opinwd,
                                                complex_opinwd=complex_opinwd,
                                                test=test)
            sentence.generate_label(test=test)
            X_tmp, y_tmp = sentence.generate_each_sentence()
            #  X_tmp, y_tmp = sentence.generate_each_sentence_with_conv(w=10)
            X.extend(X_tmp)
            y.extend(y_tmp)
        i += 1
        filename = os.path.join(pickle_dir, "bootstrapping",
                                "bootstrapping_sentences_%d.pickle" % i)
    return X, y


def generate_test_datas(test_dir, sentences, test=True, complex_opinwd=None):
    X, y = [], []
    for sentence in sentences:
        sentence.generate_candidate_relation(table_opinwd,
                                             complex_opinwd=complex_opinwd,
                                             test=test)
        sentence.generate_label(test=test)
        X_tmp, y_tmp = sentence.generate_each_sentence()
        #  X_tmp, y_tmp = sentence.generate_each_sentence_with_conv(w=10)
        X.extend(X_tmp)
        y.extend(y_tmp)
    if test:
        save_pickle_file(os.path.join(test_dir, "sentences.candidate.pickle"),
                         sentences)
    return X, y


def padding(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence[:sequence_length]
        for j in range(num_padding):
            new_sentence.append(padding_word)
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sequences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    counts = Counter()
    for sequence in sequences:
        for e in sequence:
            counts[e] += 1
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.insert(0, "UNK")
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def build_input_data(sequences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sequence in sequences:
        t = []
        for e in sequence:
            if e in vocabulary:
                t.append(vocabulary[e])
            else:
                t.append(vocabulary["UNK"])
        x.append(t)
    return x


def get_complex_opinwd(filename):
    if not os.path.exists(filename):
        print("%s not exists" % filename)
        return None
    print("%s exists" % filename)
    complex_opinwd = set()
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            complex_opinwd.add(line.strip())
    return complex_opinwd


def load_datas(domain_dir, start_index, end_index, complex_op=0):
    test_dir = os.path.join(domain_dir, "relation", "test")
    complex_dir = os.path.join(domain_dir, "complex")
    if complex_op == 1:
        complex_opinwd_train = get_complex_opinwd(os.path.join(complex_dir,
                                                        "candidate_clean",
                                                        "0.5-complex.train"))
        complex_opinwd_test= get_complex_opinwd(os.path.join(complex_dir,
                                                        "candidate_clean",
                                                        "0.5-complex.test"))
    elif complex_op == 2:
        complex_opinwd_train = get_complex_opinwd(os.path.join(complex_dir,
                                                        "candidate_clean",
                                                        "0.8-complex.train"))

        complex_opinwd_test= get_complex_opinwd(os.path.join(complex_dir,
                                                        "candidate_clean",
                                                        "0.8-complex.test"))
    else:
        complex_opinwd_train = None
        complex_opinwd_test = None
    X, y = generate_train_datas(domain_dir, start_index, end_index,
                                complex_opinwd=complex_opinwd_train)

    sentences = load_pickle_file(os.path.join(test_dir,
                                              "sentences.ann.pickle"))

    rr = len(y)
    X_test, y_test = generate_test_datas(test_dir,
                                         sentences,
                                         test=True,
                                         complex_opinwd=complex_opinwd_test)
    X.extend(X_test)
    y.extend(y_test)

    sequence_len = []
    for e in X:
        sequence_len.append([max(1, len(ee)) for ee in e])
    for i in range(1, len(X[0])):
        features = [e[i] for e in X]
        features_pad = padding(features)
        for j in range(len(X)):
            X[j][i] = features_pad[j]

    word_chain = chain([])
    postag_chain = chain([])
    for i in range(1, len(X[0])-1):
        if i % 2 == 0:
            postag_chain = chain(postag_chain, [e[i] for e in X])
        else:
            word_chain = chain(word_chain, [e[i] for e in X])
    voca, voca_inv = {}, {}
    word_voca, word_voca_inv = build_vocab(word_chain)
    voca["word"] = word_voca
    voca_inv["word"] = word_voca_inv
    postag_voca, postag_voca_inv = build_vocab(postag_chain)
    voca["postag"] = postag_voca
    voca_inv["postag"] = postag_voca_inv
    voca["dep"] = {"<": 0, ">": 1, "<PAD/>": 2}
    for i in range(1, len(X[0])-1):
        if i % 2 == 0:
            v = voca["postag"]
        else:
            v = voca["word"]
        features = [e[i] for e in X]
        features = build_input_data(features, v)
        for j in range(len(X)):
            X[j][i] = features[j]
    features = [e[-1] for e in X]
    features = build_input_data(features, voca["dep"])
    for j in range(len(X)):
        X[j][-1] = features[j]
    for i in range(len(X)):
        X[i].append(sequence_len[i])
    return np.array(X), np.array(y), voca, rr


def have_overlap(e1, e2):
    if set(e1[0]) & set(e2[0]) != set() and set(e1[1]) & set(e2[1]) != set():
        return True
    return False


def combine_result(f1, f2, outfile):
    out = open(outfile, "w", encoding="utf8")
    ann = {}
    for e in parse(f2):
        if e["S"] not in ann:
            ann[e["S"]] = e["R"]
        else:
            ann[e["S"]].extend(e["R"])
    for e in parse(f1):
        if e["S"] not in ann:
            ann[e["S"]] = e["R"]
        else:
            ann[e["S"]].extend(e["R"])
    for key, value in ann.items():
        print("S\t%s"%key, file=out)
        t = []
        m = set()
        for v in value:
            if (v[1], v[2]) not in m:
                m.add((v[1], v[2]))
                t.append(v)
        for v in t:
            print("R\t%s"%("\t".join(v)), file=out)
    out.close()


def convert_index(sentences, begin, end):
    begin, end = int(begin), int(end)
    j = 0
    for i, sentence in enumerate(sentences):
        k = 1
        for w in sentence:
            if j == begin:
                first = k
            if j == end:
                last = k
                return i, tuple(range(first, last+1))
            if w == ' ' or w == '\n':
                k += 1
            j += 1


def handle_review(review_name, ann_name, sent_ann):
    f = open(review_name, "r", encoding="utf8")
    g = open(ann_name, "r", encoding="utf8")
    sentences = f.readlines()
    sents = [e.strip() for e in sentences]
    review = "".join(sentences)
    ann_dict = {}
    for line in g:
        line_strip = line.strip()
        if line_strip.startswith("T"):
            T, O, t = line_strip.split('\t')
            offset = O.split(' ')
            ann_dict[T] = convert_index(sentences, offset[1], offset[2])
        elif line_strip.startswith("R"):
            R, s = line_strip.split('\t')
            arg1, arg2 = s[17:].split(' ')
            arg1, arg2 = arg1.split(':')[1], arg2.split(':')[1]
            opwd = ann_dict[arg1]
            optg = ann_dict[arg2]
            if sents[opwd[0]] not in sent_ann:
                sent_ann[sents[opwd[0]]] = set()
            sent_ann[sents[opwd[0]]].add((optg[1], opwd[1]))
    f.close()
    g.close()


def get_ann(ann_dir):
    i = 1
    review_name = os.path.join(ann_dir, "review_%d.txt" % i)
    ann_name = os.path.join(ann_dir, "review_%d.ann" % i)
    sent_ann = {}
    while os.path.exists(review_name):
        handle_review(review_name, ann_name, sent_ann)
        i += 1
        review_name = os.path.join(ann_dir, "review_%d.txt" % i)
        ann_name = os.path.join(ann_dir, "review_%d.ann" % i)
    return sent_ann


def calcu_PRF(filename, sent_ann):
    TP_FP = 0
    TP = 0
    TP_TN = 0
    for e in parse(filename):
        text = e['S']
        TP_FP += len(e['R'])
        unique_set = set()
        if text in sent_ann:
            tokens = text.split(' ')
            for rr in e['R']:
                r = (tuple(eval(rr[2])), tuple(eval(rr[3])))
                for ee in sent_ann[text]:
                    if have_overlap(r, ee):
                        unique_set.add(ee)
            if len(unique_set) != len(sent_ann[text]):
                pass
                #  print(text)
                #  for ee in sent_ann[text]:
                    #  if ee not in unique_set:
                        #  print("%s\t\t%s" %
                              #  (" ".join([tokens[e-1] for e in ee[0]]),
                               #  " ".join([tokens[e-1] for e in ee[1]])))
                #  print()
        TP += len(unique_set)
    for key, value in sent_ann.items():
        TP_TN += len(value)
    P = TP / TP_FP
    R = TP / TP_TN
    F = 2 * P * R / (P + R)
    print(TP, TP_FP, TP_TN)
    return P, R, F

def dump(filename, sentences, pred):
    f = open(filename, "w", encoding="utf8")
    i = 0
    #  k = 0
    for sentence in sentences:
        R = []
        #  k += len(sentence.candidate_relation)
        #  print("k=   ", k)
        for profeat_idx, opinwd_idx in sentence.candidate_relation:
            if pred[i] == 1:
                R.append((profeat_idx, opinwd_idx))
            i += 1
        if len(R) > 0:
            print("S\t%s" % sentence.text, file=f)
            for profeat_idx, opinwd_idx in R:
                print("R\t{0}\t{1}\t{2}\t{3}".format(
                    sentence.print_phrase(profeat_idx),
                    sentence.print_phrase(opinwd_idx),
                    list(profeat_idx), list(opinwd_idx)), file=f)
    f.close()


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
            if start_index >= end_index:
                continue
            yield shuffled_data[start_index:end_index]


def add_test_ann(test_dir):
    sentences = load_pickle_file(os.path.join(test_dir, "sentences.pickle"))
    sent_ann = get_ann(os.path.join(test_dir, "ann"))
    f = open(os.path.join(test_dir, "test.ann"), "w", encoding="utf8")
    for sentence in sentences:
        sentence.relation = {}
        if sentence.text in sent_ann:
            print("S\t%s" % sentence.text, file=f)
            print("S\t%s" % sentence.text)
            for e1, e2 in sent_ann[sentence.text]:
                print("R\t{0}\t{1}\t{2}\t{3}".format(
                    sentence.print_phrase(e1),
                    sentence.print_phrase(e2),
                    list(e1), list(e2)), file=f)
                sentence.relation[(e1, e2)] = "test"
    f.close()
    save_pickle_file(os.path.join(test_dir, "sentences.ann.pickle"), sentences)

if __name__ == "__main__":
    domain_dir = "/home/zhi/Project/opie/data/domains/reviews_Cell_Phones_and_Accessories"
    #  domain_dir = "/home/zhi/Project/opie/data/domains/reviews_Grocery_and_Gourmet_Food"
    #  domain_dir = "/home/zhi/Project/opie/data/domains/reviews_Pet_Supplies"
    #  domain_dir = "/home/zhi/Project/opie/data/domains/reviews_Movies_and_TV"
    pickle_dir = os.path.join(domain_dir, "pickles")
    relation_dir = os.path.join(domain_dir, "relation")
    test_dir = os.path.join(relation_dir, "test")
    #  X, y = generate_train_datas(domain_dir, 1, 2,
                                #  complex_opinwd=None)
    #  add_test_ann(test_dir)
    X, y, _, _ = load_datas(domain_dir, 1, 2)
    print(len(X[0]))
    print(X[0][-2])
    print(X[0][-1])
