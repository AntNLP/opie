#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/07/14 17:46:45

@author: Changzhi Sun
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_recall_curve
from relation import get_ann
from normalization import handle_normalize
from normalization import parse
from relation import get_ann
from relation import have_overlap
from my_package.scripts import load_pickle_file

def calcu_PRF(filename, sent_ann):
    TP_FP = 0
    TP = 0
    TP_TN = 0
    for e in parse(filename):
        text = e['S']
        TP_FP += len(e['R'])
        unique_set = set()
        if text in sent_ann:
            for rr in e['R']:
                r = (tuple(eval(rr[2])), tuple(eval(rr[3])))
                for ee in sent_ann[text]:
                    if have_overlap(r, ee):
                        unique_set.add(ee)
        TP += len(unique_set)
    for key, value in sent_ann.items():
        TP_TN += len(value)
    P = TP / TP_FP
    R = TP / TP_TN
    if P + R == 0:
        F = 0
    else:
        F = 2 * P * R / (P + R)
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

def fun(scores, sentences, ann_dict):
    precision, recall = [], []
    for t in np.linspace(0, 1, num=1500, endpoint=False):
        b = scores > t
        pred = np.array(b, dtype=np.int32)
        dump("tmp", sentences, pred)
        handle_normalize("tmp")
        p, r, f = calcu_PRF("tmp.normalize", ann_dict)
        precision.append(p)
        recall.append(r)
    return precision, recall

if __name__ == "__main__":
    domains = ["reviews_Grocery_and_Gourmet_Food",
               "reviews_Movies_and_TV",
               "reviews_Cell_Phones_and_Accessories",
               "reviews_Pet_Supplies"]
    names = ["Food", "Movie", "Phone", "Pet"]
    digit = [12, 12, 21, 11]

    while True:
        for i in range(3, 4):
            #  digit[i-1] = int(input())
            domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domains[i-1])
            relation_dir = os.path.join(domain_dir, "relation")
            test_dir = os.path.join(relation_dir, "test")
            print(digit[i-1])
            result = load_pickle_file(os.path.join(relation_dir, "add.result_%d.pickle" % digit[i-1]))
            sentences = result["sentences"]
            scores = result["scores"]
            print(scores.shape)
            ann_dict = get_ann(os.path.join(test_dir, "ann"))
            precision, recall = fun(scores, sentences, ann_dict)
            print(len(recall))
            plt.plot(recall, precision, label="NN($\gamma=0.8$)")

            #  scores = np.load(os.path.join(relation_dir, "without.scores.npy"))
            #  sentences = load_pickle_file(os.path.join(relation_dir, "without.sents.pickle"))
            scores = np.load(os.path.join(relation_dir, "scores.npy"))
            print(scores.shape)
            sentences = load_pickle_file(os.path.join(relation_dir, "sents.pickle"))
            precision, recall = fun(scores, sentences, ann_dict)
            x = [e for e in recall if e < 0.3]
            print(x)
            print(len(recall))
            plt.plot(recall, precision, label="LR($\gamma=0.8$)")
            #  plt.xlabel('Recall', fontsize=20)
            #  plt.ylabel('Precision', fontsize=18)
            #  plt.title(names[i-1], fontsize=25)
            #  plt.legend(loc="low left", numpoints=3)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(names[i-1])
            plt.legend(loc="lower left")
            plt.savefig(names[i-1]+".pdf")
            plt.show()
