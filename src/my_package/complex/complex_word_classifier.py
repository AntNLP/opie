#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/12 22:20:04

@author: Changzhi Sun
"""
import os
import sys
import getopt

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.static import Static


class ComplexWordClassifier:

    def __init__(self, domain):
        self.domain = domain
        self.domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                                       "data/domains", domain)
        self.pickle_dir = os.path.join(self.domain_dir, "pickles")
        self.train_dir = os.path.join(self.domain_dir, "train")
        self.test_dir = os.path.join(self.domain_dir, "test")
        self.near_dir = os.path.join(self.domain_dir, "complex", "near")
        self.table_opinwd = set(Static.opinwd.keys())
        self.lexcion = {
            "unigram": {
                "word": {},
                "pos_tag": {},
                "word_pos_tag": {}
                },
            "bigram": {
                "word": {},
                "pos_tag": {},
                "word_pos_tag": {}
                }
            }
        self.featvect_item = {
            "unigram": {
                "word": [],
                "pos_tag": [],
                "word_pos_tag": []
                },
            "bigram": {
                "word": [],
                "pos_tag": [],
                "word_pos_tag": []
                }
            }
        self.train_label = []
        self.train_lens = []
        self.train_is_general_opinwd = []
        self.train_opinwd_str = []
        self.test_label = []
        self.test_lens = []
        self.test_is_general_opinwd = []
        self.test_opinwd_str = []

    def generate_featvect_item(self, filename, test=False):
        if test:
            self.label = self.test_label
            self.lens = self.test_lens
            self.is_general_opinwd = self.test_is_general_opinwd
            self.opinwd_str = self.test_opinwd_str
        else:
            self.label = self.train_label
            self.lens = self.train_lens
            self.is_general_opinwd = self.train_is_general_opinwd
            self.opinwd_str = self.train_opinwd_str
        with open(os.path.join(self.near_dir, filename),
                  "r", encoding="utf8") as out:
            for line in out:
                label, word, pos_tag = line.strip().split('\t')
                self.label.append(int(label))
                complex_word_list = word.split(' ')
                pos_tag_list = pos_tag.split(' ')
                self.lens.append(len(complex_word_list))
                self.create_unigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["unigram"], self.featvect_item["unigram"], 0, test)
                self.create_unigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["unigram"], self.featvect_item["unigram"], 1, test)
                self.create_unigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["unigram"], self.featvect_item["unigram"], 2, test)
                self.create_bigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["bigram"], self.featvect_item["bigram"], 0, test)
                self.create_bigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["bigram"], self.featvect_item["bigram"], 1, test)
                self.create_bigram(
                    complex_word_list, pos_tag_list,
                    self.lexcion["bigram"], self.featvect_item["bigram"], 2, test)
                m = False
                for w in complex_word_list:
                    if w in self.table_opinwd:
                        m = True
                self.is_general_opinwd.append(m)
                self.opinwd_str.append(word)

    def create_unigram(self,
                       word_list,
                       pos_tag_list,
                       lexcion,
                       featvect_item,
                       opt,
                       test=False):
        '''
        @opt: 0 -- word
            1 -- pos_tag
            2 -- word_pos_tag
        '''
        if opt == 0:
            lex, featvect_item = lexcion["word"], featvect_item["word"]
            target_list = word_list
        elif opt == 1:
            lex, featvect_item = lexcion["pos_tag"], featvect_item["pos_tag"]
            target_list = pos_tag_list
        elif opt == 2:
            lex = lexcion["word_pos_tag"]
            featvect_item = featvect_item["word_pos_tag"]
            target_list = []
            for i in range(len(word_list)):
                target_list.append("#".join([word_list[i], pos_tag_list[i]]))
        ret = {}
        for each in target_list:
            if not test:
                if each not in lex:
                    lex[each] = len(lex) + 1
                if lex[each] not in ret:
                    ret[lex[each]] = 0
                ret[lex[each]] += 1
            else:
                if each in lex:
                    if lex[each] not in ret:
                        ret[lex[each]] = 0
                    ret[lex[each]] += 1
        featvect_item.append(ret)

    def create_bigram(self,
                      word_list,
                      pos_tag_list,
                      lexcion,
                      featvect_item,
                      opt,
                      test=False):
        '''
        @opt: 0 -- word
            1 -- pos_tag
            2 -- word_pos_tag
        '''
        if opt == 0:
            lex, featvect_item = lexcion["word"], featvect_item["word"]
            target_list = ["#"]
            target_list.extend(word_list)
            target_list.append("#")
        elif opt == 1:
            lex, featvect_item = lexcion["pos_tag"], featvect_item["pos_tag"]
            target_list = ["#"]
            target_list.extend(pos_tag_list)
            target_list.append("#")
        else:
            lex = lexcion["word_pos_tag"]
            featvect_item = featvect_item["word_pos_tag"]
            target_list = ["#|#"]
            for i in range(len(word_list)):
                target_list.append("|".join([word_list[i], pos_tag_list[i]]))
            target_list.append("#|#")
        ret = {}
        for i in range(0, len(target_list)-1):
            each = " ".join(target_list[i:i+2])
            if not test:
                if each not in lex:
                    lex[each] = len(lex) + 1
                if lex[each] not in ret:
                    ret[lex[each]] = 0
                ret[lex[each]] += 1
            else:
                if each in lex:
                    if lex[each] not in ret:
                        ret[lex[each]] = 0
                    ret[lex[each]] += 1
        featvect_item.append(ret)

    def output_feature_vector(self, filename, have_general=True):
        with open(os.path.join(self.near_dir, filename),
                  "w", encoding="utf8") as out:
            for i in range(len(self.label)):
                if not have_general and self.is_general_opinwd[i]:
                    continue
                base = 0
                print("%d" % self.label[i], end="", file=out)
                base = self.write_feature_vector(
                    base, self.lexcion["unigram"]["word"],
                    self.featvect_item["unigram"]["word"][i], out)
                base = self.write_feature_vector(
                    base, self.lexcion["unigram"]["pos_tag"],
                    self.featvect_item["unigram"]["pos_tag"][i], out)
                base = self.write_feature_vector(
                    base, self.lexcion["unigram"]["word_pos_tag"],
                    self.featvect_item["unigram"]["word_pos_tag"][i], out)
                base = self.write_feature_vector(
                    base, self.lexcion["bigram"]["word"],
                    self.featvect_item["bigram"]["word"][i], out, True)
                base = self.write_feature_vector(
                    base, self.lexcion["bigram"]["pos_tag"],
                    self.featvect_item["bigram"]["pos_tag"][i], out, True)
                base = self.write_feature_vector(
                    base, self.lexcion["bigram"]["word_pos_tag"],
                    self.featvect_item["bigram"]["word_pos_tag"][i], out, True)
                print(" %d:%d" % (base+1, self.lens[i]), end="", file=out)
                base += 1
                print(file=out)
        self.opinwd_str = np.array(self.opinwd_str)
        self.is_general_opinwd = np.array(self.is_general_opinwd)

    def write_feature_vector(self,
                             base,
                             lexcion,
                             featvect_item,
                             out,
                             num=False):
        for key in sorted(featvect_item.keys()):
            if num:
                print(" %d:%d" % (base+key, featvect_item[key]),
                      end="", file=out)
            else:
                print(" %d:1" % (base+key), end="", file=out)
        return base + len(lexcion)


def usage():
    '''print help information'''
    print("complex_word_classifier.py 用法:")
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
    c = ComplexWordClassifier(domain)

    c.generate_featvect_item("complex_data.ann")
    c.output_feature_vector("train_feature_vector")

    c.generate_featvect_item("complex_pickle_test", True)
    c.output_feature_vector("pickle_test_feature_vector")

    X_train, y_train  = load_svmlight_file(os.path.join(c.near_dir, "train_feature_vector"))
    X_test, y_test = load_svmlight_file(os.path.join(c.near_dir, "pickle_test_feature_vector"))

    #  r = int(len(y) * 0.6)
    #  X_train, X_test = X[:r, :], X[r:, :]
    #  y_train, y_test = y[:r], y[r:]
    clf = LogisticRegression(C=1.0, intercept_scaling=1, dual=False,
            fit_intercept=True, penalty="l2", tol=0.0001)
    print("fit..")
    clf.fit(X_train, y_train)
    print("fit end...")
    y = clf.predict(X_test)
    #  y_rand = np.random.randint(0, 2, (len(y)))
    #  y = [1 if e == 0 else 0 for e in y]
    #  print("P", precision_score(y_test, y))
    #  print("R", recall_score(y_test, y))
    #  print("F", f1_score(y_test, y))
    #  print("P", precision_score(y_test, y_rand))
    #  print("R", recall_score(y_test, y_rand))
    #  print("F", f1_score(y_test, y_rand))
    f = open(os.path.join(c.domain_dir, "complex", "candidate_raw", "step"),
             "w", encoding="utf8")
    with open(os.path.join(c.near_dir, "complex_pickle_test"), "r", encoding="utf8") as out:
        i = 0
        for line in out:
            _, word, _ = line.strip().split('\t')
            if y[i]:
                print(word, file=f)
            i += 1
    f.close()
