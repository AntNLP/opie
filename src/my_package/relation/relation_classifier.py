#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/10 17:44:42

@author: Changzhi Sun
"""
import configparser
import os
import sys
import getopt

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import remove
from my_package.scripts import mkdir
from my_package.static import Static


class RelationClassifier:

    def __init__(self, domain, mysql_db=None, complex_file="0.8-complex"):
        self.table_opinwd = set(Static.opinwd.keys())
        self.mysql_db = mysql_db
        self.domain = domain
        self.domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                                       "data/domains", domain)
        self.pickle_dir = os.path.join(self.domain_dir, "pickles")
        self.pickle_btsp = os.path.join(self.pickle_dir, "bootstrapping")
        self.pickle_featvect = os.path.join(self.pickle_dir, "feature_vector")
        self.train_dir = os.path.join(self.domain_dir, "relation", "train")
        self.test_dir = os.path.join(self.domain_dir, "relation", "test")
        self.test_path = os.path.join(self.test_dir, "sentences.pickle")
        self.complex_dir = os.path.join(self.domain_dir, "complex")
        self.complex_opinwd_train = self.get_complex_opinwd(
            os.path.join(self.complex_dir,
                         "candidate_clean",
                         "%s.train" % complex_file))
        self.complex_opinwd_test = self.get_complex_opinwd(
            os.path.join(self.complex_dir,
                         "candidate_clean",
                         "%s.test" % complex_file))
        self.lexcion = {
            "unigram": {
                "word": {},
                "pos_tag": {},
                "pos_tags": {},
                "joint_pos_tag": {},
                "word_pos_tag": {},
                "dep": {}
                }
            }
        mkdir(self.train_dir)
        mkdir(self.test_dir)
        mkdir(self.pickle_featvect)

    def get_complex_opinwd(self, filename):
        if not os.path.exists(filename):
            print("%s not exists" % filename)
            return None
        print("%s exists" % filename)
        complex_opinwd = set()
        with open(filename, "r", encoding="utf8") as f:
            for line in f:
                complex_opinwd.add(line.strip())
        return complex_opinwd

    def handle_sentence(self, sentence, test,
                        complex_opinwd_train, complex_opinwd_test):
        if not test:
            sentence.generate_candidate_relation(
                self.table_opinwd,
                self.mysql_db,
                complex_opinwd=complex_opinwd_train,
                test=test)
        else:
            sentence.generate_candidate_relation(
                self.table_opinwd,
                self.mysql_db,
                complex_opinwd=complex_opinwd_test,
                test=test)
        sentence.generate_candidate_featvect_item(self.lexcion, test)
        sentence.generate_label(test)

    def handle_sentences(self, sentences, use_complex, test=False):
        print("handle sentences")
        if use_complex:
            print("USE COMPLEX WORD")
            complex_opinwd_train = self.complex_opinwd_train
            complex_opinwd_test = self.complex_opinwd_test
        else:
            print("WITHOUT USE COMPLEX WORD")
            complex_opinwd_train = None
            complex_opinwd_test= None
        i = 1
        for sentence in sentences:
            self.handle_sentence(sentence, test,
                                 complex_opinwd_train,
                                 complex_opinwd_test)
            if i % 1000 == 0:
                print("sentence index: %d" % i)
            i += 1

    def output_sentence_feature_vector(self, sentence, f, g):
        featvects = sentence.generate_candidate_featvect(self.lexcion)
        print(sentence.text, file=f)
        for i, featvect in enumerate(featvects):
            print(sentence.candidate_relation[i], end="\t\t", file=f)
            print(sentence.print_phrase(sentence.candidate_relation[i][0]),
                  end="\t\t",
                  file=f)
            print(sentence.print_phrase(sentence.candidate_relation[i][1]),
                  file=f)
            print(sentence.label[i], end="", file=g)
            for e in sorted(set(featvect)):
                print(" {}:1".format(e), end="", file=g)
            print("", file=g)
        print("\n", file=f)

    def output_sentences_feature_vector(self, sentences, output_dir):
        f = open(os.path.join(output_dir, "candidates"), "a", encoding="utf8")
        g = open(os.path.join(output_dir, "feature_vector"),
                 "a", encoding="utf8")
        for sentence in sentences:
            self.output_sentence_feature_vector(sentence, f, g)
        f.close()
        g.close()

    def run_test(self, sentences, use_complex):
        remove(os.path.join(self.test_dir, "candidates"))
        remove(os.path.join(self.test_dir, "feature_vector"))
        self.handle_sentences(sentences, use_complex, test=True)
        self.output_sentences_feature_vector(sentences, self.test_dir)

def usage():
    '''print help information'''
    print("relation_classifier.py 用法:")
    print("-h, --help: help information")
    print("-d, --domain: the domain's name")
    print("-t, --test: only for test set")
    print("-b, --begin: bootstrapping pickel 文件的开始编号(包含此文件)")
    print("-e, --end: bootstrapping pickel 文件的结束编号(不包含此文件)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], "htd:b:e:",
            ["help", "test", "domain=", "begin=", "end="])
    except getopt.GetoptError:
        print("命令行参数输入错误！")
        usage()
        sys.exit(1)
    test = False
    for op, value in opts:
        if op in ("-h", "--help"):
            usage()
            sys.exit()
        if op in ("-d", "--domain"):
            domain = value
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)
        if op in ("-t", "--test"):
            test = True
    r = RelationClassifier(domain)
    if test:
        r = load_pickle_file(os.path.join(r.pickle_dir,
                                          "relation_classifier.pickle"))
        sentences = load_pickle_file(r.test_path)
        r.run_test(sentences)
        sys.exit()
    i = b
    spath = os.path.join(r.pickle_btsp,
                         "bootstrapping_sentences_%d.pickle" % i)
    while i < e and os.path.exists(spath + ".bz2"):
        print("pickle index: %d loading" % i)
        sentences = load_pickle_file(spath)
        print("pickle index: %d loaded" % i)
        r.handle_sentences(sentences)
        save_pickle_file(
            os.path.join(r.pickle_featvect,
                         "feature_vector_sentences_%d.pickle" % i),
            sentences)
        i += 1
        spath = os.path.join(r.pickle_btsp,
                             "bootstrapping_sentences_%d.pickle" % i)
    remove(os.path.join(r.train_dir, "candidates"))
    remove(os.path.join(r.train_dir, "feature_vector"))
    i = b
    spath = os.path.join(r.pickle_featvect,
                         "feature_vector_sentences_%d.pickle" % i)
    while i < e and os.path.exists(spath + ".bz2"):
        print("pickle index: %d loading" % i)
        sentences = load_pickle_file(spath)
        print("pickle index: %d loaded" % i)
        r.output_sentences_feature_vector(sentences, r.train_dir)
        i += 1
        spath = os.path.join(r.pickle_featvect,
                             "feature_vector_sentences_%d.pickle" % i)
    save_pickle_file(
        os.path.join(r.pickle_dir, "relation_classifier.pickle"), r)
    r.run_test(sentences)
