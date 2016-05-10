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

    def __init__(self, domain, mysql_db=None):
        self.table_opinwd = set(Static.opinwd.keys())
        self.mysql_db = mysql_db
        self.domain = domain
        self.domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                                       "data/domains", domain)
        self.pickle_dir = os.path.join(self.domain_dir, "pickles")
        self.pickle_btsp = os.path.join(self.pickle_dir, "bootstrapping")
        self.pickle_featvect = os.path.join(self.pickle_dir, "feature_vector")
        self.train_dir = os.path.join(self.domain_dir, "train")
        self.test_dir = os.path.join(self.domain_dir, "test")
        self.test_path = os.path.join(self.test_dir, "sentences.pickle")
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

    def handle_sentence(self, sentence, test):
        sentence.generate_candidate_relation(self.table_opinwd,
                                             self.mysql_db, test)
        sentence.generate_candidate_featvect_item(self.lexcion, test)
        sentence.generate_label(test)

    def handle_sentences(self, sentences, test=False):
        print("handle sentences")
        i = 1
        for sentence in sentences:
            self.handle_sentence(sentence, test)
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
            for e in featvect:
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

    def run_test(self, sentences):
        print("test")
        remove(os.path.join(self.test_dir, "candidates"))
        remove(os.path.join(self.test_dir, "feature_vector"))
        #  sentences = sentences[:4000]
        self.handle_sentences(sentences, test=True)
        self.output_sentences_feature_vector(sentences, self.test_dir)

def usage():
    '''print help information'''
    print("relation_classifier.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
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
    r.run_test()
