#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/07/11 15:01:11

@author: Changzhi Sun
"""
import os
import sys
import getopt
import configparser

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score

from my_package.static import Static
from my_package.static import get_wpath
from my_package.scripts import mkdir
from my_package.scripts import remove
from normalization import parse
from normalization import handle_normalize
from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from pattern_extractor import PatternExtractor
from relation_classifier import RelationClassifier
from pattern_extractor import write
from relation import get_ann, calcu_PRF

def usage():
    '''print help information'''
    print("pattern_extractor.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:",
                                   ["help", "domain=", "begin=", "end="])
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
    b = 1
    e = 10000
    r = PatternExtractor(domain)
    # pattern extractor train
    #  print("########## pattern extractor for train ##########")
    #  f = open(os.path.join(r.btsp_dir, "bootstrapping.relation"),
             #  "w", encoding="utf8")
    #  i = b
    #  spath = os.path.join(r.pickle_parse, "parse_sentences_%d.pickle" % i)
    #  while i < e and os.path.exists(spath + ".bz2"):
        #  print("pickle index: %d  loading" % i)
        #  sentences = load_pickle_file(spath)
        #  print("pickle index: %d  loaded" % i)
        #  r.handle_sentences(sentences)
        #  write(sentences, f)
        #  save_pickle_file(
            #  os.path.join(r.pickle_btsp,
                         #  "bootstrapping_sentences_%d.pickle" % i),
            #  [e for e in sentences if e.relation])
        #  i += 1
        #  spath = os.path.join(r.pickle_parse, "parse_sentences_%d.pickle" % i)
    #  f.close()
    #  save_pickle_file(os.path.join(r.domain_dir, "pickles", "pattern_extractor.pickle"), r)
    r = load_pickle_file(os.path.join(r.domain_dir, "pickles", "pattern_extractor.pickle"))
    test_dir = os.path.join(r.domain_dir, "relation", "test")
    sentences = load_pickle_file(os.path.join(test_dir, "sentences.pickle"))

    r.handle_sentences(sentences)
    sent_ann = get_ann(os.path.join(test_dir, "ann"))
    with open(os.path.join(test_dir, "relation.pattern.bootstrapping"), "w", encoding="utf8") as f:
        write(sentences, f)
    handle_normalize(os.path.join(test_dir, 'relation.pattern.bootstrapping'))
    calcu_PRF(os.path.join(test_dir, 'relation.pattern.bootstrapping.normalize'), sent_ann)
