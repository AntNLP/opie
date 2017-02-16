#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/09/22 16:18:08

@author: Changzhi Sun
"""
import getopt
import sys
import os
from nltk.tokenize import word_tokenize
from my_package.static import Static
from my_package.scripts import load_pickle_file
from process_raw_data import parse

def extract_summary(path):
    print("extracting summary")
    summ_count = {}
    for e in parse(path):
        summ = e['summary'].lower().strip()
        flag = False
        for w in word_tokenize(summ):
            if w in Static.opinwd:
                flag = True
                break
        if not flag:
            if summ not in summ_count:
                summ_count[summ] = 0
            summ_count[summ] += 1
    return summ_count


def usage():
    '''print help information'''
    print("summary_output.py :")
    print("-h, --help: print help information")
    print("-d, --domain: domain name")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:", ["help", "domain"])
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
    raw_dpath = os.path.join(os.getenv("OPIE_DIR"), "data/raw/domains")
    domain_path = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
    multi_opin_expr_dir = os.path.join(domain_path, "multiopinexpr")
    summ_path = os.path.join(multi_opin_expr_dir, "summary_sort_by_freq")
    fname = domain + ".json.gz"
    MIN_FREQ = 0
    if not os.path.exists(summ_path):
        summ_count = extract_summary(os.path.join(raw_dpath, fname))
        f = open(summ_path, "w", encoding="utf8")
        print("print summary sort by freq to {0}".format(summ_path))
        for e in sorted(summ_count.items(), key=lambda x: x[1], reverse=True):
            print("{0}\t{1}".format(e[0], e[1]), file=f)
        f.close()
    summ_phrase_set = set()
    with open(summ_path, "r", encoding="utf8") as f:
        for line in f:
            phrase = line.split('\t')[0]
            ct = int(line.split('\t')[1])
            if ct <= MIN_FREQ:
                continue
            phrase = "^".join(phrase.split(' '))
            summ_phrase_set.add(phrase)
    print("summary phrase count(remove general and freq > {0}):".format(MIN_FREQ), len(summ_phrase_set))
    vocab_path = os.path.join(multi_opin_expr_dir, "word2vec", "vocab.txt")
    vocab_phrase_set = set()
    vocab_general_set = set()
    with open(vocab_path, "r", encoding="utf8") as f:
        for line in f:
            phrase = line.split(' ')[0]
            flag = False
            for w in phrase.split('^'):
                if w in Static.opinwd:
                    flag = True
                    break
            if not flag and len(phrase.split('^')) > 1:
                    vocab_phrase_set.add(phrase)
            if flag and len(phrase.split('^')) > 1:
                vocab_general_set.add(phrase)
    print("vocab phrase count(remove general):", len(vocab_phrase_set))
    f = open(os.path.join(multi_opin_expr_dir, "distant_phrase"), "w", encoding="utf8")
    diff_set = set(e for e in vocab_phrase_set if e not in summ_phrase_set)
    #  print("#########")
    #  print("vocab phrase set \ summ phrase set: ", len(diff_set))
    for e in sorted(diff_set):
        print("{0}\t0".format(e), file=f)
    diff_set = set(e for e in vocab_phrase_set if e in summ_phrase_set)
    #  print("#########")
    #  print("vocab phrase set & summ phrase set: ", len(diff_set))
    for e in sorted(diff_set):
        print("{0}\t1".format(e), file=f)
        #  print(e)
    #  diff_set = set(e for e in summ_phrase_set if e not in vocab_phrase_set)
    #  print("#########")
    #  print("summary phrase set & vocab phrase set: ", len(diff_set))
    #  for e in diff_set:
        #  print(e)
    for e in sorted(vocab_general_set):
        print("{0}\t2".format(e), file=f)
    f.close()
