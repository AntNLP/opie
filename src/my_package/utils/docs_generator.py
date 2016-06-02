#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/05/10 16:46:35

@author: Changzhi Sun
"""
import os
import sys
import getopt
import configparser

from my_package.scripts import mkdir
from my_package.scripts import load_pickle_file


def usage():
    '''print help information'''
    print("docs_generator.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-e, --end: end pickle")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:e:",
                                   ["help", "domain=", "end="])
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
        if op in ("-e", "--end"):
            end = int(value)
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data/domains", domain)
    pickle_without = os.path.join(domain_dir,
                                  "pickles/without_parse_sentences")
    mkdir(os.path.join(domain_dir, "docs"))
    f = open(os.path.join(domain_dir, "docs/text"), "w", encoding="utf8")
    i = 1
    filename = os.path.join(pickle_without,
                            "without_parse_sentences_%d.pickle" % i)
    while i < end and os.path.exists(filename + ".bz2"):
        print("pickle index: ", i)
        sentences = load_pickle_file(filename)
        for j, sentence in enumerate(sentences):
            print("%d-%d\t%s" % (i, j, sentence.text), file=f)
        i += 1
        filename = os.path.join(pickle_without,
                                "without_parse_sentences_%d.pickle" % i)
    f.close()
