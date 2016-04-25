#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/21 16:05:54

@author: Changzhi Sun
"""
import os
import sys
import getopt


from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import mkdir


def usage():
    '''print help information'''
    print("test_generator.py 用法:")
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain)
    test_dir = os.path.join(domain_dir, "test")
    review_dir = os.path.join(test_dir, "reviews")
    mkdir(review_dir)
    print("test pickle loading")
    raw_sentences = load_pickle_file(os.path.join(test_dir,
                                              "sentences_raw.pickle"))
    #  raw_sentences = raw_sentences[29420:]
    sentences = []
    print("test pickle loaded")
    first_review_index = raw_sentences[0].review_index
    review_index = 0
    sentence_count = 0
    i = 0
    while i < len(raw_sentences):
        if raw_sentences[i].review_index != first_review_index:
            current_review_index = raw_sentences[i].review_index
            review_index += 1
            f = open(os.path.join(review_dir, "review_%d.txt" % review_index),
                    "w", encoding="utf8")
            while (i < len(raw_sentences) and
                    raw_sentences[i].review_index == current_review_index):
                sentence = raw_sentences[i]
                sentences.append(sentence)
                print(sentence.text, file=f)
                #  print(sentence.text)
                i += 1
                sentence_count += 1
            f.close()
            if sentence_count >= 1000:
                print(sentence_count)
                break
        else:
            i += 1
    save_pickle_file(os.path.join(test_dir, "sentences.pickle"), sentences)
    print(len(sentences))
