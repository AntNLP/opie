#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/05 19:10:25

@author: Changzhi Sun
"""
import configparser
import os

from nltk.corpus import stopwords

from my_package.scripts import load_file_line


def get_wpath(wname):
    '''read config file'''
    cf = configparser.ConfigParser()
    cf.read(os.path.join(os.getenv("OPIE_DIR"), "my.cnf"))
    return abs_path(cf.get("word", wname))


def abs_path(fpath):
    return os.path.join(os.getenv("OPIE_DIR"), fpath)


class Static:

    ADJ = set(["JJ", "JJR", "JJS"])

    NOUN = set(["NN", "NNS", "NNP", "NNPS"])

    VERB = set(["VB", "VBZ", "VBD", "VBG", "VBN", "VBP"])

    ADV = set(["RB", "RBR", "RBS"])

    SENTIMENT = VERB | ADV | ADJ

    BE = set(load_file_line(get_wpath("BE")))

    tersignal = set(load_file_line(get_wpath("terminal_signal")))

    weak_opinwd = set(load_file_line(get_wpath("weak_opinion_words")))

    posword = set(load_file_line(get_wpath("positive_words")))

    negword = set(load_file_line(get_wpath("negative_words")))

    weak_profeat = set(load_file_line(get_wpath("weak_product_feature")))

    opinwd = {e: 1 for e in posword}

    opinwd.update({e: -1 for e in negword})

    stopwords = set(stopwords.words('english'))

    def __init__(self):
        pass

if __name__ == "__main__":
    print(Static.opinword)
    print("end")
