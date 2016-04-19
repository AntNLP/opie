# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
import re
import sys
import getopt
import cProfile
from timeit import timeit
from nltk.corpus import stopwords
from collections import Counter

import pymysql

from my_package.class_define import Static
from my_package.scripts import load_pickle_file, return_none
from my_package.scripts import save_pickle_file, save_json_file
from my_package.scripts import create_content


def train_feature_solve(field_content, lexcion,
                        sentences, connection, table_lm, sentiments):
    it = 0
    for sentence in sentences:
        sentence.generate_candidate(sentiments, connection, table_lm)
        sentence.generate_candidate_feature_vector(lexcion)
        sentence.generate_train_label()
        if it % 1000 == 0:
            print(it)
        it += 1


def extract_feature_vector(sentences, lexcion, f, g):

    k = 0
    for sentence in sentences:
        print("{0}:{1}".format(k, sentence.text), file=g)
        if k % 1000 == 0:
            print("k=", k)
        k += 1
        sentence.generate_feature_vector(lexcion)
        for i in range(len(sentence.feature_vector)):
            print(sentence.all_match_label[i], end="", file=f)
            for e in sentence.feature_vector[i]:
                print(" {0}:1".format(e), end="", file=f)
            print(file=f)
            feat = sentence.candidate_pairs[i][0]
            sent = sentence.candidate_pairs[i][1]
            if sentence.all_match_label[i] != 0:
                print("{0}\t\t{1}".format(
                    sentence.get_phrase(feat).lower(),
                    sentence.get_phrase(sent).lower()),
                    file=g)
        print(file=g)


def extract_test_feature_vector(content, connection, table_lm, sentiments):
    field_content = r"../../data/soft_domains/" + content + r"/"
    sentences = load_pickle_file(field_content+r"test/test_sentences.pickle")
    sentences = sentences[:4000]
    lexcion = load_pickle_file(field_content + "pickles/lexicon.pickle")
    print(len(sentences))
    it = 0
    for sentence in sentences:
        sentence.generate_candidate(sentiments,
                                    connection, table_lm, test=True)
        sentence.generate_candidate_feature_vector(lexcion, test=True)
        sentence.generate_test_label()
        if it % 100 == 0:
            print(it)
        it += 1
    save_pickle_file(field_content + r"test/feature_vector_sentences.pickle",
                     sentences)
    f = open(field_content + r"test/test_candidate", "w", encoding="utf8")
    out = open(field_content + r"test/feature_vectors",
               mode="w", encoding="utf8")
    for sentence in sentences:
        sentence.generate_feature_vector(lexcion)
        print(sentence.text, file=f)
        #  print(sentence.text_and_pos_tag)
        for i in range(len(sentence.feature_vector)):
            print(sentence.candidate_pairs[i], end="   ", file=f)
            print(sentence.get_phrase(sentence.candidate_pairs[i][0]),
                  end="   ", file=f)
            print(sentence.get_phrase(sentence.candidate_pairs[i][1]), file=f)
            print(sentence.all_match_label[i], end="", file=out)
            for e in sentence.feature_vector[i]:
                print(" {}:1".format(e), end="", file=out)
            print("", file=out)
        print("\n", file=f)
    out.close()
    f.close()


def get_count(str1, str2, field_content):
    c = 0
    with open(field_content+"text", "r", encoding="utf8") as f:
        for line in f:
            if re.search(str1, line) and re.search(str2, line):
                c += 1
    return c


def usage():
    '''打印帮助信息'''
    print("extract_feature.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-b, --begin: bootstrap_sentences pickel 文件的开始编号(包含此文件)")
    print("-e, --end: bootstrap_sentences pickel 文件的结束编号(不包含此文件)")

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
            content = value
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)
    field_content = r"../../data/soft_domains/" + content + r"/"
    table_lm = content + "_lm"
    connection = pymysql.connect(host="console",
                                 user="u20130099",
                                 passwd="u20130099",
                                 db="u20130099",
                                 charset="utf8",
                                 cursorclass=pymysql.cursors.DictCursor)
    create_content(field_content + r"train")
    create_content(field_content + r"pickles/feature_vectors")
    lexcion = {
        "unigram": {
            "word":{},
            "pos_tag":{},
            "pos_tags":{},
            "joint_pos_tag":{},
            "word_pos_tag":{},
            "dep":{}
            }
        }
    sentiments = set(Static.sentiment_word.keys())
    i = b
    while i < e and os.path.exists(
            field_content +
            "pickles/bootstrap_sentences/bootstrap_sentences_" +
            str(i) + ".pickle.bz2"):
        print("loading")
        sentences = load_pickle_file(
            field_content +
            "pickles/bootstrap_sentences/bootstrap_sentences_" +
            str(i) + ".pickle")
        print("loaded")
        print(len(sentences))
        train_feature_solve(field_content, lexcion,
                            sentences, connection, table_lm, sentiments)
        save_pickle_file(field_content +
                         "pickles/feature_vectors/sentences_" + str(i) +
                         ".pickle", sentences)
        i += 1

    save_pickle_file(field_content + "pickles/lexicon.pickle", lexcion)
    save_json_file(field_content + "pickles/lexicon.json", lexcion)
    print("word:", len(lexcion['unigram']['word']))
    print("pos_tag:", len(lexcion['unigram']['pos_tag']))
    print("word_pos_tag:", len(lexcion['unigram']['word_pos_tag']))
    print("dep:", len(lexcion['unigram']['dep']))
    print("joint_pos_tag:", len(lexcion["unigram"]["joint_pos_tag"]))
    i = b
    f = open(field_content + r"train/raw_all_match_feature_vectors",
             mode="w", encoding="utf8")
    g = open(field_content + r"train/train_candidate",
             mode="w", encoding="utf8")
    while i < e and os.path.exists(field_content +
                                   "pickles/feature_vectors/sentences_" +
                                   str(i) + ".pickle.bz2"):
        sentences = load_pickle_file(field_content +
                                     "pickles/feature_vectors/sentences_" +
                                     str(i) + ".pickle")
        extract_feature_vector(sentences, lexcion, f, g)
        i += 1
    f.close()
    g.close()
    extract_test_feature_vector(content, connection, table_lm, sentiments)
    connection.close()
    print("end")
