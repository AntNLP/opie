# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
import re
import sys
import getopt
from collections import Counter
from itertools import chain

import pymysql

from my_package.scripts import load_pickle_file, return_none, save_pickle_file
from my_package.scripts import save_json_file, create_content, inquire_content
from my_package.scripts import have_overlap, have_dependent, filter_word
from my_package.class_define import Static


def f1(sentence, i, j, m, connection, table_name, mark, ff):
    if sentence.pos_tag[m] in Static.NN:
        np = sentence.get_np(m, i)
        if have_overlap(np, list(range(i, j))):
            return False
        if filter_word(sentence, np):
            return False
        new_feature_string = sentence.get_phrase(np).lower()

        if sentence.is_weak_feature(new_feature_string):
            return False
        #  语言模型过滤
        if (inquire_content(connection, new_feature_string, table_name) and
                have_dependent(sentence.dependency_tree,
                               np, list(range(i, j)))):
            if mark[0] == False:
                print("S\t{0}".format(sentence.text), file=ff)
                mark[0] = True
            key, value = np, list(range(i, j))
            print("R\t{0}\t{1}\t{2}\t{3}".format(
                sentence.get_phrase(key).lower(),
                sentence.get_phrase(value).lower(),
                key, value), file=ff)
            return True

    elif sentence.pos_tag[m] in Static.VB:
        vp = sentence.get_vp(m, i)

        if have_overlap(vp, list(range(i, j))):
            return False
        if filter_word(sentence, vp):
            return False
        new_feature_string = sentence.get_phrase(vp).lower()

        if sentence.is_weak_feature(new_feature_string):
            return False

        #  语言模型过滤
        if (inquire_content(connection, new_feature_string, table_name) and
                have_dependent(sentence.dependency_tree,
                               vp, list(range(i, j)))):
            if mark[0] == False:
                print("S\t{0}".format(sentence.text), file=ff)
                mark[0] = True
            key, value = vp, list(range(i, j))
            print("R\t{0}\t{1}\t{2}\t{3}".format(
                sentence.get_phrase(key).lower(),
                sentence.get_phrase(value).lower(),
                key,
                value), file=f)
            return True
    return False

def extract_relation(sentence, word_lists, connection, table_name, ff, w=5):
    n = len(sentence.pos_tag)
    for i in range(1, n+1):
        mark = [False]
        for word_list in word_lists:
            j, k = i, 0
            while k < len(word_list) and sentence.tokens[j] == word_list[k]:
                j += 1
                k += 1
            if k == len(word_list):
                obj_index = list(range(i, j))
                b = i - w if i - w >= 1 else 1
                e = j + w if j + w <= n + 1 else n + 1
                for m in range(i, b-1, -1):
                    if f1(sentence, i, j, m, connection, table_name, mark, ff):
                        print(word_list)
                        break
                for m in range(j, e):
                    if f1(sentence, i, j, m, connection, table_name, mark, ff):
                        print(word_list)
                        break

def usage():

    '''打印帮助信息'''
    print("find_near_word.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    #  t = Trie_Tree()
    #  t.build_suffix_tree(["a", "b", "c"])
    #  t.build_suffix_tree(["c", "b", "c"])

    #  print(t.inquire_word_string_count(["b", "c"]))


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
            content = value
    #  field_content = r"../../data/domains/" + content + r"/"
    field_content = r"../../data/soft_domains/" + content + r"/"
    create_content(field_content + "near")
    sentiment_dict = dict(Static.sentiment_word)
    sentiments = set(sentiment_dict.keys())
    word_lists = [["this", "bad"], ["wonderfully", "enjoyable"]]
    connection = pymysql.connect(host="localhost",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    table_name = content + "_lm"
    f = open(field_content+"near/complex_word_apply", "w", encoding="utf8")
    i = 1
    filename = (field_content +
               "pickles/parse_sentences/parse_sentences_%d.pickle"%i)
    while os.path.exists(filename+".bz2"):
        print(filename)
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            extract_relation(sentence, word_lists, connection, table_name, f)
        if i == 5:
            break
        i += 1
        filename = (field_content +
                   "pickles/parse_sentences/parse_sentences_%d.pickle"%i)
    connection.close()
    f.close()
