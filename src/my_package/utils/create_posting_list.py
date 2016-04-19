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

from my_package.scripts import save_json_file, create_content, load_json_file
from my_package.scripts import load_pickle_file, return_none, save_pickle_file
from my_package.class_define import Static


def get_pp_index(sentence, var, pos_tag):
    ret_set = set()
    for key, values in var.items():
        if sentence.pos_tag[key] in pos_tag:
            for value in values:
                ret_set |= set(value)
    return ret_set


def get_posting_list(i, pickle_content, post_list):
    filename = (pickle_content +
                "parse_sentences/parse_sentences_" + str(i) + ".pickle")
    if os.path.exists(filename+".bz2"):
        sentences = load_pickle_file(filename)
        print(filename)
        k = 0
        for sentence in sentences:
            if "dictionary_of_adjp" in dir(sentence):
                adjp = sentence.dictionary_of_adjp
            else:
                adjp = sentence.get("ADJP")
            i_set = set()
            i_set |= get_pp_index(sentence, adjp, Static.JJ)
            i_set |= get_pp_index(sentence,
                                  sentence.dictionary_of_np, Static.NN)
            i_set |= get_pp_index(sentence,
                                  sentence.dictionary_of_vp, Static.VB)
            for e in i_set:
                word = sentence.tokens[e].lower()
                if word not in post_list:
                    post_list[word] = {}
                if i not in post_list[word]:
                    post_list[word][i] = {}
                if k not in post_list[word][i]:
                    post_list[word][i][k] = []
                post_list[word][i][k].append(e)
            k += 1
    else:
        print(filename + "not exists!")


def usage():

    '''打印帮助信息'''
    print("create_posting_list.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-p, --part: 每个领域句子均分为 8 份, 该参数指定 parse 哪部分(1,2, ..., 8)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:p:",
                                   ["help", "domain=", "part="])
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
        if op in ("-p", "--part"):
            part_count = int(value)
    print(content)
    pickle_content = r"../../data/domains/" + content + r"/pickles/"
    pickle_size = len(os.listdir(pickle_content + "without_parse_sentences"))
    block_size = int(pickle_size / 8)
    aa = (part_count - 1) * block_size + 1
    bb = (pickle_size + 1) if part_count == 8 else (aa + block_size)

    post_list = {}
    for i in range(aa, bb):
        get_posting_list(i, pickle_content, post_list)
    #  save_pickle_file(pickle_content+"post_list_"+str(part_count)+".pickle", post_list)
    #  save_json_file(pickle_content+"post_list_"+str(part_count)+".json", post_list)
    load_pickle_file(pickle_content+"post_list_"+str(part_count)+".pickle")
    print("end")
