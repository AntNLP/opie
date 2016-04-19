# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
import sys
import getopt
import xml.etree.ElementTree as etree

import nltk

from my_package.scripts import load_pickle_file, save_pickle_file
from my_package.scripts import mkdir


def add_parse_info(kk, parse_path, pickle_path):
    without_parse_path = os.path.join(pickle_path, "without_parse_sentences")
    parse_sentence_path = os.path.join(pickle_path, "parse_sentences")
    print(os.path.join(without_parse_path,
                       "without_parse_sentences_%d.pickle" % kk))
    sentences = load_pickle_file(os.path.join(without_parse_path,
                                 "without_parse_sentences_%d.pickle" % kk))
    index_of_sentences, k = 10 * (kk-1) + 1, 0
    parse_file = os.path.join(parse_path,
                              "sentences_%d.txt.xml" % index_of_sentences)
    while os.path.exists(parse_file):
        tree = etree.parse(parse_file)
        root = tree.getroot()
        i_sent = 1
        for sentence in root[0][0]:
            i_sent += 1
            tokens = {}
            pos_tag = {}
            id_index = 1
            for token in sentence[0]:
                tokens[id_index] = token[0].text
                pos_tag[id_index] = token[3].text
                id_index += 1
            dependency_tree = {}
            for d in sentence[2]:
                up, down = int(d[0].attrib['idx']), int(d[1].attrib['idx'])
                if up not in dependency_tree:
                    dependency_tree[up] = [{'id': down,
                                            'type': d.attrib['type']}]
                else:
                    temp = {'id': down, 'type': d.attrib['type']}
                    if temp not in dependency_tree[up]:
                        dependency_tree[up].append(temp)
            parse_string = sentence[1].text
            sentences[k].set_parse_info(tokens, pos_tag,
                                        parse_string, dependency_tree)
            k += 1
            if k % 1000 == 0:
                print("kk:", kk, "k:", k)
            if k == len(sentences):
                break
        if k == len(sentences):
            break
        index_of_sentences += 1
        parse_file = os.path.join(parse_path,
                                  "sentences_%d.txt.xml" % index_of_sentences)
    save_pickle_file(os.path.join(parse_sentence_path,
                                  "parse_sentences_%d.pickle" % kk), sentences)


def usage():
    '''打印帮助信息'''
    print("process_parse_data.py 用法:")
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
            domain = value
        if op in ("-p", "--part"):
            part_count = int(value)
    parse_path = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain, "parse")
    pickle_path = os.path.join(os.getenv("OPIE_DIR"),
                               "data/domains", domain, "pickles")
    mkdir(os.path.join(pickle_path, "parse_sentences"))
    pickle_size = len(os.listdir(os.path.join(pickle_path,
                                              "without_parse_sentences")))
    block_size = int(pickle_size / 8)
    start = (part_count - 1) * block_size + 1
    end = (pickle_size + 1) if part_count == 8 else (start + block_size)
    for i in range(start, end):
        add_parse_info(i, parse_path, pickle_path)
    print("end")
