# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
from collections import defaultdict
from my_package.scripts import load_pickle_file, save_pickle_file, return_none,\
    save_json_file, create_content
import xml.etree.ElementTree as etree
import sys, getopt

def add_parse_info(kk, parse_content, pickle_content):
    ''''''
    print("kk", kk)
    without_parse_path = pickle_content + "without_parse_sentences/"
    parse_path = pickle_content + "parse_sentences/"
    sentences = load_pickle_file(without_parse_path+ r"without_parse_sentences_" + str(kk) + ".pickle")
    index_of_sentences, k = 10 * (kk-1) + 1, 0
    parse_file = parse_content +"sentences_" + str(index_of_sentences) +".txt.xml"
    print(parse_file)
    print(os.path.exists(parse_file))
    while os.path.exists(parse_file):
        tree = etree.parse(parse_file)
        root = tree.getroot()
        i_sent = 1
        for sentence in root[0][0]:
            i_sent += 1
            tokens, pos_tag = defaultdict(return_none), defaultdict(return_none)
            id_index = 1
            for token in sentence[0]:
                tokens[id_index], pos_tag[id_index] = token[0].text, token[3].text
                id_index += 1

            dependency_tree = defaultdict(return_none)
            for d in sentence[2]:
                up, down = int(d[0].attrib['idx']), int(d[1].attrib['idx'])
                if dependency_tree[up] == None:
                    dependency_tree[up] = [{'id': down, 'type': d.attrib['type']}]
                else:
                    temp = {'id': down, 'type': d.attrib['type']}
                    if temp not in dependency_tree[up]:
                        dependency_tree[up].append(temp)
            parse_string = sentence[1].text
            sentences[k].set_parse_info(tokens, pos_tag, parse_string, dependency_tree)
            k += 1
            if k % 1000 == 0:
                print("kk:", kk, "k:", k)
            if k == len(sentences):
                break
        if k == len(sentences):
            break;
        index_of_sentences += 1
        parse_file = parse_content +"sentences_" + str(index_of_sentences) +".txt.xml"
    save_pickle_file(parse_path + r"parse_sentences_" + str(kk) + ".pickle", sentences)

def usage():
    '''打印帮助信息'''
    print("process_parse_data.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-p, --part: 每个领域句子均分为 8 份, 该参数指定 parse 哪部分(1,2, ..., 8)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:p:", ["help", "domain=", "part="])
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
    parse_content = r"../../data/soft_domains/" + content + r"/parse/"
    pickle_content = r"../../data/soft_domains/" + content + r"/pickles/"
    parse_path = pickle_content + "parse_sentences/"

    pickle_size = len(os.listdir(pickle_content + "without_parse_sentences"))
    print("pickle_size:", pickle_size)
    block_size = int(pickle_size / 8)
    create_content(parse_path)
    aa = (part_count - 1) * block_size + 1
    bb = (pickle_size + 1) if part_count == 8 else (aa + block_size)
    for i in range(aa, bb):
        add_parse_info(i, parse_content, pickle_content)
    print("end")
