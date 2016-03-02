# -*- coding: utf-8 -*-
'''
Created on 2015年8月28日

@author: Changzhi Sun
'''
import os
import gzip
import shutil
from my_package.class_define import SentenceTokenizer, Sentence
from my_package.scripts import create_content, save_pickle_file, save_json_file
import re
import sys, getopt

def parse_old(file_name):
    ''' 解析亚马逊某个领域的原始数据 '''
    with gzip.open(file_name) as file:
        entry = {}
        for line in file:
            line = line.decode("utf8")
            line = line.strip()
            colon_pos = line.find(":")
            if colon_pos == -1:
                yield entry
                entry = {}
                continue
            eName = line[:colon_pos]
            rest = line[colon_pos + 2:]
            entry[eName] = rest

def parse(path):
    ''' 解析亚马逊某个领域的原始数据 '''
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

def usage():
    '''打印帮助信息'''
    print("process_raw_data.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

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
            content = value
    print(content)
    raw_data_path = r"../../data/raw/domains"
    myTokenizer = SentenceTokenizer()
    mk_path = r"../../data/domains/" + content
    #  mk_path = r"../../data/soft_domains/" + content
    file_path = content + ".json.gz"
    create_content(mk_path)
    if os.path.exists(mk_path + r"/sentences"):
        shutil.rmtree(mk_path + r"/sentences")
    create_content(mk_path + r"/sentences")
    create_content(mk_path + r"/pickles")
    create_content(mk_path + r"/pickles/without_parse_sentences")

    sentences, i, k, review_index = [], 0, 1, 1
    kk = 1
    f = open(mk_path + r"/sentences/sentences_1.txt", "a", encoding="utf8")
    flag = False
    for e in parse(raw_data_path + r"/" + file_path):
        text, score = e['reviewText'], float(e['overall'])
        text = re.sub(r'[\x00-\x1f]', '', text) # 去除所有的控制字符，防止parse出错
        sents = myTokenizer.segment_text(text)
        for sent in sents:
            t = Sentence()
            t.set_text_score_review(sent, score, review_index)
            if len(sent.split(' ')) > 50:
                continue
            sentences.append(t)
            print(sent, file=f)
            if len(sentences) == 60000:  # 60000个句子序列化一次
                save_pickle_file(mk_path + r"/pickles/without_parse_sentences/without_parse_sentences_" +str(kk) + ".pickle", sentences)
                sentences = []
                kk += 1
            i += 1
            if i == 6000:
                f.close()
                k += 1
                i = 0
                flag = True
                #break # 现在只产生前面 6000 个句子
                f = open(mk_path + r"/sentences/sentences_"+str(k) + r".txt", "a", encoding="utf8")

        #if flag:
           # break
        review_index += 1
    f.close()
    if sentences != []:
        save_pickle_file(mk_path + r"/pickles/without_parse_sentences/without_parse_sentences_" +str(kk) + ".pickle", sentences)

