# -*- coding: utf-8 -*-
'''
Created on 2015年8月28日

@author: Changzhi Sun
'''
import os
import gzip
import shutil
import re
import sys
import getopt

from my_package.sentence import Sentence
from sentence_tokenizer import SentenceTokenizer
from my_package.scripts import mkdir, save_pickle_file, save_json_file


def parse(path):
    ''' 解析亚马逊某个领域的原始数据 '''
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def usage():
    '''print help information'''
    print("process_raw_data.py :")
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
    pickle_head = "/pickles/without_parse_sentences/without_parse_sentences_"
    fname = domain + ".json.gz"
    mkdir(domain_path)
    spath = os.path.join(domain_path, "sentences")
    if os.path.exists(spath):
        shutil.rmtree(spath)
    mkdir(os.path.join(domain_path, "sentences"))
    mkdir(os.path.join(domain_path, "pickles"))
    mkdir(os.path.join(domain_path, "pickles/without_parse_sentences"))
    sentences, i, k, review_index = [], 0, 1, 1
    kk = 1
    f = open(domain_path + "/sentences/sentences_1.txt", "a", encoding="utf8")
    flag = False
    myTokenizer = SentenceTokenizer()
    for e in parse(os.path.join(raw_dpath, fname)):
        text, score = e['reviewText'], float(e['overall'])
        # 去除所有的控制字符，防止parse出错
        text = re.sub(r'[\x00-\x1f]', '', text)
        sents = myTokenizer.segment_text(text)
        for sent in sents:
            t = Sentence()
            t.set_text_score_review(sent, score, review_index)
            if len(sent.split(' ')) > 50:
                continue
            sentences.append(t)
            print(sent, file=f)
            # 60000个句子序列化一次
            if len(sentences) == 60000:
                save_pickle_file(domain_path + pickle_head +
                                 str(kk) + ".pickle", sentences)
                sentences = []
                kk += 1
            i += 1
            if i == 6000:
                f.close()
                k += 1
                i = 0
                flag = True
                f = open(domain_path + "/sentences/sentences_" +
                         str(k) + ".txt", "a", encoding="utf8")
        review_index += 1
    f.close()
    if sentences:
        save_pickle_file(domain_path + pickle_head + str(kk) + ".pickle",
                         sentences)
