# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
from my_package.scripts import load_pickle_file, return_none, save_pickle_file, save_json_file, create_content, load_json_file
import os
from collections import Counter
from my_package.class_define import Static
import sys, getopt


def get_sentiment_count(sentence, sentiments):
        sentiment_count = 0
        i = 1
        while sentence.tokens.get(i) != None:
            if sentence.tokens[i].lower() in sentiments:
                sentiment_count += 1
            i += 1
        return sentiment_count


def pos_and_num(i, pickle_content, positives, negatives, out):
    filename = pickle_content + "parse_sentences/parse_sentences_" + str(i) + ".pickle"
    if os.path.exists(filename+".bz2"):
        sentences = load_pickle_file(filename)
        print(filename)
        k = 0
        for sentence in sentences:
            if k % 1000 == 0:
                print(k)
            print("%d\t%d\t%d\t%d\t%d"%(i,
                k,
                get_sentiment_count(sentence, positives),
                get_sentiment_count(sentence, negatives),
                sentence.review_index), file=out)
            k += 1

def usage():

    '''打印帮助信息'''
    print("get_pos_num.py 用法:")
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
    pickle_content = r"../../data/soft_domains/" + content + r"/pickles/"
    pickle_size = len(os.listdir(pickle_content + "without_parse_sentences"))
    block_size = int(pickle_size / 8)
    aa = (part_count - 1) * block_size + 1
    bb = (pickle_size + 1) if part_count == 8 else (aa + block_size)
    positives = set([key for key, value in Static.sentiment_word.items() if value == 1])
    negatives = set([key for key, value in Static.sentiment_word.items() if value == -1])
    out = open(pickle_content+"seed_pos_neg_%d"%part_count, "w", encoding="utf8")
    for i in range(aa, bb):
        pos_and_num(i, pickle_content, positives, negatives, out)
    out.close()
    print("end")
