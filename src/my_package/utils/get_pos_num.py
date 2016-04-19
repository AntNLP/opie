# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
import re
import sys
import getopt
import numpy as np
from collections import Counter
from itertools import chain

import pymysql

from my_package.scripts import save_pickle_file, save_json_file
from my_package.scripts import create_content, load_json_file
from my_package.scripts import load_pickle_file, return_none
from my_package.class_define import Static


def get_sentiment_count(sentence, sentiments):
        sentiment_count = 0
        i = 1
        while sentence.tokens.get(i) != None:
            if sentence.tokens[i].lower() in sentiments:
                sentiment_count += 1
            i += 1
        return sentiment_count


def inquire_lm(connection, var, table_name):
    try:
        # 游标
        with connection.cursor() as cursor:
            sql = "select * from {0} where content=\"{1}\"".format(table_name,
                                                                   var)
            cursor.execute(sql)
            res = cursor.fetchall()
            if len(res) == 0:
                return None
            else:
                return res[0]["id"]
    except Exception as err:
        print(err)
        return None
    finally:
        pass


def insert_posting(tmp, sent_index, connection, i, k, f, b, e, g, table_lm):
    if re.search(r"^\W*$", tmp):
        return
    if len(tmp) > 225:
        return
    if tmp in sent_index:
        i_sent = sent_index[tmp]
    else:
        i_sent = inquire_lm(connection, tmp, table_lm)
        if i_sent != None:
            sent_index[tmp] = i_sent
        else:
            return
    if (b, e, i_sent) not in g:
        print("{0}\t{1}\t{2}\t{3}\t{4}".format(i,k,b,e,i_sent),file=f)
        g.add((b, e, i_sent))


def get_dict_pos(connection, sentence,
                 dict_pp, pos_tag, sent_index, i, k, f, g, table_lm):
    for key, values in dict_pp.items():
        if sentence.pos_tag[key] in pos_tag:
            for value in values:
                if len(value) > 15:
                    continue
                tmp = sentence.get_phrase(value).lower()
                insert_posting(tmp, sent_index, connection,
                               i, k, f,value[0], value[-1]+1, g, table_lm)


def pos_and_num(i, pickle_content, sentiments, out, connection, f, table_lm):
    filename = (pickle_content +
                "parse_sentences/parse_sentences_" + str(i) + ".pickle")
    sent_index = {}
    if os.path.exists(filename+".bz2"):
        sentences = load_pickle_file(filename)
        print(filename)
        k = 0
        for sentence in sentences:
            if k % 1000 == 0:
                print(k)
            print("%d\t%d\t%d\t%d\t"%(
                i, k, get_sentiment_count(sentence, sentiments),
                sentence.review_index), file=out)
            if "dictionary_of_adjp" in dir(sentence):
                adjp = sentence.dictionary_of_adjp
            else:
                adjp = sentence.get("ADJP")
            g = set()
            get_dict_pos(connection, sentence, adjp,
                         Static.JJ, sent_index, i, k, f, g, table_lm)
            get_dict_pos(connection, sentence, sentence.dictionary_of_vp,
                         Static.VB, sent_index, i, k, f, g, table_lm)
            get_dict_pos(connection, sentence, sentence.dictionary_of_np,
                         Static.NN, sent_index, i, k, f, g, table_lm)
            for key, value in sentence.pos_tag.items():
                if value in Static.SENTIMENT:
                    tmp = sentence.tokens[key].lower()
                    insert_posting(tmp, sent_index, connection,
                                   i, k, f, key, key+1, g, table_lm)
            k += 1
    else:
        print(filename + "not exists!")


def usage():

    '''打印帮助信息'''
    print("get_pos_num.py 用法:")
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
    connection = pymysql.connect(host="console",
                                 user="u20130099",
                                 passwd="u20130099",
                                 db="u20130099",
                                 charset="utf8",
                                 cursorclass=pymysql.cursors.DictCursor)
    pickle_content = r"../../data/soft_domains/" + content + r"/pickles/"
    pickle_size = len(os.listdir(pickle_content + "without_parse_sentences"))
    block_size = int(pickle_size / 8)
    aa = (part_count - 1) * block_size + 1
    bb = (pickle_size + 1) if part_count == 8 else (aa + block_size)
    table_lm = content+"_lm"

    sentiment_dict = dict(Static.sentiment_word)
    sentiments = set(sentiment_dict.keys())
    out = open(pickle_content+"seed_sent_num_%d"%part_count,
               "w", encoding="utf8")
    f = open(pickle_content+"posting_"+str(part_count)+".txt",
             "w", encoding="utf8")
    for i in range(aa, bb):
        pos_and_num(i, pickle_content,
                    sentiments, out, connection, f, table_lm)
    f.close()
    out.close()
    connection.close()
    print("end")
