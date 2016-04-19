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
from my_package.scripts import get_index, get_position, filter_word
from my_package.class_define import Static

def get_sentiment(sentence, wrod_list, sentiments,
                  connection, complex_word_pos, table_name, w=5):
    n = len(sentence.pos_tag)
    for i in range(1, n+1):
        j, k = i, 0
        while k < len(word_list) and sentence.tokens[j] == word_list[k]:
            j += 1
            k += 1
        if k == len(word_list):
            obj_index = list(range(i, j))
            b = i - w if i - w >= 1 else 1
            e = j + w if j + w <= n + 1 else n + 1
            for m in range(b, i):
                #  if sentence.pos_tag[m] in Static.NN:
                    #  np = sentence.get_np(m, i)
                    #  if len(np) == 1:
                        #  continue
                    #  f = False
                    #  for x in np:
                        #  if sentence.tokens[x].lower() in sentiments:
                            #  f = True
                            #  break
                    #  if f:
                        #  continue
                    #  new_sentiment_string = sentence.get_phrase(np).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                        #  yield new_sentiment_string

                if sentence.pos_tag[m] in Static.VB:
                    vp = sentence.get_vp(m, i)
                    if len(vp) == 1:
                        continue
                    if filter_word(sentence, vp):
                        continue
                    new_sentiment_string = sentence.get_phrase(vp).lower()

                    #  语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in vp])


                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in adjp])

                    adjp = sentence.get_min_adjp(m, obj_index)
                    new_sentiment_string = sentence.get_phrase(adjp).lower()
                    if len(adjp) == 1:
                        continue

                    if filter_word(sentence, adjp):
                        continue

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in adjp])

            for m in range(j, e):
                #  if sentence.pos_tag[m] in Static.NN:
                    #  np = sentence.get_np(m, j)
                    #  if len(np) == 1:
                        #  continue

                    #  f = False
                    #  for x in np:
                        #  if sentence.tokens[x].lower() in sentiments:
                            #  f = True
                            #  break
                    #  if f:
                        #  continue
                    #  new_sentiment_string = sentence.get_phrase(np).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                        #  yield new_sentiment_string

                if sentence.pos_tag[m] in Static.VB:
                    vp = sentence.get_vp(m, j)
                    if len(vp) == 1:
                        continue
                    if filter_word(sentence, vp):
                        continue
                    new_sentiment_string = sentence.get_phrase(vp).lower()

                    #  语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in vp])

                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in adjp])
                    adjp = sentence.get_min_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection,
                                       new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join(
                                [sentence.pos_tag[e] for e in adjp])


def usage():

    '''打印帮助信息'''
    print("find_near_word.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
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
    word_list = ["product"]
    connection = pymysql.connect(host="console",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    table_name = content + "_lm"
    table_posting = content + "_posting"
    f = open(field_content+"near/sentiment_word_near", "w", encoding="utf8")
    i = 1
    filename = (field_content +
               "pickles/parse_sentences/parse_sentences_%d.pickle"%i)
    complex_word_pos_tag = {}
    while os.path.exists(filename+".bz2"):
        print(filename)
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            get_sentiment(sentence, word_list, sentiments,
                          connection, complex_word_pos_tag, table_name)
        if i == 1:
            break
        i += 1
        filename = (field_content +
                   "pickles/parse_sentences/parse_sentences_%d.pickle"%i)
    save_pickle_file(field_content + "pickles/complex_word_pos_tag.pickle",
                     complex_word_pos_tag)
    word_pickle_sentence = {}
    for word_string, word_pos in complex_word_pos_tag.items():
        word_index = get_index(connection, table_name, word_string)
        if word_index == None:
            continue
        res = get_position(connection, table_posting, word_index)
        res_set = set(((e['i_pickle'], e['i_sentence']) for e in res))
        word_pickle_sentence[word_string] = res_set
        print("0\t%d\t%s\t%s"%(len(res_set), word_string, word_pos), file=f)
    connection.close()
    f.close()
    save_pickle_file(field_content + "pickles/word_pickle_sentence.pickle",
                     word_pickle_sentence)
