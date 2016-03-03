# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
from my_package.scripts import load_pickle_file, return_none, save_pickle_file, save_json_file, create_content
import os
from collections import Counter
from itertools import chain
from my_package.class_define import Static
import re
import sys, getopt
import pymysql

class Node:
    def __init__(self):
        self.count = 0
        self.next_node = {}

class Trie_Tree:
    def __init__(self):
        self.root = Node()

    def build_trie_tree(self, word_list):

        if word_list == []:
            return

        p = self.root
        for word in word_list:
            if p.next_node.get(word) == None:
                q = Node()
                p.next_node[word] = q
            p = p.next_node[word]
        p.count += 1

    def build_suffix_tree(self, word_list):
        for i in range(len(word_list)):
            self.build_trie_tree(word_list[i:])

    def inquire_word_string_count(self, word_list):
        p = self.root
        for word in word_list:
            if p.next_node.get(word) == None:
                return 0
            p = p.next_node[word]
        return p.count

def inquire_content(connection, var, table_name, t=-25):
    try:

        # 游标
        with connection.cursor() as cursor:
            sql = "select * from {0} where content=\"{1}\" and score>={2}".format(table_name, var, t)
            #  sql = "select * from lm_db where content=%s"
            #  sql = "select * from lm_db where content=\"{0}\"".format(var)
            #  cursor.execute(sql, (var))
            cursor.execute(sql)
            res = cursor.fetchall()
            if len(res) == 0:
                return False
            else:
                return True

    except Exception as err:
        print(err)
        print(var)
        return False
    finally:
        pass

def filter_word(sentence, pp):
    for i in pp:
        if re.search(r"^\W*$", sentence.tokens[i]) != None:
            return True
    return False

def get_sentiment(sentence, wrod_list, sentiments, connection, complex_word_pos, table_name, w=5):
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
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in vp])


                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])

                    adjp = sentence.get_min_adjp(m, obj_index)
                    new_sentiment_string = sentence.get_phrase(adjp).lower()
                    if len(adjp) == 1:
                        continue

                    if filter_word(sentence, adjp):
                        continue
                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])

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
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in vp])

                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
                    adjp = sentence.get_min_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    #  if new_sentiment_string in score_pp_set:
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in complex_word_pos:
                            complex_word_pos[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
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
    word_list = ["product"]
    connection = pymysql.connect(host="127.0.0.1",
                                user="u20130099",
                                passwd="u20130099",
                                db="u20130099",
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    table_name = content + "_lm"
    f = open(field_content+"near/sentiment_word_near", "w", encoding="utf8")
    i = 1
    filename = field_content + "pickles/parse_sentences/parse_sentences_%d.pickle"%i
    complex_word_pos = {}
    while os.path.exists(filename+".bz2"):
        print(filename)
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            get_sentiment(sentence, word_list, sentiments, connection, complex_word_pos, table_name)
        if i == 10:
            break
        i += 1
        filename = field_content + "pickles/parse_sentences/parse_sentences_%d.pickle"%i
    for key, value in complex_word_pos.items():
        print("0 %s\t%s"%(key, value), file=f)
    connection.close()
    f.close()
