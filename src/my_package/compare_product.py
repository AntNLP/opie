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

def get_sentiment(sentence, wrod_list, sentiments, connection, product_word, all_word, table_name, w=5):
    n = len(sentence.pos_tag)
    i = 1
    mark = False
    while i in sentence.tokens:
        j, k = i, 0
        while k < len(word_list) and sentence.tokens[j] == word_list[k]:
            j += 1
            k += 1
        if k == len(word_list):
            mark = True
            obj_index = list(range(i, j))
            b = i - w if i - w >= 1 else 1
            e = j + w if j + w <= n + 1 else n + 1
            for m in range(b, i):
                if sentence.pos_tag[m] in Static.VB:
                    vp = sentence.get_vp(m, i)
                    if len(vp) == 1:
                        continue
                    if filter_word(sentence, vp):
                        continue
                    new_sentiment_string = sentence.get_phrase(vp).lower()

                    #  语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in vp])


                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])

                    adjp = sentence.get_min_adjp(m, obj_index)
                    new_sentiment_string = sentence.get_phrase(adjp).lower()
                    if len(adjp) == 1:
                        continue

                    if filter_word(sentence, adjp):
                        continue
                    # 语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])

            for m in range(j, e):
                if sentence.pos_tag[m] in Static.VB:
                    vp = sentence.get_vp(m, j)
                    if len(vp) == 1:
                        continue
                    if filter_word(sentence, vp):
                        continue
                    new_sentiment_string = sentence.get_phrase(vp).lower()

                    #  语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in vp])

                if sentence.pos_tag[m] in Static.JJ:
                    adjp = sentence.get_max_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
                    adjp = sentence.get_min_adjp(m, obj_index)
                    if len(adjp) == 1:
                        continue
                    if filter_word(sentence, adjp):
                        continue
                    new_sentiment_string = sentence.get_phrase(adjp).lower()

                    # 语言模型过滤
                    if inquire_content(connection, new_sentiment_string, table_name):
                        if new_sentiment_string not in product_word:
                            product_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
        i += 1
    if mark:
        for i in range(1, len(sentence.pos_tag)+1):
            if i not in sentence.tokens:
                continue
            if sentence.pos_tag[i] in Static.VB:
                vp = sentence.get_vp(i)
                if len(vp) == 1:
                    continue
                if filter_word(sentence, vp):
                    continue
                new_sentiment_string = sentence.get_phrase(vp).lower()

                #  语言模型过滤
                if inquire_content(connection, new_sentiment_string, table_name):
                    if new_sentiment_string not in all_word:
                        all_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in vp])

            if sentence.pos_tag[i] in Static.JJ:
                adjp = sentence.get_max_adjp(i)
                if len(adjp) == 1:
                    continue
                if filter_word(sentence, adjp):
                    continue
                new_sentiment_string = sentence.get_phrase(adjp).lower()

                # 语言模型过滤
                if inquire_content(connection, new_sentiment_string, table_name):
                    if new_sentiment_string not in all_word:
                        all_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
                adjp = sentence.get_min_adjp(i)
                if len(adjp) == 1:
                    continue
                if filter_word(sentence, adjp):
                    continue
                new_sentiment_string = sentence.get_phrase(adjp).lower()

                # 语言模型过滤
                if inquire_content(connection, new_sentiment_string, table_name):
                    if new_sentiment_string not in all_word:
                        all_word[new_sentiment_string] = " ".join([sentence.pos_tag[e] for e in adjp])
    return int(mark)

def usage():

    '''打印帮助信息'''
    print("compare_product.py 用法:")
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
    field_content = r"../../data/domains/" + content + r"/"
    create_content(field_content + "near")
    sentiment_dict = dict(Static.sentiment_word)
    sentiments = set(sentiment_dict.keys())
    word_list = ["product"]
    connection = pymysql.connect(host="127.0.0.1",
                                user="root",
                                passwd="100704048",
                                db=content,
                                charset="utf8",
                                cursorclass=pymysql.cursors.DictCursor)
    table_name = content + "_lm"
    f = open(field_content+"near/product_near", "w", encoding="utf8")
    g = open(field_content+"near/all_near", "w", encoding="utf8")
    sents = []
    i = 1
    k = 0
    filename = field_content + "pickles/parse_sentences/parse_sentences_%d.pickle"%i
    product_word, all_word = {}, {}
    while os.path.exists(filename+".bz2"):
        print(filename)
        sentences = load_pickle_file(filename)
        for sentence in sentences:
            k += get_sentiment(sentence, word_list, sentiments, connection, product_word, all_word, table_name)
            if k == 1000:
                break
        if k == 1000:
            break
        i += 1
        filename = field_content + "pickles/parse_sentences/parse_sentences_%d.pickle"%i
    for key, value in product_word.items():
        print("%s\t%s"%(key, value), file=f)
    for key, value in all_word.items():
        print("%s\t%s"%(key, value), file=g)
    connection.close()
    f.close()
    g.close()
