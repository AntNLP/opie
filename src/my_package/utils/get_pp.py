# -*- coding: utf-8 -*-
'''
Created on 2015年8月29日

@author: Changzhi Sun
'''
import os
import sys
import getopt
import re
import random

import kenlm
import pymysql
from pybloom import ScalableBloomFilter

from my_package.class_define import Static
from my_package.scripts import load_json_file, load_pickle_file
from my_package.scripts import create_content, save_pickle_file, save_json_file


def usage():
    '''打印帮助信息'''
    print("get_pp.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")


def execute(connection, sql):
    try:
        # 游标
        with connection.cursor() as cursor:
            cursor.execute(sql)
        connection.commit()
    except Exception as err:
        print(err)
    finally:
        pass


def inquire_word(connection, inquir_list):

    try:
        # 游标
        with connection.cursor() as cursor:
            for key in inquir_list:
                sql = "select * from lm_db where content=\"{0}\"".format(key)
                cursor.execute(sql)
                res = cursor.fetchall()
    except Exception as err:
        print(err)
    finally:
        pass


def get_pp_condition(sentence, var, pos_tag, res_dict, model):
    for key, values in var.items():
        if sentence.pos_tag[key] not in pos_tag:
            continue
        for value in values:
            if len(value) > 15:
                continue
            tmp = sentence.get_phrase(value).lower()
            if len(tmp) > 225:
                continue
            if re.search(r"^\W*$", tmp):
                continue
            if tmp in res_dict:
                continue
            tmp_sc = model.score(tmp)
            res_dict[tmp] = tmp_sc


def get_pp_sentences(sentences, model):

    res_dict ={}
    for sentence in sentences:
        if "dictionary_of_adjp" in dir(sentence):
            adjp = sentence.dictionary_of_adjp
        else:
            adjp = sentence.get("ADJP")
        get_pp_condition(sentence,
                         sentence.dictionary_of_np, Static.NN, res_dict, model)
        get_pp_condition(sentence,
                         sentence.dictionary_of_vp, Static.VB, res_dict, model)
        get_pp_condition(sentence, adjp, Static.JJ, res_dict, model)
        for key, value in sentence.pos_tag.items():
            if value in Static.SENTIMENT:
                tmp = sentence.tokens[key].lower()
                if re.search(r"^\W*$", tmp):
                    continue
                if len(tmp) > 225:
                    continue
                if tmp in res_dict:
                    continue
                tmp_sc = model.score(tmp)
                res_dict[tmp] = tmp_sc
    return res_dict

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
    field_content = r"../../data/domains/" + content + r"/"
    print(content)
    #  connection = pymysql.connect(host="127.0.0.1",
                                #   user="root",
                                #   passwd="100704048",
                                #   local_infile=True,
                                #   max_allowed_packet=1024*1024*1024,
                                #   db=content,
                                #   charset="utf8",
                                #   cursorclass=pymysql.cursors.DictCursor)
    model = kenlm.LanguageModel('../../data/domains/' +
                                content + r'/lm/text.arpa')
    i = 1
    filename = (field_content + "pickles/parse_sentences/parse_sentences_" +
                str(i) + ".pickle")
    f = open(field_content+"lm/data.txt", "w", encoding="utf8")
    while os.path.exists(filename+".bz2"):
        print(filename)
        print("loading...")
        sentences = load_pickle_file(filename)
        print("loaded")
        res_dict = get_pp_sentences(sentences, model)
        for key, value in res_dict.items():
            print("NULL\t{0}\t{1}".format(key, value), file=f)
        i += 1
        filename = (field_content +
                    "pickles/parse_sentences/parse_sentences_" +
                    str(i) + ".pickle")


    sbf = ScalableBloomFilter(initial_capacity=150000000,
                              mode=ScalableBloomFilter.LARGE_SET_GROWTH)
    out = open(field_content + "lm/data.new", "w", encoding="utf8")
    with open(field_content + "lm/data.txt", "r", encoding="utf8") as f:
        for line in f:
            w1, w2, w3 = line.strip().split('\t')
            if w2 not in sbf:
                sbf.add(w2)
                print("{0}\t{1}\t{2}".format(w1, w2, w3), file=out)
    out.close()

    #  print("insert db")
    #  path = '"/home/zhi/Project/sentiment_relation_extraction_new_data/data/domains/{0}/lm/data.new"'.format(content)
    #  sql = "load data local infile "+path+ " into table lm_db fields escaped by ''"
    #  print(sql)
    #  execute(connection, sql)
    #  print("insert end")

    #  print("create index")
    #  sql = "alter lm_db add index on (content)"
    #  print(sql)
    #  execute(connection, sql)
    #  connection.close()
    print("end")
