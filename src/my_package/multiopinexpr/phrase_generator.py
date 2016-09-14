#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
iCreated on 16/07/28 10:26:59

@author: Changzhi Sun
"""
import getopt
import sys
import os
import math
import re
import zipfile
import numpy as np

from my_package.scripts import load_pickle_file
from my_package.scripts import save_pickle_file
from my_package.scripts import load_json_file
from my_package.scripts import save_json_file
from my_package.scripts import remove
from my_package.static import Static


def get_k_gram(text, k):
    tokens = []
    tokens.extend(["<S/>"]*(k-1))
    tokens.extend(text.lower().split(' '))
    tokens.extend(["<E/>"]*(k-1))
    for i in range(0, len(tokens)-k+1):
        yield " ".join(tokens[i:i+k])


def get_all_phrase(sentence):
    idx_candidate_opinwd = sentence.generate_candidate_opinwd()
    n = len(sentence.tokens)
    for idx_opinwd in idx_candidate_opinwd:
        word = sentence.print_phrase(idx_opinwd).lower()
        if len(idx_opinwd) == 1:
            continue
        if re.search(r"^[-_>< #'+]*$", word):
            continue
        yield word


def parse_queries_result(queries_file, save_file):
    f = open(queries_file, "r", encoding="utf8")
    g = open(save_file, "w", encoding="utf8")
    i = 1
    for line in f:
        if i % 3 == 1:
            phrase = line[15:-1]
            if phrase[0] == '"':
                phrase = phrase[1:-1]
        elif i % 3 == 2:
            count = int(line.split(' ')[0])
            print("%s\t%d" % (phrase, count), file=g)
        i += 1
    f.close()
    g.close()


def search(domain, save_file):

    cmd = "%s/src/my_package/utils/search.sh %s %s" % (os.getenv("OPIE_DIR"),
                                                       domain,
                                                       save_file)
    os.system(cmd)


def index(domain):

    cmd = "%s/src/my_package/utils/index.sh %s" % (os.getenv("OPIE_DIR"),
                                                   domain)
    os.system(cmd)


def generate_data():
    """Extract the first file enclosed in a zip file as a list of words"""

    f = open(os.path.join(multi_opin_expr_dir, "clean_replace_text"),
             "w", encoding="utf8")
    g = open(os.path.join(multi_opin_expr_dir, "replace_text"),
             "r", encoding="utf8")
    ftext = open(os.path.join(multi_opin_expr_dir, "clean_replace_sentences"),
                 "w", encoding="utf8")
    for line in g:
        text, review_id = line.split('\t')
        tokens = []
        for token in text.split(' '):
            if token in Static.stopwords:
                continue
            if re.search(r"^\W*$", token):
                continue
            tokens.append(token)
            print("%s " % token, end="", file=f)
        print("%s\t%s" % (" ".join(tokens), review_id), end="", file=ftext)
    g.close()
    f.close()
    ftext.close()
    filename = os.path.join(multi_opin_expr_dir, "text.zip")
    with zipfile.ZipFile(filename, "w") as myzip:
        myzip.write(os.path.join(multi_opin_expr_dir, "clean_replace_text"))
    return filename


def usage():
    '''print help information'''
    print("phrase_generator.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            "hd:",
            ["help", "domain="])
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"), "data", "domains", domain)
    pickle_parse_dir = os.path.join(domain_dir, "pickles", "parse_sentences")
    pickle_without_parse_dir = os.path.join(domain_dir,
                                            "pickles",
                                            "without_parse_sentences")
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")
    docs_dir = os.path.join(domain_dir, "docs")

    print("DOMAIN: %s" % domain)
    ### 为全领域生成 index
    #  print("######  GENERATE INDEX IN ALL DOMAIN ######")
    #  index(domain)

    ### 生成 queries 以及统计词频率
    #  fphrase = open(os.path.join(multi_opin_expr_dir, "phrases"),
                   #  "w", encoding="utf8")
    #  freplace = open(os.path.join(multi_opin_expr_dir, "replace_text"),
                    #  "w", encoding="utf8")
    #  fscore = open(os.path.join(multi_opin_expr_dir, "score"),
                  #  "w", encoding="utf8")
    #  i = 1
    #  filename = os.path.join(pickle_parse_dir, "parse_sentences_%d.pickle" % i)
    #  while os.path.exists(filename + ".bz2"):
        #  print("pickle index: % d  loading" % i)
        #  sentences = load_pickle_file(filename)
        #  print("pickle index: % d  loaded" % i)

        #  ######  GERERATE QUERIES  ######
        #  mark = set()
        #  print("######  GENERATE QUERIES  ######")
        #  fquery = open(os.path.join(docs_dir, "queries"), "w", encoding="utf8")
        #  for sentence in sentences:
            #  print(sentence.score, file=fscore)
            #  for phrase in get_all_phrase(sentence):
                #  if phrase not in mark:
                    #  print(phrase, file=fquery)
                    #  mark.add(phrase)
        #  fquery.close()
        #  remove(os.path.join(docs_dir, "queries_phrases_result"))
        #  print("######  SEARCHING  ######")
        #  search(domain, "queries_phrases_result")
        #  parse_queries_result(os.path.join(docs_dir, "queries_phrases_result"),
                            #  os.path.join(docs_dir, "phrases_freq"))

        #  ######  filter by the number of occurence  ######
        #  print("######  FILTERING  ######")
        #  phrases = []
        #  with open(
            #  os.path.join(docs_dir, "phrases_freq"),
            #  "r", encoding="utf8") as f:
            #  for line in f:
                #  phrase, num = line.strip().split('\t')
                #  if int(num) >= 10:
                    #  if len(phrase.split(' ')) > 1:
                        #  phrases.append(phrase)
                        #  print(phrase, file=fphrase)

        #  ######  RELACE TEXT  ######
        #  print("######  REPLACE TEXT  ######")
        #  phrases = sorted(phrases, key=lambda x : len(x.split(' ')), reverse=True)
        #  #  for phrase in phrases:
            #  #  print(phrase)
        #  for sentence in sentences:
            #  text = sentence.text.lower()
            #  for phrase in phrases:
                #  replace_text = text.replace(phrase, "^".join(phrase.split(' ')))
                #  if replace_text != text:
                    #  text = replace_text
            #  print("%s\t%d" % (text, sentence.review_index), file=freplace)
        #  i += 1
        #  filename = os.path.join(pickle_parse_dir,
                                #  "parse_sentences_%d.pickle" % i)
        #  if i == 41:
            #  break
    #  fphrase.close()
    #  freplace.close()
    #  fscore.close()
    generate_data()
