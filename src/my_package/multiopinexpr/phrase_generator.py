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
        if sentence.tokens[idx_opinwd[0]] in Static.BE:
            continue
        if len(idx_opinwd) == 1 and re.search(r"^\W*$", word):
            continue
        if re.search(r"^[-_>< #'+]*$", word):
            continue
        #  if len(idx_opinwd) > 20:
            #  continue
        #  if sentence.is_weak_opinwd(word):
            #  continue
        #  mark = False
        #  for token in word.split(' '):
            #  if re.search(r"^[^a-zA-Z0-9_$&%]*$", token):
                #  mark = True
                #  break
        #  if mark:
            #  continue
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
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")
    docs_dir = os.path.join(domain_dir, "docs")

    print("DOMAIN: %s" % domain)
    ### 为全领域生成 index
    #  print("######  GENERATE INDEX IN ALL DOMAIN ######")
    #  index(domain)

    ### 生成 queries 以及统计词频率
    #  print("######  GENERATE PHEASES  ######") 
    #  f = open(os.path.join(docs_dir, "phrases-0"), "w", encoding="utf8")
    #  i = 1
    #  filename = os.path.join(pickle_parse_dir, "parse_sentences_%d.pickle" % i)
    #  while os.path.exists(filename + ".bz2"):
        #  print("pickle index: % d  loading" % i)
        #  sentences = load_pickle_file(filename)
        #  print("pickle index: % d  loaded" % i)
        #  mark = set()
        #  for sentence in sentences:
            #  for phrase in get_all_phrase(sentence):
                #  if phrase not in mark:
                    #  print(phrase, file=f)
                    #  mark.add(phrase)
        #  i += 1
        #  filename = os.path.join(pickle_parse_dir,
                                #  "parse_sentences_%d.pickle" % i)
        #  break
    #  f.close()

    ### 统计全领域出现词频
    #  print("######  PHRASES QUERIES  ######")
    #  remove(os.path.join(docs_dir, "phrases_queries_result"))
    #  f = open(os.path.join(docs_dir, "phrases-0"), "r", encoding="utf8")
    #  g = open(os.path.join(docs_dir, "queries"), "w", encoding="utf8")
    #  i = 0
    #  for line in f:
        #  print(line, end="", file=g)
        #  i += 1
        #  if i % 10000 == 0:
            #  g.close()
            #  search(domain, "queries_phrases_result")
            #  print("NUM: %d" % i)
            #  g = open(os.path.join(docs_dir, "queries"), "w", encoding="utf8")
    #  f.close()
    #  g.close()
    #  search(domain, "queries_phrases_result")
    #  print("NUM: %d" % i)

    ###  根据 phrase 出现次数过滤 ###
    #  print("######  过滤出现次数  ######")
    #  parse_queries_result(os.path.join(docs_dir, "queries_phrases_result"),
                         #  os.path.join(docs_dir, "phrases-1_freq"))
    #  vocab = set()
    #  f = open(os.path.join(docs_dir, "phrases-1_freq"), "r", encoding="utf8")
    #  g = open(os.path.join(docs_dir, "queries"), "w", encoding="utf8")
    #  h = open(os.path.join(docs_dir, "phrases-2_freq_filter"), "w", encoding="utf8")
    #  for line in f:
        #  phrase, num = line.strip().split('\t')
        #  if int(num) >= 20:
            #  print(line, end="", file=h)
            #  for token in phrase.split(' '):
                #  vocab.add(token)
    #  for token in vocab:
        #  print(token, file=g)
    #  h.close()
    #  g.close()
    #  f.close()
    #  print("######  VOCABULARY QUERIES  ######")
    #  remove(os.path.join(docs_dir, "tokens_queries_result"))
    #  search(domain, "tokens_queries_result")
    #  parse_queries_result(os.path.join(docs_dir, "queries_tokens_result"),
                         #  os.path.join(docs_dir, "tokens_freq"))

    ###  PMI  ###
    word_freq = {}
    f = open(os.path.join(docs_dir, "tokens_freq"), "r", encoding="utf8")
    for line in f:
        word, freq = line.strip().split('\t')
        word_freq[word] = int(freq)
    f.close()
    n = 0
    f = open(os.path.join(docs_dir, "doc", "text"), "r", encoding="utf8")
    for line in f:
        n += len(line.strip().split('\t')[-1].split(' '))
    f.close()
    f = open(os.path.join(docs_dir, "phrases-2_freq_filter"),
             "r", encoding="utf8")
    g = open(os.path.join(docs_dir, "phrases-3_general"),
             "w", encoding="utf8")
    h = open(os.path.join(docs_dir, "phrases-3_without_general"),
             "w", encoding="utf8")
    for line in f:
        phrase, phrase_fq = line.strip().split('\t')
        phrase_fq = int(phrase_fq)
        tokens = phrase.split(' ')
        if len(tokens) == 1:
            continue
        has_general = False
        for token in tokens:
            if token in Static.opinwd:
                has_general = True
        if has_general:
            t = g
        else:
            t = h
        score = 1
        for token in tokens:
            score *= phrase_fq / (word_freq[token] - phrase_fq)
        print("%s\t%f" % (phrase, score), file=t)
    h.close()
    g.close()
    f.close()
