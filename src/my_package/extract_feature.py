# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
from my_package.scripts import load_pickle_file, return_none, save_pickle_file, save_json_file, create_content
from nltk.corpus import stopwords
import os
from collections import Counter
from my_package.class_define import Static
import re
from my_package.process_test_data import extract_test_feature_vector
import sys, getopt
from timeit import timeit
import cProfile


#  def filter_word(word):
    #  '''判断是否是单词

    #  keyword argument:

    #  word -- 输入单词

    #  return 如果是单词，返回True，否则，返回False

    #  '''
    #  word_pattern = r"^\W+$"

    #  if re.search(word_pattern, word) == None:
        #  return True
    #  return False


#  def get_n_gram(sent, k):
    #  '''返回句子的k-gram 元组序列

     #  argument keyword:

    #  sent -- 句子的列表
    #  k -- k gram

    #  '''
    #  for i in range(len(sent) - k + 1):
        #  string_list = [e.lower() for e in sent[i:i+k]]
        #  yield " ".join(string_list)


#  def is_stopword(word):
    #  '''判断是否是停用词

    #  keyword argument:

    #  word -- 输入单词

    #  return 如果是停用词，返回True，否则，返回False

    #  '''
    #  stops = set(stopwords.words('english'))
    #  if word in stops:
        #  return True
    #  return False

#  def create_unigram_lexicon(sentences, word_count, pos_tag_count):
    #  ''''''
    #  i = 1
    #  for sentence in sentences:
        #  if i % 100 == 0:
            #  print("i=", i)
        #  i += 1
        #  for k in range(1, len(sentence.tokens)+1):
            #  unigram_word = sentence.tokens[k].lower()
            #  unigram_pos_tag = sentence.pos_tag[k].lower()
            #  #  if is_stopword(unigram_word):
                #  #  continue
            #  if not filter_word(unigram_word):
                #  continue
            #  word_count[unigram_word] += 1
            #  pos_tag_count[unigram_pos_tag] += 1

#  def create_lexicon(filepath, b, e, k=1):

    #  unigram_word_count, bigram_word_count = Counter(), Counter()
    #  unigram_pos_tag_count, bigram_pos_tag_count = Counter(), Counter()
    #  i = b
    #  filename = filepath + "bootstrap_sentences_" + str(i) + ".pickle"
    #  while i < e and os.path.exists(filename):
        #  print(filename)
        #  print("loading...")
        #  sentences = load_pickle_file(filename)
        #  print("loaded...")
        #  create_unigram_lexicon(sentences, unigram_word_count, unigram_pos_tag_count)
        #  i += 1
        #  filename = filepath + "bootstrap_sentences_" + str(i) + ".pickle"
    #  unigram_word_set = set(key for key, value in unigram_word_count.items() if value >= k)
    #  unigram_pos_tag_set = set(key for key, value in unigram_pos_tag_count.items() if value >= k)
    #  #bigram_word_set = set(key for key, value in bigram_word_count if value >= k)
    #  #bigram_pos_tag_set = set(key for key, value in bigram_pos_tag_count if value >= k)

    #  save_json_file(r"unigram_word_count.json", unigram_word_count)
    #  save_json_file(r"unigram_pos_tag_count.json", unigram_pos_tag_count)
    #  ret_lexicon = {}
    #  ret_lexicon['unigram'] = {"word":{key : value for value, key in enumerate(unigram_word_set)},
                              #  "pos_tag":{key : value for value, key in enumerate(unigram_pos_tag_set)}}
    #  #ret_lexicon['bigram'] = {"word":{key : value for value, key in enumerate(bigram_word_set)},
    #  #                          "pos_tag":{key : value for value, key in enumerate(bigram_pos_tag_set)}}
    #  return ret_lexicon

def train_feature_solve(field_content, lexcion, sentences, score_pp_set, sentiments):
    ''''''
    it = 0
    for sentence in sentences:
        #  if it != 62:
            #  it += 1
            #  continue
        sentence.generate_candidate(sentiments, score_pp_set)
        sentence.generate_candidate_feature_vector(lexcion)
        sentence.generate_train_label()
        if it % 1000 == 0:
            print(it)
        it += 1

def extract_feature_vector(sentences, lexcion, f, g):

    k = 0
    for sentence in sentences:
        print("{0}:{1}".format(k, sentence.text), file=g)
        if k % 1000 == 0:
            print("k=", k)
        k += 1
        sentence.generate_feature_vector(lexcion)
        for i in range(len(sentence.feature_vector)):
            print(sentence.all_match_label[i], end="", file=f)
            for e in sentence.feature_vector[i]:
                print(" {0}:1".format(e), end="", file=f)
            print(file=f)
            feat, sent = sentence.candidate_pairs[i][0], sentence.candidate_pairs[i][1]
            if sentence.all_match_label[i] != 0:
                print("{0}\t\t{1}".format(
                sentence.get_phrase(feat).lower(),
                sentence.get_phrase(sent).lower()),
                # sentence.all_match_label[i]),
                file=g)
            #  print("{0}\t\t{1}\t\t{2}".format(
                #  sentence.get_phrase(feat).lower(),
                #  sentence.get_phrase(sent).lower(),
                #  sentence.all_match_label[i]), file=g)
        print(file=g)

def get_count(str1, str2, field_content):
    c = 0
    with open(field_content+"text", "r", encoding="utf8") as f:
        for line in f:
            if re.search(str1, line) and re.search(str2, line):
                c += 1
    return c

def usage():
    '''打印帮助信息'''
    print("extract_feature.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")
    print("-b, --begin: bootstrap_sentences pickel 文件的开始编号(包含此文件)")
    print("-e, --end: bootstrap_sentences pickel 文件的结束编号(不包含此文件)")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:", ["help", "domain=", "begin=", "end="])
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
        if op in ("-b", "--begin"):
            b = int(value)
        if op in ("-e", "--end"):
            e = int(value)
    field_content = r"../../data/domains/" + content + r"/"
    score_pp_dict = load_pickle_file(field_content + r"pickles/score_pp.pickle")
    score_pp_set = set(score_pp_dict.keys())
    create_content(field_content + r"train")
    create_content(field_content + r"pickles/feature_vectors")
    lexcion = {"unigram":
               {"word":{},
                "pos_tag":{},
                 "dep":{}}}
    sentiments = load_pickle_file(field_content + r"pickles/sentiments.pickle")
    i = b
    while i < e and os.path.exists(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle.bz2"):
        print("loading")
        sentences = load_pickle_file(field_content + r"pickles/bootstrap_sentences/bootstrap_sentences_" + str(i) + ".pickle")
        print("loaded")
        train_feature_solve(field_content, lexcion, sentences, score_pp_set, sentiments)
        save_pickle_file(field_content + r"pickles/feature_vectors/sentences_" + str(i) +".pickle", sentences)
        i += 1

    save_pickle_file(field_content + "pickles/lexicon.pickle", lexcion)
    save_json_file(field_content + "pickles/lexicon.json", lexcion)
    print("word:", len(lexcion['unigram']['word']))
    print("pos_tag:", len(lexcion['unigram']['pos_tag']))
    print("dep:", len(lexcion['unigram']['dep']))
    i = b
    f = open(field_content + r"train/raw_all_match_feature_vectors", mode="w", encoding="utf8")
    g = open(field_content + r"train/train_candidate", mode="w", encoding="utf8")
    while i < e and os.path.exists(field_content + r"pickles/feature_vectors/sentences_" + str(i) + ".pickle.bz2"):
        sentences = load_pickle_file(field_content + r"pickles/feature_vectors/sentences_" + str(i) + ".pickle")
        extract_feature_vector(sentences, lexcion, f, g)
        i += 1
    f.close()
    g.close()
    extract_test_feature_vector(content, score_pp_set, sentiments)
    print("end")
