# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
#  from my_package.class_define import Sentence, Static
from my_package.scripts import create_content, save_pickle_file,load_pickle_file, return_none
import re
import xml.etree.ElementTree as etree
from collections import defaultdict
import sys, getopt

def extract_test_feature_vector(content, score_pp_set, sentiments):
    field_content = r"../../data/domains/" + content + r"/"
    sentences = load_pickle_file(field_content+r"test/test_sentences.pickle")
    lexcion = load_pickle_file(field_content + "pickles/lexicon.pickle")
    it = 0
    for sentence in sentences:
        sentence.generate_candidate(sentiments, score_pp_set, test=True)
        sentence.generate_candidate_feature_vector(lexcion, test=True)
        sentence.generate_test_label()
        if it % 100 == 0:
            print(it)
        it += 1
    save_pickle_file(field_content + r"test/feature_vector_sentences.pickle", sentences)
    f = open(field_content + r"test/test_candidate", "w", encoding="utf8")
    out = open(field_content + r"test/feature_vectors", mode="w", encoding="utf8")
    for sentence in sentences:
        sentence.generate_feature_vector(lexcion)
        print(sentence.text, file=f)
        #  print(sentence.text_and_pos_tag)
        for i in range(len(sentence.feature_vector)):
            print(sentence.candidate_pairs[i], end="   ", file=f)
            print(sentence.get_phrase(sentence.candidate_pairs[i][0]), end="   ", file=f)
            print(sentence.get_phrase(sentence.candidate_pairs[i][1]), file=f)
            print(sentence.all_match_label[i], end="", file=out)
            for e in sentence.feature_vector[i]:
                print(" {}:1".format(e), end="", file=out)
            print("", file=out)
        print("\n", file=f)
    out.close()
    f.close()
        
def usage():
    '''打印帮助信息'''
    print("process_test_data.py 用法:")
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
    field_content = r"../../data/domains/" + content + r"/"
    score_pp_dict = load_pickle_file(field_content + r"pickles/score_pp.pickle")
    score_pp_set = set(score_pp_dict.keys())
    sentiments = load_pickle_file(field_content + r"pickles/sentiments.pickle")
    extract_test_feature_vector(content, score_pp_set, sentiments)