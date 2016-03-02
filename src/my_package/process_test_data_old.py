# -*- coding: utf-8 -*-
'''
Created on 2015年9月1日

@author: Changzhi Sun
'''
import os
from my_package.class_define import Sentence, Static
from my_package.scripts import create_content, save_pickle_file,\
    load_pickle_file, return_none
import re
import xml.etree.ElementTree as etree
from collections import defaultdict
import sys, getopt


def process_raw(content):
    patt = re.compile(r"^A\|R\|O\|(\d+)\|(\d+).*\|T\|(\d+)\|(\d+)")
    field_content = r"../../data/domains/" + content + r"/"
    create_content(field_content + r"test")
    sentences = []
    i, k = 0, 1
    fout = open(field_content+r"sentences/test_1.txt", "w", encoding="utf8")
    for test_file in os.listdir(field_content+r"test"):
        fin = open(field_content+r"test/" + test_file, "r", encoding="utf8")
        for line_string in fin:
            line = line_string.strip()
            line = re.sub(r'[\x01-\x1f]','', line)
            if line == "":
                continue
            if line[0] == "S" and len(line) > 2:
                tmp_line = line.replace("(", "-LRB-")
                tmp_line = tmp_line.replace(")", "-RRB-")
                sentence = Sentence()
                sentence.set_text_score_review(tmp_line[2:], 5, -1)
                sentence.feature_sentiment = []
                sentence.fs_dict = {}
                sentences.append(sentence)
                print(tmp_line[2:], file=fout)
                i += 1
                if i == 6000:
                    i = 0
                    k += 1
                    fout.close()
                    fout = open(field_content + r"sentences/test_" + str(k) + ".txt", "w", encoding="utf8")
            elif line[0] == "A" and line[2] == "R":
                res = patt.search(line).groups()
                if res != None:
                    feature_set = tuple(range(int(res[2])+1, int(res[3])+1))
                    sentiment_set = tuple(range(int(res[0])+1, int(res[1])+1))
                    tmp_tuple = (feature_set,sentiment_set, 1)
                    sentence.feature_sentiment.append(tmp_tuple)
                    sentence.fs_dict[(feature_set,sentiment_set)] = 1
        fin.close()
    fout.close()
    save_pickle_file(field_content+r"pickles/test_without_parse_sentences.pickle", sentences) 

def process_parse(content):
    field_content = r"../../data/domains/" + content + r"/"
    sentences = load_pickle_file(field_content+r"pickles/test_without_parse_sentences.pickle")
    index_of_sentences, k = 1, 0
    parse_file = field_content + "parse/test_" + str(index_of_sentences) + r".txt.xml"
    while os.path.exists(parse_file):
        tree = etree.parse(parse_file)
        root = tree.getroot()
        for sentence in root[0][0]:
            tokens, pos_tag = defaultdict(return_none), defaultdict(return_none)
            id_index = 1
            for token in sentence[0]:
                tokens[id_index], pos_tag[id_index] = token[0].text, token[3].text
                id_index += 1
            dependency_tree = defaultdict(return_none)
            for d in sentence[2]:
                up, down = int(d[0].attrib['idx']), int(d[1].attrib['idx'])
                if dependency_tree[up] == None:
                    dependency_tree[up] = [{'id': down, 'type': d.attrib['type']}]
                else:
                    temp = {'id': down, 'type': d.attrib['type']}
                    if temp not in dependency_tree[up]:
                        dependency_tree[up].append(temp)
            parse_string = sentence[1].text
            sentences[k].set_parse_info(tokens, pos_tag, parse_string, dependency_tree)
            k += 1
            if k % 100 == 0:
                print(k)
        index_of_sentences += 1
        parse_file = field_content + "parse/test_" + str(index_of_sentences) + r".txt.xml" 
    save_pickle_file(field_content + r"pickles/test_parse_sentences.pickle", sentences[:k])

def test_feature_solve(field_content, lexcion, sentences):
    ''''''
    pass
    

def extract_test_feature_vector(content, score_pp_set, sentiment_dict):
    field_content = r"../../data/domains/" + content + r"/"
    sentences = load_pickle_file(field_content+r"pickles/test_parse_sentences.pickle")
    lexcion = load_pickle_file(field_content + "pickles/lexicon.pickle")
    sentiment_dict = load_pickle_file(field_content + r"pickles/bootstrap_sentiment_dict.pickle")
    
    it = 0
    for sentence in sentences:
        #  sentence.generate_candidate(Static.sentiment_word, score_pp_set)
        sentence.generate_candidate(sentiment_dict, score_pp_set)
        sentence.generate_candidate_feature_vector(lexcion, test=True)
        sentence.generate_test_label()
        if it % 100 == 0:
            print(it)
        it += 1
    save_pickle_file(field_content + r"pickles/test_feature_vector_sentences.pickle", sentences)
    create_content(field_content + "results")
    f = open(field_content + r"results/test_feature_text.txt", "w", encoding="utf8")
    out = open(field_content + r"feature_vectors/test_feature_vectors.txt", mode="w", encoding="utf8")
    for sentence in sentences:
        sentence.generate_feature_vector(lexcion)
        print(sentence.text, file=f)
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
    #  process_raw(content)
    #  process_parse(content)
    field_content = r"../../data/domains/" + content + r"/"
    score_pp_dict = load_pickle_file(field_content + r"pickles/score_pp.pickle")
    score_pp_set = set(score_pp_dict.keys())
    sentiment_dict = load_pickle_file(field_content + r"pickles/bootstrap_sentiment_dict.pickle")

    extract_test_feature_vector(content, score_pp_set, sentiment_dict)
    
