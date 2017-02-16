#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/10/09 16:50:10

@author: Changzhi Sun
"""
import getopt
import sys
import os
import xml.etree.ElementTree as etree
from my_package.static import Static

def parse_text(in_dir, out_dir):
    old_pwd = os.getcwd()
    pwd_path = os.path.join(os.getenv("OPIE_DIR"),
                            "tools/stanford-corenlp-full-2014-08-27")
    os.chdir(pwd_path)
    cmd_string = "java -cp "
    cmd_string += "stanford-corenlp-3.4.1.jar"
    cmd_string += ":stanford-corenlp-3.4.1-models.jar"
    cmd_string += ":xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx3g "
    cmd_string += "edu.stanford.nlp.pipeline.StanfordCoreNLP "
    cmd_string += "-ssplit.eolonly true "
    cmd_string += "-tokenize.whitespace true "
    cmd_string += "-annotators tokenize,ssplit,pos,parse -file "
    cmd_string += in_dir
    cmd_string += " -outputDirectory " + out_dir + " -noClobber"
    cmd_string += " 2>&1 | tee ../summary.log"
    os.system(cmd_string)
    os.chdir(old_pwd)


def dump_phrase_info(filename, f):
    tree = etree.parse(filename)
    root = tree.getroot()
    for sentence in root[0][0]:
        tokens = []
        pos_tag = []
        for token in sentence[0]:
            tokens.append(token[0].text)
            pos_tag.append(token[3].text)
        print("{0}\t{1}".format(" ".join(tokens), " ".join(pos_tag)), file=f)


def usage():
    '''print help information'''
    print("reduce_summary_phrase.py 用法:")
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
    multi_opin_expr_dir = os.path.join(domain_dir, "multiopinexpr")

    with open(os.path.join(multi_opin_expr_dir, "summary_sort_by_freq"), "r", encoding="utf8") as f:
        phrases = []
        for line in f:
            phrases.append(line.split('\t')[0])
    i = 0
    j = 1
    while i < len(phrases):
        with open(os.path.join(multi_opin_expr_dir, "summary", "phrase", "phrase_%d" % j), "w", encoding="utf8") as f:
            for e in phrases[i:i+60000]:
                print(e, file=f)
        i += 60000
        j += 1
    parse_text(os.path.join(multi_opin_expr_dir, "summary", "phrase"), os.path.join(multi_opin_expr_dir, "summary", "parse"))


    with open(os.path.join(multi_opin_expr_dir, "summary", "summary_phrase_info"), "w", encoding="utf8") as f:
        i = 1
        filename = os.path.join(multi_opin_expr_dir, "summary", "parse", "phrase_%d.xml" % i)
        while os.path.exists(filename):
            dump_phrase_info(filename, f)
            i += 1
            filename = os.path.join(multi_opin_expr_dir, "summary", "parse", "phrase_%d.xml" % i)


    POSTAG = Static.NOUN | set(["CC", "LS", "DT"])
    with open(os.path.join(multi_opin_expr_dir, "summary", "summary_phrase_info"), "r", encoding="utf8") as f:
        with open(os.path.join(multi_opin_expr_dir, "summary", "summary_phrase.remove.nn"), "w", encoding="utf8") as g:
            for line in f:
                phrase, pos_tag = line.strip().split('\t')
                flag = False
                for w in pos_tag.split(' '):
                    if w not in POSTAG:
                        flag = True
                if flag:
                    print(line, end="", file=g)
                else:
                    for e1, e2 in zip(phrase.split(' '), pos_tag.split(' ')):
                        if e2 == 'DT' and e1 == 'no':
                            print(line, end="", file=g)
                            break
    with open(os.path.join(multi_opin_expr_dir, "summary", "summary_phrase.remove.nn"), "r", encoding="utf8") as f:
        with open(os.path.join(multi_opin_expr_dir, "summary", "summary_multiopinexpr.remove.nn"), "w", encoding="utf8") as g:
            for line in f:
                phrase, pos_tag = line.strip().split('\t')
                if len(phrase.split(' ')) == 1:
                    continue
                if pos_tag.find("NN CD") != -1:
                    continue
                print(line, end="", file=g)
