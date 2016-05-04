#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 16/04/24 13:15:04

@author: Changzhi Sun
"""
import os
import sys
import getopt


def parse_ann(ann_dir, filename, sent_ann):
    f = open(os.path.join(ann_dir, filename), "r", encoding="utf8")
    for line in f:
        line_strip = line.strip()
        if line_strip.startswith("S"):
            text = line_strip.split('\t')[1]
            if text not in sent_ann:
                sent_ann[text] = set()
        else:
            label = line_strip[-1]
            line_strip = line_strip[:-1].strip()
            items = line_strip.split('\t')
            sent_ann[text].add((items[3], items[4], label))
    f.close()


def convert_offset_begin(text, idx):
    tokens = text.split(' ')
    offset = idx - 1
    for i in range(0, idx-1):
        offset += len(tokens[i])
    return offset


def convert_offset_end(text, idx):
    tokens = text.split(' ')
    return convert_offset_begin(text, idx) + len(tokens[idx-1])


def parse_review(filename, sent_ann, f):
    with open(filename, "r", encoding="utf8") as out:
        lines = out.readlines()
    base = 0
    t = 1
    for line in lines:
        text = line[:-1]
        if text in sent_ann:
            for profeat, opinwd, label in sent_ann[text]:
                #  print(sent_ann[text])
                if label != "1":
                    continue
                profeat, opinwd = eval(profeat), eval(opinwd)
                #  print(text)
                #  print(profeat, opinwd, label)
                b = convert_offset_begin(text, profeat[0])
                e = convert_offset_end(text, profeat[-1])
                #  print(b, e)
                print("T%d\tOpinionTarget %d %d\t%s" % (
                      t, base+b, base+e, text[b:e]), file=f)
                b = convert_offset_begin(text, opinwd[0])
                e = convert_offset_end(text, opinwd[-1])
                #  print(b, e)
                print("T%d\tOpinionExpression %d %d\t%s" % (
                      t+1, base+b, base+e, text[b:e]), file=f)
                print("R%d\tExpressionTarget Arg1:T%d Arg2:T%d"% (
                      t+2, t+1, t), file=f)
                t += 3
        base += len(line)


def usage():
    '''print help information'''
    print("brat_ann_generator.py 用法:")
    print("-h, --help: 打印帮助信息")
    print("-d, --domain: 需要处理的领域名称")

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:e:",
                                   ["help", "domain=", "begin=", "end="])
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
    domain_dir = os.path.join(os.getenv("OPIE_DIR"),
                              "data/domains", domain)
    test_dir = os.path.join(domain_dir, "test")
    review_dir = os.path.join(test_dir, "reviews")
    ann_dir = os.path.join(test_dir, "ann.pre")
    sent_ann = {}
    for filename in os.listdir(ann_dir):
        #  if filename != "bootstrap_test_general_relation.ann":
            #  continue
        parse_ann(ann_dir, filename, sent_ann)
    #  with open("s", "w", encoding="utf8") as f:
        #  for text in sent_ann.keys():
            #  print(text, file=f)
    i = 1
    filename = os.path.join(review_dir, "review_%d.txt" % i)
    while os.path.exists(filename):
        annfile = os.path.join(review_dir, "review_%d.ann" % i)
        f = open(annfile, "w", encoding="utf8")
        parse_review(filename, sent_ann, f)
        f.close()
        i += 1
        filename = os.path.join(review_dir, "review_%d.txt" % i)
